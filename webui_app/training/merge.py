"""出口 · LoRA 权重合并回 gpt.pth / s2mel.pth + 推理端挂载。

两条用 adapter 的路，各有各的场合：

  · **挂载**（`mount_run` / `unmount` / `set_scale`）：引擎上直接挂 LoRA，
    强度旋钮实时可调、随时可卸 —— 试用、A/B、找最佳强度用这条。
    代价：每次推理多一层旁路计算，且分享模型时得连底座一起带。

  · **合并**（`merge_lora_to_checkpoint`）：把 ΔW 烘进权重，产出一份
    **独立的** gpt.pth / s2mel.pth。合出来的文件可以脱离训练目录直接
    用（官方 webui、别的机器、部署脚本）。代价：不可逆、不可调强度 ——
    所以合并前的最后一件事应该是用评测台确认这个强度就是想要的。

**红线**（BaseGuard 兜底）：合并输出**永远不写进 checkpoints/**。
那是底座目录，写进去等于静默替换用户的底座 —— 校验、快照、
「相对底座的改善」全部失去意义。

合并数学：W' = W + scale · (α/r) · B·A（rsLoRA 时 α/√r）。
scale 就是强度旋钮，`merge_and_unload()` 之前先 `set_adapter_scale`
即可半强度合并。Conv1D（GPT2）与 Linear（DiT）的转置差异由
PEFT 的 `get_delta_weight` 统一处理，这里不自己手搓。
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from webui_app.config import PROJECT_ROOT
from webui_app.training import cfm_lora as CL
from webui_app.training import gpt_lora as GL
from webui_app.training import guard as GD
from webui_app.training import runs as RN

__all__ = ["MergeOptions", "MergeReport", "merge_lora_to_checkpoint",
           "mount_run", "unmount", "set_scale", "resolve_mount_dir"]


# ===========================================================================
# 1. 推理端挂载（薄封装：UI 与评测台共用同一套语义）
# ===========================================================================

def resolve_mount_dir(run: str, checkpoint: str = "best") -> Tuple[str, str]:
    """run + 档位 → (adapter 目录, 架构)。与续训/评测同一套解析约定。"""
    rj = RN.read_run(run)
    arch = str(rj.get("arch") or "gpt")
    which = (checkpoint or "best").lower()
    if which in ("best", "final"):
        d = RN.adapter_dir(run)
    else:
        d = os.path.join(RN.checkpoints_dir(run), checkpoint)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"adapter 目录不存在：{d}（run={run}, checkpoint={checkpoint}）")
    return d, arch


def mount_run(engine, run: str, checkpoint: str = "best",
              scale: float = 1.0) -> str:
    """把某个 run 的 adapter 挂到引擎上并设强度。返回 tag。"""
    d, arch = resolve_mount_dir(run, checkpoint)
    tag = engine.attach_lora(d, target=arch)
    if abs(float(scale) - 1.0) > 1e-9:
        set_scale(engine, float(scale), arch)
    return tag


def unmount(engine, target: str = "gpt") -> None:
    """卸掉引擎上 target（gpt|cfm）的 adapter。"""
    engine.detach_lora(target=target)


def set_scale(engine, factor: float, target: str = "gpt") -> int:
    """推理期强度旋钮：0=纯底座，1=完整 LoRA，0.6~0.8=常见折中。"""
    mod = (getattr(engine.tts, "gpt", None) if target == "gpt"
           else engine.tts.s2mel.models.get("cfm"))
    if mod is None:
        return 0
    return GD.set_adapter_scale(mod, factor)


# ===========================================================================
# 2. 合并
# ===========================================================================

@dataclass
class MergeOptions:
    """合并的输入与去向。"""
    adapter_dir: str = ""          # PEFT 的 adapter 目录（含 adapter_model.*）
    arch: str = "gpt"              # gpt | cfm —— 决定合并进哪份底座
    out_path: str = ""             # 输出 pth（必须在 checkpoints/ 之外）
    scale: float = 1.0             # 合并强度（一般与评测确认过的旋钮一致）
    device: str = "cpu"            # 合并在 CPU 上做就够（一次性矩阵加法）
    keep_backup: bool = True       # 输出已存在时先备份成 .bak（不覆盖历史）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MergeReport:
    ok: bool = False
    out_path: str = ""
    seconds: float = 0.0
    n_merged: int = 0              # 合了多少层
    max_rel: float = 0.0           # 合并后相对原权重的最大改动（漂移体检）
    global_rel: float = 0.0
    missing: int = 0
    unexpected: int = 0
    backup: str = ""
    notes: List[str] = None

    def __post_init__(self):
        self.notes = self.notes or []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        icon = "✅" if self.ok else "🔴"
        L = [f"### {icon} 合并 `{self.out_path}`", "",
             "| 项 | 值 |", "|---|---|",
             f"| 合并层数 | {self.n_merged} |",
             f"| 耗时 | {self.seconds:.1f}s |",
             f"| 权重改动（全局） | {self.global_rel:.5f} |",
             f"| 权重改动（最大单层） | {self.max_rel:.5f} |",
             f"| 校验 missing/unexpected | {self.missing}/{self.unexpected} |"]
        if self.backup:
            L.append(f"| 备份 | `{self.backup}` |")
        for n in self.notes:
            L.append(f"- {n}")
        return "\n".join(L)


def merge_lora_to_checkpoint(opts: MergeOptions,
                             guard: Optional[GD.BaseGuard] = None,
                             model_dir: Optional[str] = None
                             ) -> MergeReport:
    """把 adapter 的 ΔW 烘进底座，产出一份独立可用的 pth。

    流程：load 底座 → PeftModel.from_pretrained → set_adapter_scale →
    merge_and_unload →（校验 missing/unexpected = 0）→ 按原文件的
    包装格式写出。Conv1D / Linear、rsLoRA、多 rank 的差异全部由
    PEFT 处理，这里只做账目与安全检查。
    """
    rep = MergeReport(out_path=opts.out_path)
    t0 = time.perf_counter()
    guard = guard or GD.BaseGuard(model_dir=model_dir)
    model_dir = model_dir or os.path.join(PROJECT_ROOT, GD.MODEL_DIR_NAME)

    if not opts.adapter_dir or not os.path.isdir(opts.adapter_dir):
        rep.notes.append(f"adapter 目录不存在：{opts.adapter_dir}")
        return rep
    if opts.arch not in ("gpt", "cfm"):
        rep.notes.append(f"arch={opts.arch!r} 不合法（gpt | cfm）")
        return rep
    if not opts.out_path:
        rep.notes.append("out_path 不能为空")
        return rep
    # 红线：输出不准落在底座目录里（哪怕子目录也不行 ——
    # 用户可能把 checkpoints/ 整个拷走，混进去就说不清了）
    try:
        safe = guard.assert_safe_output(opts.out_path, must_not_exist=False)
        opts.out_path = safe
    except Exception as e:
        rep.notes.append(f"输出路径被防线拦下：{e}")
        return rep

    # ---- 备份 ----
    if opts.keep_backup and os.path.isfile(opts.out_path):
        bak = opts.out_path + ".bak"
        shutil.copy2(opts.out_path, bak)
        rep.backup = bak

    from peft import PeftModel

    if opts.arch == "gpt":
        base, info = GL.load_base_gpt(opts.device, model_dir)
        src_ckpt = info["path"]
    else:
        base, info = CL.load_base_cfm(opts.device, model_dir)
        src_ckpt = info["path"]

    pm = PeftModel.from_pretrained(base, opts.adapter_dir)
    if abs(float(opts.scale) - 1.0) > 1e-9:
        GD.set_adapter_scale(pm, float(opts.scale))
    n_layers = sum(1 for _n, m in pm.named_modules() if hasattr(m, "lora_A"))

    # 漂移体检必须在 merge_and_unload **之前**：解包之后 LoRA 层已经没了，
    # 再体检只会得到 0 —— 那不是「没漂」，是「体检器找不到了」。
    dr = GD.analyze_drift(pm)
    rep.global_rel = round(dr.global_rel, 6)
    rep.max_rel = round(dr.max_rel, 6)

    merged = pm.merge_and_unload()          # 返回解包后的底座（权重已并）
    sd_new = merged.state_dict()
    sd_old = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    # 按原文件的包装格式写回（官方 gpt.pth 带 "model" 键，s2mel 带 net.cfm）
    if opts.arch == "gpt":
        if isinstance(sd_old, dict) and "model" in sd_old:
            payload = {"model": sd_new}
        else:
            payload = sd_new
    else:
        net = (sd_old or {}).get("net") if isinstance(sd_old, dict) else None
        if not isinstance(net, dict) or "cfm" not in net:
            rep.notes.append(f"源 s2mel.pth 结构不是预期的 net.cfm：{src_ckpt}")
            return rep
        import copy
        payload = copy.deepcopy(sd_old)
        payload["net"]["cfm"] = sd_new

    # ---- 校验：合并结果必须能被一个全新底座**完整**吃下 ----
    if opts.arch == "gpt":
        checker, _ = GL.load_base_gpt("cpu", model_dir)
    else:
        checker, _ = CL.load_base_cfm("cpu", model_dir)
    target = payload["model"] if (opts.arch == "gpt"
                                  and isinstance(payload, dict)
                                  and "model" in payload) else (
        payload["net"]["cfm"] if opts.arch == "cfm" else payload)
    missing, unexpected = checker.load_state_dict(target, strict=False)
    rep.missing, rep.unexpected = len(missing), len(unexpected)
    del checker
    if rep.missing or rep.unexpected:
        rep.notes.append(
            f"合并结果与底座结构对不上（missing={rep.missing}, "
            f"unexpected={rep.unexpected}）。**没有写出任何文件**。")
        return rep

    os.makedirs(os.path.dirname(os.path.abspath(opts.out_path)) or ".",
                exist_ok=True)
    torch.save(payload, opts.out_path)
    rep.ok = True
    rep.n_merged = n_layers
    rep.seconds = round(time.perf_counter() - t0, 1)
    rep.notes.append(f"已合并 {n_layers} 层（scale={opts.scale:g}）"
                     f"→ {opts.out_path}")
    if rep.global_rel >= 0.25:
        rep.notes.append(f"⚠ 权重改动 {rep.global_rel:.4f} 偏大：合并后通用能力"
                         "大概率退化，建议先在评测台 A/B 确认过再合并。")
    return rep
