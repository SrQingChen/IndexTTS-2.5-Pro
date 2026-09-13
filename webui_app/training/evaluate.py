"""L3 · 自动评测台：A/B 对比 + 指标报告 + 试听文件。

训练完一个 adapter，「感觉像了」不算数 —— 同一段文本、同一个参考音色，
让**两个配置**（底座 vs adapter、adapter 强度 0.6 vs 1.0、新旧两个 run……）
各合成一遍，`reward.py` 打分，逐条对比。回答三个问题：

  · 说得更对了没有（WER）
  · 像不像（声纹相似度）
  · 综合起来谁好（reward 的胜负表）

**公平性**：A/B 每条都用**同一个种子**起采（差异只来自模型本身）；
打分器只看音频，不知道也不关心它来自哪个选手。

**产物**（`outputs/eval/<时间戳>/`）：
  · `report.json`  全部数字（UI 与后续脚本消费）
  · `report.md`    人看的对比表
  · `<样本>_<选手>.wav`  逐条试听 —— 指标说 A 好，耳朵说了才算

引擎侧的 adapter 挂载/卸载用 `engine.attach_lora / detach_lora`
（services/engine.py 的挂载点），强度旋钮用 `GD.set_adapter_scale`
—— 与训练端的防线 9 是同一个机制，不会出现「评测用的强度用户听不到」。
"""

from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import dataset as DS
from webui_app.training import guard as GD
from webui_app.training import reward as RW
from webui_app.training import runs as RN

__all__ = ["Contender", "EvalOptions", "run_eval", "eval_report_markdown"]


# ===========================================================================
# 1. 配置
# ===========================================================================

@dataclass
class Contender:
    """一个参赛配置。name 用于文件名与表格（别用中文）。"""
    name: str
    run: str = ""                # 空 = 纯底座（不挂 adapter）
    checkpoint: str = "best"     # best | 档位名 | "final"
    adapter_scale: float = 1.0   # 强度旋钮（防线 9）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        if not self.run:
            return f"{self.name}(底座)"
        return f"{self.name}({self.run}/{self.checkpoint}@{self.adapter_scale:g})"


@dataclass
class EvalOptions:
    dataset: str = ""
    out_dir: str = ""            # 空 = outputs/eval/<时间戳>
    n_samples: int = 0           # 0 = 全部 ready 样本
    seed: int = 42
    # 共同推理参数（A/B 一致，保证差异只来自模型）
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    max_mel_tokens: int = 900
    duration_factor: float = 1.0
    # 打分
    whisper_size: str = RW.DEFAULT_WHISPER
    language: str = "zh"
    wer_weight: float = 0.6
    sim_weight: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []

        def err(m): n.append(GD.Notice("error", m))
        def warn(m): n.append(GD.Notice("warn", m))

        if not self.dataset or not DS.exists(self.dataset):
            err(f"数据集 `{self.dataset}` 不存在")
        if int(self.n_samples) < 0:
            err("n_samples 不能为负")
        if not 0.0 < float(self.temperature) <= 2.0:
            err(f"temperature={self.temperature} 超出 (0, 2]")
        if float(self.wer_weight) < 0 or float(self.sim_weight) <= 0:
            err("打分权重必须是正数（且不全为零）")
        if not self.language:
            warn("language 为空将走 whisper 自动检测，转写可能不稳定")
        return n


def resolve_adapter_dir(c: Contender) -> Tuple[str, str]:
    """选手 → (adapter 目录, 目标架构 gpt|cfm)。返回 ("", "") 表示纯底座。

    best/final 走 run 的顶层目录（保险库每次改善都同步过去），
    档位名走保险库目录 —— 与续训的 `_resolve_ckpt_dir` 同一套约定。
    """
    if not c.run:
        return "", ""
    rj = RN.read_run(c.run)
    arch = str(rj.get("arch") or "gpt")
    which = (c.checkpoint or "best").lower()
    if which in ("best", "final"):
        d = RN.adapter_dir(c.run)
    else:
        d = os.path.join(RN.checkpoints_dir(c.run), c.checkpoint)
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"选手 {c.name} 的 adapter 目录不存在：{d}"
            f"（run={c.run}, checkpoint={c.checkpoint}）")
    return d, arch


# ===========================================================================
# 2. 主流程
# ===========================================================================

def run_eval(engine, a: Contender, b: Optional[Contender],
             opts: EvalOptions,
             progress: Optional[Callable[[float, str], None]] = None,
             should_stop: Optional[Callable[[], bool]] = None,
             scorer: Optional[RW.RewardScorer] = None,
             ) -> Dict[str, Any]:
    """跑一场 A/B（或单选手体检）。返回完整结果 dict（并落盘）。

    engine 必须已加载。评测期间会按选手挂/卸 adapter、调强度旋钮，
    结束后**恢复原状**（原来挂着什么就还原成什么 —— 用户引擎的状态
    不能被评测台偷偷改掉）。
    """
    out: Dict[str, Any] = {"ok": False, "rows": [], "errors": [],
                           "warnings": [], "a": a.to_dict(),
                           "b": b.to_dict() if b else None}
    notes = opts.validate()
    out["errors"] += [x.message for x in notes if x.level == "error"]
    out["warnings"] += [x.message for x in notes if x.level == "warn"]
    if a.name == (b.name if b else ""):
        out["errors"].append("两个选手名字不能相同")
    if out["errors"]:
        return out
    if engine is None or getattr(engine, "tts", None) is None:
        out["errors"].append("引擎未加载。请先在「系统」页加载模型。")
        return out

    items = [u for u in DS.load_meta(opts.dataset)
             if u.status == "ready" and (u.text or "").strip()]
    if not items:
        out["errors"].append(f"`{opts.dataset}` 里没有带文本的 ready 样本")
        return out
    if opts.n_samples > 0:
        items = items[:int(opts.n_samples)]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = opts.out_dir or os.path.join(
        PROJECT_ROOT, "outputs", "eval",
        f"{ts}_{a.name}" + (f"_vs_{b.name}" if b else ""))
    os.makedirs(out_dir, exist_ok=True)
    out["out_dir"] = out_dir

    sc = scorer or RW.RewardScorer(RW.RewardOptions(
        whisper_size=opts.whisper_size, language=opts.language,
        wer_weight=opts.wer_weight, sim_weight=opts.sim_weight))

    # ---- 选手的 adapter 目录先验好：别等合成到一半才发现路径是错的 ----
    for c in [a] + ([b] if b else []):
        try:
            resolve_adapter_dir(c)
        except FileNotFoundError as e:
            out["errors"].append(str(e))
    if out["errors"]:
        return out

    # ---- 引擎状态管理 ----
    # 开场先卸掉用户挂着的 adapter（记录在案）：attach 会把 PEFT 包在
    # 当前模块上，双重包装后 detach 一次解不干净。评测完再尽力还原。
    pre_attached = list(getattr(getattr(engine, "stats", None),
                                "lora_adapters", []) or [])
    if pre_attached:
        out["warnings"].append(
            f"评测开始前引擎挂着 adapter（{', '.join(pre_attached)}），"
            "已临时卸载，评测结束后会重新挂回。")
        for tag in list(pre_attached):
            try:
                engine.detach_lora(target=tag.split(":", 1)[0])
            except Exception:
                pass

    import torch

    def synth_one(c: Contender, u, dst: str) -> Optional[str]:
        """挂上选手的 adapter → 同种子合成 → 卸载。"""
        adir, arch = "", ""
        try:
            adir, arch = resolve_adapter_dir(c)
            if adir:
                engine.attach_lora(adir, target=arch)
                if float(c.adapter_scale) != 1.0:
                    mod = (getattr(engine.tts, "gpt", None) if arch == "gpt"
                           else engine.tts.s2mel.models.get("cfm"))
                    if mod is not None:
                        GD.set_adapter_scale(mod, float(c.adapter_scale))
            # 每条样本对 A/B 用同一个种子：差异只来自模型，不来自采样起点。
            # zlib.crc32 是稳定哈希（Python 内置 hash() 每个进程都会变，
            # 用它的话报告永远无法复现）。
            torch.manual_seed(int(opts.seed) + zlib.crc32(u.id.encode()) % 100000)
            engine.infer(
                spk_audio_prompt=os.path.join(DS.dir_of(opts.dataset),
                                              str(u.audio or "")),
                text=u.text, output_path=dst, verbose=False,
                lang=(u.lang or "ZH").upper(),
                do_sample=True, top_p=float(opts.top_p),
                top_k=(int(opts.top_k) or None),
                temperature=float(opts.temperature),
                duration_factor=float(opts.duration_factor),
                max_mel_tokens=int(opts.max_mel_tokens))
            return dst if os.path.isfile(dst) else None
        finally:
            if adir:
                try:
                    engine.detach_lora(target=arch)
                except Exception:
                    pass

    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        for i, u in enumerate(items):
            if should_stop and should_stop():
                out["warnings"].append("已手动停止：报告包含已完成的样本")
                break
            if progress:
                progress(i / max(1, len(items)),
                         f"{i+1}/{len(items)}：{u.text[:16]}…")
            row: Dict[str, Any] = {"id": u.id, "text": u.text}
            ok_all = True
            for c in [a] + ([b] if b else []):
                dst = os.path.join(out_dir,
                                   f"{u.id}_{c.name}.wav")
                try:
                    got = synth_one(c, u, dst)
                except Exception as e:
                    got = None
                    row[f"{c.name}_error"] = f"{type(e).__name__}: {e}"
                if not got:
                    ok_all = False
                    continue
                r = sc.score(dst, u.text,
                             os.path.join(DS.dir_of(opts.dataset),
                                          str(u.audio or "")),
                             lang=u.lang)
                row[c.name] = r
                row[f"{c.name}_wav"] = dst
                ok_all = ok_all and bool(r.get("ok"))
            row["ok"] = ok_all
            if b and row.get(a.name, {}).get("ok") and row.get(b.name, {}).get("ok"):
                row["delta_reward"] = round(
                    float(row[a.name]["reward"]) - float(row[b.name]["reward"]), 4)
                row["delta_wer"] = round(
                    float(row[a.name]["wer"]) - float(row[b.name]["wer"]), 4)
                row["delta_sim"] = round(
                    float(row[a.name]["sim"]) - float(row[b.name]["sim"]), 4)
            rows.append(row)

        out["rows"] = rows
        out["summary"] = summarize(a, b, rows)
        out["seconds"] = round(time.perf_counter() - t0, 1)
        out["ok"] = bool(rows) and not out["errors"]

        with open(os.path.join(out_dir, "report.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"options": opts.to_dict(), "a": a.to_dict(),
                       "b": b.to_dict() if b else None,
                       "summary": out["summary"], "rows": rows,
                       "seconds": out["seconds"]},
                      f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, "report.md"), "w",
                  encoding="utf-8") as f:
            f.write(eval_report_markdown(a, b, rows, out_dir, opts))
    finally:
        # ---- 还原引擎：卸掉评测期挂的，再把用户原来挂的挂回去 ----
        try:
            for tag in list(getattr(getattr(engine, "stats", None),
                                    "lora_adapters", []) or []):
                engine.detach_lora(target=tag.split(":", 1)[0])
            for tag in pre_attached:
                tgt, name = tag.split(":", 1)
                d = _find_adapter_dir(name)
                if d:
                    engine.attach_lora(d, target=tgt)
                    mod = (getattr(engine.tts, "gpt", None) if tgt == "gpt"
                           else engine.tts.s2mel.models.get("cfm"))
                    if mod is not None:
                        GD.set_adapter_scale(mod, 1.0)
                else:
                    out["warnings"].append(
                        f"无法还原评测前挂着的 adapter `{tag}`："
                        "目录找不到了，请到「训练」页重新挂载")
        except Exception:
            pass
        sc.unload()
    return out


def _find_adapter_dir(name: str) -> str:
    """按目录名在 training_runs/ 里找 adapter 目录（tag 只记了名字）。

    检查两个位置：保险库里的档位目录（ckpt-e003-s000036 这种名字）
    和各 run 的同步目录 adapter/。找不到返回空串（尽力而为的还原）。
    """
    top = RN.root()
    if not os.path.isdir(top):
        return ""
    for run in sorted(os.listdir(top)):
        cands = [os.path.join(top, run, RN.CHECKPOINT_DIR, name)]
        if name == RN.ADAPTER_DIR or name in ("final",):
            # 同步目录名固定叫 adapter/；「final」是没 val 时的最终权重目录
            cands.append(os.path.join(top, run, RN.ADAPTER_DIR))
            cands.append(os.path.join(top, run, "final"))
        for sub in cands:
            if os.path.isdir(sub):
                return sub
    return ""


# ===========================================================================
# 3. 汇总与报告
# ===========================================================================

def summarize(a: Contender, b: Optional[Contender],
              rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def agg(name: str) -> Dict[str, float]:
        rs = [r[name] for r in rows if r.get(name, {}).get("ok")]
        if not rs:
            return {"n": 0}
        n = len(rs)
        return {"n": n,
                "wer": round(sum(x["wer"] for x in rs) / n, 4),
                "sim": round(sum(x["sim"] for x in rs) / n, 4),
                "reward": round(sum(x["reward"] for x in rs) / n, 4)}

    s: Dict[str, Any] = {"a": agg(a.name)}
    if b:
        s["b"] = agg(b.name)
        deltas = [r["delta_reward"] for r in rows
                  if r.get("delta_reward") is not None]
        if deltas:
            eps = 0.01          # 小于一个百分点的差当平局（转写噪声量级）
            s["win_a"] = sum(1 for d in deltas if d > eps)
            s["win_b"] = sum(1 for d in deltas if d < -eps)
            s["tie"] = sum(1 for d in deltas if abs(d) <= eps)
            s["mean_delta_reward"] = round(sum(deltas) / len(deltas), 4)
    return s


def eval_report_markdown(a: Contender, b: Optional[Contender],
                         rows: Sequence[Dict[str, Any]], out_dir: str,
                         opts: EvalOptions) -> str:
    s = summarize(a, b, rows)
    title = f"A/B 评测：{a.label()}" + (f" vs {b.label()}" if b else "")
    try:
        where = os.path.relpath(out_dir, PROJECT_ROOT)
    except ValueError:
        where = out_dir        # 跨盘符（Windows）时 relpath 会炸，退回绝对路径
    L = [f"### {title}", "",
         f"- 数据集 `{opts.dataset}` · {len(rows)} 条 · 种子 {opts.seed}"
         f" · 打分 whisper={opts.whisper_size}"
         f"（WER×{opts.wer_weight:g} + SS×{opts.sim_weight:g}）",
         f"- 试听目录：`{where}`", ""]
    sa, sb = s.get("a", {}), s.get("b", {})
    if sb:
        L += ["| 指标 | " + f"{a.name} | {b.name} | 越好 |", "|---|---|---|---|",
              f"| WER | {sa.get('wer', '—')} | {sb.get('wer', '—')} | ↓ |",
              f"| 声纹相似 | {sa.get('sim', '—')} | {sb.get('sim', '—')} | ↑ |",
              f"| reward | **{sa.get('reward', '—')}** | "
              f"**{sb.get('reward', '—')}** | ↑ |", ""]
        if "win_a" in s:
            verdict = (f"🏆 {a.name} 胜 {s['win_a']} / "
                       f"{s['win_b']} 负 {a.name} / {s['tie']} 平")
            L += [f"**胜负**：{verdict} · 平均 Δreward = "
                  f"{s['mean_delta_reward']:+.4f}（正 = {a.name} 好）", ""]
    else:
        L += [f"| 指标 | {a.name} | 越好 |", "|---|---|---|",
              f"| WER | {sa.get('wer', '—')} | ↓ |",
              f"| 声纹相似 | {sa.get('sim', '—')} | ↑ |",
              f"| reward | **{sa.get('reward', '—')}** | ↑ |", ""]

    L += ["<details><summary>逐条明细</summary>", "",
          "| 样本 | 文本 | " + (f"{a.name} reward | {b.name} reward | Δ"
                                if b else f"{a.name} reward") + " |",
          "|---|---|---|" + ("---|---|" if b else "")]
    for r in rows:
        ra = (r.get(a.name) or {}).get("reward")
        if b:
            rb = (r.get(b.name) or {}).get("reward")
            d = r.get("delta_reward")
            L.append(f"| `{r['id']}` | {r['text'][:18]}… "
                     f"| {ra if ra is not None else '🔴'} "
                     f"| {rb if rb is not None else '🔴'} "
                     f"| {d if d is not None else '—'} |")
        else:
            L.append(f"| `{r['id']}` | {r['text'][:18]}… "
                     f"| {ra if ra is not None else '🔴'} |")
    L += ["", "</details>", ""]
    return "\n".join(L)
