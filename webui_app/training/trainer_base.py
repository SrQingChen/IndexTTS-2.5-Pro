"""训练器共用骨架。

GPT(T2S) 与 CFM(S2M) 两个 LoRA 训练器的**数据与前向**完全不同，但
「登记 → 校验底座 → 注入 → 优化器 → 评估 → 早停 → 存盘 → 收尾 → 释放」
这条流水线是一模一样的。把它抽出来的理由不是省事，而是这些环节里的坑
（释放顺序、续训要恢复权重、重复评估会误触发早停）都是实打实踩出来的，
写两份就一定会有一份忘记改。

子类只需要实现四个钩子：
    `_build_model()`       加载底座 + 注入 LoRA + dtype + 切训练态
    `_epoch_batches()`     这一轮跑哪些 batch（含回放混合）
    `_train_micro_batch()` 一个微批的前向与反向
    `evaluate()`           val 指标
"""

from __future__ import annotations

import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import guard as GD
from webui_app.training import runs as RN

# 一个 epoch 里为了「长度接近的排一起」而打包的大块尺寸（× batch_size）
MEGA_BATCH = 40


# ===========================================================================
# 0. 批规划（两个训练器共用）
# ===========================================================================

def plan_batches(items: Sequence[Any], batch_size: int, sort_by_length: bool,
                 seed: int = 42,
                 len_of: Optional[Callable[[Any], int]] = None,
                 mega: int = MEGA_BATCH) -> List[List[Any]]:
    """把一个 epoch 的取样顺序切成 batch。

    `sort_by_length` 用的是「大块内排序 + 大块间打乱」：
    先按长度排序，切成 mega×batch_size 的大块，块内打乱后再切 batch。
    这样同一 batch 里长度接近（padding 少），但 batch 的**顺序**每个
    epoch 都不同 —— 完全排序会让模型每轮都按同一个由短到长的顺序看数据，
    等于给了它一个额外的伪特征。

    `items` 的元素类型由调用方定（GPT 是 (池名, 下标)，CFM 是
    (池名, target 下标, prompt 下标)），所以长度要由 `len_of` 回调给。
    """
    out_items = list(items)
    bs = max(1, int(batch_size))
    if sort_by_length and bs > 1 and len(out_items) > bs and len_of is not None:
        out_items.sort(key=len_of)
        chunk_n = max(1, int(mega)) * bs
        rnd = random.Random(seed)
        shuffled: List[Any] = []
        for s in range(0, len(out_items), chunk_n):
            c = out_items[s:s + chunk_n]
            rnd.shuffle(c)
            shuffled.extend(c)
        out_items = shuffled
    else:
        random.Random(seed).shuffle(out_items)
    return [out_items[i:i + bs] for i in range(0, len(out_items), bs)]


# ===========================================================================
# 0b. 特征样本池
# ===========================================================================

class FeaturePool:
    """把一个数据集的 `.pt` 特征缓存变成可索引的样本池。

    懒加载 + 有上限的缓存：几百条的数据集全放进内存也就几百 MB，
    但几千条就不行了，所以给个条数上限，超了按 LRU 淘汰。
    被淘汰的只是**张量**，元信息（长度/语言）一直留着 ——
    排序与过滤都只看元信息，不该触发磁盘 IO。

    子类只需要实现 `_convert(raw)`：GPT 要 text_tokens/codes，
    CFM 要 mel/mu，两边从同一个 .pt 里取不同字段。
    """

    #: 排序用的长度字段（GPT = n_codes，CFM = mel_len）
    LEN_FIELD = "mel_len"

    def __init__(self, dataset: str, ids: Sequence[str], kind: str = "target",
                 cache_limit: int = 200, backfill: bool = True):
        from webui_app.training import dataset as DS
        from webui_app.training import features as FT

        self.dataset = dataset
        self.ids = list(ids)
        self.kind = kind                       # "target" | "replay"
        self.ds_dir = DS.dir_of(dataset)
        self.cache_limit = max(8, int(cache_limit))
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []            # LRU 顺序，末尾是最近用的
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.missing: List[str] = []

        # 一次 load_meta 建索引。逐条调 DS.get() 会每次重读整个 meta.jsonl，
        # 1000 条的数据集就是 1000 次全文件解析。
        by_id = {u.id: u for u in DS.load_meta(dataset)}
        for uid in self.ids:
            p = FT.FeatureExtractor.feature_path(self.ds_dir, uid)
            if not FT.is_usable(p):
                self.missing.append(uid)
                continue
            m = by_id.get(uid)
            self.meta[uid] = {
                "n_codes": int(getattr(m, "n_codes", 0) or 0),
                "n_text": int(getattr(m, "n_text_tokens", 0) or 0),
                "mel_len": int(getattr(m, "mel_len", 0) or 0),
                "lang": (getattr(m, "lang", "ZH") or "ZH"),
                "duration": float(getattr(m, "duration", 0.0) or 0.0),
                "text": (getattr(m, "text", "") or ""),
            }
        self.ids = [u for u in self.ids if u in self.meta]

        # meta 里的长度字段是后来加的，老数据集没有。
        # 不补写的话排序与过滤全失效（所有样本看起来都是 0 帧），
        # 而 backfill_meta 是幂等的，第二次几乎瞬间返回。
        self.stale = sum(1 for v in self.meta.values() if v["n_codes"] <= 0)
        if backfill and self.stale:
            FT.backfill_meta(dataset)
            by_id = {u.id: u for u in DS.load_meta(dataset)}
            for uid, v in self.meta.items():
                m = by_id.get(uid)
                if m is None:
                    continue
                v["n_codes"] = int(m.n_codes or 0)
                v["n_text"] = int(m.n_text_tokens or 0)
                v["mel_len"] = int(m.mel_len or 0)
            self.stale = sum(1 for v in self.meta.values() if v["n_codes"] <= 0)

    def __len__(self) -> int:
        return len(self.ids)

    # ---------------- 子类钩子 ----------------
    def _convert(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # ---------------- 读取 ----------------
    def load(self, uid: str) -> Optional[Dict[str, Any]]:
        """取一条样本的张量（带缓存）。"""
        hit = self._cache.get(uid)
        if hit is not None:
            if uid in self._order:
                self._order.remove(uid)
            self._order.append(uid)
            return hit
        from webui_app.training import features as FT
        p = FT.FeatureExtractor.feature_path(self.ds_dir, uid)
        if not FT.is_usable(p):
            return None
        import torch
        try:
            raw = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            return None
        try:
            s = self._convert(raw)
        except Exception:
            return None
        finally:
            del raw
        if s is None:
            return None
        s["id"] = uid
        s["kind"] = self.kind
        self._cache[uid] = s
        self._order.append(uid)
        while len(self._order) > self.cache_limit:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return s

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()

    def restrict(self, ids: Sequence[str]) -> int:
        """把池子缩到给定的 id（保持原顺序），返回剩下的条数。

        过滤掉超长/缺特征的样本后用。不能只改 `self.ids`：
        `_order` 里残留的 id 会占着 LRU 名额把真正要用的样本挤出去。
        """
        keep = set(ids)
        self.ids = [u for u in self.ids if u in keep]
        self.meta = {u: v for u, v in self.meta.items() if u in keep}
        self._cache = {u: v for u, v in self._cache.items() if u in keep}
        self._order = [u for u in self._order if u in keep]
        return len(self.ids)

    def length_of(self, uid: str) -> int:
        return int((self.meta.get(uid) or {}).get(self.LEN_FIELD) or 0)


# ===========================================================================
# 0c. LoRA 注入与精度（GPT 与 CFM 完全共用）
# ===========================================================================

def inject_lora(base, cfg: GD.LoRAConfig) -> Tuple[Any, Dict[str, Any]]:
    """冻结底座 → 注入 LoRA → 复核。返回 (peft_model, 报告)。

    两边都是「先冻结、再注入、再复核」，差别只在 `scan_targets` 扫出来的
    模块名（GPT 是 Conv1D，DiT 是 Linear）—— 而那个差别已经被
    `pattern_of` / `build_target_regex` 吸收了，所以这里不需要分支。
    """
    from peft import LoraConfig, get_peft_model

    groups = GD.scan_targets(base)
    pats = GD.resolve_target_patterns(groups, cfg.target_modules)
    if not pats:
        raise ValueError(
            f"注入面 {cfg.target_modules} 在这个模型里一个都没匹配上。"
            "请到「训练」页重新扫描可用模块。")
    dropped = sorted(set(cfg.target_modules) - set(pats))
    regex = GD.build_target_regex(pats)

    frozen = GD.freeze_base(base)                # 必须在注入**之前**
    lc = LoraConfig(r=int(cfg.rank), lora_alpha=int(cfg.alpha),
                    lora_dropout=float(cfg.dropout), target_modules=regex,
                    bias="none", task_type=None, use_rslora=bool(cfg.use_rslora))
    pm = get_peft_model(base, lc)

    not_frozen = GD.assert_base_frozen(pm)
    if not_frozen:
        raise RuntimeError(f"注入后仍有 {len(not_frozen)} 个底座参数带梯度："
                           f"{not_frozen[:5]}")
    cnt = GD.count_params(pm)
    n_layers = sum(1 for _n, m in pm.named_modules()
                   if hasattr(m, "lora_A") and hasattr(m, "scaling"))
    report = {
        "regex": regex,
        "patterns": pats,
        "dropped_patterns": dropped,
        "lora_layers": n_layers,
        "base_params": cnt["base"],
        "adapter_params": cnt["trainable"],
        "total_params": cnt["total"],
        "trainable_pct": cnt["trainable_pct"],
        "estimated_params": GD.estimate_adapter_params(
            [g for g in groups if g.pattern in set(pats)], cfg),
        "frozen": frozen,
    }
    return pm, report


def promote_adapter_fp32(model) -> int:
    """把 adapter 参数从 bf16 提回 fp32，返回改动的张量数。

    底座可以是 bf16（省一半显存），但 **AdamW 的一阶/二阶动量必须是 fp32**：
    bf16 只有 8 位尾数，lr=1e-4 级别的更新量加到动量上会直接下溢成 0，
    表现为「loss 不降但也不报错」。
    """
    import torch

    n = 0
    for _name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
            n += 1
    return n


def cast_base_dtype(model, bf16: bool) -> Dict[str, int]:
    """只把**冻结的底座**转成 bf16，adapter 留在 fp32。"""
    import torch

    want = torch.bfloat16 if bf16 else torch.float32
    n_p = n_b = 0
    for _name, p in model.named_parameters():
        if not p.requires_grad and p.dtype != want:
            p.data = p.data.to(want)
            n_p += 1
    for _name, b in model.named_buffers():
        if b.is_floating_point() and b.dtype != want:
            b.data = b.data.to(want)
            n_b += 1
    return {"params": n_p, "buffers": n_b, "dtype": str(want).replace("torch.", "")}


# ===========================================================================
# 1. 训练报告
# ===========================================================================

@dataclass
class TrainReport:
    """一次训练的完整结果。既是返回值，也是 run.json 的主体。"""
    ok: bool = False
    run: str = ""
    arch: str = ""
    steps: int = 0
    epochs: int = 0
    seconds: float = 0.0
    best_val: Optional[float] = None
    first_val: Optional[float] = None       # step-0 的评估值 = **纯底座**的水平
    final_train: Optional[float] = None
    stopped_early: bool = False
    stop_reason: str = ""
    adapter_dir: str = ""
    n_checkpoints: int = 0
    vram_peak_gb: float = 0.0
    adapter_params: int = 0
    base_params: int = 0
    drift: Dict[str, Any] = field(default_factory=dict)
    base_verify: Dict[str, Any] = field(default_factory=dict)
    notes: List[Dict[str, str]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    replay: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def note(self, level: str, message: str) -> None:
        self.notes.append({"level": level, "message": message})

    def merge_notes(self, notices: Sequence[GD.Notice]) -> None:
        for x in notices or []:
            self.note(x.level, x.message)

    @property
    def improved(self) -> Optional[float]:
        """相对底座降了多少。None = 还没两次可比的评估。"""
        if self.first_val is None or self.best_val is None or self.first_val <= 0:
            return None
        return 1.0 - self.best_val / self.first_val

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        # ok 只说明「训练器自己没出错且产出了 adapter」；手动停/早停/max_steps
        # 都属于 ok 但没跑完，用 🟡 区分开，否则用户会把提前终止误读成完整训练。
        icon = "🔴" if not self.ok else ("🟡" if self.stop_reason else "✅")
        head = f"{icon} 训练 `{self.run}`" + ("" if self.ok else "（未正常完成）")
        L = [f"### {head}", "",
             "| 项 | 值 |", "|---|---|",
             f"| 目标 | {RN.ARCH_LABELS.get(self.arch, self.arch)} |",
             f"| 步数 / 轮数 | {self.steps} / {self.epochs} |",
             f"| 耗时 | {self.seconds/60:.1f} 分钟 |"]
        if self.vram_peak_gb:
            L.append(f"| 峰值显存 | {self.vram_peak_gb:.2f} GB |")
        if self.adapter_params:
            L.append(f"| 可训练参数 | {self.adapter_params/1e6:.2f} M |")
        if self.first_val is not None:
            L.append(f"| val（底座） | {self.first_val:.4f} |")
        if self.best_val is not None:
            imp = self.improved
            L.append(f"| val（最好） | **{self.best_val:.4f}**"
                     + (f" · 降 {imp*100:.1f}%" if imp is not None else "") + " |")
        if self.final_train is not None:
            L.append(f"| train（最后） | {self.final_train:.4f} |")
        if self.stopped_early:
            L.append(f"| 早停 | {self.stop_reason} |")
        elif self.stop_reason:
            L.append(f"| 提前结束 | {self.stop_reason} |")
        L.append(f"| checkpoint | {self.n_checkpoints} 份 |")
        L.append(f"| adapter | `{self.adapter_dir or '—'}` |")
        L.append("")

        bv = self.base_verify or {}
        if bv:
            mark = "✅ 未被改动" if bv.get("ok") else "🔴 **发生变化，结果不可信**"
            L.append(f"**底座完整性**：{mark}（校验 {bv.get('checked', 0)} 个文件）")
            L.append("")
        if self.drift:
            L += [f"**权重漂移**：全局 {self.drift.get('global_rel', 0.0):.4f}"
                  f" · 最大单层 `{self.drift.get('max_name', '')}`"
                  f" = {self.drift.get('max_rel', 0.0):.4f}"
                  f" · 等级 {self.drift.get('level', '')}", ""]
        rp = self.replay or {}
        if rp:
            L += [f"**回放**：池 {rp.get('pool', 0)} 条 · 目标比例 "
                  f"{float(rp.get('ratio', 0.0)):.0%} · 实际 "
                  f"{float(rp.get('achieved', 0.0)):.0%}"
                  + (f" · {rp['note']}" if rp.get("note") else ""), ""]
        if self.notes:
            mark = {"error": "✖", "warn": "⚠️", "info": "ℹ️"}
            L.append("<details open><summary>体检结论（%d 条）</summary>"
                     % len(self.notes))
            L.append("")
            for x in self.notes:
                L.append(f"> {mark.get(x['level'], '·')} **{x['level'].upper()}** "
                         f"{x['message']}")
                L.append(">")
            if L[-1] == ">":
                L.pop()
            L += ["", "</details>", ""]
        if self.error:
            L += [f"> 🔴 **异常**：`{self.error}`", ""]
        if self.history:
            L += ["<details><summary>训练曲线（%d 个记录点）</summary>"
                  % len(self.history),
                  "", "| step | epoch | train | val | lr |", "|---|---|---|---|---|"]
            for h in self.history:
                tv, vv = h.get("train"), h.get("val")
                L.append(f"| {h.get('step', 0)} | {h.get('epoch', 0)} "
                         f"| {('%.4f' % tv) if isinstance(tv, (int, float)) else '—'} "
                         f"| {('%.4f' % vv) if isinstance(vv, (int, float)) else '—'} "
                         f"| {h.get('lr', 0.0):.2e} |")
            L += ["", "</details>"]
        return "\n".join(L)


# ===========================================================================
# 2. 基类
# ===========================================================================

class BaseTrainer:
    """三段式：`preflight()`（不加载模型）→ `prepare()`（建模型）→ `run()`。

    分开的原因：UI 需要在用户点「开始」之前就把风险、显存、数据量全部摆出来，
    而那一步不应该花几十秒去读几个 GB 的权重。
    """

    ARCH = ""

    def __init__(self, dataset: str, cfg: Optional[GD.LoRAConfig] = None,
                 run_name: Optional[str] = None, device: Optional[str] = None,
                 model_dir: Optional[str] = None, val_ratio: float = 0.05,
                 resume_from: Optional[str] = None,
                 train_root: Optional[str] = None):
        self.dataset = dataset
        self.cfg = cfg or GD.LoRAConfig.preset("balanced")
        self.run_name = RN.safe_run_name(run_name) if run_name else ""
        self.model_dir = model_dir or os.path.join(PROJECT_ROOT, GD.MODEL_DIR_NAME)
        self.val_ratio = float(val_ratio)
        self.resume_from = resume_from

        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.guard = GD.BaseGuard(model_dir=self.model_dir, train_root=train_root)
        self.model = None                 # PEFT 包装后的底座
        self.prev_state: Dict[str, Any] = {}
        self.opt = None
        self.sched = None
        self.stopper: Optional[GD.EarlyStopper] = None
        self.vault_obj: Optional[GD.CheckpointVault] = None
        self.total_steps = 0
        self.train_ids: List[str] = []
        self.val_ids: List[str] = []
        self.replay_ids: List[str] = []
        self.replay_src = "none"
        self.replay_dataset = ""

        self.report = TrainReport(run=self.run_name, arch=self.ARCH)
        self._pf: Dict[str, Any] = {}
        self._prep: Dict[str, Any] = {}
        self._prepared = False
        self._step = 0
        self._accum = 0
        self._synced = False
        self._last_val: Optional[float] = None
        self._last_eval_step = -1
        self._resume_epoch = 0

    # ------------------------------------------------------------------
    # 子类钩子
    # ------------------------------------------------------------------
    def preflight(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_model(self, pf: Dict[str, Any],
                     progress: Optional[Callable[[float, str], None]]
                     ) -> Dict[str, Any]:
        """加载底座 → 注入 LoRA → dtype → 切训练态。返回要记进 run.json 的信息。"""
        raise NotImplementedError

    def _epoch_batches(self, epoch: int) -> Tuple[List[List[Any]], Dict[str, Any]]:
        """返回 (这一轮的 batch 列表, 回放信息)。"""
        raise NotImplementedError

    def _train_micro_batch(self, items: Sequence[Any]) -> Optional[Dict[str, float]]:
        """一个微批：前向 + 反向。返回统计值，没数据可跑时返回 None。"""
        raise NotImplementedError

    def evaluate(self, max_batches: int = 0) -> Optional[float]:
        raise NotImplementedError

    def _train_snapshot(self, max_batches: int = 4) -> Optional[float]:
        return None

    def _restore_model(self) -> None:
        """把 `_build_model` 改过的模型状态还原（GPT 需要，CFM 不需要）。"""

    def _drop_model_refs(self) -> None:
        """断开子类持有的模型引用。基类只负责 self.model。"""

    # ------------------------------------------------------------------
    # 日志与进度
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.run_name:
            RN.append_log(self.run_name, msg)

    def _progress(self, fn: Optional[Callable[[float, str], None]],
                  frac: float, msg: str) -> None:
        if fn:
            fn(float(min(1.0, max(0.0, frac))), msg)

    def _set_seed(self) -> None:
        import torch
        s = int(self.cfg.seed)
        random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        try:
            import numpy as np
            np.random.seed(s % (2 ** 32))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 优化器与调度器
    # ------------------------------------------------------------------
    def _lr_lambda(self) -> Callable[[int], float]:
        """线性 warmup → 余弦退火到 `lr × min_lr_ratio`。

        不退回 0 的原因：最后几个 step 的 lr 如果趋近 0，那一段训练等于白跑，
        而早停选中的 checkpoint 很可能就落在那一段里。
        """
        total = max(1, int(self.total_steps))
        warm = max(1, int(round(total * float(self.cfg.warmup_ratio or 0.0))))
        floor = max(0.0, min(0.99, float(self.min_lr_ratio())))
        decay = max(1, total - warm)

        def fn(s: int) -> float:
            if s < warm:
                return float(s + 1) / float(warm)
            p = min(1.0, max(0.0, (s - warm) / float(decay)))
            cos = 0.5 * (1.0 + math.cos(math.pi * p))
            return floor + (1.0 - floor) * cos
        return fn

    def min_lr_ratio(self) -> float:
        return 0.1

    def _cur_lr(self) -> float:
        try:
            return float(self.opt.param_groups[0]["lr"]) if self.opt else 0.0
        except Exception:
            return 0.0

    def _optimizer_step(self) -> float:
        """裁切 → 更新 → 退火。返回裁切前的梯度范数。"""
        import torch
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        # 按**实际**累积到的微批数归一：有些微批会因为读不到样本而整个跳过，
        # 固定除 grad_accum 会把有效梯度按比例缩小（跳一个就少 1/4），
        # 表现为「同样配置下 loss 降得比预期慢」，而且很难查。
        inv = 1.0 / max(1, self._accum)
        for p in trainable:
            if p.grad is not None:
                p.grad.mul_(inv)
        gn = 0.0
        if float(self.cfg.grad_clip or 0.0) > 0.0:
            gn = float(torch.nn.utils.clip_grad_norm_(trainable, self.cfg.grad_clip))
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        self._step += 1
        self._accum = 0
        return gn

    # ------------------------------------------------------------------
    # checkpoint / 续训
    # ------------------------------------------------------------------
    def _state_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, "train_state.pt")

    def _resolve_ckpt_dir(self, ref: str) -> str:
        """把一个 checkpoint 引用解析成目录。

        支持三种写法：绝对/相对路径、本 run 保险库里的档位名、
        以及其他 run 下的档位名（跨 run 续训是常见需求：
        先用均衡档跑一轮，再用保守档接着跑）。
        """
        if os.path.isdir(ref):
            return os.path.abspath(ref)
        base = os.path.basename(str(ref).rstrip("\\/"))
        # 候选 = 本 run 保险库 + 所有 run 的保险库（跨 run 续训）。
        # "final" 特殊：它就在 run 目录下而不是 checkpoints/ 里。
        cands = [os.path.join(RN.checkpoints_dir(self.run_name), base)]
        top = RN.root()
        if os.path.isdir(top):
            for d in sorted(os.listdir(top)):
                cands.append(os.path.join(top, d, RN.CHECKPOINT_DIR, base))
                if base == "final":
                    cands.append(os.path.join(top, d, "final"))
        for p in cands:
            if os.path.isdir(p):
                return os.path.abspath(p)
        raise FileNotFoundError(
            f"找不到 checkpoint：{ref}\n已搜索：{cands[:3]}…\n"
            "可以传绝对路径，或传保险库里的档位名（如 ckpt-e003-s000036）。")

    def name_conflict(self) -> str:
        """用户指定的运行名若已存在训练记录，返回错误文案；无冲突返回空串。

        prepare() 会直接复用同名目录：旧 run.json 被覆盖（历史丢失），
        新 checkpoint 还会混进旧保险库参与排序剪枝。必须在门口拦住，
        preflight 与训练页也调它，让问题在开跑前就可见。
        """
        if self.run_name and RN.read_run(self.run_name):
            return (f"运行名 `{self.run_name}` 已有训练记录，为避免覆盖"
                    "请换一个名字或留空自动生成"
                    "（确要重用请先在「训练记录」区删除旧记录）。")
        return ""

    def _load_adapter_weights(self, ckpt_dir: str) -> int:
        """把 checkpoint 里的 adapter 权重灌回刚注入的模型，返回张量数。

        **这一步不能省**。`train_state.pt` 里只有优化器/调度器/早停状态，
        没有权重。少了它，续训会从「B 矩阵全零 = 等价于纯底座」重新开始，
        而优化器却带着上一轮的动量 —— 相当于拿着一份不属于当前权重的历史
        继续走，表现是续训之后 loss 突然弹回去，而且**不会报任何错**。
        """
        from peft import load_peft_weights, set_peft_model_state_dict

        sd = load_peft_weights(ckpt_dir, device="cpu")
        if not sd:
            raise RuntimeError(
                f"{ckpt_dir} 里没有 adapter 权重。这个目录可能不是 PEFT 存的 checkpoint。")
        res = set_peft_model_state_dict(self.model, sd, adapter_name="default")
        bad = list(getattr(res, "unexpected_keys", None) or [])
        lack = [k for k in (getattr(res, "missing_keys", None) or []) if "lora_" in k]
        if bad or lack:
            raise RuntimeError(
                f"adapter 权重与当前注入面对不上（unexpected={bad[:4]} "
                f"missing={lack[:4]}）。\n续训必须用与原训练完全相同的 "
                "rank / target_modules，否则低秩矩阵的形状对不上。")
        self._promote_fp32()
        self._log(f"已加载 adapter 权重：{len(sd)} 个张量 ← {ckpt_dir}")
        return len(sd)

    def _promote_fp32(self) -> int:
        """把可训练参数提回 fp32（子类可覆盖）。"""
        return 0

    def _save_train_state(self, dest: str, epoch: int,
                          val: Optional[float]) -> None:
        """把优化器/调度器/早停状态存进 checkpoint 目录。

        与 adapter 同目录而不是另开一个：回滚时两者必须成对生效，
        分开放迟早会出现「权重是第 3 轮、动量是第 7 轮」这种无法察觉的错配。
        """
        import torch
        st = {
            "optimizer": self.opt.state_dict() if self.opt else None,
            "scheduler": self.sched.state_dict() if self.sched else None,
            "stopper": self.stopper.state_dict() if self.stopper else None,
            "epoch": int(epoch), "step": int(self._step),
            "best_val": (self.stopper.best if self.stopper else None),
            "val": val,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            "arch": self.ARCH,
            "cfg": self.cfg.to_dict(),
        }
        torch.save(st, self._state_path(dest))

    def _load_train_state(self, ckpt_dir: str) -> Dict[str, Any]:
        """恢复优化器/调度器/早停状态。**权重由 `_load_adapter_weights` 负责**。"""
        import torch
        ckpt_dir = self._resolve_ckpt_dir(ckpt_dir)
        p = self._state_path(ckpt_dir)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} 不存在。续训需要当初存了优化器状态"
                "（options.save_optimizer_state=True）。")
        st = torch.load(p, map_location=self.device, weights_only=False)
        if st.get("arch") and st["arch"] != self.ARCH:
            raise RuntimeError(
                f"这份 checkpoint 是 `{st['arch']}` 训练的，不能用来续训 "
                f"`{self.ARCH}`。两个目标的 adapter 形状完全不同。")
        if self.opt and st.get("optimizer"):
            self.opt.load_state_dict(st["optimizer"])
        if self.sched and st.get("scheduler"):
            self.sched.load_state_dict(st["scheduler"])
        if self.stopper and st.get("stopper"):
            self.stopper.load_state_dict(st["stopper"])
        if st.get("torch_rng") is not None:
            torch.set_rng_state(st["torch_rng"].cpu())
        if st.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in st["cuda_rng"]])
        self._step = int(st.get("step") or 0)
        self._resume_epoch = int(st.get("epoch") or 0)
        self._log(f"已恢复训练状态 ← {os.path.basename(ckpt_dir)}（step={self._step}）")
        return {k: st.get(k) for k in ("epoch", "step", "best_val", "val")}

    def _write_adapter(self, dest: str, epoch: int, val: Optional[float]) -> None:
        os.makedirs(dest, exist_ok=True)
        self.model.save_pretrained(dest)
        if self.save_optimizer_state() and self.opt is not None:
            self._save_train_state(dest, epoch, val)

    def save_optimizer_state(self) -> bool:
        return True

    def _save_ckpt(self, epoch: int, val: float) -> Optional[str]:
        """存一份 checkpoint 并同步到 `adapter/`。

        每次都同步而不是只在最后同步：训练可能因为显存、断电、用户关页面
        而在任何时刻死掉。`adapter/` 永远是「目前为止最好的那份」，
        推理端只认这一个路径（见 runs.sync_adapter 的原子替换）。
        """
        if self.vault_obj is None:
            return None
        info = self.vault_obj.save(lambda d: self._write_adapter(d, epoch, val),
                                   epoch, self._step, float(val),
                                   extra={"arch": self.ARCH, "run": self.run_name,
                                          "val": float(val), "dataset": self.dataset})
        RN.sync_adapter(self.run_name, info.path)
        self._synced = True
        self.report.n_checkpoints = len(self.vault_obj.list())
        self.report.adapter_dir = RN.adapter_dir(self.run_name)
        self._log(f"已存 checkpoint {info.name}（val={val:.4f}）并同步到 adapter/")
        return info.path

    # ------------------------------------------------------------------
    # 评估与早停
    # ------------------------------------------------------------------
    def _handle_eval(self, val: Optional[float], epoch: int,
                     save: bool = True) -> bool:
        """评估 → 早停 → 存盘。返回是否应该停下。

        基线评估要传 save=False：LoRA 的 B 矩阵是零初始化的，此刻输出等于
        纯底座。存下来只会白占一个 keep 名额（adapter 只有几 MB，但名额只有 3 个），
        而「回到底座」这件事推理端把 adapter_scale 调成 0 就行了，不需要存权重。
        """
        if val is None:
            return False
        self._last_val = float(val)
        self._last_eval_step = self._step
        rep = self.report
        # 自己算「是否改善」而不用 sr.improved：
        # EarlyStopper 在 patience=0（早停关闭）时把 improved **写死为 True**，
        # 直接拿来用就会变成「每次评估都存一份 checkpoint」，
        # 包括 val 变差的那些 —— 保险库会被一堆越来越差的存档洗一遍。
        prev_best = self.stopper.best if self.stopper is not None else rep.best_val
        delta = float(self.cfg.val_min_delta or 0.0)
        improved = prev_best is None or float(val) < float(prev_best) - delta
        sr = self.stopper.step(val, epoch) if self.stopper else None
        self._record(self._step, epoch, val=val)
        if rep.first_val is None:
            rep.first_val = float(val)
        if sr is not None and sr.best is not None:
            rep.best_val = float(sr.best)
        elif improved or rep.best_val is None:
            rep.best_val = float(val)
        self._log(f"评估 step={self._step} epoch={epoch} val={val:.4f} "
                  f"best={rep.best_val:.4f}"
                  + (f" bad={sr.bad_count}" if sr else "")
                  + ("" if improved else "（未改善）"))
        if save and improved:
            self._save_ckpt(epoch, float(val))
        if sr is not None and sr.should_stop:
            rep.stopped_early = True
            rep.stop_reason = sr.reason
            self._log(f"早停触发：{sr.reason}")
            return True
        return False

    def _record(self, step: int, epoch: int, train: Optional[float] = None,
                val: Optional[float] = None, lr: Optional[float] = None,
                **extra) -> None:
        h: Dict[str, Any] = {
            "step": int(step), "epoch": int(epoch),
            "train": (round(float(train), 5) if train is not None else None),
            "val": (round(float(val), 5) if val is not None else None),
            "lr": float(lr if lr is not None else self._cur_lr()),
        }
        h.update({k: v for k, v in extra.items() if v is not None})
        self.report.history.append(h)

    @staticmethod
    def _new_acc() -> Dict[str, float]:
        return {"loss": 0.0, "nb": 0, "n": 0, "gn": 0.0, "skip": 0, "sec": 0.0}

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def prepare(self, progress: Optional[Callable[[float, str], None]] = None
                ) -> Dict[str, Any]:
        """把模型建起来。失败时抛异常，且**已占用的显存会在 except 里归还**。"""
        import torch

        if self._prepared:
            return self._prep
        pf = self._pf or self.preflight()
        if not pf.get("ok"):
            raise RuntimeError("预检未通过：\n· "
                               + "\n· ".join(pf.get("errors") or ["未知"]))
        try:
            self._progress(progress, 0.02, "建立训练记录")
            if not self.run_name:
                self.run_name = RN.suggest_run_name(self.ARCH, self.dataset)
            self.run_name = RN.safe_run_name(self.run_name)
            conflict = self.name_conflict()
            if conflict:
                raise RuntimeError(conflict)
            self.report.run = self.run_name
            RN.run_dir(self.run_name, create=True)

            self._progress(progress, 0.05, "校验底座完整性")
            self._guard_base()

            if self.device.startswith("cuda"):
                self._progress(progress, 0.12, "显存体检")
                GD.vram_preflight(float(pf.get("est_vram_gb") or 0.0),
                                  raise_on_short=True)

            self._set_seed()
            extra = self._build_model(pf, progress)

            self._progress(progress, 0.76, "建立 AdamW 与余弦退火")
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("没有任何可训练参数 —— LoRA 注入失败或底座没冻住")
            self.opt = torch.optim.AdamW(
                params, lr=float(self.cfg.lr), betas=(0.9, 0.999), eps=1e-8,
                weight_decay=float(self.cfg.weight_decay))
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda())

            self.stopper = GD.EarlyStopper(
                patience=int(self.cfg.val_patience),
                min_delta=float(self.cfg.val_min_delta), mode="min")
            if not self.val_ids:
                # 没 val 就没得早停。与其让 stopper 空转，不如明确关掉并告知。
                self.stopper = GD.EarlyStopper(patience=0, mode="min")
                self.report.note("warn", "验证集为空，早停已自动关闭")

            self.vault_obj = RN.vault(self.run_name,
                                      keep=int(self.cfg.keep_checkpoints))

            # 续训必须在建完优化器之后（要往 opt/sched 里灌状态），
            # 但在写 run.json 之前（那里要记下**解析后**的 checkpoint 目录）。
            resume_dir, n_loaded = "", 0
            if self.resume_from:
                self._progress(progress, 0.86, f"续训：{self.resume_from}")
                resume_dir = self._resolve_ckpt_dir(self.resume_from)
                # 先权重后状态：少了权重这一步，续训会从「等于纯底座」
                # 重新开始，而优化器却带着上一轮的动量 —— 不报错，只是跑偏。
                n_loaded = self._load_adapter_weights(resume_dir)
                self._load_train_state(resume_dir)

            self._prep = dict(extra or {})
            self._prep.update({"ok": True, "run": self.run_name,
                               "total_steps": self.total_steps,
                               "preflight": pf, "resume_dir": resume_dir,
                               "resumed_tensors": n_loaded})
            self._write_run_json(pf, extra)
            self._prepared = True
            if self.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            self._progress(progress, 0.95, "准备完成")
            return self._prep
        except Exception:
            self.release()
            raise

    def _guard_base(self) -> None:
        """防线 1：训练前建立/校验底座快照。"""
        if not self.guard.load_manifest():
            self.guard.snapshot(hashes=True)
            self._log("已建立底座快照（SHA-256）")
        vr = self.guard.verify(hashes=False)
        if not vr.ok:
            raise RuntimeError(
                f"底座校验不通过：变更 {vr.changed} · 缺失 {vr.missing}\n"
                "训练前必须确认底座与快照一致，否则无法证明结果可信。")
        self.report.base_verify = {"ok": True, "checked": vr.checked,
                                   "changed": [], "missing": [], "when": "before"}

    def _write_run_json(self, pf: Dict[str, Any], extra: Dict[str, Any]) -> None:
        cfg_d = self.cfg.to_dict()
        cfg_d["replay_dataset"] = self.replay_dataset      # 存**解析后**的名字
        data = {
            "arch": self.ARCH, "run": self.run_name, "dataset": self.dataset,
            "status": RN.STATUS_RUNNING, "created_at": time.time(),
            "config": cfg_d, "options": self.options_dict(),
            "device": self.device, "model_dir": self.model_dir,
            "total_steps": self.total_steps,
            "steps_per_epoch": int(pf.get("steps_per_epoch") or 0),
            "adapter_params": int(self.report.adapter_params or 0),
            "base_params": int(self.report.base_params or 0),
            "data": dict(self.report.data),
            "notes": [dict(x) for x in self.report.notes],
            "steps": 0, "epochs": 0, "history": [],
            "resume_from": (self._prep.get("resume_dir") or self.resume_from or ""),
            "resumed_tensors": int(self._prep.get("resumed_tensors") or 0),
        }
        data.update(self._extra_run_fields(pf, extra))
        RN.write_run(self.run_name, data)

    def _extra_run_fields(self, pf: Dict[str, Any],
                          extra: Dict[str, Any]) -> Dict[str, Any]:
        """子类往 run.json 顶层补的字段（必须是 JSON 可序列化的）。"""
        return {}

    def options_dict(self) -> Dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, progress: Optional[Callable[[float, str], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> TrainReport:
        """跑完一次完整训练。**任何异常都不会把显存留在卡上**。"""
        if not self._prepared:
            self.prepare(progress)
        rep = self.report
        cfg = self.cfg
        t0 = time.perf_counter()
        status = RN.STATUS_DONE
        n_epochs = max(1, int(cfg.epochs))
        halt = ""

        try:
            # ---- 基线：LoRA 的 B 矩阵零初始化，此刻输出严格等于纯底座 ----
            # 这个数很关键：它把「降了多少」从「相对随机初始化」变成
            # 「相对底座在目标数据上的真实水平」，后者才是用户关心的。
            self._progress(progress, 0.05, "评估底座基线")
            base_val = self.evaluate()
            if base_val is None:
                rep.note("warn", "无法评估基线（验证集为空），"
                                 "报告里不会有「相对底座的改善」这个数")
            else:
                self._handle_eval(base_val, epoch=0, save=False)
                self._log(f"底座基线 val={base_val:.4f}"
                          "（LoRA 零初始化，此刻等价于不加载 adapter）")

            for epoch in range(self._resume_epoch, n_epochs):
                if halt:
                    break
                batches, replay_info = self._epoch_batches(epoch)
                if replay_info:
                    rep.replay = replay_info
                    if replay_info.get("note") and epoch == self._resume_epoch:
                        self._log(f"回放：{replay_info['note']}")
                n_rep = int(replay_info.get("n_replay") or 0) if replay_info else 0
                self._log(f"=== epoch {epoch+1}/{n_epochs} · {len(batches)} 个 batch"
                          f" · 回放 {n_rep} ===")

                # 两套累加器：win 是「自上次写日志以来」（写完就清），
                # epo 是整轮。混用一套会让日志里的 loss 除错分母。
                win = self._new_acc()
                epo = self._new_acc()
                te = time.perf_counter()
                for items in batches:
                    if should_stop and should_stop():
                        halt, status = "用户手动停止", RN.STATUS_STOPPED
                        rep.stop_reason = halt
                        self._log(halt)
                        break
                    if cfg.max_steps > 0 and self._step >= int(cfg.max_steps):
                        halt = f"达到 max_steps={cfg.max_steps}（硬刹车）"
                        rep.stop_reason = halt
                        self._log(halt)
                        break

                    tb = time.perf_counter()
                    r = self._train_micro_batch(items)
                    dt = time.perf_counter() - tb
                    if r is None:
                        win["skip"] += 1
                        epo["skip"] += 1
                        continue
                    for a in (win, epo):
                        a["loss"] += float(r.get("loss", 0.0))
                        a["nb"] += 1
                        a["n"] += int(r.get("n", 0))
                    win["sec"] += dt
                    epo["sec"] += dt

                    if self._accum < int(cfg.grad_accum):
                        continue
                    gn = self._optimizer_step()
                    win["gn"] = epo["gn"] = gn

                    log_every = int(self.log_every())
                    if log_every > 0 and self._step % log_every == 0:
                        el = time.perf_counter() - t0
                        eta = el / max(1, self._step) * max(
                            0, self.total_steps - self._step)
                        wl = win["loss"] / max(1, win["nb"])
                        self._log(
                            f"step {self._step}/{self.total_steps} loss={wl:.4f} "
                            f"lr={self._cur_lr():.2e} grad={gn:.3f} "
                            f"前向占比={win['sec']/max(1e-6, el):.0%} "
                            f"ETA {eta/60:.1f}min")
                        self._record(self._step, epoch, train=wl,
                                     grad_norm=round(gn, 4), samples=win["n"])
                        self._progress(
                            progress,
                            min(0.97, 0.05 + 0.9 * self._step / self.total_steps),
                            f"epoch {epoch+1} · step {self._step}/{self.total_steps}"
                            f" · loss {wl:.4f}")
                        win = self._new_acc()

                    # 按 optimizer step 评估。小数据集上 eval_every 可能整个 run
                    # 都触发不到（20 条数据、global_batch=4 → 每轮 5 个 step，
                    # eval_every=100 永远不到），所以每轮末尾必定再评一次兜底。
                    if cfg.eval_every > 0 and self._step % int(cfg.eval_every) == 0:
                        self._progress(progress,
                                       min(0.97, self._step / self.total_steps),
                                       f"评估中（step {self._step}）")
                        if self._handle_eval(self.evaluate(), epoch):
                            halt, status = rep.stop_reason, RN.STATUS_STOPPED
                            break

                # 本轮没凑满的梯度也要冲掉，否则它会混进下一轮第一个 batch，
                # 相当于跨 epoch 拼了一个更大的 batch，学习率与梯度就对不上了。
                if self._accum > 0 and not halt:
                    left = self._accum
                    self._optimizer_step()
                    self._log(f"epoch {epoch+1} 末尾冲掉不足一批的 {left} 个微批梯度")

                rep.epochs = epoch + 1
                self._log(f"epoch {epoch+1} 完成："
                          f"{(time.perf_counter()-te)/60:.1f} 分钟 · "
                          f"均 loss {epo['loss']/max(1, epo['nb']):.4f} · "
                          f"{epo['n']} 条样本"
                          + (f" · 跳过 {epo['skip']} 个空 batch" if epo["skip"] else ""))
                rep.final_train = epo["loss"] / max(1, epo["nb"])
                RN.update_run(self.run_name, steps=self._step, epochs=rep.epochs,
                              seconds=time.perf_counter() - t0,
                              best_val=rep.best_val, final_train=rep.final_train)

                # 轮末兜底评估。但如果刚刚已经在这个 step 上评过（最后一个 batch
                # 恰好撞上 eval_every），再评一次就是**重复计数**：
                # 两次 val 完全相同 → EarlyStopper 认为「没改善」→ bad_count +1，
                # 白白吞掉一次耐心，运气差能把早停误触发。
                # halt 之后也要评：这是拿到最终数字的唯一机会，而且保险库
                # 需要一个 val 当 metric 才能把最终权重存进去。
                if self._last_eval_step != self._step:
                    self._progress(progress, min(0.97, (epoch + 1) / n_epochs),
                                   f"epoch {epoch+1} 轮末评估")
                    if self._handle_eval(self.evaluate(), epoch) and not halt:
                        halt, status = rep.stop_reason, RN.STATUS_STOPPED
                if halt:
                    break

            if not halt:
                snap = self._train_snapshot()
                if snap is not None:
                    rep.final_train = float(snap)
                    if rep.best_val is not None and snap < rep.best_val * 0.7:
                        rep.note("warn",
                                 f"train loss {snap:.4f} 远低于 val "
                                 f"{rep.best_val:.4f}（不到它的 70%）—— 模型在背"
                                 "训练集而不是学规律。建议降 rank、提高 "
                                 "dropout/weight_decay 或加大回放比例。")
        except Exception as e:
            import traceback
            status = RN.STATUS_FAILED
            rep.error = f"{type(e).__name__}: {e}"
            rep.note("error", rep.error)
            self._log("训练异常：" + rep.error)
            self._log(traceback.format_exc(limit=6))
        finally:
            self._finalize(status, t0)
        return rep

    def log_every(self) -> int:
        return 10

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _finalize(self, status: str, t0: float) -> None:
        """正常、早停、手动停、异常四条路径都要走这里。

        顺序不能改：峰值显存 / 底座复校 / 存最终权重 / 漂移体检
        都需要模型还在，必须全部排在 `release()` 之前。
        """
        import torch
        rep = self.report
        rep.seconds = time.perf_counter() - t0
        rep.steps = self._step

        try:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                rep.vram_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:
            pass

        # ---- 防线 1：训练后再校一次底座 ----
        # size+mtime 足以发现「文件被重写过」（任何写入都会改 mtime）。
        # 不用 SHA-256：再哈希一遍几个 GB 要几十秒，而它只能多发现
        # 「内容变了但 size 与 mtime 都没变」这种现实中不会发生的情况。
        try:
            vr = self.guard.verify(hashes=False)
            rep.base_verify = {"ok": bool(vr.ok), "checked": vr.checked,
                               "changed": list(vr.changed),
                               "missing": list(vr.missing), "when": "after"}
            if not vr.ok:
                rep.note("error",
                         f"**底座发生变化**：{vr.changed or vr.missing}。"
                         "本次训练结果不可信，请到「泛化保护」页重建快照。")
                self._log(f"⚠ 底座校验不通过：{vr.changed or vr.missing}")
            else:
                self._log(f"底座复校通过（{vr.checked} 个文件未被改动）")
        except Exception as e:
            rep.base_verify = {"ok": None, "error": f"{type(e).__name__}: {e}"}

        # ---- 存最终权重 ----
        fin_path = ""
        try:
            if self.model is not None and self.vault_obj is not None and self._step > 0:
                if self._last_val is not None:
                    info = self.vault_obj.save(
                        lambda d: self._write_adapter(d, rep.epochs, self._last_val),
                        rep.epochs, self._step, float(self._last_val),
                        extra={"kind": "final", "arch": self.ARCH,
                               "run": self.run_name})
                    fin_path = info.path
                    rep.n_checkpoints = len(self.vault_obj.list())
                else:
                    # 没 val 时**不能**进保险库：保险库按 metric 排序（mode=min），
                    # 把 train loss 和 val loss 混在同一个排序里，
                    # `activate("best")` 就会挑中过拟合的那份。另存一份。
                    fin_path = os.path.join(RN.run_dir(self.run_name), "final")
                    shutil.rmtree(fin_path, ignore_errors=True)
                    os.makedirs(fin_path, exist_ok=True)
                    self._write_adapter(fin_path, rep.epochs, None)
                if not self._synced and fin_path:
                    RN.sync_adapter(self.run_name, fin_path)
                    self._synced = True
                    self._log("全程没有一次 val 改善，已把最终权重同步到 adapter/"
                              "（它很可能不如底座，用前请先看漂移体检）")
                rep.adapter_dir = RN.adapter_dir(self.run_name)
        except Exception as e:
            rep.note("error", f"保存最终权重失败：{type(e).__name__}: {e}")
            self._log(f"保存最终权重失败：{e}")

        # ---- 漂移体检（防线 9）----
        try:
            if self.model is not None:
                dr = GD.analyze_drift(self.model, device="cpu")
                lv, advice = dr.level
                rep.drift = {"global_rel": round(dr.global_rel, 6),
                             "max_rel": round(dr.max_rel, 6),
                             "max_name": dr.max_name,
                             "mean_rel": round(dr.mean_rel, 6),
                             "n_layers": dr.n_layers,
                             "seconds": round(dr.seconds, 2),
                             "level": lv, "advice": advice}
                self._log(f"漂移体检：全局 {dr.global_rel:.4f} · {lv}")
                if dr.global_rel >= 0.12:
                    rep.note("warn",
                             f"权重漂移 {dr.global_rel:.4f} 已进入「偏大」区间："
                             "底座被改写得比较多，通用能力大概率受损。"
                             "建议把推理端的 adapter 强度调到 0.6~0.8，或降 rank 重训。")
                elif dr.global_rel < 0.005 and self._step > 0:
                    rep.note("warn",
                             f"漂移只有 {dr.global_rel:.4f}，几乎没改到底座。"
                             "可能是学习率太小、步数太少，或者梯度根本没回传"
                             "（看日志里的 grad 列）。")
        except Exception as e:
            rep.drift = {"error": f"{type(e).__name__}: {e}"}

        # ---- 写 run.json ----
        try:
            imp = rep.improved
            summary = (f"val {rep.first_val:.4f} → {rep.best_val:.4f}"
                       f"（降 {imp*100:.1f}%）"
                       if (imp is not None and rep.first_val and rep.best_val)
                       else (f"跑了 {self._step} 步，无可比的 val" if self._step
                             else "未开始训练"))
            RN.update_run(
                self.run_name, status=status,
                finished_at=time.time(), seconds=round(rep.seconds, 1),
                steps=rep.steps, epochs=rep.epochs,
                best_val=rep.best_val, first_val=rep.first_val,
                final_train=rep.final_train, improved=imp,
                stopped_early=rep.stopped_early, stop_reason=rep.stop_reason,
                adapter_params=rep.adapter_params, base_params=rep.base_params,
                n_checkpoints=rep.n_checkpoints,
                vram_peak_gb=rep.vram_peak_gb,
                history=[dict(h) for h in rep.history],
                notes=[dict(x) for x in rep.notes],
                drift=rep.drift, base_verify=rep.base_verify,
                replay=rep.replay, data=dict(rep.data),
                adapter_dir=rep.adapter_dir, error=rep.error, note=summary)
        except Exception as e:
            self._log(f"写 run.json 失败：{e}")

        rep.ok = (status != RN.STATUS_FAILED and self._step > 0 and not rep.error)
        self._log(f"训练结束（{RN.STATUS_LABELS.get(status, (status,))[0]}）："
                  f"{self._step} 步 · {rep.seconds/60:.1f} 分钟"
                  + (f" · 峰值显存 {rep.vram_peak_gb:.2f} GB"
                     if rep.vram_peak_gb else ""))
        self.release()

    def release(self) -> None:
        """归还显存。**这里的顺序是实打实踩坑踩出来的，不要改**：

            ① 还原模型状态 → ② 断开所有引用 → ③ gc.collect() → ④ empty_cache()

        gc 放在断引用之前是没用的：那时循环引用对 gc 而言仍然可达，
        根本不会被回收。features_probe 里就因为这个顺序错，
        卸载后实测残留 1.82 GB（整个 GPT2 钉在卡上）。
        训练器退出时做不对这件事，下一次推理就会静默溢出。
        """
        import gc
        try:
            if self.model is not None and self.prev_state:
                self._restore_model()
        except Exception:
            pass
        try:
            self._drop_model_refs()
        except Exception:
            pass
        # opt / sched 都持有参数引用（AdamW 的 state 里存着动量张量），
        # 不断干净就等同于没释放。
        self.model = None
        self.opt = None
        self.sched = None
        self.prev_state = {}
        self.vault_obj = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        self._prepared = False

    # ------------------------------------------------------------------
    # 回放池解析（两个训练器共用）
    # ------------------------------------------------------------------
    def resolve_replay(self, select: Optional[Callable[[str, Sequence[str]],
                                                       Sequence[str]]] = None
                       ) -> Tuple[List[GD.Notice], str]:
        """定下回放集，返回 (提示列表, 解析出的数据集名)。

        `select(name, ready_ids)` 由子类提供长度/可用性筛选 ——
        GPT 看 codes 帧数，CFM 看 mel 帧数，两边的阀值完全不同。

        一个必须处理的陷阱：回放集**不能**包含验证集样本，
        否则 val loss 里混进了模型每轮都看过的数据，早停就废了。
        """
        from webui_app.training import dataset as DS

        notes: List[GD.Notice] = []
        src = (self.cfg.replay_source or "none").lower()
        ratio = float(self.cfg.replay_ratio or 0.0)
        self.replay_src, self.replay_ids, self.replay_dataset = "none", [], ""

        if ratio <= 0.0:
            if src != "none":
                notes.append(GD.Notice("info",
                                       "replay_ratio=0，回放源已忽略。"
                                       "这是灾难性遗忘最主要的成因，确认再开训。"))
            return notes, ""
        if src == "none":
            notes.append(GD.Notice("warn",
                                   f"replay_ratio={ratio:.0%} 但回放源是 none，"
                                   "实际不会有任何回放。请选一个数据集或先生成底座蒸馏集。"))
            return notes, ""

        name = GD.DISTILL_DATASET if src == "base_distill" else self.cfg.replay_dataset
        if not name or not DS.exists(name):
            hint = ("到「训练」页点「生成底座回放集」" if src == "base_distill"
                    else "到「数据集」页选一个现成的")
            notes.append(GD.Notice("warn",
                                   f"回放数据集 `{name or '(未选)'}` 不存在，{hint}。"
                                   "本次训练将**没有回放**，泛化风险显著上升。"))
            return notes, name or ""

        ids = list(DS.ids(name, only_ready=True))
        if select:
            ids = list(select(name, ids))
        if name == self.dataset:
            # 与目标集重叠时，回放等于没回放（反而把目标数据的权重变相提高了）。
            # 宁可报个空池让用户去配，也不要静默地把「30% 回放」变成假的。
            banned = set(self.train_ids) | set(self.val_ids)
            ids = [u for u in ids if u not in banned]
            if not ids:
                notes.append(GD.Notice("warn",
                                       "回放数据集与目标数据集相同，剔除重叠后回放池为空。"
                                       "请另备一份通用语音数据集，或生成底座蒸馏集。"))
        self.replay_src, self.replay_ids, self.replay_dataset = src, ids, name
        if ids and src == "base_distill":
            notes.append(GD.Notice("info",
                                   f"回放源 = 底座蒸馏集 `{name}`（{len(ids)} 条）。"
                                   "它教的是「底座自己遇到这些通用文本时该怎么说」，"
                                   "正好把 LoRA 想覆盖掉的那部分能力钉住。"))
        return notes, name
