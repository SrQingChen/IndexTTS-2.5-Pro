"""LoRA 泛化保护 —— 底座保护 / 参数可配置 / 漂移体检 / 早停 / 权重保险库。

微调最常见的翻车方式不是「学不像」，而是**学会了角色、忘了怎么正常说话**：
读错字变多、韵律发飘、没见过的句式直接崩。本模块把所有**在本机可控**的
防退化手段集中在一处，训练器（gpt_lora / cfm_lora）只消费这里的接口。

十道防线（按重要性排序）：

    1. 底座只读     原始 gpt.pth / s2mel.pth 全程不被改写，训练前做哈希快照，
                    每次训练与合并前重新校验；输出路径落在 checkpoints/ 里直接拒绝
    2. LoRA 本身    底座 requires_grad=False，ΔW 是低秩旁路，天然限制了偏移上限
    3. rank/alpha   容量旋钮。rank 越小、alpha/rank 越低，能记住的东西越少，
                    但也越难把底座带偏
    4. target_modules 只注入注意力投影，不动 MLP / 词嵌入 / 输出头 ——
                    注入面越窄，通用能力损失越小
    5. dropout + weight_decay  常规正则
    6. 回放混合     按 replay_ratio 掺入通用样本，直接对抗灾难性遗忘
    7. 早停         用自己数据的 val 集盯 loss，反弹即停
    8. 保险库       只保留 top-K checkpoint，原子写入，随时回滚
    9. 强度旋钮     **推理期**把 adapter 缩放 0~1，0 = 纯底座。
                    不用重训就能在「像角色」与「稳」之间连续调
   10. 漂移体检     训练后量化 ‖ΔW‖/‖W‖，给出过拟合风险等级

本机**无法控制**的部分（不在这里管）：训练集本身的多样性、录音质量、
参考音频的语种覆盖 —— 这些属于数据问题，见 dataset.py 的体检与统计。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT

TRAINING_ROOT = os.path.join(PROJECT_ROOT, "training_runs")
ADAPTER_SUBDIR = "adapter"
ACTIVE_SUBDIR = "active"          # 推理端实际加载的 adapter（回滚就是改写这里）
MANIFEST_NAME = "base_manifest.json"

# 底座文件：这些**永远**不能作为训练输出被写入
PROTECTED_FILES = (
    "gpt.pth", "s2mel.pth", "bigvgan.pth", "codec.pth",
    "campplus.pth", "speech_tokenizer_v2.onnx", "qwen0.6b",
    "config.yaml", "pinyin.vocab",
)
MODEL_DIR_NAME = "checkpoints"

_HASH_CHUNK = 1 << 22             # 4 MiB

# 漂移风险阈值（‖ΔW‖_F / ‖W‖_F，按参数量加权）
DRIFT_LEVELS: Tuple[Tuple[float, str, str], ...] = (
    (0.02, "🟢 保守", "底座几乎没被带动，通用能力基本无损。若角色相似度不够，可提高 rank / alpha"),
    (0.05, "🟢 正常", "典型的成功 LoRA 区间，音色/语气有明显改变而通用能力保持"),
    (0.12, "🟡 偏激进", "已开始明显改写底座行为。建议用「强度旋钮」降到 0.6~0.8 再试听"),
    (0.25, "🟠 过拟合风险", "偏移过大，很可能出现读错字、韵律发飘。建议回滚到更早的 checkpoint"),
    (float("inf"), "🔴 严重漂移", "底座已被大幅改写，通用能力大概率严重退化。强烈建议回滚"),
)


@dataclass
class Notice:
    """一条配置提示。level: error | warn | info"""
    level: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "message": self.message}


# ===========================================================================
# 1. 可配置的 LoRA / 训练参数
# ===========================================================================

# 注入面预设（数据驱动：具体可用项由 scan_targets() 从真实模型里扫出来，
# 这里只给几个常见组合的**建议值**，UI 会把扫描结果一并列出供勾选）
TARGET_PRESETS: Dict[str, Dict[str, Any]] = {
    "attn": {
        "label": "仅注意力投影（最抗遗忘，推荐）",
        "gpt": ["attn/c_attn", "attn/c_proj"],
        # DiT 的注意力叫 wqkv / wo（gpt_fast/model.py:253,254），不是 qkv / out_proj。
        # 旧值在真模型上一个都匹不上，resolve_target_patterns 会返回空，
        # 然后 inject_lora 报「注入面一个都没匹配上」—— CFM 训练直接跑不了。
        "cfm": ["attention/wqkv", "attention/wo"],
    },
    "attn_mlp": {
        "label": "注意力 + MLP（容量更大，遗忘风险上升）",
        "gpt": ["attn/c_attn", "attn/c_proj", "mlp/c_fc", "mlp/c_proj"],
        "cfm": ["attention/wqkv", "attention/wo",
                "feed_forward/w1", "feed_forward/w2", "feed_forward/w3"],
    },
    "all_linear": {
        "label": "全部线性层（最激进）",
        "gpt": ["c_attn", "c_proj", "c_fc"],
        "cfm": ["*"],
    },
}

# `target_modules=["*"]` 展开时要跳过的 pattern：这两个模块是 DiT 里的**死代码**。
#   · cond_embedder：diffusion_transformer.py:206-207 把 cond_in_module **硬编码**
#     成 cond_projection，用 cond_embedder 的那一行已经被注释掉了；
#   · content_mask_embedder：整个 DiT.forward 从头到尾没引用过它。
# 给死模块注 LoRA 会得到一堆**永远拿不到梯度**的可训练参数：不报错，
# 但「可训练参数量」与「漂移体检」两个数字都会失真（分母里多了不动的层）。
# 注：`nn.Conv1d` / `nn.Conv2d`（CFM 里只有 `estimator.conv2`）本来就不在
# `scan_targets` 的收录范围内，所以 `"*"` 展开后也不包含它们 ——
# 这里的「全部线性层」指的是**全部被扫描到的**线性层，账目因此始终对得上。
DEAD_MODULE_PATTERNS = {"estimator/cond_embedder", "estimator/content_mask_embedder"}

# 三档出厂预设：保守 / 均衡 / 激进
CONFIG_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": dict(
        rank=4, alpha=8, dropout=0.10, use_rslora=True,
        lr=5e-5, weight_decay=0.05, warmup_ratio=0.10,
        epochs=2, replay_ratio=0.5, val_patience=2,
        grad_clip=0.5, keep_checkpoints=5, eval_every=50,
        target_preset="attn",
    ),
    "balanced": dict(
        rank=8, alpha=16, dropout=0.05, use_rslora=False,
        lr=1e-4, weight_decay=0.01, warmup_ratio=0.06,
        epochs=4, replay_ratio=0.3, val_patience=3,
        grad_clip=1.0, keep_checkpoints=3, eval_every=100,
        target_preset="attn",
    ),
    "aggressive": dict(
        rank=32, alpha=64, dropout=0.0, use_rslora=False,
        lr=2e-4, weight_decay=0.0, warmup_ratio=0.03,
        epochs=8, replay_ratio=0.0, val_patience=0,
        grad_clip=1.0, keep_checkpoints=2, eval_every=200,
        target_preset="attn_mlp",
    ),
}

PRESET_NOTES: Dict[str, str] = {
    "conservative": "最抗遗忘：rank=4 + rsLoRA + 50% 回放 + 早停。角色相似度提升有限，"
                    "但几乎不可能把底座带坏。**数据少于 5 分钟时请用这档**。",
    "balanced": "默认档。rank=8、30% 回放、按 val loss 早停。"
                "大多数「学某个角色的音色和语气」的需求都够用。",
    "aggressive": "最像角色，也最容易过拟合：无回放、无早停、注入 MLP。"
                  "只建议在数据充足（≥30 分钟）且已用均衡档试过之后使用。",
}


@dataclass
class LoRAConfig:
    """一次 LoRA 训练的全部可调参数。

    每个字段都直接对应一道防线，没有「写了但不生效」的摆设项。
    序列化成 JSON 存进训练目录，保证任何一次训练都能被精确复现。
    """

    # ---- 容量（防线 3）----
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    use_rslora: bool = False          # rank-stabilized：scaling = alpha/√rank

    # ---- 注入面（防线 4）----
    # 元素形如 "attn/c_proj"（用 / 分隔以区分 attn.c_proj 与 mlp.c_proj），
    # 注入时由 build_target_regex() 合成一条正则交给 PEFT。
    target_modules: List[str] = field(
        default_factory=lambda: list(TARGET_PRESETS["attn"]["gpt"]))
    target_preset: str = "attn"

    # ---- 优化器（防线 5）----
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    epochs: int = 4
    max_steps: int = -1               # >0 时优先于 epochs，作为硬刹车
    batch_size: int = 1
    grad_accum: int = 4

    # ---- 回放（防线 6）----
    replay_ratio: float = 0.3         # 每个 epoch 里通用样本占比
    replay_source: str = "none"       # none | dataset | base_distill
    replay_dataset: str = ""          # replay_source=dataset 时的数据集名

    # ---- 早停与保险库（防线 7、8）----
    eval_every: int = 100             # 每多少个 optimizer step 评估一次
    val_patience: int = 3             # 0 = 关闭早停
    val_min_delta: float = 1e-4
    keep_checkpoints: int = 3

    # ---- 推理期强度（防线 9）----
    adapter_scale: float = 1.0        # 0 = 完全等于底座，1 = 完整 LoRA

    # ---- 显存 / 复现 ----
    bf16: bool = True
    grad_checkpointing: bool = True
    seed: int = 42

    # ------------------------------------------------------------------
    @property
    def effective_scale(self) -> float:
        """旁路增益 = alpha/rank（rsLoRA 时 alpha/√rank）。

        这个数比 rank 本身更能说明「LoRA 有多大话语权」：
        rank=32/alpha=32 的增益只有 1.0，比 rank=4/alpha=16（增益 4.0）温和得多。
        """
        if self.rank <= 0:
            return 0.0
        return self.alpha / math.sqrt(self.rank) if self.use_rslora \
            else self.alpha / self.rank

    @property
    def global_batch(self) -> int:
        return max(1, int(self.batch_size) * int(self.grad_accum))

    @classmethod
    def preset(cls, name: str) -> "LoRAConfig":
        kw = dict(CONFIG_PRESETS.get(name, CONFIG_PRESETS["balanced"]))
        tp = kw.pop("target_preset", "attn")
        cfg = cls(**kw)
        cfg.target_preset = tp
        cfg.apply_target_preset(tp, "gpt")
        return cfg

    def apply_target_preset(self, preset: str, arch: str = "gpt") -> None:
        """按预设与目标架构（gpt / cfm）填入 target_modules。"""
        p = TARGET_PRESETS.get(preset) or TARGET_PRESETS["attn"]
        mods = p.get(arch) or p.get("gpt") or []
        self.target_preset = preset
        self.target_modules = list(mods)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["_effective_scale"] = self.effective_scale
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoRAConfig":
        known = {f for f in cls.__dataclass_fields__}      # noqa: 忽略 _effective_scale 等
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    # ------------------------------------------------------------------
    def validate(self, n_samples: Optional[int] = None) -> List[Notice]:
        """检查配置自身是否合理。返回问题列表（空 = 没问题）。

        n_samples 给定时会额外做「数据量 vs 训练强度」的匹配检查 ——
        这是过拟合最强的预测因子，比任何超参都重要。
        """
        n: List[Notice] = []

        def err(m): n.append(Notice("error", m))
        def warn(m): n.append(Notice("warn", m))
        def info(m): n.append(Notice("info", m))

        if not 1 <= int(self.rank) <= 256:
            err(f"rank={self.rank} 超出 1~256")
        if int(self.alpha) < 1:
            err(f"alpha={self.alpha} 必须 ≥ 1")
        if not 0.0 <= float(self.dropout) <= 0.9:
            err(f"dropout={self.dropout} 超出 0~0.9")
        if not 0.0 < float(self.lr) < 0.1:
            err(f"lr={self.lr} 不合理（LoRA 常用 1e-5 ~ 5e-4）")
        if not 0.0 <= float(self.replay_ratio) < 0.95:
            err(f"replay_ratio={self.replay_ratio} 超出 0~0.95")
        if not 0.0 <= float(self.adapter_scale) <= 2.0:
            err(f"adapter_scale={self.adapter_scale} 超出 0~2")
        if int(self.epochs) < 1 and int(self.max_steps) <= 0:
            err("epochs 与 max_steps 至少要有一个是正的，否则什么都不会训")
        if n:
            return n

        # ---- 泛化风险（warn / info）----
        if self.rank >= 32:
            warn(f"rank={self.rank} 容量偏大。小数据集上高 rank 会**记住**训练样本"
                 "而不是学到音色，表现为：训练集里的句子完美，换个说法就崩。"
                 "8GB 显存下建议 4~16。")
        if self.effective_scale > 4.0:
            warn(f"旁路增益 alpha/rank = {self.effective_scale:.2f} 偏大（>4）。"
                 "增益越大，LoRA 对输出的话语权越强，底座被带偏得越快。")
        if self.dropout <= 0.0 and self.rank >= 16:
            warn("dropout=0 且 rank≥16：没有任何随机正则。建议至少设 0.05。")
        if self.weight_decay <= 0.0:
            info("weight_decay=0：不对 adapter 权重做衰减。"
                 "小数据集上设 0.01~0.05 能明显压制过拟合。")
        if self.replay_ratio <= 0.0:
            warn("**回放比例 = 0**，这是灾难性遗忘最主要的成因。"
                 "训练全程只看目标角色的数据，模型会逐步丢掉通用发音能力。"
                 "建议 ≥0.2，数据越少比例越高。")
        if self.val_patience <= 0:
            warn("早停已关闭（val_patience=0）。训练会一路跑到 epochs 结束，"
                 "val loss 反弹之后继续练出来的都是过拟合。")
        if self.target_preset == "all_linear":
            warn("注入面 = 全部线性层。注入越多，通用能力损失越大。"
                 "只注入注意力投影通常就足够改变音色与语气。")
        if not self.grad_checkpointing:
            info("关闭了梯度检查点：速度快约 30%，但显存占用明显上升。")

        # ---- 数据量 vs 训练强度 ----
        if n_samples is not None and n_samples > 0:
            steps_per_epoch = max(1, n_samples // self.global_batch)
            total = steps_per_epoch * max(1, int(self.epochs))
            if n_samples < 20:
                err(f"只有 {n_samples} 条可用样本。这个量级不足以微调，"
                    "请先到「数据集」页导入更多音频，或改用参考音频 + 情感向量。")
            elif n_samples < 60:
                warn(f"可用样本 {n_samples} 条（约 {total} 个 step）。"
                     "属于极小数据集，请用「保守」档并把 replay_ratio 提到 0.5 以上。")
            if total > 20000:
                warn(f"总步数 {total} 偏多（>20000）。8GB 单卡上这会跑很久，"
                     "且早停大概率会先触发 —— 建议直接调低 epochs。")
            if n_samples < self.global_batch * 2:
                warn(f"样本数 {n_samples} 不足一个 epoch 两个 batch"
                     f"（global_batch={self.global_batch}）。"
                     "请把 batch_size × grad_accum 调小，否则 val 集会是空的。")
        return n

    def risk_markdown(self, n_samples: Optional[int] = None,
                      adapter_params: Optional[int] = None,
                      base_params: Optional[int] = None) -> str:
        """把配置与风险渲染成 Markdown（训练页右侧的「本次训练体检」）。"""
        L = ["| 项 | 值 | 含义 |", "|---|---|---|",
             f"| rank | `{self.rank}` | 旁路矩阵的秩，容量上限 |",
             f"| alpha | `{self.alpha}` | 缩放系数 |",
             f"| 旁路增益 | `{self.effective_scale:.2f}` | alpha/rank，"
             f"越大越容易带偏底座 |",
             f"| dropout | `{self.dropout}` | 随机正则 |",
             f"| 注入面 | `{self.target_preset}` | "
             f"{len(self.target_modules)} 类模块 |",
             f"| lr | `{self.lr:g}` | 学习率 |",
             f"| weight_decay | `{self.weight_decay}` | 权重衰减 |",
             f"| 回放比例 | `{self.replay_ratio:.0%}` | 通用样本占比，抗遗忘 |",
             f"| 早停耐心 | `{self.val_patience}` | "
             f"{'**已关闭**' if self.val_patience <= 0 else str(self.val_patience) + ' 次'} |",
             f"| 保留 ckpt | `{self.keep_checkpoints}` | 可回滚的档位数 |",
             f"| 推理强度 | `{self.adapter_scale:.2f}` | 0=纯底座，1=完整 LoRA |"]
        if adapter_params and base_params:
            pct = adapter_params / base_params * 100 if base_params else 0.0
            L.append(f"| 可训练参数 | `{adapter_params/1e6:.2f} M` | "
                     f"占底座 {base_params/1e6:.0f} M 的 **{pct:.2f}%** |")
        if n_samples:
            spe = max(1, n_samples // self.global_batch)
            L.append(f"| 数据量 | `{n_samples}` 条 | 每 epoch {spe} step，"
                     f"共约 {spe * max(1, self.epochs)} step |")
        L.append("")

        notes = self.validate(n_samples)
        if not notes:
            L.append("> ✅ 配置检查通过，没有发现泛化风险项。")
        else:
            icon = {"error": "✖", "warn": "⚠️", "info": "ℹ️"}
            for x in notes:
                L.append(f"> {icon[x.level]} **{x.level.upper()}** {x.message}")
                L.append(">")
            if L[-1] == ">":
                L.pop()
        return "\n".join(L)


# ===========================================================================
# 2. 注入面扫描与正则合成
# ===========================================================================

@dataclass
class TargetGroup:
    """一组可注入的模块。pattern 用 / 分隔以区分同名的不同层级。"""
    pattern: str          # "attn/c_proj"
    tail: str             # "attn.c_proj"
    kind: str             # Linear | Conv1D | Embedding
    count: int = 0
    params: int = 0
    in_features: int = 0
    out_features: int = 0
    examples: List[str] = field(default_factory=list)


def _module_io(m) -> Optional[Tuple[str, int, int]]:
    """返回 (类型, in, out)；不是可注入的层就返回 None。"""
    cls = type(m).__name__
    if cls == "Conv1D":                      # transformers.pytorch_utils.Conv1D
        w = getattr(m, "weight", None)
        if w is None or w.ndim != 2:
            return None
        return "Conv1D", int(w.shape[0]), int(w.shape[1])
    if cls in ("Linear", "NonDynamicallyQuantizableLinear"):
        return "Linear", int(m.in_features), int(m.out_features)
    if cls == "Embedding":
        return "Embedding", int(m.num_embeddings), int(m.embedding_dim)
    return None


def pattern_of(name: str) -> str:
    """从一个完整模块路径提出分组用的 pattern（用 `/` 分隔）。

    两种数字段要分开处理，否则 UI 上的注入面列表会变成一堆垃圾：

        · 数字在**中间**（`layers.0.skip_in_linear`）：它只是循环下标，
          跳过它取上一个有意义的名字 → `layers/skip_in_linear`，
          13 层归成一组。不跳的话会得到 13 个 `0/skip_in_linear`…`12/…`。
        · 数字在**末尾**（`t_embedder.mlp.0`）：nn.Sequential 的下标
          **就是叶子模块的名字本身**，只能连它一起用 → `mlp/0`。

    GPT2 的命名（`gpt.h.0.attn.c_attn`）两种情况都不踩，
    结果与旧版一致，所以现有预设不会失效。
    """
    parts = [p for p in str(name).split(".") if p]
    if not parts:
        return ""
    leaf = parts[-1]
    if leaf.isdigit():
        return "/".join(parts[-2:]) if len(parts) >= 2 else leaf
    keep = [p for p in parts[:-1] if not p.isdigit()]
    return f"{keep[-1]}/{leaf}" if keep else leaf


def scan_targets(model) -> List[TargetGroup]:
    """从真实模型里扫出所有可注入的层，按 pattern 分组。

    不硬编码模块名 —— IndexTTS 的 GPT 用 Conv1D、DiT 用 Linear，
    两边命名规则完全不同，写死的清单迟早对不上。扫出来的结果直接进 UI。

    分组必须看**两级**：GPT2 里 `attn.c_proj` 和 `mlp.c_proj` 同名，
    只看最后一级会把 MLP 一起注进去，注入面直接翻倍。
    """
    groups: Dict[str, TargetGroup] = {}
    for name, m in model.named_modules():
        io = _module_io(m)
        if io is None:
            continue
        kind, i, o = io
        pat = pattern_of(name)
        if not pat:
            continue
        key = f"{pat}|{kind}"
        g = groups.get(key)
        if g is None:
            g = TargetGroup(pattern=pat, tail=pat.replace("/", "."), kind=kind,
                            in_features=i, out_features=o)
            groups[key] = g
        g.count += 1
        g.params += sum(p.numel() for p in m.parameters(recurse=False))
        if len(g.examples) < 2:
            g.examples.append(name)
    out = list(groups.values())
    out.sort(key=lambda g: (-g.count, g.tail))
    return out


def build_target_regex(patterns: Sequence[str]) -> str:
    """把 `["attn/c_proj", "c_attn"]` 合成一条 PEFT 用的正则。

    PEFT 对 `target_modules` 的两种处理：
        · 传**列表** → 逐项 `key.endswith("." + item)`，无法区分同名层级
        · 传**字符串** → `re.fullmatch(s, key)`
    所以这里必须合成单条正则，才能精确控制注入面。

    每一级之间插入 `\\.(\\d+\\.)?`：pattern 里的循环下标已经被 `pattern_of`
    跳过了，但 PEFT 匹的是**真实模块名**（仍带着 `.0.` `.1.`），
    不插这个通配就一个也匹不上。
    注意连接符里的 `\\.` 不能省：只写 `(\d+\.)?` 会把原本的分隔点也后吃掉，
    变成 `attn(\d+\.)?c_attn` —— 它反而匹不上 `attn.c_attn`。
    """
    alts: List[str] = []
    for p in patterns or []:
        s = str(p).strip().strip("/").replace("/", ".")
        if not s:
            continue
        if s == "*":
            return ".*"
        alts.append(r"\.(\d+\.)?".join(re.escape(x) for x in s.split(".") if x))
    if not alts:
        return ""
    return r"(.*\.)?(" + "|".join(alts) + r")$"


def resolve_target_patterns(groups: Sequence[TargetGroup],
                            wanted: Sequence[str]) -> List[str]:
    """把用户勾的 pattern 对齐到扫描结果，丢掉模型里根本不存在的项。

    训练器换目标（GPT ↔ CFM）时预设清单里总有一半对不上，
    与其报错不如静默丢弃 + 记录，UI 会显示实际生效的注入面。

    `"*"` 必须在这里**展开成具体 pattern**，不能直接往下传：
    `build_target_regex(["*"])` 得到的是裸 `.*`，PEFT 拿它 `fullmatch`
    每一个模块名 —— 而 `BASECFM.criterion`（一个 `L1Loss`）也是注册在
    模块树里的，于是 PEFT 直接抛
    `ValueError: Target module L1Loss() is not supported`。
    CFM 的 all_linear 预设用的正是 `"*"`，所以这不是理论风险，是必崩。
    展开成 `scan_targets` 的结果就天然安全：那里只收 Linear / Conv1D / Embedding。
    """
    if any(str(w).strip().strip("/") == "*" for w in (wanted or [])):
        return [g.pattern for g in groups if g.pattern not in DEAD_MODULE_PATTERNS]
    have = {g.pattern for g in groups}
    out: List[str] = []
    for w in wanted or []:
        s = str(w).strip().strip("/")
        if s in have and s not in out:
            out.append(s)
    return out


def estimate_adapter_params(groups: Sequence[TargetGroup], cfg: LoRAConfig) -> int:
    """按 rank 估算 adapter 参数量：每层 r×(in+out)。

    估算而非实测 —— 注入前就要在 UI 上告诉用户「这会占多少显存」。
    """
    r = max(1, int(cfg.rank))
    total = 0
    for g in groups:
        if g.kind == "Embedding":
            total += r * (g.in_features + g.out_features) * g.count
        else:
            total += r * (g.in_features + g.out_features) * g.count
    return int(total)


def scan_targets_markdown(model, cfg: Optional[LoRAConfig] = None) -> str:
    groups = scan_targets(model)
    if not groups:
        return "_这个模型里没有可注入的 Linear / Conv1D / Embedding 层。_"
    L = ["| 模块 | 类型 | 层数 | in→out | 权重参数 | 已勾选 |", "|---|---|---|---|---|---|"]
    picked = set(cfg.target_modules) if cfg else set()
    for g in groups:
        L.append(f"| `{g.pattern}` | {g.kind} | {g.count} "
                 f"| {g.in_features}→{g.out_features} | {g.params/1e6:.2f} M "
                 f"| {'✅' if g.pattern in picked else ''} |")
    if cfg:
        est = estimate_adapter_params([g for g in groups if g.pattern in picked], cfg)
        L += ["", f"> 按 rank={cfg.rank} 估算，勾选部分共 **{est/1e6:.2f} M** 可训练参数。"]
    return "\n".join(L)


# ===========================================================================
# 3. 底座保护
# ===========================================================================

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_HASH_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class FileState:
    name: str
    path: str
    size: int
    mtime: float
    sha256: str = ""
    read_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifyReport:
    ok: bool = True
    checked: int = 0
    unchanged: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    size_only: bool = False          # True = 只比了 size+mtime，没算哈希
    seconds: float = 0.0

    def markdown(self) -> str:
        if self.ok:
            head = (f"> ✅ **底座完好**：{self.checked} 个文件"
                    f"{'（size+mtime 快速校验）' if self.size_only else '（SHA-256 全量校验）'}"
                    f"全部与训练前一致，耗时 {self.seconds:.1f}s")
        else:
            head = f"> ✖ **底座校验失败**：{self.seconds:.1f}s"
        L = [head]
        for f in self.changed:
            L.append(f"> ✖ `{f}` 内容已改变 —— 底座被动过了，训练结果不可信")
        for f in self.missing:
            L.append(f"> ✖ `{f}` 不存在")
        return "\n".join(L)


class BaseGuard:
    """底座只读保护。

    做三件事：训练前给 checkpoints/ 里的权重做哈希快照；
    训练与合并前重新校验；拒绝任何指向 checkpoints/ 的输出路径。

    哈希 3.2 GB 的 gpt.pth 约需数秒，所以 `verify()` 默认只比
    size+mtime（足以发现「文件被重写过」），深度校验要显式开。
    """

    def __init__(self, model_dir: Optional[str] = None,
                 train_root: Optional[str] = None):
        self.model_dir = os.path.abspath(model_dir or
                                         os.path.join(PROJECT_ROOT, MODEL_DIR_NAME))
        self.train_root = os.path.abspath(train_root or TRAINING_ROOT)
        self.manifest_path = os.path.join(self.train_root, MANIFEST_NAME)

    # ---------------- 快照 ----------------
    def protected_paths(self) -> List[str]:
        """底座目录里所有与 PROTECTED_FILES 同名的文件。"""
        out: List[str] = []
        if not os.path.isdir(self.model_dir):
            return out
        wanted = set(PROTECTED_FILES)
        for root, _dirs, files in os.walk(self.model_dir):
            for fn in files:
                if fn in wanted:
                    out.append(os.path.join(root, fn))
        return sorted(out)

    def snapshot(self, files: Optional[Sequence[str]] = None,
                 hashes: bool = True,
                 progress: Optional[Callable[[float, str], None]] = None
                 ) -> Dict[str, Any]:
        """记录底座文件的 size/mtime/sha256，写入 base_manifest.json。"""
        files = list(files) if files else self._default_files()
        states: Dict[str, FileState] = {}
        total = max(1, len(files))
        for i, p in enumerate(files):
            if progress:
                progress(i / total, f"哈希底座 {os.path.basename(p)}")
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
            rel = os.path.relpath(p, self.model_dir).replace("\\", "/")
            states[rel] = FileState(
                name=rel, path=os.path.abspath(p), size=st.st_size,
                mtime=st.st_mtime,
                sha256=_sha256(p) if hashes else "",
                read_only=not (st.st_mode & stat.S_IWRITE))
        man = {
            "created_at": time.time(),
            "model_dir": self.model_dir,
            "hashed": bool(hashes),
            "files": {k: v.to_dict() for k, v in states.items()},
        }
        os.makedirs(self.train_root, exist_ok=True)
        _atomic_write_json(self.manifest_path, man)
        if progress:
            progress(1.0, "底座快照完成")
        return man

    def _default_files(self) -> List[str]:
        """默认保护 checkpoints/ 下的全部 .pth / .onnx / .yaml / .vocab。"""
        if not os.path.isdir(self.model_dir):
            return []
        exts = (".pth", ".onnx", ".yaml", ".yml", ".vocab", ".bin", ".safetensors")
        out = []
        for root, _dirs, files in os.walk(self.model_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    out.append(os.path.join(root, fn))
        return sorted(out)

    def load_manifest(self) -> Dict[str, Any]:
        if not os.path.isfile(self.manifest_path):
            return {}
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # ---------------- 校验 ----------------
    def verify(self, hashes: bool = False,
               progress: Optional[Callable[[float, str], None]] = None
               ) -> VerifyReport:
        man = self.load_manifest()
        rep = VerifyReport(size_only=not hashes)
        files = (man or {}).get("files") or {}
        if not files:
            rep.ok = False
            rep.missing = ["<没有底座快照，请先执行「建立底座快照」>"]
            return rep
        t0 = time.perf_counter()
        total = max(1, len(files))
        for i, (rel, d) in enumerate(sorted(files.items())):
            if progress:
                progress(i / total, f"校验 {rel}")
            rep.checked += 1
            p = d.get("path") or os.path.join(self.model_dir, rel)
            if not os.path.isfile(p):
                rep.missing.append(rel)
                continue
            st = os.stat(p)
            same = (st.st_size == d.get("size"))
            if same and hashes and d.get("sha256"):
                same = (_sha256(p) == d["sha256"])
            if same:
                rep.unchanged.append(rel)
            else:
                rep.changed.append(rel)
        rep.seconds = time.perf_counter() - t0
        rep.ok = not rep.changed and not rep.missing
        return rep

    # ---------------- 只读锁 ----------------
    def set_read_only(self, on: bool = True,
                      files: Optional[Sequence[str]] = None) -> Tuple[int, List[str]]:
        """给底座文件加/去 Windows 只读属性。

        这是**第二道**保险：哈希校验是事后发现，只读属性是事前阻止。
        代价是自己想更新底模时得先解锁，所以 UI 里做成显式按钮。
        """
        man = self.load_manifest()
        paths = [d.get("path") for d in ((man or {}).get("files") or {}).values()] \
            if not files else list(files)
        done, failed = 0, []
        for p in paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                st = os.stat(p)
                mode = st.st_mode
                mode = mode & ~stat.S_IWRITE if on else mode | stat.S_IWRITE
                os.chmod(p, mode)
                done += 1
            except Exception:
                failed.append(os.path.basename(p))
        return done, failed

    def read_only_state(self) -> Dict[str, bool]:
        man = self.load_manifest()
        out = {}
        for rel, d in ((man or {}).get("files") or {}).items():
            p = d.get("path") or ""
            out[rel] = bool(p and os.path.isfile(p)
                            and not (os.stat(p).st_mode & stat.S_IWRITE))
        return out

    # ---------------- 输出路径安全 ----------------
    def assert_safe_output(self, path: str, must_not_exist: bool = False) -> str:
        """确认一个输出路径不会破坏底座。返回规范化后的绝对路径。

        三条硬规则：
            · 不许落在 checkpoints/ 里（包括子目录）
            · 不许覆盖任何 PROTECTED_FILES 里的文件名
            · must_not_exist 时不许覆盖已有文件
        """
        ap = os.path.abspath(path or "")
        md = self.model_dir.rstrip("\\/") + os.sep
        if ap.startswith(md) or ap == self.model_dir.rstrip("\\/"):
            raise PermissionError(
                f"拒绝写入底座目录：{ap}\n"
                f"训练产物必须放在 {self.train_root} 下。"
                "原始 gpt.pth / s2mel.pth 一旦覆盖就再也回不去了。")
        base = os.path.basename(ap)
        if base in PROTECTED_FILES:
            raise PermissionError(
                f"拒绝生成名为 `{base}` 的文件 —— 它与底座文件同名，"
                "极易在后续操作中被误当成底座加载。请换个名字。")
        if must_not_exist and os.path.exists(ap):
            raise FileExistsError(f"目标已存在：{ap}")
        return ap

    def markdown(self) -> str:
        man = self.load_manifest()
        if not man:
            return ("> ⚠️ **尚未建立底座快照**。第一次训练前必须做一次 —— "
                    "没有快照就无法证明训练过程没有改动底座。")
        files = man.get("files") or {}
        total = sum(int(d.get("size") or 0) for d in files.values())
        ro = self.read_only_state()
        n_ro = sum(1 for v in ro.values() if v)
        L = [f"**底座目录**：`{man.get('model_dir')}`",
             f"**快照时间**：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(man.get('created_at') or 0))}",
             f"**受保护文件**：{len(files)} 个，共 {total / 1e9:.2f} GB"
             f"{'（已算 SHA-256）' if man.get('hashed') else '（仅 size+mtime）'}",
             f"**只读锁**：{n_ro}/{len(files)} 个已锁定", "",
             "| 文件 | 大小 | SHA-256（前 12） | 只读 |", "|---|---|---|---|"]
        for rel, d in sorted(files.items()):
            h = (d.get("sha256") or "")[:12] or "—"
            L.append(f"| `{rel}` | {int(d.get('size') or 0)/1e6:.1f} MB "
                     f"| `{h}` | {'🔒' if ro.get(rel) else '🔓'} |")
        return "\n".join(L)


def _atomic_write_json(path: str, obj: Any) -> None:
    """先写 .tmp 再 os.replace —— 中途断电不会留下半个 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ===========================================================================
# 4. 冻结底座 / 参数计数
# ===========================================================================

def freeze_base(model) -> Dict[str, int]:
    """把整个模型 requires_grad=False，返回统计。

    PEFT 注入 LoRA 时会自己解冻 adapter，所以这一步要在**注入之前**做，
    注入之后再调一次 `assert_base_frozen` 确认底座没被意外解冻。
    """
    total = changed = already = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            p.requires_grad_(False)
            changed += p.numel()
        else:
            already += p.numel()
    return {"total_params": total, "newly_frozen": changed,
            "already_frozen": already}


def count_params(model) -> Dict[str, int]:
    """分开统计底座与可训练（adapter）参数量。

    `trainable_pct` 不在这里四舍五入 —— 返回给调用方的数据应该保持
    全精度，取多少位小数是**展示**的事，搞反了会让下游对不上账。
    """
    base = train = 0
    for n, p in model.named_parameters():
        is_adapter = any(k in n for k in ("lora_", "lora.", ".lora"))
        if p.requires_grad or is_adapter:
            train += p.numel()
        else:
            base += p.numel()
    total = base + train
    return {"base": base, "trainable": train, "total": total,
            "trainable_pct": (train / total * 100) if total else 0.0}


def assert_base_frozen(model) -> List[str]:
    """返回「本该冻结却仍带梯度」的参数名（不含 lora_ 开头的）。"""
    bad = []
    for n, p in model.named_parameters():
        if p.requires_grad and "lora_" not in n and ".lora" not in n:
            bad.append(n)
    return bad


# ===========================================================================
# 5. 推理期强度旋钮（防线 9）
# ===========================================================================

def iter_lora_layers(model) -> Iterator[Tuple[str, Any]]:
    """遍历模型里所有 PEFT LoRA 层。"""
    for name, m in model.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "scaling") and hasattr(m, "r"):
            yield name, m


def nominal_scaling(layer, adapter: str) -> float:
    """这层在 factor=1 时应有的 scaling（不依赖当前值，所以可反复调用）。"""
    r = float(layer.r.get(adapter, 0) or 0)
    a = float(layer.lora_alpha.get(adapter, 0) or 0)
    if r <= 0:
        return 0.0
    if layer.use_rslora.get(adapter):
        return a / math.sqrt(r)
    return a / r


def set_adapter_scale(model, factor: float, adapter: str = "default") -> int:
    """把所有 LoRA 层的增益乘上 factor。

    **这是整套保护里最实用的一招**：不用重训，就能在
    「像目标角色」和「保持通用能力」之间连续调节。
        factor = 0.0 → 数值上完全等价于底座（ΔW 被清零）
        factor = 1.0 → 完整 LoRA
        factor = 0.7 → 常见的折中：保住八成相似度，显著降低读错字概率

    实现上直接写 `layer.scaling[adapter]`，并且每次都从
    `nominal_scaling()` 重算基准，所以反复调用不会累积放大。
    """
    factor = max(0.0, min(2.0, float(factor)))
    n = 0
    for _name, layer in iter_lora_layers(model):
        if adapter not in layer.scaling:
            continue
        layer.scaling[adapter] = nominal_scaling(layer, adapter) * factor
        n += 1
    return n


def get_adapter_scale(model, adapter: str = "default") -> Dict[str, float]:
    """反查当前生效的 factor（用实际 scaling ÷ 名义 scaling）。"""
    out: Dict[str, float] = {}
    for _name, layer in iter_lora_layers(model):
        if adapter not in layer.scaling:
            continue
        nom = nominal_scaling(layer, adapter)
        if nom <= 0:
            continue
        out[_name] = float(layer.scaling[adapter]) / nom
    if not out:
        return {}
    vals = list(out.values())
    return {"_min": min(vals), "_max": max(vals),
            "_mean": sum(vals) / len(vals), "_layers": len(vals)}


def disable_adapters(model, off: bool = True) -> int:
    """整体开关 adapter（比 factor=0 更彻底：连旁路计算都跳过）。"""
    n = 0
    for _name, layer in iter_lora_layers(model):
        try:
            layer.enable_adapters(not off)
            n += 1
        except Exception:
            pass
    return n


# ===========================================================================
# 6. 权重漂移体检（防线 10）
# ===========================================================================

@dataclass
class DriftRow:
    name: str
    kind: str
    base_params: int
    base_norm: float
    delta_norm: float
    rel: float                 # ‖ΔW‖ / ‖W‖
    rank: int = 0


@dataclass
class DriftReport:
    rows: List[DriftRow] = field(default_factory=list)
    global_rel: float = 0.0    # 参数量加权：√Σ‖ΔW‖² / √Σ‖W‖²
    max_rel: float = 0.0
    max_name: str = ""
    mean_rel: float = 0.0
    n_layers: int = 0
    seconds: float = 0.0

    @property
    def level(self) -> Tuple[str, str]:
        for th, name, advice in DRIFT_LEVELS:
            if self.global_rel < th:
                return name, advice
        return DRIFT_LEVELS[-1][1], DRIFT_LEVELS[-1][2]

    def markdown(self, top: int = 12) -> str:
        if not self.rows:
            return "_没有检测到 LoRA 层，无法做漂移体检。_"
        lv, advice = self.level
        L = [f"### 权重漂移体检 · {lv}", "",
             f"**全局相对漂移** `√Σ‖ΔW‖² / √Σ‖W‖²` = "
             f"**{self.global_rel:.4f}**（{self.global_rel*100:.2f}%）",
             f"**最大单层** `{self.max_name}` = {self.max_rel:.4f}",
             f"**平均** {self.mean_rel:.4f} · 共 {self.n_layers} 层 · "
             f"耗时 {self.seconds:.1f}s", "",
             f"> {advice}", "",
             f"<details><summary>漂移最大的 {min(top, len(self.rows))} 层</summary>", "",
             "| 层 | 类型 | rank | ‖W‖ | ‖ΔW‖ | 相对 |", "|---|---|---|---|---|---|"]
        for r in sorted(self.rows, key=lambda x: -x.rel)[:top]:
            L.append(f"| `{r.name}` | {r.kind} | {r.rank} | {r.base_norm:.2f} "
                     f"| {r.delta_norm:.4f} | **{r.rel:.4f}** |")
        L += ["", "</details>"]
        return "\n".join(L)


def analyze_drift(model, adapter: str = "default",
                  device: str = "cpu") -> DriftReport:
    """量化 adapter 对底座权重的实际改动幅度。

    ΔW = B @ A × scaling，形状 (out, in)；Conv1D 的 weight 是 (in, out)，
    要转置后才能和它比 —— 这个坑不看形状就会静默算出错误的范数。
    """
    import torch
    t0 = time.perf_counter()
    rep = DriftReport()
    sum_d2 = sum_w2 = 0.0
    rels: List[float] = []
    for name, layer in iter_lora_layers(model):
        if adapter not in layer.lora_A or adapter not in layer.lora_B:
            continue
        try:
            A = layer.lora_A[adapter].weight.detach()      # (r, in)
            B = layer.lora_B[adapter].weight.detach()      # (out, r)
            base = layer.get_base_layer() if hasattr(layer, "get_base_layer") \
                else layer.base_layer
            W = base.weight.detach()
        except Exception:
            continue
        sc = float(layer.scaling.get(adapter, 0.0) or 0.0)
        with torch.no_grad():
            dW = (B.float().to(device) @ A.float().to(device)) * sc
            Wf = W.float().to(device)
            if Wf.shape != dW.shape and tuple(Wf.shape) == tuple(reversed(dW.shape)):
                dW = dW.t()                       # Conv1D：weight 是 (in, out)
            if Wf.shape != dW.shape:
                continue
            nd = float(dW.norm()); nw = float(Wf.norm())
        sum_d2 += nd * nd; sum_w2 += nw * nw
        rel = nd / nw if nw > 0 else float("inf")
        rels.append(rel)
        rep.rows.append(DriftRow(
            name=name, kind=type(base).__name__, base_params=int(W.numel()),
            base_norm=nw, delta_norm=nd, rel=rel,
            rank=int(layer.r.get(adapter, 0) or 0)))
    rep.n_layers = len(rep.rows)
    rep.global_rel = math.sqrt(sum_d2) / math.sqrt(sum_w2) if sum_w2 > 0 else 0.0
    rep.mean_rel = sum(rels) / len(rels) if rels else 0.0
    if rep.rows:
        worst = max(rep.rows, key=lambda r: r.rel)
        rep.max_rel, rep.max_name = worst.rel, worst.name
    rep.seconds = time.perf_counter() - t0
    return rep


# ===========================================================================
# 7. 早停（防线 7）
# ===========================================================================

@dataclass
class StepResult:
    improved: bool
    best: float
    bad_count: int
    should_stop: bool
    reason: str = ""


class EarlyStopper:
    """盯 val 指标，连续 patience 次不改善就叫停。

    小数据集微调里 val loss 的典型曲线是「先降、到某个点开始升」——
    升上去的部分全是过拟合。不早停就等于主动把模型练坏。
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4,
                 mode: str = "min"):
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.mode = mode                  # min | max
        self.best: Optional[float] = None
        self.best_epoch: int = -1
        self.bad_count = 0
        self.history: List[Tuple[int, float]] = []

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def _better(self, v: float) -> bool:
        if self.best is None:
            return True
        return (v < self.best - self.min_delta) if self.mode == "min" \
            else (v > self.best + self.min_delta)

    def step(self, value: float, epoch: int = 0) -> StepResult:
        v = float(value)
        self.history.append((epoch, v))
        if not self.enabled:
            if self.best is None or self._better(v):
                self.best, self.best_epoch = v, epoch
            return StepResult(True, self.best, 0, False, "早停已关闭")
        if self._better(v):
            self.best, self.best_epoch, self.bad_count = v, epoch, 0
            return StepResult(True, self.best, 0, False)
        self.bad_count += 1
        stop = self.bad_count >= self.patience
        reason = ""
        if stop:
            reason = (f"val 指标连续 {self.bad_count} 次未改善"
                      f"（最好 {self.best:.5f} @ epoch {self.best_epoch}），"
                      "继续训练只会加剧过拟合")
        return StepResult(False, self.best, self.bad_count, stop, reason)

    def state_dict(self) -> Dict[str, Any]:
        return {"patience": self.patience, "min_delta": self.min_delta,
                "mode": self.mode, "best": self.best,
                "best_epoch": self.best_epoch, "bad_count": self.bad_count,
                "history": self.history}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        for k in ("patience", "min_delta", "mode", "best", "best_epoch",
                  "bad_count", "history"):
            if k in (d or {}):
                setattr(self, k, d[k])


# ===========================================================================
# 8. Checkpoint 保险库（防线 8）
# ===========================================================================

@dataclass
class CkptInfo:
    name: str
    path: str
    epoch: int
    step: int
    metric: float
    saved_at: float
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointVault:
    """只保留 top-K 的 checkpoint 目录，原子写入，可回滚。

    `save()` 收一个 `save_fn(dest_dir)` 回调而不是直接调 PEFT ——
    GPT 与 CFM 的保存内容不同，保险库不该关心这些细节。

    回滚 = 把某个 checkpoint 复制到 `<run>/active/`，
    **永远不写 checkpoints/**（那由 BaseGuard 兜底拦截）。
    """

    INDEX = "index.json"

    def __init__(self, root: str, keep: int = 3, mode: str = "min"):
        self.root = os.path.abspath(root)
        self.keep = max(1, int(keep))
        self.mode = mode
        self.index_path = os.path.join(self.root, self.INDEX)

    # ---------------- 索引 ----------------
    def list(self) -> List[CkptInfo]:
        if not os.path.isfile(self.index_path):
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        out = []
        for d in data.get("checkpoints") or []:
            if os.path.isdir(d.get("path") or ""):
                out.append(CkptInfo(**{k: v for k, v in d.items()
                                       if k in CkptInfo.__dataclass_fields__}))
        return out

    def _write_index(self, items: Sequence[CkptInfo]) -> None:
        best_path = self._best_of(items)
        best_path = best_path.path if best_path else None
        _atomic_write_json(self.index_path, {
            "mode": self.mode, "keep": self.keep,
            "updated_at": time.time(),
            "checkpoints": [dict(i.to_dict(), is_best=(i.path == best_path))
                            for i in items],
        })

    def _best_of(self, items: Sequence[CkptInfo]) -> Optional[CkptInfo]:
        """从**给定的**列表里取 best。

        不能调 self.best()：那个会去读磁盘上的旧索引，
        而保存/剪枝过程中新 checkpoint 还没写进去，拿到的是陈旧的 best。
        """
        items = [i for i in items]
        if not items:
            return None
        return min(items, key=lambda i: i.metric) if self.mode == "min" \
            else max(items, key=lambda i: i.metric)

    def best(self) -> Optional[CkptInfo]:
        return self._best_of(self.list())

    # ---------------- 保存 ----------------
    def save(self, save_fn: Callable[[str], None], epoch: int, step: int,
             metric: float, extra: Optional[Dict[str, Any]] = None) -> CkptInfo:
        os.makedirs(self.root, exist_ok=True)
        name = f"ckpt-e{int(epoch):03d}-s{int(step):06d}"
        dest = os.path.join(self.root, name)
        tmp = dest + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        save_fn(tmp)
        if extra:
            _atomic_write_json(os.path.join(tmp, "extra.json"), extra)
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)          # 目录级原子替换（Windows 上要求目标不存在）

        info = CkptInfo(name=name, path=dest, epoch=int(epoch), step=int(step),
                        metric=float(metric), saved_at=time.time())
        items = self.list() + [info]
        items = self._prune(items)
        self._write_index(items)
        return info

    def _prune(self, items: List[CkptInfo]) -> List[CkptInfo]:
        """按指标排序，保留前 keep 个，其余删掉。

        best 永远保留 —— 否则 keep=1 时可能把最好的那份挤掉。
        """
        items = sorted(items, key=lambda i: i.metric,
                       reverse=(self.mode == "max"))
        keep = items[:self.keep]
        b = self._best_of(items)
        if b is not None and not any(k.path == b.path for k in keep):
            keep.append(b)
        for drop in items[len(keep):]:
            shutil.rmtree(drop.path, ignore_errors=True)
        return sorted(keep, key=lambda i: i.step)

    # ---------------- 回滚 ----------------
    @property
    def active_dir(self) -> str:
        return os.path.join(self.root, ACTIVE_SUBDIR)

    def rollback(self, which: str = "best") -> Optional[str]:
        """把某个 checkpoint 复制到 active/，推理端只加载 active/。

        which: "best" | checkpoint 名 | 绝对路径
        """
        items = self.list()
        if not items:
            return None
        if which == "best":
            src = self.best()
        else:
            src = next((i for i in items if i.name == which or i.path == which), None)
        if src is None:
            return None
        act = self.active_dir
        tmp = act + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src.path, tmp)
        shutil.rmtree(act, ignore_errors=True)
        os.replace(tmp, act)
        _atomic_write_json(os.path.join(act, "rollback.json"),
                           {"from": src.path, "metric": src.metric,
                            "epoch": src.epoch, "step": src.step,
                            "at": time.time()})
        return act

    def markdown(self) -> str:
        items = self.list()
        if not items:
            return "_还没有保存任何 checkpoint。_"
        b = self.best()
        L = [f"共 {len(items)} 档（最多保留 {self.keep}），"
             f"best = `{b.name if b else '—'}`", "",
             "| checkpoint | epoch | step | 指标 | 保存时间 | |",
             "|---|---|---|---|---|---|"]
        for i in sorted(items, key=lambda x: -x.saved_at):
            t = time.strftime("%m-%d %H:%M", time.localtime(i.saved_at))
            L.append(f"| `{i.name}` | {i.epoch} | {i.step} | {i.metric:.5f} "
                     f"| {t} | {'⭐' if b and i.path == b.path else ''} |")
        return "\n".join(L)


# ===========================================================================
# 9. 显存余量体检（开工前必查）
# ===========================================================================
#
# 这一项看着像运维细节，实际上是**训练能不能用**的生死线：
# Windows 的 WDDM 驱动在物理显存不够时不会报 OOM，而是静默把张量放进
# 「共享 GPU 内存」（即系统 RAM），速度直接掉 20~30 倍且不报任何错。
# 实测对比：CFM 跑 25 步扩散，显存充足时 2.42s，溢出后 66s。
# 而 torch.cuda.memory_allocated() 只看得到自己进程，看不到其他应用
# （浏览器、IDE、残留的旧服务）占掉的那部分，所以必须用 mem_get_info()。

# 训练/提取前建议保留的余量（GB）。CUDA 上下文本身就要吃 300~500 MB，
# 再加上激活值与碎片，留太少一样会溢出。
VRAM_SAFETY_MARGIN_GB = 0.6


@dataclass
class VramReport:
    ok: bool = True
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0            # 整卡已用（**含其他进程**）
    torch_alloc_gb: float = 0.0     # 本进程已分配
    torch_reserved_gb: float = 0.0  # 本进程向驱动要的缓存池
    need_gb: float = 0.0
    shortfall_gb: float = 0.0
    level: str = ""
    message: str = ""

    def markdown(self) -> str:
        icon = "✅" if self.ok else ("🟡" if self.shortfall_gb <= 0 else "🔴")
        L = [f"{icon} **{self.level}** — {self.message}", "",
             "| 项 | 值 |", "|---|---|",
             f"| 整卡容量 | {self.total_gb:.2f} GB |",
             f"| 当前空闲 | **{self.free_gb:.2f} GB** |",
             f"| 已被占用（含其他进程） | {self.used_gb:.2f} GB |",
             f"| 本进程已分配 | {self.torch_alloc_gb:.2f} GB |",
             f"| 本进程缓存池 | {self.torch_reserved_gb:.2f} GB |",
             f"| 预计还需 | {self.need_gb:.2f} GB |"]
        if self.shortfall_gb > 0:
            L += ["", f"> 🔴 缺口 **{self.shortfall_gb:.2f} GB**。在 Windows 上这不会报错，"
                      "而是静默溢出到系统内存，速度会掉 20~30 倍。"
                      "请先关掉占显存的应用（浏览器/IDE/残留的服务进程），"
                      "或调小 batch / 序列长度 / 注入面。"]
        return "\n".join(L)


def free_vram(tag: str = "", logger=None) -> Dict[str, Any]:
    """尽量把显存还回去，并报告「清理前 → 清理后 → 释放了多少」。

    在**每次加载模型之前**调用它 —— 理由是实测出来的：8 GB 卡上先常驻推理引擎
    （4.94 GB）再叠一个 whisper medium（1.6 GB）就只剩几百 MB，Windows 会把计算
    挤到共享内存，表现为**静默降速 20~30 倍**（实测一条 10 秒音频的转写卡了
    4 分钟以上），而不是报 OOM。

    三件事按顺序做：
        gc.collect()              回收 Python 侧的引用环（模型对象常在里面）
        torch.cuda.empty_cache()  把缓存块还给驱动
        torch.cuda.ipc_collect()  回收其它进程遗留 IPC 句柄占用的显存

    注意：`empty_cache` 只能归还**已释放**的块。若还有对象持有张量（比如引擎
    还挂着），一点也不会少 —— 所以调用方要先把模型真的卸掉；本函数只负责把
    已断开的引用真正还给驱动，不负责卸载模型。
    返回 {"before_gb", "after_gb", "freed_gb", "ok", "tag"}。
    """
    out: Dict[str, Any] = {"before_gb": 0.0, "after_gb": 0.0,
                           "freed_gb": 0.0, "ok": False, "tag": tag}
    try:
        import gc

        import torch
    except Exception:
        return out

    def _free_gb() -> float:
        try:
            free, _total = torch.cuda.mem_get_info(0)
            return round(free / (1024 ** 3), 2)
        except Exception:
            return 0.0

    try:
        if not torch.cuda.is_available():
            return out
        out["before_gb"] = _free_gb()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        out["after_gb"] = _free_gb()
        out["freed_gb"] = round(out["after_gb"] - out["before_gb"], 2)
        out["ok"] = True
        if logger is not None:
            logger.info("显存清理%s：空闲 %.2f → %.2f GB（释放 %.2f GB）",
                        f"（{tag}）" if tag else "", out["before_gb"],
                        out["after_gb"], out["freed_gb"])
    except Exception as e:
        if logger is not None:
            try:
                logger.warning("显存清理失败：%s: %s", type(e).__name__, e)
            except Exception:
                pass
    return out


def vram_headroom(need_gb: float = 0.0) -> VramReport:
    """查整卡显存的**真实**余量。

    need_gb 是「除了当前已占用的，还需要多少」，会额外加上安全余量。
    没有 CUDA 时返回一个 ok=True 的空报告（CPU 训练不受这个限制）。
    """
    rep = VramReport(need_gb=max(0.0, float(need_gb)))
    try:
        import torch
    except Exception:
        rep.level, rep.message = "无法判定", "torch 不可用"
        return rep
    if not torch.cuda.is_available():
        rep.level, rep.message = "CPU 模式", "没有可用 GPU，不涉及显存溢出问题（但会非常慢）"
        return rep
    free_b, total_b = torch.cuda.mem_get_info()
    rep.total_gb = total_b / 1e9
    rep.free_gb = free_b / 1e9
    rep.used_gb = (total_b - free_b) / 1e9
    rep.torch_alloc_gb = torch.cuda.memory_allocated() / 1e9
    rep.torch_reserved_gb = torch.cuda.memory_reserved() / 1e9

    need = rep.need_gb + (VRAM_SAFETY_MARGIN_GB if rep.need_gb > 0 else 0.0)
    rep.shortfall_gb = max(0.0, need - rep.free_gb)
    if rep.shortfall_gb <= 0:
        rep.ok = True
        rep.level = "显存充足"
        rep.message = (f"空闲 {rep.free_gb:.2f} GB，预计还需 {need:.2f} GB"
                       f"（含 {VRAM_SAFETY_MARGIN_GB} GB 安全余量）")
    else:
        rep.ok = False
        rep.level = "显存不足，会静默降速"
        rep.message = (f"空闲 {rep.free_gb:.2f} GB < 需要 {need:.2f} GB，"
                       f"缺 {rep.shortfall_gb:.2f} GB")
    return rep


def vram_preflight(need_gb: float, raise_on_short: bool = False) -> VramReport:
    """开工前的显存体检。raise_on_short=True 时缺口直接抛异常。"""
    rep = vram_headroom(need_gb)
    if not rep.ok and raise_on_short:
        raise RuntimeError(
            f"显存不足：空闲 {rep.free_gb:.2f} GB，需要约 {rep.need_gb:.2f} GB。"
            "Windows 下这不会 OOM，而是静默溢出到系统内存使速度掉 20~30 倍，"
            "所以直接拦住。请关闭其他占显存的程序后重试。")
    return rep


# ===========================================================================
# 10. 回放混合（防线 6）
# ===========================================================================

# replay_source="base_distill" 时用的那个数据集名（由 replay.py 生成）。
# 固定名字而不是每次让用户填：回放集是**工具产物**不是用户素材，
# 让它出现在「数据集」列表里但不需要用户管理，重训时能直接复用。
DISTILL_DATASET = "base_distill"


@dataclass
class ReplayPlan:
    n_target: int
    n_replay: int
    ratio: float
    achieved: float
    with_replacement: bool = False
    note: str = ""


def build_epoch_plan(n_target: int, n_replay_pool: int, ratio: float,
                     seed: int = 42) -> Tuple[List[Tuple[str, int]], ReplayPlan]:
    """生成一个 epoch 的取样顺序：[("target", i) | ("replay", j), ...]。

    ratio 定义成「回放样本占整个 epoch 的比例」，所以
        n_replay = n_target × ratio / (1 - ratio)
    这样 ratio=0.3 时，10 条目标数据会配 4.3≈4 条回放，4/14≈29%。

    回放池不够时**有放回**采样并给出提示 —— 宁可重复也不要不回放，
    因为重复的通用样本仍然在提醒模型「这些发音方式不能忘」。
    """
    import random
    n_target = max(0, int(n_target))
    ratio = max(0.0, min(0.95, float(ratio)))
    plan: List[Tuple[str, int]] = [("target", i) for i in range(n_target)]
    n_replay = 0
    if ratio > 0 and n_target > 0:
        n_replay = int(round(n_target * ratio / (1.0 - ratio)))
    with_rep = False
    note = ""
    if n_replay > 0:
        if n_replay_pool <= 0:
            note = "回放池为空：本次训练实际上没有任何回放，等价于 replay_ratio=0"
            n_replay = 0
        elif n_replay <= n_replay_pool:
            idx = random.Random(seed).sample(range(n_replay_pool), n_replay)
            plan += [("replay", j) for j in idx]
        else:
            with_rep = True
            rnd = random.Random(seed)
            plan += [("replay", rnd.randrange(n_replay_pool)) for _ in range(n_replay)]
            note = (f"回放池只有 {n_replay_pool} 条，需要 {n_replay} 条，"
                    "已启用**有放回**采样（部分通用样本会重复出现）")
    random.Random(seed + 1).shuffle(plan)
    total = len(plan)
    achieved = (sum(1 for k, _ in plan if k == "replay") / total) if total else 0.0
    return plan, ReplayPlan(n_target=n_target, n_replay=n_replay, ratio=ratio,
                            achieved=achieved, with_replacement=with_rep, note=note)


# ===========================================================================
# 11. 说明文档（UI 用）
# ===========================================================================

PROTECTION_DOC = """
### LoRA 微调为什么会掉泛化，以及本工具做了哪些防护

**掉泛化的两个机理**（要分开看，对策完全不同）：

| 机理 | 表现 | 对策 |
|---|---|---|
| **过拟合** | 训练集里的句子完美，换个说法就崩；读没见过的词开始出错 | 降 rank、加 dropout/weight_decay、早停、缩小注入面 |
| **灾难性遗忘** | 音色像了，但普通话整体变差、多音字读错率上升、韵律发飘 | **回放混合**、降低旁路增益、推理期调低强度 |

小数据集微调里，**遗忘比过拟合更常见也更隐蔽** —— 因为 val loss 只测目标数据，
遗忘在 val 上看不出来。所以本工具的默认档强制 30% 回放。

---

### 十道防线

**1 · 底座只读**
原始 `checkpoints/gpt.pth`、`s2mel.pth` 全程不被改写。训练前做 SHA-256 快照，
每次训练与合并前重新校验；任何指向 `checkpoints/` 的输出路径会被直接拒绝，
连「生成一个同名文件」都会被拦下。另外可以给底座加 Windows 只读属性做事前阻止。
LoRA adapter 存在 `training_runs/<run>/adapter/`，与底座物理隔离。

**2 · LoRA 结构本身就是保护**
底座权重 `requires_grad=False`，训练只更新低秩旁路 A/B。
底座数值从头到尾一个字节都没变 —— 所以「回到底座」永远是可能的。

**3 · rank / alpha = 容量旋钮**
真正决定影响力的是 **旁路增益 alpha/rank**，不是 rank 本身：
`rank=32, alpha=32`（增益 1.0）比 `rank=4, alpha=16`（增益 4.0）温和得多。
增益越大，LoRA 对输出的话语权越强，底座行为被改写越多。
勾 `rsLoRA` 会用 `alpha/√rank`，高 rank 时更稳。

**4 · 注入面越窄越安全**
只注入注意力投影（`attn/c_attn`、`attn/c_proj`）通常就足够改变音色与语气。
把 MLP 也注进去，可训练参数翻一倍，通用能力损失也明显上升。
**词嵌入和输出头永远不要注入** —— 那直接改的是「文本→语义」的映射表。
UI 里的候选清单是从真实模型扫出来的，不是硬编码，所以 GPT(Conv1D) 和
CFM(Linear) 两边都能对上。

**5 · dropout + weight_decay**
常规正则。`rank≥16` 时 dropout 别设 0；小数据集 weight_decay 给 0.01~0.05。

**6 · 回放混合（对抗遗忘的主力）**
按 `replay_ratio` 在每个 epoch 里掺入通用样本。
ratio=0.3 意味着 10 条目标数据配 4 条通用数据（4/14≈29%）。
回放池不足时会**有放回**采样并提示 —— 重复的通用样本仍然在提醒模型
「这些发音方式不能忘」，比完全不回放好。

**7 · 早停**
盯自己数据的 val loss，连续 `val_patience` 次不改善就停。
小数据集的典型曲线是「先降后升」，升上去的部分全是过拟合。
注意：**val loss 测不出遗忘**，所以早停不能替代回放。

**8 · Checkpoint 保险库**
原子写入（先 `.tmp` 再 `os.replace`），只保留指标最好的 top-K，
best 永不被挤掉。随时可一键回滚到 `active/`，推理端只加载 `active/`。

**9 · 推理期强度旋钮 ← 最实用**
`adapter_scale` 把所有 LoRA 层的增益乘上一个系数：
- `0.0` = 数值上完全等价于底座（可用来做 A/B 对照）
- `0.6~0.8` = 保住大部分角色相似度，同时显著降低读错字概率
- `1.0` = 完整 LoRA

**这一招不需要重训**。发现微调后通用能力变差，第一件事就是把强度调到 0.7 再听。
实现上每次都从 `alpha/rank` 重算基准再乘系数，所以反复调不会累积放大。

**10 · 权重漂移体检**
训练后计算每层的 `‖ΔW‖_F / ‖W‖_F`（ΔW = B@A×scaling），
给出全局加权漂移与风险等级：

| 全局漂移 | 判定 |
|---|---|
| < 2% | 🟢 保守，通用能力基本无损（相似度不够就提 rank/alpha） |
| 2% ~ 5% | 🟢 正常，典型成功区间 |
| 5% ~ 12% | 🟡 偏激进，建议强度调到 0.6~0.8 |
| 12% ~ 25% | 🟠 过拟合风险，建议回滚到更早的 checkpoint |
| > 25% | 🔴 严重漂移，通用能力大概率已明显退化 |

---

### 开工前必查：显存余量

这一项不是「泛化」问题，但它会让训练**慢 20~30 倍而不报任何错**，
所以也算一道防线。

Windows 的 WDDM 驱动在物理显存不够时**不会 OOM**，而是静默把张量放进
「共享 GPU 内存」（其实就是系统 RAM），于是每一步都在走 PCIe。
实测对比（RTX 4060 Laptop 8GB，CFM 跑 25 步扩散）：

| 情况 | 25 步耗时 | 每步 |
|---|---|---|
| 显存充足（仅加载 s2mel，峰值 740 MB） | **2.42 s** | 96.9 ms |
| 显存被其他进程挤爆（引擎 5.7 GB + 残留服务 1.7 GB） | **66 s** | 2640 ms |

关键在于 `torch.cuda.memory_allocated()` **只看得到自己进程**，
看不到浏览器、IDE、以及上次忘了关的旧服务占掉的那 1~2 GB。
所以 `vram_headroom()` 用的是 `torch.cuda.mem_get_info()`，看的是整卡。

**训练前发现缺口时的处理顺序**：
1. 关掉占显存的应用（浏览器标签页、IDE 的 GPU 加速、残留的 python 服务进程）
2. 卸载推理引擎再训练（两者同时要 5 GB 以上）
3. 调小 `batch_size` / 序列长度 / `rank` / 注入面
4. 开 `grad_checkpointing`（用时间换显存，约慢 30%）

> 排查提示：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
> 能列出占卡的进程。Windows 上 `.venv/Scripts/python.exe` 是个启动器，
> 真正占显存的是它拉起的子进程，所以会看到成对的两个 python。

---

### 本机管不了的部分

- **训练集多样性**：只有 3 分钟单一情绪的录音，任何正则都救不回来。
  数据侧能做的都放在「数据集」页（体检、切片、SNR 过滤、时长统计与分级建议）
- **录音质量**：底噪、混响、削波会直接被学进去
- **底座模型自身的能力边界**：IndexTTS-2.5 没学过的语种/发音，微调也变不出来

> 遇到读错字，**先试读音标注**（合成页「🔤 读音纠正」）——
> 那是零风险的精确修正，比微调可靠得多。微调用来改音色和语气，不用来纠读音。
"""
