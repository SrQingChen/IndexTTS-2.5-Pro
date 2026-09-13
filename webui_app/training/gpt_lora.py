"""L1 · GPT(T2S) LoRA SFT 训练器。

**这个目标练的是什么**：GPT 决定「怎么说」—— 语气、节奏、停顿习惯、重音位置、
在什么文本上出什么语义 token。音色本身主要靠 CFM（见 `cfm_lora.py`），
但韵律骨架在 GPT 这边。只练 CFM 会得到「音色对但语气平」的结果，
只练 GPT 会得到「说话像但音质发飘」的结果 —— 想真像本人，两个都得练。

**为什么 8GB 卡训得动 813M 的底座**：训练时完全不碰 w2v-BERT / conformer /
perceiver / CAMPPlus / semantic codec。这些在 `features.py` 的离线提取阶段
就算完并缓存了（`style` 192 维、`emo_vec` 1280 维、`codes` 25Hz），
训练只跑 GPT2 主干 + 两个输出头。省掉的是显存大头，也是速度大头。

**前向不自己写**：一律走 `forward.py::gpt_training_forward`，
那一份已经被 `tools/features_probe.py` 在真实权重上验证过（含 lang_embedding
消融、mask 校验、reentrant 梯度检查点陷阱）。这里再写一份就一定会漂移。

**流水线也不自己写**：登记 / 底座校验 / 优化器 / 早停 / 保险库 / 收尾 / 释放
全在 `trainer_base.BaseTrainer` 里，与 CFM 训练器共用同一份。
本文件只剩 GPT 特有的东西：样本池、批组装、前向、评估。
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import dataset as DS
from webui_app.training import features as FT
from webui_app.training import forward as FW
from webui_app.training import guard as GD
from webui_app.training import trainer_base as TB

ARCH = "gpt"

# 兼容旧的导入路径：报告类只有一份，住在 trainer_base 里
TrainReport = TB.TrainReport

# 底座权重的关键子串：加载时少任何一个都说明 gpt.pth 与 config 不匹配，
# 而官方 `load_checkpoint` 用的是 strict=False —— 少键只 print 一行就过去了。
CRITICAL_KEYS = (
    "spk_emb_proj.weight", "lang_embedding.weight",
    "text_embedding.weight", "mel_embedding.weight",
    "text_pos_embedding.emb.weight", "mel_pos_embedding.emb.weight",
    "final_norm.weight", "mel_head.weight", "text_head.weight",
    "gpt.h.0.attn.c_attn.weight", "gpt.h.23.mlp.c_proj.weight",
)

# 一个 epoch 里为了「长度接近的排一起」而打包的大块尺寸（实现在 trainer_base）
MEGA_BATCH = TB.MEGA_BATCH

# 预检阶段还没加载模型，但用户必须**在开工前**就知道装不装得下 ——
# 等到 WDDM 静默溢出再发现就晚了（实测溢出后 CFM 25 步从 2.4s 变 66s，不报 OOM）。
# 下面两个数是 features_probe 在真实 gpt.pth 上量出来的，不是拍的：
#   底座 813M；attn-only r=16 → 2,949,120；全 96 个 Conv1D r=16 → 7,864,320
EST_BASE_PARAMS = 813_000_000
EST_ADAPTER_PER_RANK = {"attn": 184_320, "attn_mlp": 491_520, "all_linear": 491_520}


# ===========================================================================
# 1. 配置
# ===========================================================================

@dataclass
class GptTrainOptions:
    """GPT 训练专属的旋钮（通用的那些在 `guard.LoRAConfig` 里）。

    每一项都对应一个真实的取舍，没有摆设。
    """
    # text 头学的是「下一个文本 token」，对音色/语气没有帮助，
    # 却会和 mel 抢同一批 LoRA 参数的容量。默认 0 = 只学 mel。
    text_loss_weight: float = 0.0

    # 序列长度上限。总长 = 3(conds) + (Lt+2) + (Lc+2)，
    # 必须 ≤ n_positions(2420)，且 Lc+2 ≤ mel_pos_embedding 容量(1818)。
    # 卡这两个数不是为了合规，是为了显存：激活值随长度平方增长（注意力）。
    max_codes: int = 900
    max_text_tokens: int = 500

    # 把长度接近的样本排进同一个 batch，padding 能少一半以上。
    # batch_size=1 时无所谓，>1 时收益明显。
    sort_by_length: bool = True

    # 底座 GPT2 的 attn/resid/embd dropout。默认 0：
    # 底座是冻结的，它的 dropout 不是我们能调的正则，只会让 val loss 抖动、
    # 让早停误判；随机正则统一交给 lora_dropout。
    base_dropout: float = 0.0

    # 余弦退火的地板。0.1 = 最后 lr 降到峰值的 10%。
    min_lr_ratio: float = 0.1

    log_every: int = 10          # 每多少个 optimizer step 往日志写一行
    save_optimizer_state: bool = True    # 关掉能省 2/3 的 checkpoint 体积，代价是不能续训

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GptTrainOptions":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []
        if not 0.0 <= self.text_loss_weight <= 2.0:
            n.append(GD.Notice("error", f"text_loss_weight={self.text_loss_weight} 超出 0~2"))
        if not 16 <= int(self.max_codes) <= 1816:
            n.append(GD.Notice("error",
                               f"max_codes={self.max_codes} 超出 16~1816"
                               "（1816 = mel_pos_embedding 容量 1818 − 2）"))
        if not 4 <= int(self.max_text_tokens) <= 600:
            n.append(GD.Notice("error",
                               f"max_text_tokens={self.max_text_tokens} 超出 4~600"
                               "（600 = text_pos_embedding 容量 602 − 2）"))
        if 3 + (self.max_text_tokens + 2) + (self.max_codes + 2) > 2420:
            n.append(GD.Notice("error",
                               "max_codes + max_text_tokens 太大：总序列会超过 GPT2 的 "
                               f"n_positions=2420（当前算出 "
                               f"{3 + self.max_text_tokens + 2 + self.max_codes + 2}）"))
        if not 0.0 <= self.base_dropout <= 0.5:
            n.append(GD.Notice("error", f"base_dropout={self.base_dropout} 超出 0~0.5"))
        if not 0.0 <= self.min_lr_ratio < 1.0:
            n.append(GD.Notice("error", f"min_lr_ratio={self.min_lr_ratio} 超出 0~1"))
        if self.text_loss_weight > 0:
            n.append(GD.Notice("info",
                               f"text 头参与训练（权重 {self.text_loss_weight}）。"
                               "它会和 mel 抢 LoRA 容量，除非明确要复现双头训练，否则建议设 0"))
        return n


# ===========================================================================
# 2. 样本池与批组装
# ===========================================================================

class SamplePool(TB.FeaturePool):
    """GPT 样本池：从 .pt 里取 text_tokens / codes / style / emo_vec。

    缓存、LRU、meta 索引、老数据集补写全在 `FeaturePool` 里（CFM 共用）；
    这里只剩 GPT 特有的两件事：取哪些字段、以及按**两个**长度上限过滤。
    """

    LEN_FIELD = "n_codes"          # 排序按语义 token 帧数，不是 mel 帧数

    def _convert(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return FT.build_gpt_sample(raw)

    def usable_ids(self, max_codes: int, max_text: int) -> Tuple[List[str], List[str]]:
        """返回 (可用 id, 被刷掉的 id)。长度信息在 meta 里，不碰磁盘。

        两个上限都得卡：codes 决定 mel_pos_embedding 会不会越界，
        text 决定 text_pos_embedding；而且它们加起来还受 n_positions=2420 限制。
        """
        ok, bad = [], []
        for uid in self.ids:
            m = self.meta.get(uid) or {}
            if int(m.get("n_codes") or 0) < 2 or int(m.get("n_text") or 0) < 1:
                bad.append(uid)
            elif int(m["n_codes"]) > max_codes or int(m["n_text"]) > max_text:
                bad.append(uid)
            else:
                ok.append(uid)
        return ok, bad


def collate(samples: Sequence[Dict[str, Any]]):
    """把若干条样本拼成一个 batch。

    padding 值给 0 就行：`set_text_padding` / `set_mel_padding` 会按
    `*_lengths` 把填充区重写成 stop token，我们只需要把**真实长度**传对。
    """
    import torch

    text_tokens, text_lengths = FT.pad_stack([s["text_tokens"] for s in samples],
                                             value=0, dim=0)
    codes, code_lengths = FT.pad_stack([s["codes"] for s in samples], value=0, dim=0)
    style = torch.stack([s["style"] for s in samples])
    emo_vec = torch.stack([s["emo_vec"] for s in samples])
    langs = torch.LongTensor([int(s["lang_token"]) for s in samples])
    return {
        "text_tokens": text_tokens, "text_lengths": text_lengths,
        "codes": codes, "code_lengths": code_lengths,
        "style": style, "emo_vec": emo_vec, "langs": langs,
        "ids": [s.get("id", "") for s in samples],
        "kinds": [s.get("kind", "target") for s in samples],
    }


def plan_batches(order: Sequence[Tuple[str, int]], pools: Dict[str, SamplePool],
                 batch_size: int, sort_by_length: bool, seed: int = 42,
                 len_key: str = "n_codes"
                 ) -> List[List[Tuple[str, int]]]:
    """薄封装：真正的实现在 `trainer_base.plan_batches`（CFM 也用那一份）。

    这里只负责把「(池名, 下标) → 长度」这个查询接上。
    排序策略的理由（大块内排序 + 大块间打乱）写在基类里。
    """
    def len_of(ki: Tuple[str, int]) -> int:
        k, i = ki
        p = pools.get(k)
        if p is None or i >= len(p.ids):
            return 0
        return int((p.meta.get(p.ids[i]) or {}).get(len_key) or 0)

    return TB.plan_batches(order, batch_size, sort_by_length, seed=seed,
                           len_of=len_of)


# ===========================================================================
# 3. 底座加载与 LoRA 注入
# ===========================================================================

def load_base_gpt(device: str = "cuda", model_dir: Optional[str] = None,
                  verify_keys: bool = True):
    """加载一份**独立于推理引擎**的 UnifiedVoice。

    不复用引擎里那份的原因：
      · 引擎的 GPT 是 `eval()` + 可能被 `.bfloat16()` 过的，训练要 `train()`；
      · 训练期间用户如果去合成一句，两边会互相踩（dropout 开关、KV 状态）；
      · PEFT 会**原地**改写模块树，引擎那份被改了就没法再正常推理。
    代价是多一份权重（bf16 约 1.6 GB），在 8GB 卡上必须先卸载引擎。
    """
    import torch
    from omegaconf import OmegaConf

    from indextts.gpt.model_v2 import UnifiedVoice        # 真正的类，不是 model_v2_5

    model_dir = model_dir or os.path.join(PROJECT_ROOT, GD.MODEL_DIR_NAME)
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"找不到 {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    m = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    pth = os.path.join(model_dir, str(cfg.gpt_checkpoint))
    if not os.path.isfile(pth):
        raise FileNotFoundError(f"找不到底座权重 {pth}")

    # 不用官方 load_checkpoint：它 strict=False 且少键只 print 一行。
    # gpt.pth 与 config.yaml 不匹配时（比如换了个版本的权重），
    # 缺 lang_embedding 会让训练静默地学不到语言条件 —— 必须硬拦。
    ck = torch.load(pth, map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        ck = ck["model"]
    missing, unexpected = m.load_state_dict(ck, strict=False)
    del ck
    if verify_keys:
        have = set(m.state_dict().keys())
        got = have - set(missing)
        lack = [k for k in CRITICAL_KEYS if k in have and k not in got]
        if lack:
            raise RuntimeError(
                f"gpt.pth 缺少关键权重：{lack}\n"
                "底座与 config.yaml 不匹配，训练出来的东西不可信。"
                "请到「模型」页重新校验/下载 gpt.pth。")
    m = m.to(device)
    return m, {"missing": len(missing), "unexpected": len(unexpected),
               "path": pth, "model_dir": model_dir}


def promote_adapter_fp32(model) -> int:
    """把 adapter 参数从 bf16 提回 fp32，返回改动的张量数。

    底座可以是 bf16（省一半显存），但 **AdamW 的一阶/二阶动量必须是 fp32**：
    bf16 只有 8 位尾数，lr=1e-4 级别的更新量加到动量上会直接下溢成 0，
    表现为「loss 不降但也不报错」。autocast 会在前向时把它们临时转回 bf16，
    所以计算精度不受影响，只有主权重是 fp32。
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


def inject_lora(base, cfg: GD.LoRAConfig) -> Tuple[Any, Dict[str, Any]]:
    """冻结底座 → 注入 LoRA → 复核。返回 (peft_model, 报告)。

    GPT 与 CFM 共用这一份：两边都是「先冻结、再注入、再复核」，
    差别只在 `scan_targets` 扫出来的模块名（Conv1D vs Linear）。
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


def estimate_vram_gb(base_params: int, adapter_params: int, bf16: bool,
                     batch_size: int, max_codes: int) -> float:
    """训练峰值显存的粗估（GB）。

    粗估而非实测：这一步要在**加载底座之前**就告诉用户装不装得下，
    否则等到 OOM（或者更糟，WDDM 静默溢出）就晚了。
    系数来自 lora_probe 的实测：813M 底座 bf16 ≈ 1.63 GB，
    r=16 注意力注入 ≈ 7.86M 参数，AdamW 双动量 fp32 ≈ 63 MB。
    """
    w = 2.0 if bf16 else 4.0
    gb = base_params * w / 1e9                       # 底座权重
    gb += adapter_params * 16.0 / 1e9                # fp32 权重4 + 梯度4 + m4 + v4
    gb += 0.55                                       # CUDA 上下文
    gb += 0.30 * max(1, batch_size) * (max_codes / 500.0)   # 激活（开检查点后）
    return round(gb, 2)


# ===========================================================================
# 4. 训练器
# ===========================================================================

class GptTrainer(TB.BaseTrainer):
    """GPT(T2S) 的 LoRA SFT 训练器。流水线见 `trainer_base.BaseTrainer`。"""

    ARCH = "gpt"

    def __init__(self, dataset: str, cfg: Optional[GD.LoRAConfig] = None,
                 options: Optional[GptTrainOptions] = None,
                 run_name: Optional[str] = None, device: Optional[str] = None,
                 model_dir: Optional[str] = None, val_ratio: float = 0.05,
                 resume_from: Optional[str] = None,
                 train_root: Optional[str] = None):
        super().__init__(dataset, cfg=cfg, run_name=run_name, device=device,
                         model_dir=model_dir, val_ratio=val_ratio,
                         resume_from=resume_from, train_root=train_root)
        self.options = options or GptTrainOptions()
        self.pools: Dict[str, SamplePool] = {}
        self.val_pool: Optional[SamplePool] = None
        # val 单独一套 pools：`_gather` 按下标取样本，而 batch 里的下标是
        # 在**哪个池**里算出来的就必须回那个池取。两边共用一个默认池的话，
        # val 的下标会落到训练池上 —— loss 照样算得出来、曲线照样下降，
        # 但「验证集」实际是训练集的前几条，早停与泛化判断全部失效。
        self.val_pools: Dict[str, SamplePool] = {}

    # `pm` 是 `model` 的历史别名：探针与 UI 都按这个名字取 PEFT 包装后的
    # UnifiedVoice。做成只读属性而不是第二个字段，避免出现两份不同步的引用。
    @property
    def pm(self):
        return self.model

    # ---------------- 基类钩子的参数化 ----------------
    def min_lr_ratio(self) -> float:
        return float(self.options.min_lr_ratio)

    def log_every(self) -> int:
        return int(self.options.log_every)

    def save_optimizer_state(self) -> bool:
        return bool(self.options.save_optimizer_state)

    def options_dict(self) -> Dict[str, Any]:
        return self.options.to_dict()

    def _promote_fp32(self) -> int:
        return promote_adapter_fp32(self.model) if self.model is not None else 0

    def _restore_model(self) -> None:
        FW.restore_gpt(self.model, self.prev_state)

    def _drop_model_refs(self) -> None:
        for p in list(self.pools.values()):
            p.clear()
        self.pools = {}
        for p in list(self.val_pools.values()):
            p.clear()
        self.val_pools = {}
        if self.val_pool is not None:
            self.val_pool.clear()
            self.val_pool = None

    def _extra_run_fields(self, pf: Dict[str, Any],
                          extra: Dict[str, Any]) -> Dict[str, Any]:
        ex = extra or {}
        inj = ex.get("inject") or {}
        return {
            "lora_layers": int(inj.get("lora_layers") or 0),
            "target_regex": str(inj.get("regex") or ""),
            "target_patterns": list(inj.get("patterns") or []),
            "base_dtype": str((ex.get("cast") or {}).get("dtype") or ""),
            "base_weights": str((ex.get("base_info") or {}).get("path") or ""),
            "promoted_fp32": int(ex.get("promoted_fp32") or 0),
        }

    # =======================================================================
    # 4.1 预检（不加载模型）
    # =======================================================================
    def preflight(self, make_split_if_missing: bool = True) -> Dict[str, Any]:
        """开工前的全部检查。不读 gpt.pth，所以很快（实测 0.2s）。

        返回 dict 而不是抛异常：UI 要把**所有**问题一次展示给用户，
        而不是一条一条地报错让用户反复点。
        """
        out: Dict[str, Any] = {"ok": False, "errors": [], "warnings": [], "infos": []}

        def _add(x: GD.Notice) -> None:
            out["errors" if x.level == "error" else
                ("warnings" if x.level == "warn" else "infos")].append(x.message)

        if not DS.exists(self.dataset):
            _add(GD.Notice("error", f"数据集 `{self.dataset}` 不存在"))
            return out

        # ---- 划分 ----
        sp = DS.load_split(self.dataset)
        if not sp.get("train"):
            if not make_split_if_missing:
                _add(GD.Notice("error", "数据集还没有 train/val 划分，请先点「划分数据集」"))
                return out
            r = DS.make_split(self.dataset, val_ratio=self.val_ratio,
                              seed=self.cfg.seed)
            if not r.get("ok"):
                _add(GD.Notice("error", f"划分失败：{r.get('error', '')}"))
                return out
            sp = DS.load_split(self.dataset)
            _add(GD.Notice("info",
                           f"已自动划分：训练 {r['train']} 条 / 验证 {r['val']} 条"))
        self.train_ids = list(sp.get("train") or [])
        self.val_ids = list(sp.get("val") or [])

        # ---- 训练池（只读 meta.jsonl，不碰 .pt）----
        try:
            tgt = SamplePool(self.dataset, self.train_ids, kind="target")
        except Exception as e:
            _add(GD.Notice("error", f"读训练集失败：{type(e).__name__}: {e}"))
            return out
        if tgt.missing:
            _add(GD.Notice("warn",
                           f"{len(tgt.missing)} 条训练样本缺特征（未提取或已失效），"
                           "已自动跳过。到「数据集」页重新提取即可用。"))
        if tgt.stale:
            _add(GD.Notice("warn",
                           f"{tgt.stale} 条样本的长度字段缺失（旧版本提取的），"
                           "排序与长度过滤会不准。已尝试补写。"))
        ok_ids, bad_ids = tgt.usable_ids(self.options.max_codes,
                                         self.options.max_text_tokens)
        if bad_ids:
            _add(GD.Notice("warn",
                           f"{len(bad_ids)} 条样本因为太短/太长被剔除"
                           f"（codes 限 {self.options.max_codes}，"
                           f"text 限 {self.options.max_text_tokens}）。"
                           "单条音频硬上限是 72.6s（1815 帧 ÷ 25Hz）。"))
        tgt.restrict(ok_ids)
        self.pools["target"] = tgt
        self.train_ids = ok_ids
        n_train = len(ok_ids)
        if n_train == 0:
            _add(GD.Notice("error", "没有一条可用的训练样本"))
            return out

        # val 池不做长度以外的过滤：剔多了会让 first_val 与 best_val 不可比
        vp = SamplePool(self.dataset, self.val_ids, kind="target")
        v_ok, _v_bad = vp.usable_ids(self.options.max_codes,
                                     self.options.max_text_tokens)
        vp.restrict(v_ok)
        self.val_pool = vp
        self.val_pools = {"target": vp}
        self.val_ids = v_ok
        if not self.val_ids:
            _add(GD.Notice("warn",
                           "**验证集为空**：早停与「相对底座的改善」都将无法计算，"
                           "训练会一直跑到 epochs 结束。请提高 val_ratio 或补充数据。"))

        # ---- 回放池 ----
        notes, name_of_replay = self.resolve_replay(select=self._select_replay)
        for x in notes:
            _add(x)
            self.report.note(x.level, x.message)
        if self.pools.get("replay") is not None:
            # resolve_replay 可能在 select 之后又剔掉了与目标集重叠的 id，
            # 池子必须跟着缩，否则 plan_batches 会按下标取到已被排除的样本。
            self.pools["replay"].restrict(self.replay_ids)
        n_replay_pool = len(self.replay_ids)

        # ---- 配置体检 ----
        all_notes = self.cfg.validate(n_train) + self.options.validate()
        self.report.merge_notes(all_notes)
        for x in all_notes:
            _add(x)

        # ---- 步数与显存 ----
        per_epoch = self._samples_per_epoch(n_train, n_replay_pool)
        spe = max(1, math.ceil(per_epoch / self.cfg.global_batch))
        epochs_est = max(1, int(self.cfg.epochs))
        # max_steps 是硬刹车：它优先于 epochs，但不会把 total_steps 抬上去。
        # 调度器的余弦退火按 total_steps 算周期，这个数错了学习率就永远降不到地板。
        self.total_steps = spe * epochs_est
        if self.cfg.max_steps > 0:
            self.total_steps = min(self.total_steps, int(self.cfg.max_steps))
        self.total_steps = max(1, int(self.total_steps))

        est_adapter = EST_ADAPTER_PER_RANK.get(
            self.cfg.target_preset, EST_ADAPTER_PER_RANK["attn"]) * int(self.cfg.rank)
        need = estimate_vram_gb(EST_BASE_PARAMS, est_adapter, self.cfg.bf16,
                                self.cfg.batch_size, self.options.max_codes)
        vr = GD.vram_headroom(need)

        out.update({
            "ok": not out["errors"],
            "n_train": n_train, "n_val": len(self.val_ids),
            "n_replay_pool": n_replay_pool,
            "replay_source": self.replay_src,
            "replay_dataset": name_of_replay,
            "samples_per_epoch": per_epoch,
            "steps_per_epoch": spe,
            "total_steps": self.total_steps,
            "est_adapter_params": est_adapter,
            "est_vram_gb": need,
            "vram": vr,
            "device": self.device,
        })
        self.report.data.update({
            "n_train": n_train, "n_val": len(self.val_ids),
            "n_replay_pool": n_replay_pool,
            "steps_per_epoch": spe, "total_steps": self.total_steps,
        })
        self._pf = out
        return out

    def _select_replay(self, name: str, ready_ids: Sequence[str]) -> List[str]:
        pool = SamplePool(name, ready_ids, kind="replay")
        ok, _bad = pool.usable_ids(self.options.max_codes,
                                   self.options.max_text_tokens)
        pool.restrict(ok)
        self.pools["replay"] = pool
        return ok

    def _samples_per_epoch(self, n_train: int, n_replay_pool: int) -> int:
        """一个 epoch 实际会跑多少条样本（与 build_epoch_plan 同一套公式）。"""
        r = max(0.0, min(0.95, float(self.cfg.replay_ratio or 0.0)))
        n_rep = int(round(n_train * r / (1.0 - r))) if (r > 0 and n_train > 0
                                                        and n_replay_pool > 0) else 0
        return n_train + n_rep

    # =======================================================================
    # 4.2 建模型
    # =======================================================================
    def _build_model(self, pf: Dict[str, Any],
                     progress: Optional[Callable[[float, str], None]]
                     ) -> Dict[str, Any]:
        self._progress(progress, 0.18, "加载 gpt.pth（约 3.2 GB，请稍候）")
        base, base_info = load_base_gpt(self.device, self.model_dir)

        self._progress(progress, 0.58,
                       f"注入 LoRA（{self.cfg.target_preset}, r={self.cfg.rank}）")
        self.model, inj = inject_lora(base, self.cfg)
        base = None                        # 已被 PEFT 包住，不单独留引用

        # 估算值与实际注入数对不上就说明注入面选错了（比如正则没匹上），
        # 这是个便宜但很有用的交叉验证。
        est = int(pf.get("est_adapter_params") or 0)
        act = int(inj["adapter_params"])
        if est and abs(act - est) / max(1, est) > 0.15:
            self.report.note("warn",
                             f"实际可训练参数 {act/1e6:.2f}M 与预估 {est/1e6:.2f}M "
                             f"相差 {abs(act-est)/est*100:.0f}%。"
                             "可能是注入面与预设不一致，或 PEFT 匹到了意外的层。")

        self._progress(progress, 0.64, "对齐精度（底座 bf16 / adapter fp32）")
        cast = cast_base_dtype(self.model, bool(self.cfg.bf16))
        n_promoted = promote_adapter_fp32(self.model)

        # 两个会让训练**静默失效**的开关，见 forward.configure_gpt_for_training
        self._progress(progress, 0.70, "切换到训练态（非重入梯度检查点）")
        self.prev_state = FW.configure_gpt_for_training(
            self.model, grad_checkpointing=bool(self.cfg.grad_checkpointing),
            base_dropout=float(self.options.base_dropout))

        self.report.adapter_params = act
        self.report.base_params = int(inj["base_params"])
        self._log(f"准备完成：{inj['lora_layers']} 个 LoRA 层，"
                  f"{act/1e6:.2f}M 可训练参数（{inj['trainable_pct']:.3f}%），"
                  f"total_steps={self.total_steps}")
        return {"inject": inj, "cast": cast, "promoted_fp32": n_promoted,
                "base_info": base_info}

    # =======================================================================
    # 4.3 数据与前向
    # =======================================================================
    def _to_device(self, b: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        return {k: (v.to(self.device, non_blocking=True)
                    if torch.is_tensor(v) else v) for k, v in b.items()}

    def _gather(self, items: Sequence[Tuple[str, int]],
                pools: Dict[str, SamplePool]) -> List[Dict[str, Any]]:
        """按 (池名, 下标) 取样本。`pools` 必须显式传 ——
        给个默认值就是上面那个 val/train 串池 bug 的根源。"""
        out = []
        for kind, idx in items:
            p = pools.get(kind)
            if p is None or idx >= len(p.ids):
                continue
            s = p.load(p.ids[idx])
            if s is not None:
                out.append(s)
        return out

    def _epoch_batches(self, epoch: int) -> Tuple[List[List[Any]], Dict[str, Any]]:
        plan, rplan = GD.build_epoch_plan(
            len(self.train_ids), len(self.replay_ids),
            float(self.cfg.replay_ratio), seed=int(self.cfg.seed) + epoch)
        info = {"source": self.replay_src, "dataset": self.replay_dataset,
                "pool": len(self.replay_ids), "ratio": float(rplan.ratio),
                "n_replay": int(rplan.n_replay),
                "achieved": float(rplan.achieved),
                "with_replacement": bool(rplan.with_replacement),
                "note": rplan.note}
        batches = plan_batches(plan, self.pools, int(self.cfg.batch_size),
                               bool(self.options.sort_by_length),
                               seed=int(self.cfg.seed) + epoch, len_key="n_codes")
        return batches, info

    def _train_micro_batch(self, items: Sequence[Tuple[str, int]]
                           ) -> Optional[Dict[str, float]]:
        """一个微批：前向 + 反向。没数据可跑时返回 None。

        梯度**不在这里除 grad_accum**：基类 `_optimizer_step` 会按
        **实际累积到的个数**归一，这样跳过空批时有效梯度不会被按比例缩小。
        """
        samples = self._gather(items, self.pools)
        if not samples:
            return None
        b = self._to_device(collate(samples))
        f = FW.gpt_training_forward(
            self.model, b["style"], b["emo_vec"], b["langs"],
            b["text_tokens"], b["text_lengths"], b["codes"], b["code_lengths"])
        w = float(self.options.text_loss_weight or 0.0)
        mel_l = f.mel_loss()
        if w > 0.0:
            txt_l = f.text_loss()
            loss = mel_l + w * txt_l
        else:
            txt_l, loss = None, mel_l
        loss.backward()
        self._accum += 1
        # 先读完标量再返回：backward 之后计算图已释放，但张量值还在。
        # 不调 f.values()：那会把交叉熵**重算一遍**（8194 类 × B×900 位）。
        return {"loss": float(loss), "n": len(samples),
                "mel": float(mel_l),
                "text": (float(txt_l) if txt_l is not None else 0.0),
                "mel_tokens": int(f.mel_mask.sum()),
                "Lc": int(b["codes"].size(1)), "Lt": int(b["text_tokens"].size(1))}

    def evaluate(self, max_batches: int = 0) -> Optional[float]:
        """val loss（按有效 token 加权）。没 val 集时返回 None。

        GPT 这边**没有** CFM 那个 `mask_content` 陷阱（eval 下把条件全清零），
        所以直接 `eval()` + `no_grad` 就行。但必须记得切回 train：
        忘了的话后面所有 step 的 lora_dropout 都不生效，正则静默失效。
        用 try/finally 而不是靠记得。

        val 的 batch 划分**不排序也不洗牌**（seed 固定）：验证集每轮必须是
        同一组数，否则 val 曲线里的抖动分不清是模型变了还是样本变了，
        早停就会误判。
        """
        import torch
        if not self.val_ids or self.val_pool is None or self.model is None:
            return None
        order = [("target", i) for i in range(len(self.val_ids))]
        batches = plan_batches(order, self.val_pools,
                               self.cfg.batch_size, False, seed=self.cfg.seed)
        return self._mean_loss(batches, self.val_pools, max_batches)

    def _mean_loss(self, batches: Sequence[List[Tuple[str, int]]],
                   pools: Dict[str, SamplePool], max_batches: int = 0
                   ) -> Optional[float]:
        """在 no_grad + eval 下算 token 加权平均 loss。"""
        import torch
        was_training = bool(self.model.training)
        w = float(self.options.text_loss_weight or 0.0)
        num = den = 0.0
        self.model.eval()
        try:
            with torch.no_grad():
                for k, items in enumerate(batches):
                    if max_batches and k >= int(max_batches):
                        break
                    samples = self._gather(items, pools)
                    if not samples:
                        continue
                    b = self._to_device(collate(samples))
                    f = FW.gpt_training_forward(
                        self.model, b["style"], b["emo_vec"], b["langs"],
                        b["text_tokens"], b["text_lengths"],
                        b["codes"], b["code_lengths"])
                    wt = float(f.mel_mask.sum()) + w * float(f.text_mask.sum())
                    num += float(f.loss(w)) * wt
                    den += wt
        finally:
            self.model.train(was_training)
        return (num / den) if den > 0.0 else None

    def _train_snapshot(self, max_batches: int = 4) -> Optional[float]:
        """用训练集前几个 batch 算一个当前的 train loss。

        与 val 的差就是过拟合的直接证据：两者接近 = 学到了东西，
        train 远低于 val = 开始背训练集了。只看前几个 batch 是为了
        不在收尾时多花一分钟。
        """
        if not self.train_ids or self.model is None:
            return None
        order = [("target", i) for i in range(len(self.train_ids))]
        batches = plan_batches(order, self.pools, self.cfg.batch_size, False,
                               seed=self.cfg.seed)[:max(1, max_batches)]
        return self._mean_loss(batches, self.pools, 0)
