"""L1 · CFM(S2M) LoRA SFT 训练器。

**这个目标练的是什么**：CFM 决定「听起来像谁」—— 音色、音质、频谱细节。
GPT 出的是 25Hz 的语义 token（说什么、什么节奏），CFM 把它渲染成 mel，
再由 BigVGAN 变成波形。只练 GPT 会得到「说话像但音质发飘」的结果，
只练 CFM 会得到「音色对但语气平」的结果 —— 想真像本人，两个都得练。

**为什么它比 GPT 好训**：底座只有 **98,187,344** 参数（GPT 是 813M），
构造一次仅 0.32s，所以预检阶段可以做**精确**扫描而不必靠估算。
代价是另一头的：它的激活值随帧数**平方**增长（注意力），
而且官方在 `DiT.forward` 里造了一个 `(B, 1, T, T)` 的注意力掩码，
所以显存的压力来自长度而不是参数量 —— `max_total_frames` 才是这里真正的旋钮。

**前向不自己写**：一律走 `forward.py::cfm_training_forward`，
那一份绕开了 `BASECFM.forward` 的 `mask_content` 陷阱（eval 模式下会把
prompt_x / mu / style 全部乘 0，val loss 变成与条件无关的常数），
并且已经被 `tools/features_probe.py` 在真实权重上验证过。

**流水线也不自己写**：登记 / 底座校验 / 注入 / 优化器 / 早停 / 保险库 /
收尾 / 释放全在 `trainer_base.BaseTrainer` 里，与 GPT 训练器共用同一份。
本文件只剩 CFM 特有的东西：配对策略、批组装、评估的确定性处理。
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import dataset as DS
from webui_app.training import features as FT
from webui_app.training import forward as FW
from webui_app.training import guard as GD
from webui_app.training import trainer_base as TB

ARCH = "cfm"

# 兼容旧的导入路径：报告类只有一份，住在 trainer_base 里
TrainReport = TB.TrainReport

# 底座权重的关键子串。这些名字是**从真实 s2mel.pth 里读出来的**（256 个张量
# 全部挂在 `estimator.` 下），不是猜的：
#   missing 0 / unexpected 0 —— 与 config.yaml 的 DiT 配置严格一致。
# 不用官方 `load_checkpoint2`：它先按形状过滤再 `strict=False` 加载，
# 形状对不上的键只 print 一行 Warning 就跳过（commons.py:615-619）。
# 少了 cond_x_merge_linear 会让条件融合彻底失效，而 loss 照样是有限正数。
CRITICAL_KEYS = (
    "estimator.cond_x_merge_linear.weight",
    "estimator.cond_projection.weight",
    "estimator.x_embedder.weight_v",
    "estimator.skip_linear.weight",
    "estimator.res_projection.weight",
    "estimator.conv2.weight",
    "estimator.t_embedder.mlp.0.weight",
    "estimator.transformer.norm.project_layer.weight",
    "estimator.transformer.layers.0.attention.wqkv.weight",
    "estimator.transformer.layers.12.attention.wo.weight",
    "estimator.transformer.layers.12.feed_forward.w2.weight",
    "estimator.final_layer.linear.weight_v",
)

MEGA_BATCH = TB.MEGA_BATCH

# 实测常量（tools/_cfm_scan.py 在真实 CFM 上量的，不是拍的）：
#   底座 98,187,344；attn(wqkv+wo) 26 层 = 39,936/rank；
#   attn_mlp 65 层 = 119,808/rank；all_linear 117 层 = 193,500/rank
# （all_linear 是 119 层减去 2 个死模块，见 guard.DEAD_MODULE_PATTERNS）
EST_BASE_PARAMS = 98_187_344
EST_ADAPTER_PER_RANK = {"attn": 39_936, "attn_mlp": 119_808,
                        "all_linear": 193_500}

# 推理时参考音频被截到 **15 秒**（infer_v2_5.py:642 的
# `_load_and_cut_audio(spk_audio_prompt, 15, ...)`）。训练时的 prompt 长度
# 必须与之对齐：训练见的全是 3 秒 prompt、上线给 15 秒，
# 模型对长参考音频的行为就没被约束过。
# 帧数用 ceil：15s = 330750 样本，center=True 的 mel 帧数 = 1+⌊330750/256⌋
# = 1292；直接 int(15×86.13) 会截成 1291，比推理上限少一帧。
PROMPT_AUDIO_SECONDS = 15.0
MAX_PROMPT_FRAMES_DEFAULT = math.ceil(PROMPT_AUDIO_SECONDS * FT.MEL_FPS)   # 1292

PAIR_MODES = ("other", "self", "mix")

# 官方 config.yaml 里的值，写在这里只是为了让 validate() 的提示能说清「默认是多少」
OFFICIAL_WAVENET_DROPOUT = 0.2
OFFICIAL_CLASS_DROPOUT = 0.1


# ===========================================================================
# 1. 配置
# ===========================================================================

@dataclass
class CfmTrainOptions:
    """CFM 训练专属的旋钮（通用的那些在 `guard.LoRAConfig` 里）。

    与 GPT 的 `GptTrainOptions` 相比，这里多出来的一整块都是**配对**：
    CFM 的一条训练样本不是一个张量，而是「参考音频 + 目标音频」的一对。
    """
    # ---- 配对策略 ----
    # other：target 与 prompt 取自**不同**的句子（默认）。这才是推理时的形态 ——
    #   用户给一段参考音频，模型要说的是另一段文本。
    # self ：prompt == target。任务变简单了（x1 的前缀就是标准答案的一部分），
    #   练出来的模型对「参考音频与目标不匹配」的鲁棒性差。
    # mix  ：按 mix_self_prob 混合。数据极少（<20 条）时可用，
    #   让每条样本至少有一种配对是稳定的。
    pair_mode: str = "other"
    mix_self_prob: float = 0.25

    # ---- 帧数上限（这里真正的显存旋钮）----
    # 注意力是 O(T²)，而且 DiT 会实打实造一个 (B,1,T,T) 的掩码
    # （diffusion_transformer.py:236-238），所以 T 的影响远比参数量大。
    max_prompt_frames: int = MAX_PROMPT_FRAMES_DEFAULT    # 1292 ≈ 15s，对齐推理
    min_prompt_frames: int = 32
    max_target_frames: int = 1600                         # ≈ 18.6s
    min_target_frames: int = 32
    max_total_frames: int = 2200                          # Tp + Tt 的硬上限

    # 目标太长时**截断**而不是丢弃：flow matching 是逐帧的，
    # 截断后的 (mel[:N], mu_target[:N]) 仍然是一条自洽的样本，
    # 而丢掉一条 20s 的长句等于白白损失最有信息量的数据。
    truncate_target: bool = True

    sort_by_length: bool = True

    # ---- 两个 dropout ----
    # base_dropout：底座里唯一的 Dropout 是 `estimator.wavenet.drop`（p=0.2，
    #   被 8 层 WaveNet 循环反复调用）。-1 = 保持官方 config，0~0.9 = 强制值。
    #   默认 0.0 的理由是**训推一致**：推理走 eval()，这个 dropout 是关的。
    #   想复现官方训练时的正则强度就填 -1。
    base_dropout: float = 0.0

    # class_dropout：CFG 的条件丢弃概率（官方 0.1）。**建议保持 -1**：
    #   推理时 `inference_cfg_rate=0.7`（infer_v2_5.py:830）会真的走一条
    #   「条件全为 0」的分支（flow_matching.py:89-91）。如果训练时把
    #   class_dropout 关掉，LoRA 从没见过零条件的输入，而推理时那条分支
    #   照样要经过被 LoRA 改过的权重 —— CFG 的修正项就会漂。
    #   算 val loss 时不受这个影响：cfm_training_forward(deterministic=True)
    #   会临时置 0。
    class_dropout: float = -1.0

    # ---- 评估 ----
    # val 的噪声来源是 `BASECFM.forward` 里每次都重采的 t 与 z
    # （flow_matching.py:138-140），实测单 seed 的区间宽达 0.27 ——
    # 比训练带来的改善还大。解决办法不是多跑几次取平均那么简单：
    # 必须让**每次评估用同一组 (t, z)**，这样 val 曲线的抖动才只反映模型变化。
    # 这件事由 `_rng_guard` 保证（见 evaluate）。val_repeats 是在此之上
    # 再多平均几组固定噪声，用来降低「这一组噪声恰好偏乐观」的偶然性。
    val_repeats: int = 1

    min_lr_ratio: float = 0.1
    log_every: int = 10
    save_optimizer_state: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CfmTrainOptions":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []

        def err(m): n.append(GD.Notice("error", m))
        def warn(m): n.append(GD.Notice("warn", m))
        def info(m): n.append(GD.Notice("info", m))

        if self.pair_mode not in PAIR_MODES:
            err(f"pair_mode={self.pair_mode!r} 不合法，可选 {PAIR_MODES}")
        if not 0.0 <= self.mix_self_prob <= 1.0:
            err(f"mix_self_prob={self.mix_self_prob} 超出 0~1")
        elif self.pair_mode != "mix" and self.mix_self_prob != 0.25:
            info("pair_mode 不是 mix，mix_self_prob 不生效")

        mp, tp = int(self.min_prompt_frames), int(self.max_prompt_frames)
        mt, tt = int(self.min_target_frames), int(self.max_target_frames)
        if tp > 0 and mp >= tp:
            err(f"min_prompt_frames({mp}) 必须小于 max_prompt_frames({tp})")
        if tt > 0 and mt >= tt:
            err(f"min_target_frames({mt}) 必须小于 max_target_frames({tt})")
        if mp < 4 or mt < 4:
            err("帧数下限太小（<4）：这么短的片段几乎没有可用的频谱上下文")
        total = int(self.max_total_frames)
        if total > 0:
            need = (tp if tp > 0 else mp) + mt
            if total < need:
                err(f"max_total_frames={total} 装不下「一个最短 prompt + 一个最短 "
                    f"target」（至少 {need}），所有样本都会被丢弃")
            if total > 4096:
                warn(f"max_total_frames={total} 偏大：注意力掩码是 (B,1,T,T)，"
                     f"T={total} 时单这一项就 {total*total/1e6:.1f} MB，"
                     "8GB 卡上很容易静默溢出")
        if tp > MAX_PROMPT_FRAMES_DEFAULT:
            warn(f"max_prompt_frames={tp} 超过推理时的 {MAX_PROMPT_FRAMES_DEFAULT}"
                 f"（=15s×{FT.MEL_FPS:.2f}）。训练见到的参考音频比上线时更长，"
                 "属于训推不一致")
        elif tp < MAX_PROMPT_FRAMES_DEFAULT // 2:
            info(f"max_prompt_frames={tp} 明显短于推理上限 "
                 f"{MAX_PROMPT_FRAMES_DEFAULT}（15s）。省显存，但长参考音频的"
                 "行为没被训练约束过")

        for name, v in (("base_dropout", self.base_dropout),
                        ("class_dropout", self.class_dropout)):
            if not (float(v) < 0.0 or 0.0 <= float(v) <= 0.9):
                err(f"{name}={v} 超出 0~0.9（-1 表示保持官方 config）")
        if float(self.base_dropout) < 0.0:
            info(f"base_dropout=-1：保持官方的 wavenet p={OFFICIAL_WAVENET_DROPOUT}。"
                 "注意推理走 eval()，这个 dropout 是关的 —— 训练开着属于训推不一致，"
                 "但它是官方训练时的设置")
        if float(self.class_dropout) == 0.0:
            warn("class_dropout=0：关掉了 CFG 的条件丢弃。推理时 "
                 "inference_cfg_rate=0.7 会真的走「条件全为 0」那条分支，"
                 "而 LoRA 从没见过这种输入，CFG 修正项可能漂。建议保持 -1")

        if not 1 <= int(self.val_repeats) <= 8:
            err(f"val_repeats={self.val_repeats} 超出 1~8")
        elif int(self.val_repeats) > 1:
            info(f"val_repeats={self.val_repeats}：每次评估会平均这么多组"
                 "**固定**噪声。更稳，但评估耗时按倍数增加")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            err(f"min_lr_ratio={self.min_lr_ratio} 超出 0~1")
        if not self.truncate_target and tt > 0:
            info(f"truncate_target=False：超过 {tt} 帧的目标会被**整条丢弃**"
                 "而不是截断。长句多的数据集会损失不少数据")
        return n


def default_config(preset: str = "balanced", **over: Any) -> GD.LoRAConfig:
    """CFM 的推荐配置。与 `LoRAConfig.preset()` 的三点差别，每一条都有原因：

      · **target_modules 按 cfm 展开**。`preset()` 里写死了 `arch="gpt"`，
        直接拿来用会得到 `["attn/c_attn", "attn/c_proj"]` —— DiT 里根本没有
        这两个名字（它叫 `attention/wqkv` 与 `attention/wo`），
        `resolve_target_patterns` 返回空，注入时报「一个都没匹配上」。
      · **bf16=False**。推理时只有 GPT 被转成 bf16（infer_v2_5.py:142-143），
        CFM 是 fp32。而 CFM 的 loss 是 **L1**（回归速度场之差，量级很小），
        bf16 只有 8 位尾数，在这种小量上精度不够。底座才 98M，
        fp32 也只多占 0.2 GB，没必要省。
      · **batch_size=1 + grad_accum=8**。注意力掩码是 (B,1,T,T)，
        batch 翻倍等于显存翻倍还多。等效批量靠累积拿，不靠 batch。
    """
    cfg = GD.LoRAConfig.preset(preset)
    cfg.apply_target_preset(cfg.target_preset, ARCH)
    cfg.bf16 = False
    cfg.batch_size = 1
    cfg.grad_accum = 8
    # 梯度检查点对 CFM 不适用（见 CfmTrainer.preflight 里的说明），
    # 与其让配置里留一个不生效的 True，不如如实标成 False。
    cfg.grad_checkpointing = False
    for k, v in (over or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ===========================================================================
# 2. 样本池、配对与批组装
# ===========================================================================

class CfmSamplePool(TB.FeaturePool):
    """CFM 样本池：从 `.pt` 里取 mel / mu_prompt / mu_target / style。

    缓存、LRU、meta 索引、老数据集补写全在 `FeaturePool`（GPT 共用）；
    这里只剩 CFM 特有的两件事：取哪些字段、按 mel 帧数过滤。
    """

    LEN_FIELD = "mel_len"          # 排序按 mel 帧数，不是 codes 帧数

    def _convert(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return FT.build_cfm_sample(raw)

    def usable_ids(self, min_frames: int) -> Tuple[List[str], List[str]]:
        """返回 (可用 id, 被刷掉的 id)。长度信息在 meta 里，不碰磁盘。

        只卡下限不卡上限：超长的部分由 `make_pair` 截断处理，
        丢掉一条 20s 的长句比截断它损失大得多。
        """
        lo = max(2, int(min_frames))
        ok, bad = [], []
        for uid in self.ids:
            L = int((self.meta.get(uid) or {}).get("mel_len") or 0)
            (ok if L >= lo else bad).append(uid)
        return ok, bad


def truncate_prompt(sample: Dict[str, Any], max_frames: int) -> Dict[str, Any]:
    """把一条样本的 **prompt 侧**截到 max_frames 帧。

    `mel` 与 `mu_prompt` 必须**同步**截：`build_cfm_pair` 用
    `prompt["mel"].shape[1]` 当 Tp，然后拼 `prompt["mu_prompt"]`。
    两者长度不一致时它会返回 None（x1 与 mu 帧数对不上），
    整个 batch 就少一条 —— 不报错，只是数据悄悄变少。
    """
    T = int(sample["mel"].shape[1])
    if max_frames <= 0 or T <= max_frames:
        return sample
    out = dict(sample)
    out["mel"] = sample["mel"][:, :max_frames]
    out["mu_prompt"] = sample["mu_prompt"][:max_frames]
    out["mel_len"] = max_frames
    return out


def truncate_target(sample: Dict[str, Any], max_frames: int) -> Dict[str, Any]:
    """把一条样本的 **target 侧**截到 max_frames 帧（同理，mel 与 mu_target 同步）。"""
    T = int(sample["mel"].shape[1])
    if max_frames <= 0 or T <= max_frames:
        return sample
    out = dict(sample)
    out["mel"] = sample["mel"][:, :max_frames]
    out["mu_target"] = sample["mu_target"][:max_frames]
    out["mel_len"] = max_frames
    return out


def make_pair(prompt: Dict[str, Any], target: Dict[str, Any],
              opt: CfmTrainOptions) -> Optional[Dict[str, Any]]:
    """按 options 截断并拼出一个 CFM 训练对，不合规时返回 None。

    结构（与 `features.build_cfm_pair` 一致，那里已对齐 infer:839-845）：
        x1  = [prompt_mel | target_mel]     (80, Tp+Tt)
        mu  = [mu_prompt  | mu_target]      (Tp+Tt, 512)
        prompt_lens = Tp，style = prompt 的 style
    loss 只算 `[Tp:x_lens]` 这一段（BASECFM.forward:156），
    与推理时 `vc_target[:, :, ref_mel.size(-1):]` 切掉 prompt 段是同一件事。
    """
    p = truncate_prompt(prompt, int(opt.max_prompt_frames))
    t = target
    Tt = int(t["mel"].shape[1])
    if Tt < int(opt.min_target_frames):
        return None
    if opt.truncate_target and int(opt.max_target_frames) > 0:
        t = truncate_target(t, int(opt.max_target_frames))
        Tt = int(t["mel"].shape[1])
    elif int(opt.max_target_frames) > 0 and Tt > int(opt.max_target_frames):
        return None
    if int(p["mel"].shape[1]) < int(opt.min_prompt_frames):
        return None
    # 总长预算（Tp+Tt ≤ max_total_frames）同样按「截 target」处理：
    # prompt 已可截到 1292、target 到 1600，两者相加 2892 默认就超 2200 ——
    # 若在这里返回 None，长 prompt 配长句的样本会被**静默整对丢掉**，
    # 与 preflight「截断而不是丢弃」的承诺矛盾，步数预估也会虚高。
    total = int(opt.max_total_frames)
    if total > 0:
        budget = total - int(p["mel"].shape[1])
        if budget < int(opt.min_target_frames):
            return None          # prompt 独吞了预算，这条对确实拼不出来
        if Tt > budget:
            t = truncate_target(t, budget)
    return FT.build_cfm_pair(p, t, max_frames=int(opt.max_total_frames))


def collate_cfm(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把若干个训练对拼成一个 batch。

    padding 用 0 就够了，理由有三层，每一层都验过：
      · 注意力：`DiT.forward` 用 `sequence_mask(x_lens)` 造掩码
        （diffusion_transformer.py:236），填充位不会被看到；
      · WaveNet：同一个掩码会乘进去（:249），填充位输出为 0；
      · loss：`BASECFM.forward:156` 只切 `[prompt_lens[b]:x_lens[b]]`，
        填充位在区间之外。
    所以唯一必须传对的是 **x_lens 与 prompt_lens** 这两个真实长度。
    """
    import torch

    B = len(pairs)
    Tmax = max(int(p["total_len"]) for p in pairs)
    Dmu = int(pairs[0]["mu"].shape[1])
    Cin = int(pairs[0]["x1"].shape[0])
    x1 = torch.zeros(B, Cin, Tmax, dtype=torch.float32)
    mu = torch.zeros(B, Tmax, Dmu, dtype=torch.float32)
    for i, p in enumerate(pairs):
        T = int(p["total_len"])
        x1[i, :, :T] = p["x1"]
        mu[i, :T, :] = p["mu"]
    return {
        "x1": x1, "mu": mu,
        "style": torch.stack([p["style"] for p in pairs]),
        "x_lens": torch.LongTensor([int(p["total_len"]) for p in pairs]),
        "prompt_lens": torch.LongTensor([int(p["prompt_len"]) for p in pairs]),
        "target_frames": torch.LongTensor(
            [int(p["total_len"]) - int(p["prompt_len"]) for p in pairs]),
    }


def plan_batches(order: Sequence[Tuple[str, int, int]],
                 pools: Dict[str, CfmSamplePool], batch_size: int,
                 sort_by_length: bool, seed: int = 42,
                 max_prompt_frames: int = 0) -> List[List[Tuple[str, int, int]]]:
    """薄封装：真正的实现在 `trainer_base.plan_batches`（GPT 也用那一份）。

    这里只负责把「(池名, target 下标, prompt 下标) → 总帧数」接上。
    prompt 侧要先按 `max_prompt_frames` 截一刀再相加，否则排序用的长度
    与实际进 batch 的长度不一致 —— 长短混排，padding 白做。
    """
    def len_of(item: Tuple[str, int, int]) -> int:
        kind, ti, pi = item
        p = pools.get(kind)
        if p is None:
            return 0
        tt = p.length_of(p.ids[ti]) if ti < len(p.ids) else 0
        tp = p.length_of(p.ids[pi]) if pi < len(p.ids) else 0
        if max_prompt_frames > 0:
            tp = min(tp, max_prompt_frames)
        return tt + tp

    return TB.plan_batches(order, batch_size, sort_by_length, seed=seed,
                           len_of=len_of)


# ===========================================================================
# 3. 训练态切换与两个上下文管理器
# ===========================================================================

def configure_cfm_for_training(cfm, base_dropout: float = 0.0,
                               class_dropout: float = -1.0) -> Dict[str, Any]:
    """把 CFM 切到可训练状态，返回 prev 供 `restore_cfm` 还原。

    **没有梯度检查点**，这是与 GPT 那边最大的差别：
      · `Transformer`（gpt_fast/model.py）是普通 nn.Module，不是
        HF `PreTrainedModel`，没有 `gradient_checkpointing_enable()` 可调；
      · 它的层间还有 uvit skip connection（`skip_in_x_list` 在循环里
        push/pop，:181-189），逐层包 checkpoint 需要连这个列表一起处理，
        改错了不报错，只是 skip 连错层 —— 不值得冒这个险。
    所以 CFM 省显存的杠杆是 `max_total_frames` 与 `batch_size`，不是检查点。
    """
    import torch.nn as nn

    core = FW.unwrap(cfm)
    est = core.estimator
    prev: Dict[str, Any] = {
        "training": bool(core.training),
        "cdp": float(est.class_dropout_prob),
        "drops": [],
    }
    want_drop = None if base_dropout is None or float(base_dropout) < 0.0 \
        else max(0.0, float(base_dropout))
    for m in core.modules():
        if isinstance(m, nn.Dropout):
            prev["drops"].append((m, float(m.p)))
            if want_drop is not None:
                m.p = want_drop
    if class_dropout is not None and float(class_dropout) >= 0.0:
        est.class_dropout_prob = max(0.0, float(class_dropout))
    core.train()
    return prev


def restore_cfm(cfm, prev: Dict[str, Any]) -> None:
    """把 `configure_cfm_for_training` 改过的东西全部还原，并释放 O(T²) 缓存。"""
    core = FW.unwrap(cfm)
    for m, p in (prev.get("drops") or []):
        m.p = p
    if prev.get("cdp") is not None:
        core.estimator.class_dropout_prob = float(prev["cdp"])
    core.train(bool(prev.get("training", False)))
    # causal_mask 是 (L, L) 的，只增不减；训练用的 L 比推理的 8192 小得多，
    # 不清掉的话下一次推理会抱着一个尺寸不对的旧缓存早退。
    FW.invalidate_cfm_caches(core)


@contextlib.contextmanager
def dropout_off(model):
    """临时把**所有** Dropout 的 p 置 0，退出时逐个还原。

    评估时必须开这个，而且它管的不只是底座：PEFT 的 `lora_dropout`
    也是 `nn.Dropout` 实例（默认 p=0.05）。不关掉的话 val loss 里混进了
    adapter 自己的随机丢弃 —— 曲线抖动，早停误判，而且**看不出来**，
    因为 loss 依然是有限正数、依然在下降。
    """
    import torch.nn as nn

    saved = []
    for m in model.modules():
        if isinstance(m, nn.Dropout) and float(m.p) != 0.0:
            saved.append((m, float(m.p)))
            m.p = 0.0
    try:
        yield len(saved)
    finally:
        for m, p in saved:
            m.p = p


@contextlib.contextmanager
def rng_guard(seed: int):
    """在 with 块里把全局 RNG 定到 seed，退出时**完整还原**。

    为什么必须还原：`BASECFM.forward` 的 `torch.rand` / `torch.randn_like`
    用的是**全局**生成器（flow_matching.py:138-140）。评估插在训练中间，
    如果在这里 `manual_seed` 之后不还原，后面每个训练 step 的 t 与 z
    都会从头再走一遍同一条随机序列 —— 训练悄悄变成了「在同一组噪声上反复练」。
    这种偏差不会体现在 loss 曲线上，只会体现在最终质量上。

    CPU 与 CUDA 两套状态都要存：`t` 在 `mu.device` 上采（CUDA 生成器），
    而 DiT 的 class_dropout 判定 `torch.rand(1)` 在 CPU 上采。
    """
    import torch

    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


# ===========================================================================
# 4. 底座加载与显存估算
# ===========================================================================

def load_base_cfm(device: str = "cuda", model_dir: Optional[str] = None,
                  verify_keys: bool = True):
    """加载一份**独立于推理引擎**的 CFM。

    不复用引擎里那份的三个理由（与 GPT 那边一致）：
      · 引擎的 CFM 是 `eval()`，而 CFM 的训练前向**必须**在 `train()` 下跑
        （否则 mask_content 陷阱会把条件全清零）；
      · 训练期间用户如果去合成一句，两边会互踩（class_dropout_prob 被临时改过）；
      · PEFT 会**原地**改写模块树，引擎那份被改了就没法再正常推理。
    代价很小：CFM 只有 98M 参数，fp32 也才 0.39 GB。

    只建 `MyModel(cfg.s2mel)`（`use_gpt_latent=False`）然后取 `.models["cfm"]`，
    与官方推理入口（infer_v2_5.py:190）一致 —— 那个 checkpoint 里虽然有
    `gpt_layer`，但官方的 `load_checkpoint2` 也只遍历 `model.models` 的键，
    同样不会加载它。`length_regulator` 训练时用不到（mu 已离线算好），
    所以构造完就把 MyModel 丢掉。
    """
    import torch
    from omegaconf import OmegaConf

    from indextts.s2mel.modules.commons import MyModel

    model_dir = model_dir or os.path.join(PROJECT_ROOT, GD.MODEL_DIR_NAME)
    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"找不到 {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    holder = MyModel(cfg.s2mel)
    cfm = holder.models["cfm"]
    del holder                              # length_regulator 用不到，早点释放

    pth = os.path.join(model_dir, str(cfg.s2mel_checkpoint))
    if not os.path.isfile(pth):
        raise FileNotFoundError(f"找不到底座权重 {pth}")

    ck = torch.load(pth, map_location="cpu", weights_only=False)
    net = ck.get("net", ck) if isinstance(ck, dict) else ck
    if not isinstance(net, dict) or "cfm" not in net:
        raise RuntimeError(
            f"{pth} 的结构不是预期的 {{'net': {{'cfm': ...}}}}（得到 "
            f"{list(net.keys())[:6] if isinstance(net, dict) else type(net)}）。"
            "请到「模型」页重新校验/下载 s2mel.pth。")
    sd = net["cfm"]
    missing, unexpected = cfm.load_state_dict(sd, strict=False)
    del ck, net, sd

    if verify_keys:
        have = set(cfm.state_dict().keys())
        got = have - set(missing)
        lack = [k for k in CRITICAL_KEYS if k in have and k not in got]
        if lack:
            raise RuntimeError(
                f"s2mel.pth 缺少关键权重：{lack}\n"
                "底座与 config.yaml 不匹配，训练出来的东西不可信。"
                "请到「模型」页重新校验/下载 s2mel.pth。")
        if unexpected:
            raise RuntimeError(
                f"s2mel.pth 里有 {len(unexpected)} 个模型不认识的键"
                f"（如 {list(unexpected)[:3]}）。这通常意味着权重版本比代码新，"
                "继续训会得到一个与推理不一致的模型。")
    cfm = cfm.to(device)
    return cfm, {"missing": len(missing), "unexpected": len(unexpected),
                 "path": pth, "model_dir": model_dir}


def estimate_vram_gb(base_params: int, adapter_params: int, bf16: bool,
                     batch_size: int, max_total_frames: int) -> float:
    """训练峰值显存的粗估（GB）。

    与 GPT 那边不同，这里的大头是**激活**而不是参数：底座 fp32 才 0.39 GB，
    而注意力掩码是 (B,1,T,T)、注意力分数是 (B,heads,T,T)，两者都是 T² 级。

    系数是粗估（保守偏高），不是实测拟合 —— 这一步要在**加载底座之前**
    就告诉用户装不装得下，等到 WDDM 静默溢出再发现就晚了
    （实测溢出后 CFM 25 步从 2.42s 变 66s，而且不报 OOM）。
    """
    w = 2.0 if bf16 else 4.0
    B = max(1, int(batch_size))
    T = max(64, int(max_total_frames))
    gb = base_params * w / 1e9                       # 底座权重
    gb += adapter_params * 16.0 / 1e9                # fp32 权重4 + 梯度4 + m4 + v4
    gb += 0.55                                       # CUDA 上下文
    gb += B * T * T * 1.0 / 1e9                      # (B,1,T,T) bool 掩码
    # 注意力分数：SDPA 走 mem-efficient 后端时不会实体化，
    # 但显式传了 attn_mask 就有回退到 math 后端的可能，按 25% 计入留个余量。
    gb += B * 8 * T * T * 4.0 * 0.25 / 1e9
    # 13 层 × 每层约 10 份 (B,T,512) 的 fp32 激活（含反向要保留的）
    gb += B * T * 512 * 4.0 * 13 * 10 * 0.6 / 1e9
    return round(gb, 2)


# ===========================================================================
# 5. 训练器
# ===========================================================================

class CfmTrainer(TB.BaseTrainer):
    """CFM(S2M) 的 LoRA SFT 训练器。流水线见 `trainer_base.BaseTrainer`。"""

    ARCH = "cfm"

    def __init__(self, dataset: str, cfg: Optional[GD.LoRAConfig] = None,
                 options: Optional[CfmTrainOptions] = None,
                 run_name: Optional[str] = None, device: Optional[str] = None,
                 model_dir: Optional[str] = None, val_ratio: float = 0.05,
                 resume_from: Optional[str] = None,
                 train_root: Optional[str] = None):
        super().__init__(dataset, cfg=cfg, run_name=run_name, device=device,
                         model_dir=model_dir, val_ratio=val_ratio,
                         resume_from=resume_from, train_root=train_root)
        self.options = options or CfmTrainOptions()
        self.pools: Dict[str, CfmSamplePool] = {}
        self.val_pool: Optional[CfmSamplePool] = None
        # val 单独一套 pools，理由与 GPT 那边完全相同（那里踩过一次）：
        # `_gather` 按下标取样本，而 batch 里的下标是在**哪个池**里算出来的
        # 就必须回那个池取。共用一个默认池的话，val 的下标会落到训练池上 ——
        # loss 照样算得出来、曲线照样下降，但「验证集」实际是训练集的前几条，
        # 早停与「相对底座的改善」全部失效，而且因为自洽所以测试全绿。
        self.val_pools: Dict[str, CfmSamplePool] = {}
        # val 的配对必须**固定**：配对每次重挑的话，val 曲线的抖动就分不清
        # 是模型变了还是「这次恰好挑到一对好配的样本」。所以在这里一次定死，
        # 之后每次 evaluate 都用同一份 (池名, target 下标, prompt 下标)。
        self.val_items: List[Tuple[str, int, int]] = []

    # `pm` 是 `model` 的历史别名（与 GptTrainer 保持一致，探针与 UI 都按这个名取）
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
        return TB.promote_adapter_fp32(self.model) if self.model is not None else 0

    def _restore_model(self) -> None:
        restore_cfm(self.model, self.prev_state)

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
            "pair_mode": str(self.options.pair_mode),
            "frame_caps": {"prompt": int(self.options.max_prompt_frames),
                           "target": int(self.options.max_target_frames),
                           "total": int(self.options.max_total_frames)},
        }

    # =======================================================================
    # 5.1 预检（不加载模型）
    # =======================================================================
    def preflight(self, make_split_if_missing: bool = True) -> Dict[str, Any]:
        """开工前的全部检查。不读 s2mel.pth，所以很快。

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

        opt = self.options
        lo = int(opt.min_target_frames)

        # ---- 训练池（只读 meta.jsonl，不碰 .pt）----
        try:
            tgt = CfmSamplePool(self.dataset, self.train_ids, kind="target")
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
        ok_ids, bad_ids = tgt.usable_ids(lo)
        if bad_ids:
            _add(GD.Notice("warn",
                           f"{len(bad_ids)} 条样本因为太短被剔除"
                           f"（mel 少于 {lo} 帧 ≈ {lo / FT.MEL_FPS:.2f}s）。"
                           "CFM 需要足够的频谱上下文，太短的片段学不到东西。"))
        tgt.restrict(ok_ids)
        self.pools["target"] = tgt
        self.train_ids = ok_ids
        n_train = len(ok_ids)
        if n_train == 0:
            _add(GD.Notice("error", "没有一条可用的训练样本"))
            return out

        n_long = sum(1 for u in ok_ids
                     if int((tgt.meta.get(u) or {}).get("mel_len") or 0)
                     > int(opt.max_target_frames or 0) > 0)
        if n_long and opt.truncate_target:
            _add(GD.Notice(
                "info",
                f"{n_long} 条样本超过 max_target_frames="
                f"{opt.max_target_frames}，会被**截断**到该长度"
                "（不是丢弃）。想保留完整长句就调大这个上限，"
                "代价是显存按平方涨。"))
        if int(opt.max_total_frames or 0) < (
                int(opt.max_prompt_frames) + int(opt.max_target_frames)):
            _add(GD.Notice(
                "info",
                f"max_total_frames={opt.max_total_frames} 小于 "
                f"prompt 上限与 target 上限之和"
                f"（{opt.max_prompt_frames}+{opt.max_target_frames}）："
                "两侧都顶格时 target 会被再截到剩余预算"
                "（截断，不是丢弃）。"))

        # val 池：与训练池同一套下限，但配对要单独定死（见 __init__ 的说明）
        vp = CfmSamplePool(self.dataset, self.val_ids, kind="target")
        v_ok, _v_bad = vp.usable_ids(lo)
        vp.restrict(v_ok)
        self.val_pool = vp
        self.val_pools = {"target": vp}
        self.val_ids = v_ok
        self.val_items = []
        if not self.val_ids:
            _add(GD.Notice("warn",
                           "**验证集为空**：早停与「相对底座的改善」都将无法计算，"
                           "训练会一直跑到 epochs 结束。请提高 val_ratio 或补充数据。"))
        else:
            rnd = random.Random(int(self.cfg.seed) * 31 + 7)
            self.val_items = [(("target", i, self._pick_prompt("target", i, rnd,
                                                              self.val_pools)))
                              for i in range(len(self.val_ids))]
            _add(GD.Notice("info",
                           f"验证集配对已固定（{len(self.val_items)} 对，"
                           f"pair_mode={opt.pair_mode}）。每次评估都用同一组配对"
                           "与同一组噪声，所以 val 曲线只反映模型变化。"))

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
        all_notes = self.cfg.validate(n_train) + opt.validate()
        if bool(self.cfg.grad_checkpointing):
            all_notes.append(GD.Notice(
                "warn",
                "梯度检查点对 CFM **不生效**：`Transformer`（gpt_fast/model.py）是"
                "普通 nn.Module，没有 HF 的 gradient_checkpointing_enable()；"
                "而且它的层间有 uvit skip connection，逐层包 checkpoint 改错了"
                "不会报错、只会把 skip 连错层。CFM 省显存的杠杆是 "
                "max_total_frames 与 batch_size，已按不启用处理。"))
        if bool(self.cfg.bf16):
            all_notes.append(GD.Notice(
                "warn",
                "bf16=True：CFM 在推理时是 **fp32**（infer_v2_5.py:142-143 只把 GPT "
                "转 bf16），而 CFM 的 loss 是 L1（回归速度场之差，量级很小），"
                "bf16 的 8 位尾数在这种小量上精度不够。底座只有 98M，"
                "fp32 也只多占 0.2 GB —— 建议关掉。"))
        if int(self.cfg.batch_size) > 1:
            all_notes.append(GD.Notice(
                "warn",
                f"batch_size={self.cfg.batch_size}：DiT 会实体化一个 (B,1,T,T) 的"
                "注意力掩码，T=2200 时 batch 每加 1 就多 4.8 MB 掩码，"
                "而注意力分数是 (B,8,T,T) —— 显存随 batch **和** T² 一起涨。"
                "8GB 卡上建议 batch_size=1 + grad_accum 拿等效批量。"))
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
                                self.cfg.batch_size,
                                int(opt.max_total_frames or 2200))
        vr = GD.vram_headroom(need)

        out.update({
            "ok": not out["errors"],
            "n_train": n_train, "n_val": len(self.val_ids),
            "n_val_pairs": len(self.val_items),
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
            "pair_mode": opt.pair_mode,
        })
        self.report.data.update({
            "n_train": n_train, "n_val": len(self.val_ids),
            "n_val_pairs": len(self.val_items),
            "n_replay_pool": n_replay_pool,
            "steps_per_epoch": spe, "total_steps": self.total_steps,
        })
        self._pf = out
        return out

    def _select_replay(self, name: str, ready_ids: Sequence[str]) -> List[str]:
        pool = CfmSamplePool(name, ready_ids, kind="replay")
        ok, _bad = pool.usable_ids(int(self.options.min_target_frames))
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
    # 5.2 建模型
    # =======================================================================
    def _build_model(self, pf: Dict[str, Any],
                     progress: Optional[Callable[[float, str], None]]
                     ) -> Dict[str, Any]:
        self._progress(progress, 0.18, "加载 s2mel.pth（CFM 98M 参数）")
        base, base_info = load_base_cfm(self.device, self.model_dir)

        self._progress(progress, 0.58,
                       f"注入 LoRA（{self.cfg.target_preset}, r={self.cfg.rank}）")
        self.model, inj = TB.inject_lora(base, self.cfg)
        base = None                        # 已被 PEFT 包住，不单独留引用

        # 估算值与实际注入数对不上就说明注入面选错了（比如正则没匹上），
        # 这是个便宜但很有用的交叉验证。CFM 的估算尤其可靠 ——
        # 底座构造只要 0.32s，EST_ADAPTER_PER_RANK 是在真模型上量出来的。
        est = int(pf.get("est_adapter_params") or 0)
        act = int(inj["adapter_params"])
        if est and abs(act - est) / max(1, est) > 0.15:
            self.report.note("warn",
                             f"实际可训练参数 {act/1e6:.2f}M 与预估 {est/1e6:.2f}M "
                             f"相差 {abs(act-est)/est*100:.0f}%。"
                             "可能是注入面与预设不一致，或 PEFT 匹到了意外的层。")

        self._progress(progress, 0.64, "对齐精度（底座 / adapter fp32）")
        cast = TB.cast_base_dtype(self.model, bool(self.cfg.bf16))
        n_promoted = TB.promote_adapter_fp32(self.model)

        self._progress(progress, 0.70, "切换到训练态（dropout / CFG 丢弃）")
        self.prev_state = configure_cfm_for_training(
            self.model, base_dropout=float(self.options.base_dropout),
            class_dropout=float(self.options.class_dropout))

        self.report.adapter_params = act
        self.report.base_params = int(inj["base_params"])
        self._log(f"准备完成：{inj['lora_layers']} 个 LoRA 层，"
                  f"{act/1e6:.2f}M 可训练参数（{inj['trainable_pct']:.3f}%），"
                  f"total_steps={self.total_steps}")
        return {"inject": inj, "cast": cast, "promoted_fp32": n_promoted,
                "base_info": base_info}

    # =======================================================================
    # 5.3 配对与数据
    # =======================================================================
    def _pick_prompt(self, kind: str, ti: int, rnd: random.Random,
                     pools: Optional[Dict[str, CfmSamplePool]] = None
                     ) -> int:
        """为 target 下标 `ti` 挑一个 prompt 下标（**同一个池内**）。

        同池而不是跨池：回放集的说话人与目标角色不是同一个人，
        拿目标的 prompt 去配回放的目标（或反过来）等于在教模型
        「用 A 的音色说 B 的内容」—— 那不是我们要的能力，
        而且 style 与 mu 会互相矛盾，loss 里全是噪声。

        prompt 与推理时的参考音频同源（都是「另一个人的另一句话」），
        所以默认 pair_mode="other"。只有池里就一条样本时才退回 self。
        """
        pools = pools if pools is not None else self.pools
        p = pools.get(kind)
        n = len(p.ids) if p is not None else 0
        mode = str(self.options.pair_mode)
        if n <= 1 or mode == "self":
            return ti
        if mode == "mix" and rnd.random() < float(self.options.mix_self_prob):
            return ti
        # O(1) 且均匀地取「ti 以外的一个下标」：
        # 先 randrange(n-1) 再把 ≥ti 的往后挪一位。
        # 写成 `[j for j in range(n) if j != ti]` 也行，但那是每条样本 O(n)，
        # 几千条的数据集每个 epoch 白跑几百万次比较。
        j = rnd.randrange(n - 1)
        return j if j < ti else j + 1

    def _to_device(self, b: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        return {k: (v.to(self.device, non_blocking=True)
                    if torch.is_tensor(v) else v) for k, v in b.items()}

    def _make_pair(self, pools: Dict[str, CfmSamplePool], kind: str,
                   ti: int, pi: int) -> Optional[Dict[str, Any]]:
        p = pools.get(kind)
        if p is None or ti >= len(p.ids) or pi >= len(p.ids):
            return None
        tgt = p.load(p.ids[ti])
        if tgt is None:
            return None
        # pi == ti 时复用同一个对象：池里缓存的是同一份张量，
        # 再 load 一次只是走一遍 LRU，白白多一次 dict 查找。
        prm = tgt if pi == ti else p.load(p.ids[pi])
        if prm is None:
            return None
        return make_pair(prm, tgt, self.options)

    def _gather(self, items: Sequence[Tuple[str, int, int]],
                pools: Dict[str, CfmSamplePool]) -> List[Dict[str, Any]]:
        """按 (池名, target 下标, prompt 下标) 取训练对。

        `pools` **必须显式传** —— 给它一个默认值（`self.pools`）就是 GPT
        那边 val/train 串池 bug 的根源：调用方忘了传，val 就悄悄算在训练集上，
        而且因为处处自洽，测试全绿都发现不了。
        """
        out = []
        for kind, ti, pi in items:
            pair = self._make_pair(pools, kind, ti, pi)
            if pair is not None:
                out.append(pair)
        return out

    def _epoch_batches(self, epoch: int) -> Tuple[List[List[Any]], Dict[str, Any]]:
        plan, rplan = GD.build_epoch_plan(
            len(self.train_ids), len(self.replay_ids),
            float(self.cfg.replay_ratio), seed=int(self.cfg.seed) + epoch)
        # 每个 epoch 换一个种子重挑 prompt：固定配对等于每条 target 永远
        # 只见过同一个参考音频，模型会记住「这个 target 配这个 prompt」
        # 而不是学会「按 style/mu 渲染」。
        rnd = random.Random(int(self.cfg.seed) * 1000 + epoch * 17 + 3)
        items = [(k, ti, self._pick_prompt(k, ti, rnd)) for k, ti in plan]
        info = {"source": self.replay_src, "dataset": self.replay_dataset,
                "pool": len(self.replay_ids), "ratio": float(rplan.ratio),
                "n_replay": int(rplan.n_replay),
                "achieved": float(rplan.achieved),
                "with_replacement": bool(rplan.with_replacement),
                "pair_mode": str(self.options.pair_mode),
                "note": rplan.note}
        batches = plan_batches(items, self.pools, int(self.cfg.batch_size),
                               bool(self.options.sort_by_length),
                               seed=int(self.cfg.seed) + epoch,
                               max_prompt_frames=int(self.options.max_prompt_frames))
        return batches, info

    # =======================================================================
    # 5.4 前向与评估
    # =======================================================================
    def _train_micro_batch(self, items: Sequence[Tuple[str, int, int]]
                           ) -> Optional[Dict[str, float]]:
        """一个微批：前向 + 反向。没数据可跑时返回 None。

        梯度**不在这里除 grad_accum**：基类 `_optimizer_step` 会按
        **实际累积到的个数**归一，这样跳过空批时有效梯度不会被按比例缩小。

        `deterministic=False`：训练时保留 class_dropout（CFG 需要它，
        见 CfmTrainOptions.class_dropout 的说明）与随机 t/z。
        """
        pairs = self._gather(items, self.pools)
        if not pairs:
            return None
        b = self._to_device(collate_cfm(pairs))
        loss, _y = FW.cfm_training_forward(
            self.model, b["x1"], b["x_lens"], b["prompt_lens"],
            b["mu"], b["style"], deterministic=False)
        loss.backward()
        self._accum += 1
        # 先读完标量再返回：backward 之后计算图已释放，但张量值还在。
        # `_y` 是 BASECFM.forward 顺手算的 `estimator_out + (1-σ)z`
        # （flow_matching.py:159），训练用不上，直接丢。
        return {"loss": float(loss), "n": len(pairs),
                "T": int(b["x1"].size(-1)),
                "tgt_frames": int(b["target_frames"].sum()),
                "prompt_frames": int(b["prompt_lens"].sum())}

    def evaluate(self, max_batches: int = 0) -> Optional[float]:
        """val loss（按**目标帧数**加权）。没 val 集时返回 None。

        三件事必须同时做对，少一件 val 就是错的：

        1. **保持 `train()` 模式**。这是 CFM 独有的坑：`BASECFM.forward` 把
           `prompt_lens` 传进了 `DiT.forward` 的第 7 形参 `mask_content`，
           eval 模式下 `not self.training and mask_content` 成立 →
           class_dropout=True → `x_in[..., 80:] *= 0`，prompt_x / mu / style
           **全被清零**。val loss 会变成一个与条件无关的常数（仍然是有限正数，
           仍然「在下降」，因为模型在学无条件分布），早停彻底失效。
           `cfm_training_forward` 里有一道 RuntimeError 兜底，但这里从设计上
           就不该切 eval。

        2. **关掉所有 Dropout**（`dropout_off`）。因为要留在 train() 模式，
           底座的 wavenet dropout 与 PEFT 的 lora_dropout 都会是开着的。

        3. **固定 (t, z)**（`rng_guard`）。`BASECFM.forward` 每次都重采一个
           时间步 t 和一份噪声 z，实测单 seed 的 loss 区间宽达 0.27 ——
           比训练带来的改善还大。不固定的话 val 曲线全是噪声，早停等于抛硬币。
        """
        if not self.val_items or self.val_pool is None or self.model is None:
            return None
        return self._mean_loss(self.val_batches(), self.val_pools, max_batches)

    def val_batches(self) -> List[List[Tuple[str, int, int]]]:
        """val 的 batch 划分。**不排序也不洗牌**（seed 固定）：
        验证集每轮必须是同一组数、同一个顺序，否则 val 曲线里的抖动
        分不清是模型变了还是样本变了。
        """
        return plan_batches(self.val_items, self.val_pools,
                            int(self.cfg.batch_size), False,
                            seed=int(self.cfg.seed),
                            max_prompt_frames=int(self.options.max_prompt_frames))

    def _mean_loss(self, batches: Sequence[List[Tuple[str, int, int]]],
                   pools: Dict[str, CfmSamplePool], max_batches: int = 0,
                   deterministic: bool = True) -> Optional[float]:
        """在 no_grad + **train()** 下算帧加权平均 loss。

        加权而不是直接平均：`BASECFM.forward` 对每条样本先取 L1 的 mean
        再对 batch 取 mean（:154-157），所以一条 100 帧的样本和一条 1000 帧的
        样本权重相同。算整体指标时按目标帧数加权才代表「每个 mel 帧的平均误差」。

        `val_repeats` 是在「固定噪声」之上再平均几组**不同的固定**噪声：
        每组内部完全可复现（所以跨评估可比），组间取平均降低单组噪声的偶然偏差。
        """
        import torch

        reps = max(1, int(self.options.val_repeats))
        num = den = 0.0
        with dropout_off(self.model), torch.no_grad():
            for rep in range(reps):
                # 7919 是个质数，避免与 cfg.seed 的倍数关系让两组噪声撞上
                with rng_guard(int(self.cfg.seed) + 7919 * rep):
                    for k, items in enumerate(batches):
                        if max_batches and k >= int(max_batches):
                            break
                        pairs = self._gather(items, pools)
                        if not pairs:
                            continue
                        b = self._to_device(collate_cfm(pairs))
                        loss, _y = FW.cfm_training_forward(
                            self.model, b["x1"], b["x_lens"], b["prompt_lens"],
                            b["mu"], b["style"], deterministic=deterministic)
                        w = float(b["target_frames"].clamp(min=1).sum())
                        num += float(loss) * w
                        den += w
        return (num / den) if den > 0.0 else None

    def _train_snapshot(self, max_batches: int = 4) -> Optional[float]:
        """用训练集前几个 batch 算一个当前的 train loss。

        与 val 的差就是过拟合的直接证据：两者接近 = 学到了东西，
        train 远低于 val = 开始背训练集了。只看前几个 batch 是为了
        不在收尾时多花一分钟。

        配对与噪声都固定（seed 与 val 不同源），否则这个数每次都在抖，
        拿来和 best_val 比 70% 阈值就毫无意义。
        """
        if not self.train_ids or self.model is None:
            return None
        rnd = random.Random(int(self.cfg.seed) * 131 + 11)
        items = [("target", i, self._pick_prompt("target", i, rnd))
                 for i in range(len(self.train_ids))]
        batches = plan_batches(items, self.pools, int(self.cfg.batch_size), False,
                               seed=int(self.cfg.seed),
                               max_prompt_frames=int(
                                   self.options.max_prompt_frames))[:max(1, max_batches)]
        with rng_guard(int(self.cfg.seed) + 4219):
            return self._mean_loss(batches, self.pools, 0, deterministic=True)
