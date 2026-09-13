"""训练前向：把「与推理对齐」这件事收敛到一个地方。

阶段 2 最大的风险不是「训不动」，而是「训得动但训的是错的东西」——
前向接错一根线，loss 照样下降，只是学到的映射跟推理时对不上。
所以这里只放**已经被 tools/features_probe.py 实测验证过**的前向，
训练器与探针共用同一份代码，探针绿了训练器才是对的。

三条官方陷阱（都在探针里被显式复现并断言）：

  1. `UnifiedVoice.forward()`（indextts/gpt/model_v2.py:596）**不能用于训练**：
       · :639 算 text_emb 时漏了 `+ lang_embedding(langs)`，而推理走的
         prepare_gpt_inputs（:681）有 —— 语言条件训推不一致；
       · :633 的 `torch.zeros(b, 2, d).to(device)` 没指定 dtype，
         use_bf16=True 时 torch.cat 会把 bf16 条件提升成 fp32，
         撞上 bf16 的 LayerNorm 权重直接 RuntimeError。
     → GPT 一律走 `gpt_training_forward()`。

  2. `BASECFM.forward()`（flow_matching.py:116）是**训练**函数：
     它把 `prompt_lens` 传进了 `DiT.forward` 的第 7 个形参 `mask_content`
     （diffusion_transformer.py:186）。eval 模式下
     `not self.training and mask_content` 成立 → class_dropout=True
     → `x_in[..., 80:] *= 0`，prompt_x / cond(mu) / style **全被清零**。
     后果：val loss 变成与条件无关的常数，早停彻底失效。
     → CFM 一律在 train() 下调 `cfm_training_forward()`。

  3. CFM 的 estimator 必须先 `setup_caches()`，否则
     gpt_fast/model.py:169 抛 "Caches must be initialized first"。
     而且要在 **train() 模式下**建缓存：eval 模式会顺带分配 KVCache，
     白占显存（训练根本不用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

__all__ = [
    "unwrap", "GptForward", "masked_ce", "build_length_mask",
    "gpt_training_forward",
    "configure_gpt_for_training", "restore_gpt",
    "setup_cfm_caches", "invalidate_cfm_caches", "cfm_training_forward",
]


def unwrap(model):
    """从 PEFT 包装里取出真正的模块（PeftModel → LoraModel → UnifiedVoice/CFM）。

    PEFT 的 PeftModel / LoraModel 都用 `__getattr__` 把未知属性转发给被包装的
    模块，所以直接 `peft_model.spk_emb_proj` 也能拿到。但转发链路每层都要
    抛一次 AttributeError 再接住，在训练热路径上每步要跑几十次，纯属浪费。

    两个实现细节：
      · 走 `_modules` 而不走 `getattr`：nn.Module.__setattr__ 会把 Module 类型的值
        放进 `_modules` 而不是 `__dict__`，用 `m.__dict__.get("base_model")` 是拿不到的。
      · 靠 `peft_config` 判定「还是不是包装层」：PeftModel 把内层存在
        `base_model` 下，而直接用 `get_peft_model` 得到的 LoraModel 存在 `model` 下，
        只看前者会漏掉后者。
    """
    m = model
    for _ in range(4):
        if not hasattr(m, "peft_config"):            # 不是 PEFT 包装层，到底了
            return m
        mods = getattr(m, "_modules", None) or {}
        nxt = mods.get("base_model")
        if nxt is None:
            nxt = mods.get("model")
        if nxt is None or nxt is m:
            return m
        m = nxt
    return m


# ===========================================================================
# 通用：带掩码的交叉熵
# ===========================================================================

def masked_ce(logits, targets, mask=None):
    """带掩码的交叉熵，返回**标量**（保留梯度）。

    轴约定：logits `(B, V, L)`（官方 `get_logits` 结尾 permute(0,2,1) 过），
    targets / mask `(B, L)`。

    ★ 必须先 `transpose(1, 2)` 再展平。直接 `reshape(-1, V)` 会把
      batch/vocab/seq 三个轴的内存布局搅在一起，算出一个比均匀分布还差的
      假 loss（实测 17.06 vs 基线 ln(8194)=9.01）。这个坑很隐蔽：
      形状对得上、loss 有限、还能反向传播，只是数值全是错的。
    """
    import torch.nn.functional as F

    if logits.dim() != 3:
        raise ValueError(f"masked_ce 只接受 (B,V,L) 的 logits，得到 {tuple(logits.shape)}")
    logits = logits.transpose(1, 2)                       # (B, V, L) -> (B, L, V)
    if logits.shape[:2] != tuple(targets.shape):
        raise ValueError(f"logits/targets 长度不匹配："
                         f"{tuple(logits.shape)} vs {tuple(targets.shape)}")
    per = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                          targets.reshape(-1), reduction="none")
    per = per.view(targets.shape)                          # (B, L)
    if mask is None:
        return per.mean()
    m = mask.to(per.dtype)
    # clamp(min=1)：整批都被掩掉时返回 0 而不是 NaN，
    # NaN 一旦进 AdamW 的动量就再也洗不掉了。
    return (per * m).sum() / m.sum().clamp(min=1.0)


def build_length_mask(lengths, total: int):
    """(B,) 的真实长度 → (B, total) 的 bool 掩码，前 lengths[b] 位为 True。"""
    import torch

    lengths = torch.as_tensor(lengths, dtype=torch.long).reshape(-1)
    ar = torch.arange(total, device=lengths.device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)


# ===========================================================================
# GPT(T2S)
# ===========================================================================

@dataclass
class GptForward:
    """一次 GPT 训练前向的全部产物。

    掩码是**必需**的，不是优化项：批量训练时短样本会被 `set_*_padding`
    填成 stop token，那些位置的 target 也是 stop。不加掩码等于教模型
    「一句话讲完之后继续吐 stop」，batch 里长短差得越多，这个错误信号越强。
    """
    text_logits: Any                       # (B, V_text, Lt)
    text_targets: Any                      # (B, Lt)
    mel_logits: Any                        # (B, V_mel,  Lm)
    mel_targets: Any                       # (B, Lm)
    text_mask: Any                         # (B, Lt) bool
    mel_mask: Any                          # (B, Lm) bool
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------- loss ----------------
    def mel_loss(self):
        return masked_ce(self.mel_logits, self.mel_targets, self.mel_mask)

    def text_loss(self):
        return masked_ce(self.text_logits, self.text_targets, self.text_mask)

    def loss(self, text_weight: float = 0.0):
        """mel 是主目标；text 头默认不参与。

        text 头学的是「下一个文本 token」，对 TTS 的音色/语气没有任何帮助，
        却会和 mel 抢同一批 LoRA 参数的容量。默认权重 0，
        想复现官方那种双头训练再手动调高。
        """
        out = self.mel_loss()
        w = float(text_weight or 0.0)
        if w > 0.0:
            out = out + w * self.text_loss()
        return out

    # ---------------- 只读的展示值 ----------------
    def values(self) -> Dict[str, float]:
        """detach 后的 float，给日志/UI 用（不会把计算图留在显存里）。"""
        import torch

        with torch.no_grad():
            d = {
                "mel_loss": float(self.mel_loss()),
                "text_loss": float(self.text_loss()),
                "mel_tokens": int(self.mel_mask.sum()),
                "text_tokens": int(self.text_mask.sum()),
            }
        return d


def gpt_training_forward(gpt, style, emo_vec, langs, text_tokens, text_lengths,
                         codes, code_lengths, use_lang: bool = True) -> GptForward:
    """与推理路径对齐的 GPT teacher-forcing 前向。

    参数
    ----
    gpt          : `UnifiedVoice`（PEFT 包装过的也行，属性透传）
    style        : (B, 192)  CAMPPlus 声纹
    emo_vec      : (B, 1280) 情感向量（由冻结的 conformer/perceiver 离线算好）
    langs        : (B,)      语言 token
    text_tokens  : (B, Lt)   文本 token（含语言前缀，不含 start/stop）
    text_lengths : (B,)      每条的真实文本长度
    codes        : (B, Lc)   语义 token（25Hz，训练目标）
    code_lengths : (B,)      每条的真实 codes 长度
    use_lang     : 关掉就是官方 forward 的行为，用来做消融对比

    dtype 说明：`use_bf16` 时整个 GPT 被 `.bfloat16()`（infer_v2_5.py:142-143），
    官方把推理包在 autocast 里（:757）。这里同样包 autocast，但**不够**——
    `spk(bf16) + emo_vec(fp32)` 会被类型提升成 fp32，再喂进 bf16 主干又炸。
    所以进来先把条件张量显式对齐到权重 dtype，autocast 只当双保险。
    dtype 从权重反推而不写死，fp32 / bf16 两种加载方式都能跑。
    """
    import torch
    import torch.nn.functional as F

    core = unwrap(gpt)

    b = text_tokens.size(0)
    dev = text_tokens.device
    p_dtype = next(core.spk_emb_proj.parameters()).dtype
    style = style.to(p_dtype)
    emo_vec = emo_vec.to(p_dtype)
    use_amp = p_dtype != torch.float32 and dev.type == "cuda"

    with torch.amp.autocast(dev.type, enabled=use_amp, dtype=p_dtype):
        spk = core.spk_emb_proj(style)                       # (B, dim)
        spk = spk.unsqueeze(1)                               # (B, 1, dim)
        # 与 model_v2.py:768 逐字对齐：[spk+emo, 0, 0] → (B, 3, dim)
        conds = torch.cat(
            [spk + emo_vec.unsqueeze(1),
             torch.zeros(b, 2, spk.size(-1), device=dev, dtype=spk.dtype)], dim=1)

        # ---- 文本侧：set_text_padding → 补 stop → 前后各加一个 token ----
        ti = core.set_text_padding(text_tokens.clone(), text_lengths)
        ti = F.pad(ti, (0, 1), value=core.stop_text_token)
        ti, t_tgt = core.build_aligned_inputs_and_targets(
            ti, core.start_text_token, core.stop_text_token)
        lang_term = core.lang_embedding(langs).unsqueeze(1) if use_lang else 0
        t_emb = core.text_embedding(ti) + core.text_pos_embedding(ti) + lang_term

        # ---- mel 侧：同一套 padding 规则 ----
        mc = core.set_mel_padding(codes.clone(), code_lengths)
        mc = F.pad(mc, (0, 1), value=core.stop_mel_token)
        mc, m_tgt = core.build_aligned_inputs_and_targets(
            mc, core.start_mel_token, core.stop_mel_token)
        m_emb = core.mel_embedding(mc) + core.mel_pos_embedding(mc)

        t_log, m_log = core.get_logits(conds, t_emb, core.text_head,
                                       m_emb, core.mel_head,
                                       get_attns=False, return_latent=False)

    # logits 转回 fp32 再算 loss：bf16 只有 8 位尾数，
    # 8194 类的 log-softmax 在 bf16 下会明显失真。
    t_log, m_log = t_log.float(), m_log.float()

    # 有效位 = 真实长度 + 1（那 +1 是「预测 stop」的位置，必须学）。
    # 之后的位置全是 set_*_padding 造出来的 stop，属于噪声目标，掩掉。
    t_mask = build_length_mask(text_lengths.to(dev) + 1, t_tgt.size(1)).to(dev)
    m_mask = build_length_mask(code_lengths.to(dev) + 1, m_tgt.size(1)).to(dev)

    return GptForward(text_logits=t_log, text_targets=t_tgt,
                      mel_logits=m_log, mel_targets=m_tgt,
                      text_mask=t_mask, mel_mask=m_mask,
                      meta={"n_conds": int(conds.size(1)),
                            "amp": bool(use_amp), "dtype": str(p_dtype),
                            "use_lang": bool(use_lang)})


# ---------------------------------------------------------------------------
# GPT 的训练态切换
# ---------------------------------------------------------------------------

def configure_gpt_for_training(gpt, grad_checkpointing: bool = True,
                               base_dropout: float = 0.0) -> Dict[str, Any]:
    """把 UnifiedVoice 切到可训练状态，并处理两个会让训练**静默失效**的开关。

    陷阱 A：梯度检查点 + 全冻结底座
        `build_hf_gpt_transformer` 建 GPT2Config 时 `gradient_checkpointing=checkpointing`，
        而 UnifiedVoice 的 `checkpointing` 默认 True（config.yaml 里没这一项）。
        于是只要 `.train()`，HF 就会用检查点包每个 block，而 transformers 4.52 的
        `PreTrainedModel` 默认那个 partial 是 **`use_reentrant=True`**。
        reentrant 版本要求至少一个输入 `requires_grad`；我们的 inputs_embeds
        来自冻结的 embedding，一个都不需要梯度 → 反向传播**什么都不回传**，
        loss 照降（其实是随机游走），LoRA 权重纹丝不动。
        → 显式传 `use_reentrant=False`，或干脆关掉检查点。

    陷阱 B：底座 dropout
        stock GPT2Config 的 attn/resid/embd pdrop 默认 0.1。`.train()` 后
        这 24 层主干会随机丢 10% —— 而推理时是 eval，dropout 关闭。
        底座是冻结的，它的 dropout 不是我们能调的正则，只会让 val loss 抖动、
        让早停误判。随机正则统一交给 `lora_dropout`（可配置）。
        → 训练期把底座 pdrop 置 0，退出时恢复。

        改 config 不够：config 只在 __init__ 时读一次，运行中的模块持有的是
        Dropout **实例**，必须逐个改 `m.p`。
    """
    core = unwrap(gpt)
    inner = core.gpt                                   # HF GPT2Model

    prev = {
        "training": bool(core.training),
        "pdrop": (float(inner.config.attn_pdrop), float(inner.config.resid_pdrop),
                  float(inner.config.embd_pdrop)),
        "gc": bool(getattr(inner, "gradient_checkpointing", False)),
    }

    if grad_checkpointing:
        # use_reentrant=False 是唯一能让「冻结底座 + LoRA」正确回传的选项
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        inner.gradient_checkpointing_disable()

    p = max(0.0, float(base_dropout))
    inner.config.attn_pdrop = p
    inner.config.resid_pdrop = p
    inner.config.embd_pdrop = p
    # config 只在 __init__ 时读一次，运行中的模块持有的是 Dropout 实例，
    # 必须把每个实例的 p 也改掉，否则改了 config 等于没改。
    import torch.nn as nn
    n_drop = 0
    for m in inner.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            n_drop += 1
    prev["n_dropout_modules"] = n_drop

    core.train()
    return prev


def restore_gpt(gpt, prev: Dict[str, Any]) -> None:
    """把 `configure_gpt_for_training` 改过的东西全部还原。"""
    inner = unwrap(gpt).gpt
    a, r, e = prev.get("pdrop", (0.1, 0.1, 0.1))
    inner.config.attn_pdrop, inner.config.resid_pdrop, inner.config.embd_pdrop = a, r, e
    import torch.nn as nn
    for m in inner.modules():
        if isinstance(m, nn.Dropout):
            # 三类 dropout 的原值都是同一个 pdrop，统一恢复即可
            m.p = a
    if prev.get("gc"):
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True})
    else:
        inner.gradient_checkpointing_disable()
    unwrap(gpt).train(bool(prev.get("training", False)))


# ===========================================================================
# CFM(S2M)
# ===========================================================================

def _cfm_transformer(cfm):
    """取到真正持有缓存的那个 Transformer。

    `DiT.setup_caches`（diffusion_transformer.py:183）只是个转发：
        self.transformer.setup_caches(max_batch_size, max_seq_length, use_kv_cache=False)
    freqs_cis / causal_mask / max_seq_length / max_batch_size / use_kv_cache
    全部住在 `DiT.transformer` 上。直接往 DiT 写这些属性会创建**同名影子属性**，
    真正的 Transformer 一点没变 —— 不报错，只是静默失效。
    """
    est = unwrap(cfm).estimator
    return getattr(est, "transformer", est)


def setup_cfm_caches(cfm, batch: int, seq_len: int) -> None:
    """幂等地把 estimator 的 RoPE / causal mask 缓存撑到够用。

    `Transformer.setup_caches` 自带早退（两个维度都够大就直接 return），
    所以每个 step 调也不亏；但它**只增不减**，需要缩小时得先
    `invalidate_cfm_caches()`。

    causal_mask 是 O(L²) 的 bool：官方推理入口（infer_v2_5.py:200）一律给 8192，
    那就是 64 MB。训练时按实际帧数给（典型 ~700 帧 → 0.5 MB），
    在 8GB 卡上这点差别值得计较。

    顺带说明：`DiT.setup_caches` 把 `use_kv_cache` **硬编码成 False**，
    所以无论 train 还是 eval 都不会分配 KVCache——CFM 的 25 步 Euler 每步
    都跑完整序列，KV 缓存本来就没用。这也意味着 train/eval 在 mask 处理上
    完全一致，CFM 唯一的 train/eval 差异是 class_dropout（见 cfm_training_forward）。
    """
    est = unwrap(cfm).estimator
    est.setup_caches(max_batch_size=max(1, int(batch)),
                     max_seq_length=max(8, int(seq_len)))


def invalidate_cfm_caches(cfm) -> int:
    """把缓存标记清零，强制下一次 `setup_caches` 真正重建。返回清掉的 KVCache 份数。

    两个用途：
      · 训练完把 O(L²) 的 causal_mask 显存还回去；
      · 训练期用了一个较小的 seq_len，之后要拿同一个实例去推理时，
        推理入口的 `setup_caches(1, 8192)` 会正常重建（8192 更大，不会早退），
        但反过来（先推理后训练）就会因为早退而抱着一个 64 MB 的旧 mask。

    KVCache 在当前版本永远是 0 份（DiT 硬编码 use_kv_cache=False），
    但清理循环保留：上游一旦把那个开关打开，这里不用跟着改。
    """
    tr = _cfm_transformer(cfm)
    tr.max_batch_size = -1
    tr.max_seq_length = -1
    tr.freqs_cis = None
    for attr in ("causal_mask", "mask_cache"):
        if hasattr(tr, attr):
            setattr(tr, attr, None)
    n_kv = 0
    for blk in getattr(tr, "layers", []):
        att = getattr(blk, "attention", None)
        if att is not None and getattr(att, "kv_cache", None) is not None:
            att.kv_cache = None
            n_kv += 1
    return n_kv


def cfm_training_forward(cfm, x1, x_lens, prompt_lens, mu, style,
                         deterministic: bool = False) -> Tuple[Any, Any]:
    """CFM 的训练前向，返回 `(loss, y)`。

    x1          : (B, 80, T)   [prompt_mel | target_mel]
    x_lens      : (B,)         每条的总帧数
    prompt_lens : (B,)         每条的 prompt 帧数（这段不计 loss）
    mu          : (B, T, 512)  length_regulator 的输出，[mu_prompt | mu_target]
    style       : (B, 192)     CAMPPlus 声纹
    deterministic: True = 临时把 `class_dropout_prob` 置 0。
                   **算 val loss 时必须开**，否则每条样本有 10% 概率
                   被整体丢掉条件，val 曲线会随机跳动，早停会误判。

    `BASECFM.forward` 内部会随机采一个 t 和一份噪声，所以同一条样本
    每次 loss 都不同（实测单 seed 区间宽 0.27）。这就是为什么
    deterministic 只关 CFG 丢弃、不试图固定 t —— 想降方差要靠
    多次平均，不是靠单次结果下结论。
    """
    core = unwrap(cfm)
    est = core.estimator
    if not bool(core.training):
        raise RuntimeError(
            "cfm_training_forward 必须在 cfm.train() 下调用。"
            "eval 模式下 BASECFM 传进 DiT 的 mask_content=prompt_lens 会触发 "
            "class_dropout，把 prompt_x / mu / style 全部乘 0 —— "
            "loss 仍然是有限正数，但学不到任何条件信息。")

    setup_cfm_caches(core, batch=x1.size(0), seq_len=x1.size(-1))

    saved = float(est.class_dropout_prob)
    if deterministic and saved > 0.0:
        est.class_dropout_prob = 0.0
    try:
        # zero_prompt_speech_token=False（config.yaml）时 BASECFM 不会原地改 mu；
        # 万一将来配置改了，clone 能保证缓存里的特征不被污染。
        mu_in = mu.clone() if bool(getattr(core, "zero_prompt_speech_token", False)) else mu
        return core.forward(x1, x_lens, prompt_lens, mu_in, style)
    finally:
        est.class_dropout_prob = saved
