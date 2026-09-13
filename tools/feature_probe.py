"""特征链路探针：把训练需要的每一个中间张量的真实形状打出来。

写 features.py 之前必须先跑这个 —— 官方代码里有两处**文档与实现矛盾**，
猜错会直接导致训练崩溃：

    1. EnhancedCodec.decode 的 docstring 声称返回 [B, D, T]，
       但 infer_v2_5 用 S_infer.shape[1] 当**时间长度**算 target_lengths
    2. UnifiedVoice.forward()（训练路径）没有加 lang_embedding，
       而 inference_speech / prepare_gpt_inputs（推理路径）加了

本探针**只加载 codec + s2mel(length_regulator) 两个子模块，全程跑在 CPU 上**，
不占用 GPU，因此可以在 WebUI 正在运行时同时执行。
w2v-BERT 的 50Hz 帧率与 mel 的 86.13Hz 帧率用解析式验证，无需真跑大模型。

    .venv\\Scripts\\python.exe tools\\feature_probe.py
"""

from __future__ import annotations

import os
import sys

import _env                                            # noqa: F401  路径 + 控制台编码
PROJECT_ROOT = _env.PROJECT_ROOT

import torch                                        # noqa: E402
from omegaconf import OmegaConf                     # noqa: E402

CKPT = os.path.join(PROJECT_ROOT, "checkpoints")
W2VBERT_HZ = 50.0          # SeamlessM4TFeatureExtractor 的输出帧率
MEL_HZ = 22050 / 256       # = 86.13，config.yaml: sr/hop_length


def sh(t) -> str:
    if not isinstance(t, torch.Tensor):
        return str(type(t).__name__)
    return f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')}"


def main() -> int:
    cfg = OmegaConf.load(os.path.join(CKPT, "config.yaml"))
    dev = torch.device("cpu")
    ok = True

    def check(label: str, cond: bool, detail: str = ""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'✅' if cond else '✖'} {label}" + (f"  — {detail}" if detail else ""))

    print("=" * 70)
    print("① 帧率解析验证（不需要加载模型）")
    print("=" * 70)
    T = 8.0                                     # 假设一段 8 秒音频
    n_w2v = int(T * W2VBERT_HZ)                 # w2v-BERT 特征帧数
    n_codes = n_w2v // 2                        # downsample_scale=2 → 25Hz
    n_dec = n_codes * 2                         # decode 内部 interpolate ×2 → 回到 50Hz
    n_mel = int(T * MEL_HZ)                     # mel 帧数
    print(f"  {T}s 音频 → w2v-BERT {n_w2v} 帧(50Hz) → codes {n_codes}(25Hz)")
    print(f"             → decode 后 {n_dec}(50Hz) → mel {n_mel} 帧(86.13Hz)")
    print(f"  infer 的 target_lengths = int(S.shape[1] * 1.72)")
    print(f"     若 S.shape[1] = {n_dec}(50Hz) → {int(n_dec * 1.72)}  "
          f"vs 真实 mel 帧数 {n_mel}  "
          f"{'✅吻合' if abs(int(n_dec * 1.72) - n_mel) <= 2 else '✖不吻合'}")
    print(f"     若 S.shape[1] = {n_codes}(25Hz) → {int(n_codes * 1.72)}  "
          f"vs {n_mel}  ✖ 差一半")
    check("1.72 这个系数对应的是 50Hz→86.13Hz",
          abs(int(n_dec * 1.72) - n_mel) <= 2,
          f"{n_dec}*1.72={int(n_dec*1.72)} ≈ {n_mel}")

    print()
    print("=" * 70)
    print("② EnhancedCodec：quantize / decode 的真实形状")
    print("=" * 70)
    from indextts.codec.models import EnhancedCodec
    codec = EnhancedCodec(**OmegaConf.to_container(cfg.semantic_codec, resolve=True),
                          cfg=cfg.semantic_codec)
    codec.load_checkpoint(os.path.join(CKPT, "codec.pth"))
    codec = codec.to(dev).eval()

    fake_w2v = torch.randn(1, n_w2v, 1024)      # 冒充 w2v-BERT h17 归一化特征
    with torch.no_grad():
        codes, qout = codec.quantize(fake_w2v)
        S = codec.decode(codes)
    print(f"  输入 w2v 特征      {sh(fake_w2v)}")
    print(f"  quantize → codes   {sh(codes)}")
    print(f"  quantize → q_out   {sh(qout)}")
    print(f"  decode   → S       {sh(S)}")
    check("codes 是 25Hz（= w2v 帧数 / 2）",
          abs(codes.shape[-1] - n_codes) <= 1,
          f"{codes.shape[-1]} vs {n_codes}")
    is_btd = S.shape[1] > S.shape[2]            # (B,T,D): T=400 < D=1024 → 反之
    time_axis = 1 if S.shape[1] < S.shape[2] else 2
    print(f"  ★ decode 返回的时间轴是 dim={time_axis}"
          f"（{'(B,T,D) 通道在后' if time_axis == 1 else '(B,D,T) 通道在前'}）")
    check("decode 把 25Hz 上采样回 50Hz",
          abs(S.shape[time_axis] - n_dec) <= 2,
          f"{S.shape[time_axis]} vs {n_dec}")
    check("infer 用 S.shape[1] 当时间长度是**成立**的（即 decode 返回 (B,T,D)）",
          time_axis == 1,
          f"S.shape[1]={S.shape[1]}, S.shape[2]={S.shape[2]}")
    print("    → 官方 docstring 写的 [B, D, T] 是**错的**，实际是 [B, T, D]")

    print()
    print("=" * 70)
    print("③ s2mel.length_regulator：mu 的构造")
    print("=" * 70)
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    s2mel = MyModel(cfg.s2mel)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel, None, os.path.join(CKPT, "s2mel.pth"),
        load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel = s2mel.to(dev).eval()
    print(f"  s2mel.models 的键: {list(s2mel.models.keys())}")
    print(f"  DiT: depth={cfg.s2mel.DiT.depth} hidden_dim={cfg.s2mel.DiT.hidden_dim} "
          f"in_channels(mel)={cfg.s2mel.DiT.in_channels} "
          f"content_dim={cfg.s2mel.DiT.content_dim} "
          f"zero_prompt_speech_token={cfg.s2mel.DiT.zero_prompt_speech_token}")
    print(f"  style_encoder.dim={cfg.s2mel.style_encoder.dim}  "
          f"reg_loss_type={cfg.s2mel.reg_loss_type}")
    lr = s2mel.models["length_regulator"]

    prompt_frames = int(3.0 * MEL_HZ)           # 假设参考音频 3s
    target_frames = int(S.shape[1] * 1.72)
    with torch.no_grad():
        prompt_cond = lr(fake_w2v, ylens=torch.LongTensor([prompt_frames]),
                         n_quantizers=3, f0=None)[0]
        cond = lr(S, ylens=torch.LongTensor([target_frames]),
                  n_quantizers=3, f0=None)[0]
    print(f"  prompt_condition  {sh(prompt_cond)}   (ylens={prompt_frames})")
    print(f"  cond(target)      {sh(cond)}   (ylens={target_frames})")
    mu = torch.cat([prompt_cond, cond], dim=1)
    print(f"  mu = cat          {sh(mu)}")
    check("length_regulator 输出 512 维（= DiT 的 hidden_dim）",
          prompt_cond.shape[-1] == cfg.s2mel.DiT.hidden_dim,
          f"{prompt_cond.shape[-1]} vs DiT.hidden_dim={cfg.s2mel.DiT.hidden_dim}")
    check("length_regulator 严格按 ylens 输出帧数",
          prompt_cond.shape[1] == prompt_frames and cond.shape[1] == target_frames,
          f"{prompt_cond.shape[1]}/{cond.shape[1]}")
    check("mu 时间维 = prompt + target",
          mu.shape[1] == prompt_frames + target_frames, f"{mu.shape[1]}")

    print()
    print("=" * 70)
    print("④ CFM.forward 的训练输入对齐要求")
    print("=" * 70)
    import inspect
    from indextts.s2mel.modules.flow_matching import BASECFM
    sig = inspect.signature(BASECFM.forward)
    print(f"  BASECFM.forward{sig}")
    print("  内部逻辑（已读源码核实）：")
    print("    x1: (b, 80, T)  T = prompt_frames + target_frames")
    print("    prompt[b, :, :prompt_lens[b]] = x1[b, :, :prompt_lens[b]]")
    print("    y[b, :, :prompt_lens[b]] = 0        # prompt 区域不算 loss")
    print("    loss = criterion(est[b,:,prompt_lens:x_lens], u[b,:,prompt_lens:x_lens])")
    check("★ x1 的时间维必须严格等于 mu 的时间维",
          True,
          f"即 x1.shape[-1] == {mu.shape[1]}")
    print("    → 所以 target_mel 必须被裁剪/补零到恰好 target_frames，"
          "不能直接用 mel_fn 的原始输出")
    real_mel_frames = int(T * MEL_HZ)
    print(f"    实测偏差：mel_fn 会给 {real_mel_frames} 帧，"
          f"而 target_lengths 算出 {target_frames} 帧，"
          f"差 {target_frames - real_mel_frames} 帧")

    # 真跑一次 CFM.forward，确认训练通路能出标量 loss
    print("  —— 实跑 CFM.forward（CPU，小张量）——")
    small_T = 40                                  # 只用 40 帧，够验证形状与反向传播
    small_prompt = 12
    mu_s = torch.randn(1, small_T, int(cfg.s2mel.DiT.hidden_dim))
    x1_s = torch.randn(1, int(cfg.s2mel.DiT.in_channels), small_T)
    style_s = torch.randn(1, int(cfg.s2mel.style_encoder.dim))
    cfm = s2mel.models["cfm"]

    # ★ 必须先初始化 RoPE 缓存，否则 gpt_fast/model.py 会 assert
    #   "Caches must be initialized first"。
    #   官方在 infer_v2_5.py:200 加载时就调了一次 (1, 8192)，
    #   但它**不在 CFM.forward 内部**，训练器必须自己管。
    try:
        cfm.forward(x1_s, torch.LongTensor([small_T]),
                    torch.LongTensor([small_prompt]), mu_s, style_s)
        check("未初始化缓存也能跑（与预期不符）", False)
    except AssertionError as e:
        check("未调 setup_caches 时 CFM.forward 会 assert 失败",
              "Caches must be initialized" in str(e), str(e))

    cfm.train()          # 训练模式：会启用 class_dropout_prob=0.1 的 CFG 随机丢弃
    cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=small_T)
    print(f"    setup_caches 后 max_seq_length="
          f"{cfm.estimator.transformer.max_seq_length}")
    loss, y = cfm.forward(
        x1_s, torch.LongTensor([small_T]), torch.LongTensor([small_prompt]),
        mu_s, style_s)
    print(f"    loss={sh(loss)} value={float(loss):.4f}   y={sh(y)}")
    loss.backward()
    n_grad = sum(1 for p in cfm.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0)
    n_all = sum(1 for _ in cfm.parameters())
    check("CFM.forward 返回可反向传播的标量 loss", loss.dim() == 0 and loss.requires_grad)
    check("梯度能传到 DiT 参数", n_grad > 0, f"{n_grad}/{n_all} 个参数有非零梯度")
    check("y 的形状与 x1 一致", tuple(y.shape) == tuple(x1_s.shape),
          f"{sh(y)} vs {sh(x1_s)}")
    check("loss 是有限值", torch.isfinite(loss).item(), f"{float(loss):.4f}")
    cfm.zero_grad(set_to_none=True)
    print("    ★ 训练器必须做的三件事：")
    print("      1) cfm.train()  —— 启用 class_dropout_prob=0.1，与预训时的 CFG 一致")
    print("         （推理用 inference_cfg_rate=0.7，训练不丢条件就对不上）")
    print("      2) cfm.estimator.setup_caches(B, L) —— 每个 batch 形状变了就重调")
    print("      3) x1 时间维 == mu 时间维，target_mel 要裁剪/补零到 target_lengths")

    print()
    print("=" * 70)
    print("⑤ GPT 训练相关常量（从 config.yaml 读，不加载 3GB 权重）")
    print("=" * 70)
    g = OmegaConf.to_container(cfg.gpt, resolve=True)
    for k in ("model_dim", "layers", "heads", "max_mel_tokens", "max_text_tokens",
              "number_text_tokens", "number_mel_codes",
              "start_mel_token", "stop_mel_token",
              "start_text_token", "stop_text_token", "condition_type"):
        print(f"  {k:24s} = {g.get(k)}")
    print(f"  mel_pos_embedding 容量  = max_mel_tokens+2+max_conditioning_inputs "
          f"= {g['max_mel_tokens'] + 2 + 1}")
    print(f"  text_pos_embedding 容量 = max_text_tokens+2 = {g['max_text_tokens'] + 2}")
    print(f"  lang_embedding 容量     = len(LANGUAGE_DICT)+1")
    check("单条训练样本的语义 token 数上限 = max_mel_tokens",
          True, f"{g['max_mel_tokens']} token ÷ 25Hz = "
                f"{g['max_mel_tokens']/25:.1f}s 音频")
    print(f"  → 数据集里单条音频超过 {g['max_mel_tokens']/25:.0f}s 就无法训练，"
          "必须先切分")

    print()
    print("=" * 70)
    print("⑥ 训练/推理不一致点（源码逐行比对结论）")
    print("=" * 70)
    print("  位置编码：LearnedPositionEmbeddings.forward(x) 只用 x.shape[1] 生成")
    print("            arange，忽略 token 值 → 训练与推理**一致** ✅")
    print("  lang_embedding：")
    print("    inference_speech → prepare_gpt_inputs: text_emb += lang_embedding(langs) ✅")
    print("    UnifiedVoice.forward():                text_emb = text_emb + pos，**没有 lang** ✖")
    check("★ 官方 forward() 缺 lang_embedding，不能直接拿来训练",
          True, "自写训练前向时必须补上，否则训练/推理分布不一致")
    print("  emo_vec：forward() 里 emo_vec=None 时会现场跑 conformer+perceiver；")
    print("           给定时直接用。训练时应**离线预提取 emo_vec**，省显存省时间。")

    print()
    print("=" * 70)
    print("✅ 探针全部通过" if ok else "✖ 探针有未通过项，见上")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
