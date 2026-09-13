"""CFM(S2M) LoRA 训练器的端到端验证。

**为什么不直接用真模型跑**：真 s2mel.pth 虽然只有 0.39 GB，但完整训练一轮
要几十分钟，没法当回归测试用。这个脚本改用一个**结构完全相同、尺寸极小**的
CFM（DiT 2 层 / 64 维 / 4 头 / wavenet 2 层，42 万参数），在 CPU 上几十秒
跑完整个 preflight → prepare → run → finalize 闭环。

前向本身的正确性（mask_content 陷阱、setup_caches、L1Loss 区间）
已由 `features_probe.py` 在**真实权重**上验证过，这里不重复。
这个脚本盯的是训练器自己的逻辑：配对与批组装、冻结是否真的冻结、
val 的确定性、基线是否真的等于底座、早停/checkpoint/续训/收尾释放
是否都对得上账。

**注入面的一个坑（实验结论，写在这里防止后人再踩）**：
随机冻结的小模型上，transformer 主干（attention / feed_forward）的
LoRA 梯度天然小两个数量级（实测 tiny 上 wqkv 梯度和 0.006，
而 res_projection 是 0.27）——学得动的是**外层**（res_projection /
conv1 / cond_x_merge_linear / t_embedder）。这不是训练器的 bug：
同一份代码在真模型（13 层已训练权重）上 attn 预设的梯度完全健康
（26 层 attn LoRA，一次 backward 梯度和 146，与 attn_mlp 的 MLP 102
同一量级）。所以：
  · 主训练闭环用「外层注入面」——它在小模型上真的能学；
  · attn 预设单独做**梯度连通性**检查（每个 LoRA 张量都拿到非零梯度），
    「能不能学」由真模型负责，不在这里断言 loss 降幅。

跑法：  .venv\\Scripts\\python.exe tools\\cfm_train_probe.py
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import hashlib
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from webui_app.training import cfm_lora as CL               # noqa: E402
from webui_app.training import dataset as DS                # noqa: E402
from webui_app.training import features as FT               # noqa: E402
from webui_app.training import forward as FW                # noqa: E402
from webui_app.training import guard as GD                  # noqa: E402
from webui_app.training import runs as RN                   # noqa: E402
from webui_app.training import trainer_base as TB           # noqa: E402

PASS = FAIL = 0
FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def head(t: str):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# 小尺寸同构模型
# ---------------------------------------------------------------------------

# 与官方 config.yaml 的 s2mel 段**同构**，只是尺寸缩小：
#   DiT 13层/512维/8头 → 2层/64维/4头；wavenet 8层/512维 → 2层/64维。
# length_regulator 缩小只为 MyModel 构造快（训练前向用离线 mu，不碰它）。
# style_encoder.dim 保持 192 —— 特征里的 style 就是 (192,)，不能缩。
TINY_S2MEL = dict(
    dit_type="DiT", reg_loss_type="l1",
    style_encoder=dict(dim=192),
    length_regulator=dict(channels=64, is_discrete=False, in_channels=64,
                          content_codebook_size=128, sampling_ratios=[1, 1, 1, 1],
                          vector_quantize=False, n_codebooks=1,
                          quantizer_dropout=0.0, f0_condition=False,
                          n_f0_bins=512),
    DiT=dict(hidden_dim=64, num_heads=4, depth=2, class_dropout_prob=0.1,
             block_size=8192, in_channels=80, style_condition=True,
             final_layer_type="wavenet", target="mel", content_dim=64,
             content_codebook_size=128, content_type="discrete",
             f0_condition=False, n_f0_bins=512, content_codebooks=1,
             is_causal=False, long_skip_connection=True,
             zero_prompt_speech_token=False, time_as_token=False,
             style_as_token=False, uvit_skip_connection=True,
             add_resblock_in_transformer=False),
    wavenet=dict(hidden_dim=64, num_layers=2, kernel_size=5, dilation_rate=1,
                 p_dropout=0.2, style_condition=True),
)

DMU = 64                       # 小模型里 mu 的维数（真模型是 512）
STYLE_DIM = 192                # 与真模型一致，特征 schema 也这么定


def make_tiny_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create(TINY_S2MEL)


def make_tiny_cfm(device: str = "cpu"):
    """结构与真模型一致、尺寸极小的 CFM(BASECFM+DiT)。"""
    from indextts.s2mel.modules.flow_matching import CFM
    torch.manual_seed(0)
    return CFM(make_tiny_cfg()).to(device)


def base_fingerprint(pm) -> str:
    """底座（不含 adapter）的逐字节指纹（与 gpt_train_probe 同一套）。"""
    core = FW.unwrap(pm)
    h = hashlib.sha256()
    for n, p in core.named_parameters():
        if p.requires_grad:
            continue
        h.update(n.encode("utf-8"))
        h.update(p.detach().cpu().float().numpy().tobytes())
    for n, b in core.named_buffers():
        if b.is_floating_point():
            h.update(n.encode("utf-8"))
            h.update(b.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def lora_tensors(pm) -> dict:
    return {n: p.detach().clone()
            for n, p in pm.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# 可学的合成数据：mel = A @ mu
#
# mu[t] 由 (说话人, 帧位) 完全确定，mel 是 mu 的固定线性读出 ——
# 于是速度场目标 u = x1-(1-σ)z 里由条件决定的那一半是可以精确算出来的，
# LoRA 真的在学映射时 loss 必然下降（eval 的噪声被 rng_guard 定死，
# 任何下降都只能来自模型变化）。
# ---------------------------------------------------------------------------

TEXTS = [
    "今天天气不错，我们出去走走吧。",
    "这个模型的音色克隆效果相当自然。",
    "请问附近的地铁站应该怎么走？",
    "他一边说话一边比划着手势。",
    "训练数据的质量决定了最终效果。",
    "我想听听这首歌的另一个版本。",
    "麻烦你把窗户关上，外面有点吵。",
    "这套流程已经跑通了全部环节。",
    "明天早上八点我们在门口集合。",
    "请把这段话读得再慢一些。",
    "语音合成的自然度提升很明显。",
    "他说到这里停顿了一下。",
]

_A = None          # 80×DMU 的列正交读出矩阵（进程内共享，保证可学）


def readout_matrix() -> torch.Tensor:
    global _A
    if _A is None:
        rng = np.random.default_rng(7)
        Q, _ = np.linalg.qr(rng.standard_normal((80, DMU)))
        _A = torch.from_numpy(Q.astype(np.float32)) * 2.0
    return _A


def synth_sample(spk: int, mel_len: int):
    """返回 (mu(T,D), mel(80,T), style(192,))。mu→mel 是确定性的。"""
    t = np.arange(mel_len)
    freqs = 0.05 + 0.03 * np.arange(DMU)
    mu = np.sin(2 * np.pi * np.outer(t, freqs) + spk * 1.7)
    mu = mu + 0.3 * np.sin(2 * np.pi * 0.011 * t)[:, None] * (spk - 1)
    mu_t = torch.from_numpy(mu.astype(np.float32))
    mel = readout_matrix() @ mu_t.T
    style = torch.zeros(STYLE_DIM, dtype=torch.float32)
    style[spk * 40:(spk + 1) * 40] = 1.0
    return mu_t, mel, style


def make_wav(path: str, seconds: float, sr: int, seed: int) -> None:
    import soundfile as sf
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 110.0 + seed * 7
    sig = (np.sin(2 * np.pi * f0 * t)
           + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
           + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t + seed)
    sig = sig * env
    sig = sig / (np.abs(sig).max() + 1e-9) * 0.6
    noise = rng.standard_normal(n) * 0.002
    sf.write(path, (sig + noise).astype(np.float32), sr, subtype="PCM_16")


def build_dataset(name: str, n: int, sr: int, seed: int = 0) -> dict:
    """造一个「特征已提取好」的数据集（CFM 路径用 mel/mu，GPT 字段给占位）。"""
    DS.create(name, note="cfm_train_probe 自动生成")
    ds_dir = DS.dir_of(name)
    os.makedirs(os.path.join(ds_dir, DS.FEATURE_SUBDIR), exist_ok=True)
    tmpw = os.path.join(ds_dir, "_src")
    os.makedirs(tmpw, exist_ok=True)

    wavs = []
    for i in range(n):
        p = os.path.join(tmpw, f"s{i:03d}.wav")
        make_wav(p, 1.6 + (i % 4) * 0.35, sr, seed + i)
        wavs.append(p)
    imp = DS.import_audio(name, wavs, copy=True, lang="ZH")
    shutil.rmtree(tmpw, ignore_errors=True)

    items = DS.load_meta(name)
    touched = {}
    paths = {}
    for i, u in enumerate(items):
        spk = i % 3                                   # 三个「说话人」
        mel_len = 120 + (i % 5) * 30                  # 120~240 帧（约1.4~2.8s）
        mu, mel, style = synth_sample(spk, mel_len)
        n_codes = max(1, int(round(mel_len / FT.MEL_PER_CODE)))
        feat = {
            # GPT 路径的字段：is_usable 要求齐全，但 CFM 训练永远不读
            "text_tokens": torch.full((8,), 2 + spk, dtype=torch.int64),
            "codes": torch.full((n_codes,), 10 + spk * 50, dtype=torch.int64),
            "emo_vec": torch.zeros(1280, dtype=torch.float32),
            # CFM 路径真正用的字段
            "mel": mel.to(torch.float16),                     # (80, Tm)
            "mu_prompt": mu.to(torch.float16),                # (Tm, DMU)
            "mu_target": mu.to(torch.float16),                # 同一条：prompt==target 自提示
            "style": style,
            "n_codes": int(n_codes),
            "n_text_tokens": 8,
            "mel_len": int(mel_len),
            "lang_token": 1,
            "feature_version": FT.FEATURE_VERSION,
            "extract_seconds": 0.1,
            "extracted_at": time.time(),
            "source": "synthetic",
            "warnings": [],
        }
        p = FT.FeatureExtractor.feature_path(ds_dir, u.id)
        torch.save(feat, p)
        paths[u.id] = p
        touched[u.id] = {
            "text": TEXTS[spk], "has_features": True,
            "features_at": time.time(),
            "mel_len": int(mel_len), "n_codes": int(n_codes),
            "n_text_tokens": 8, "lang_token": 1,
        }
    FT._apply_meta(name, touched)
    st = DS.refresh_all(name, require_features=True)
    return {"imported": imp, "stats": st, "paths": paths, "dir": ds_dir}


# 主训练闭环用的注入面：tiny 模型上真正拿得到梯度的外层
# （原因见文件头「注入面的一个坑」）。t_embedder 的 mlp 让 t 依赖的
# α(t)/β(t) 门控变得可学 —— CFM 的速度场目标本来就是 t 的函数。
PROBE_TARGETS = ["estimator/res_projection", "estimator/conv1",
                 "estimator/cond_x_merge_linear", "estimator/cond_projection",
                 "mlp/0", "mlp/2"]


# ===========================================================================
def main() -> int:
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="cfm_train_probe_")
    ds_root = os.path.join(tmp, "datasets")
    run_root = os.path.join(tmp, "runs")
    guard_root = os.path.join(tmp, "guard")
    model_dir = os.path.join(tmp, "checkpoints")
    os.makedirs(ds_root); os.makedirs(run_root)
    os.makedirs(guard_root); os.makedirs(model_dir)

    # 模块级路径全部指到临时目录（理由与 gpt_train_probe 相同）
    DS.DATASETS_ROOT = ds_root
    RN.ROOT = run_root
    # [6] 要测真的 load_base_cfm（含权重校验），先留一份再打补丁
    real_load_base_cfm = CL.load_base_cfm
    CL.load_base_cfm = lambda device="cpu", model_dir=None, verify_keys=True: (
        make_tiny_cfm(device), {"missing": 0, "unexpected": 0,
                                "path": "<tiny>", "model_dir": model_dir})

    # 假底座目录：给 BaseGuard 一点东西做快照与校验
    with open(os.path.join(model_dir, "s2mel.pth"), "wb") as f:
        f.write(b"\x00" * 4096)
    with open(os.path.join(model_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("s2mel_checkpoint: s2mel.pth\n")

    SR = 22050
    DS_NAME = "probe_cfm"
    RP_NAME = "probe_replay"
    N_DS = 32            # ≥27：val_ratio=0.25 拆完训练集还要 ≥20 条（防线：数据量）

    try:
        # ==============================================================
        head("[1] 造数据集与特征")
        d = build_dataset(DS_NAME, N_DS, SR, seed=1)
        items = DS.load_meta(DS_NAME)
        ready = [u for u in items if u.status == "ready"]
        check(f"{N_DS} 条样本全部 ready", len(ready) == N_DS,
              f"{len(ready)}/{N_DS} · {[(u.id, u.status, u.problems) for u in items if u.status != 'ready'][:3]}")
        check("特征文件被 is_usable 认可",
              all(FT.is_usable(p) for p in d["paths"].values()))
        check("meta 里的 mel_len 已回写",
              all(u.mel_len > 0 for u in items))
        ok_shape = True
        for p in list(d["paths"].values())[:3]:
            smp = FT.build_cfm_sample(torch.load(p, weights_only=False))
            ok_shape &= (smp["mel"].shape[1] == smp["mu_prompt"].shape[0]
                         == smp["mel_len"])
        check("mel 与 mu 帧数一致（build_cfm_pair 的前提）", ok_shape)

        rp = build_dataset(RP_NAME, 6, SR, seed=7)
        check("回放集 6 条全部 ready",
              len([u for u in DS.load_meta(RP_NAME) if u.status == "ready"]) == 6)

        # ==============================================================
        head("[2] 纯逻辑：CfmSamplePool / make_pair / collate / plan_batches")
        pool = CL.CfmSamplePool(DS_NAME, [u.id for u in items], kind="target")
        check(f"样本池收到 {N_DS} 条，没有 missing",
              len(pool) == N_DS and not pool.missing,
              f"len={len(pool)} missing={pool.missing}")
        check("LEN_FIELD 用的是 mel_len（不是 n_codes）",
              pool.LEN_FIELD == "mel_len")
        ok, bad = pool.usable_ids(min_frames=200)
        check("usable_ids 按下限刷掉太短的样本",
              len(bad) > 0 and len(ok) + len(bad) == N_DS,
              f"ok={len(ok)} bad={len(bad)}")
        s = pool.load(pool.ids[0])
        check("load 返回 build_cfm_sample 的字段",
              s is not None and {"mel", "mu_prompt", "mu_target", "style",
                                 "mel_len", "id", "kind"} <= set(s))
        check("mel 保持 fp16（缓存里不翻倍内存，拼对时才 .float()）",
              s["mel"].dtype == torch.float16)
        check("重复 load 命中缓存（同一个对象）", pool.load(pool.ids[0]) is s)
        pool.restrict(ok)
        check("restrict 后 ids/meta 一致", len(pool.ids) == len(ok) and
              set(pool.meta) == set(ok), f"{len(pool.ids)} vs {len(ok)}")

        # ---- truncate_prompt / truncate_target：mel 与 mu 必须同步截 ----
        t60 = CL.truncate_prompt(s, 60)
        check("truncate_prompt 同步截 mel 与 mu_prompt",
              t60["mel"].shape[1] == 60 and t60["mu_prompt"].shape[0] == 60
              and t60["mel_len"] == 60)
        check("truncate_prompt 不动 target 侧",
              t60["mu_target"].shape[0] == s["mu_target"].shape[0])
        check("不超长时 truncate_* 原样返回（同一个对象）",
              CL.truncate_prompt(s, 10_000) is s
              and CL.truncate_target(s, 10_000) is s)

        # ---- make_pair 的合规检查 ----
        opt = CL.CfmTrainOptions()
        pr = pool.load(pool.ids[0]); tg = pool.load(pool.ids[1])
        pair = CL.make_pair(pr, tg, opt)
        check("make_pair 正常路径拼出 (x1, mu, style)",
              pair is not None and
              pair["x1"].shape == (80, pr["mel"].shape[1] + tg["mel"].shape[1]) and
              pair["mu"].shape == (pair["total_len"], DMU) and
              pair["style"].shape == (192,) and
              pair["prompt_len"] == pr["mel"].shape[1])
        check("style 取自 prompt 侧（推理时参考音频的声纹）",
              torch.equal(pair["style"], pr["style"].float()))
        check("target 短于下限时 make_pair 返回 None",
              CL.make_pair(pr, tg, CL.CfmTrainOptions(
                  min_target_frames=10_000)) is None)
        check("prompt 短于下限时 make_pair 返回 None",
              CL.make_pair(pr, tg, CL.CfmTrainOptions(
                  min_prompt_frames=10_000)) is None)
        check("总帧数超 max_total_frames 返回 None（截断救不了）",
              CL.make_pair(pr, tg, CL.CfmTrainOptions(max_total_frames=100)) is None)
        p_tr = CL.make_pair(pr, tg, CL.CfmTrainOptions(
            max_target_frames=50, truncate_target=True))
        check("truncate_target=True 截断而不是丢弃",
              p_tr is not None and p_tr["total_len"] == pr["mel"].shape[1] + 50)
        check("truncate_target=False 整条丢弃",
              CL.make_pair(pr, tg, CL.CfmTrainOptions(
                  max_target_frames=50, truncate_target=False)) is None)
        p_pp = CL.make_pair(pr, tg, CL.CfmTrainOptions(max_prompt_frames=40))
        check("prompt 超长被截到 max_prompt_frames",
              p_pp is not None and p_pp["prompt_len"] == 40)

        # ---- collate ----
        pr2 = pool.load(pool.ids[2]); tg2 = pool.load(pool.ids[3])
        pairs = [CL.make_pair(pr, tg, opt), CL.make_pair(pr2, tg2, opt)]
        b = CL.collate_cfm(pairs)
        Tmax = max(p["total_len"] for p in pairs)
        check("collate 的 x1 是 (B,80,Tmax)、mu 是 (B,Tmax,DMU)",
              tuple(b["x1"].shape) == (2, 80, Tmax) and
              tuple(b["mu"].shape) == (2, Tmax, DMU))
        check("x_lens / prompt_lens 记的是真实长度",
              [int(v) for v in b["x_lens"]] == [p["total_len"] for p in pairs]
              and [int(v) for v in b["prompt_lens"]] == [p["prompt_len"] for p in pairs])
        check("target_frames = total - prompt（帧加权评估用）",
              [int(v) for v in b["target_frames"]] ==
              [p["total_len"] - p["prompt_len"] for p in pairs])
        check("padding 区是 0（掩码保证它不参与前向）",
              float(b["x1"][1, :, pairs[1]["total_len"]:].abs().max() if
                    pairs[1]["total_len"] < Tmax else
                    torch.zeros(1)) == 0.0)

        # ---- plan_batches：prompt 截断计入排序长度 + 不丢样本 ----
        full = CL.CfmSamplePool(DS_NAME, [u.id for u in items], kind="target")
        order = [("target", i, i) for i in range(N_DS)]
        bts = CL.plan_batches(order, {"target": full}, 4, True, seed=1,
                              max_prompt_frames=60)
        flat = [i for bb in bts for _k, i, _p in bb]
        check("plan_batches 不丢样本",
              sorted(flat) == list(range(N_DS)))
        check("batch 大小不超上限",
              all(len(bb) <= 4 for bb in bts))

        # 长度接近的排一起：排序模式下同 batch 的 (tt+min(tp,60)) 极差更小
        def spread(batches):
            out = []
            for bb in batches:
                L = [full.length_of(full.ids[i]) +
                     min(full.length_of(full.ids[i]), 60) for _k, i, _p in bb]
                out.append(max(L) - min(L))
            return sum(out) / len(out)
        bts_rand = CL.plan_batches(order, {"target": full}, 4, False, seed=1,
                                   max_prompt_frames=60)
        check("按长度排序后同 batch 极差更小",
              spread(bts) < spread(bts_rand),
              f"sorted={spread(bts):.1f} vs random={spread(bts_rand):.1f}")

        # ==============================================================
        head("[3] CfmTrainOptions.validate 的拦截")
        o = CL.CfmTrainOptions()
        check("默认配置没有 error",
              not [x for x in o.validate() if x.level == "error"])
        msgs = lambda vv: [x.message for x in vv]
        check("pair_mode 非法被拦",
              any(x.level == "error" for x in
                  CL.CfmTrainOptions(pair_mode="both").validate()))
        check("min≥max 的 prompt 帧数被拦",
              any(x.level == "error" for x in
                  CL.CfmTrainOptions(min_prompt_frames=1292,
                                     max_prompt_frames=1292).validate()))
        check("max_total_frames 装不下最短组合被拦",
              any("装不下" in m for m in msgs(
                  CL.CfmTrainOptions(max_total_frames=70).validate())))
        check("max_total_frames 偏大给 warn（掩码是 T²）",
              any("平方" in m or "MB" in m for m in msgs(
                  CL.CfmTrainOptions(max_total_frames=8192).validate())))
        check("max_prompt_frames 超过推理 15s 给 warn（训推不一致）",
              any("训推不一致" in m for m in msgs(
                  CL.CfmTrainOptions(max_prompt_frames=1400).validate())))
        check("class_dropout=0 给 warn（CFG 分支没见过零条件）",
              any("CFG" in m for m in msgs(
                  CL.CfmTrainOptions(class_dropout=0.0).validate())))
        check("base_dropout=-1 给 info（保持官方）",
              any(x.level == "info" for x in
                  CL.CfmTrainOptions(base_dropout=-1.0).validate()))
        check("val_repeats 超界被拦",
              any(x.level == "error" for x in
                  CL.CfmTrainOptions(val_repeats=9).validate()))
        check("truncate_target=False 给 info（长句会被整条丢）",
              any("整条丢弃" in m for m in msgs(
                  CL.CfmTrainOptions(truncate_target=False).validate())))

        dc = CL.default_config()
        check("default_config 的注入面是 cfm 的名字（wqkv/wo）",
              dc.target_modules == ["attention/wqkv", "attention/wo"], str(dc.target_modules))
        check("default_config：bf16=False（推理时 CFM 是 fp32）",
              dc.bf16 is False)
        check("default_config：梯度检查点如实标 False（CFM 不支持）",
              dc.grad_checkpointing is False)
        check("default_config：batch=1 + grad_accum=8（T² 掩码省显存）",
              dc.batch_size == 1 and dc.grad_accum == 8)
        check("默认 pair_mode=other（对齐推理形态）", o.pair_mode == "other")
        check("max_prompt_frames 默认对齐推理 15s",
              CL.MAX_PROMPT_FRAMES_DEFAULT == 1292
              and o.max_prompt_frames == 1292)
        check("常量与 TODO 钉死的事实一致",
              CL.EST_BASE_PARAMS == 98_187_344
              and CL.OFFICIAL_WAVENET_DROPOUT == 0.2
              and CL.OFFICIAL_CLASS_DROPOUT == 0.1)
        e22 = CL.estimate_vram_gb(CL.EST_BASE_PARAMS, 400_000, False, 1, 2200)
        e11 = CL.estimate_vram_gb(CL.EST_BASE_PARAMS, 400_000, False, 1, 1100)
        e2b = CL.estimate_vram_gb(CL.EST_BASE_PARAMS, 400_000, False, 2, 2200)
        check("estimate_vram_gb 随 T² 增长（掩码+注意力分数+激活）",
              e22 > e11 + 0.1, f"{e22:.2f} vs {e11:.2f}")
        check("estimate_vram_gb 随 batch 增长（T² 掩码随 B 线性放大）",
              e2b > e22, f"{e2b:.2f} vs {e22:.2f}")
        check("T=2200 全量估算超过 1.2GB（WDDM 拦截线之前要能喊出来）",
              e22 > 1.2, f"{e22:.2f}")

        # ==============================================================
        head("[4] 陷阱回归钉（train/eval · mask_content · RNG · dropout）")
        torch.manual_seed(0)
        raw = make_tiny_cfm()
        B, Tp, Tt = 1, 48, 90
        x1 = torch.randn(B, 80, Tp + Tt)
        mu = torch.randn(B, Tp + Tt, DMU)
        st = torch.randn(B, 192)
        xl = torch.LongTensor([Tp + Tt]); pl = torch.LongTensor([Tp])

        # 陷阱 3：eval 模式下 BASECFM.forward 的 mask_content
        raw.eval()
        FW.setup_cfm_caches(raw, 1, Tp + Tt)
        with torch.no_grad():
            torch.manual_seed(5)
            l_real = float(raw.forward(x1, xl, pl, mu.clone(), st)[0])
            torch.manual_seed(5)
            l_zero = float(raw.forward(x1, xl, pl, mu.clone() * 0, st * 0)[0])
        check("★【陷阱3】eval 下条件全清零：真条件与全零条件 loss 完全相同",
              l_real == l_zero, f"{l_real:.6f} vs {l_zero:.6f}")
        raw2 = make_tiny_cfm(); raw2.eval()
        FW.setup_cfm_caches(raw2, 2, Tp + Tt)
        try:
            raw2.forward(torch.randn(2, 80, Tp + Tt), torch.LongTensor([Tp + Tt] * 2),
                         torch.LongTensor([Tp] * 2), torch.randn(2, Tp + Tt, DMU),
                         torch.randn(2, 192))
            check("eval + B>1 直接崩（mask_content 的张量真值判断）", False, "没报错")
        except RuntimeError as e:
            check("eval + B>1 直接崩（mask_content 的张量真值判断）",
                  "ambiguous" in str(e), str(e)[:50])

        # cfm_training_forward 的兜底：eval 下拒绝调用
        try:
            FW.cfm_training_forward(raw, x1, xl, pl, mu, st)
            check("cfm_training_forward 在 eval 下拒绝调用", False, "没报错")
        except RuntimeError as e:
            check("cfm_training_forward 在 eval 下拒绝调用",
                  "train()" in str(e), str(e)[:40])

        # train 模式 + deterministic：条件是有用的
        tr_cfm = make_tiny_cfm()
        prevst = CL.configure_cfm_for_training(tr_cfm, 0.0, -1.0)
        with torch.no_grad():
            torch.manual_seed(5)
            lt_real = float(FW.cfm_training_forward(
                tr_cfm, x1, xl, pl, mu, st, deterministic=True)[0])
            torch.manual_seed(5)
            lt_zero = float(FW.cfm_training_forward(
                tr_cfm, x1, xl, pl, mu * 0, st * 0, deterministic=True)[0])
        check("★ train+deterministic 下条件有用（真条件 ≠ 全零条件）",
              abs(lt_real - lt_zero) > 1e-4,
              f"{lt_real:.4f} vs {lt_zero:.4f}")
        CL.restore_cfm(tr_cfm, prevst)

        # rng_guard：块内可复现、退出后全局 RNG 原地续走
        torch.manual_seed(11)
        a1 = torch.rand(3)
        with CL.rng_guard(123):
            r_in = torch.rand(2)
            with CL.rng_guard(123):
                r_in2 = torch.rand(2)
        a2 = torch.rand(3)
        torch.manual_seed(11)
        _ = torch.rand(3)
        b2 = torch.rand(3)
        check("rng_guard 块内同一 seed 完全可复现",
              torch.equal(r_in, r_in2))
        check("★ rng_guard 退出后全局 RNG 没被重置（序列原地续走）",
              torch.equal(a2, b2),
              f"{a2.tolist()} vs {b2.tolist()}")

        # dropout_off：临时关掉所有 Dropout（含 lora_dropout），退出还原
        from peft import LoraConfig, get_peft_model
        cfm_p = make_tiny_cfm(); GD.freeze_base(cfm_p)
        rx = GD.build_target_regex(["attention/wqkv", "attention/wo"])
        pm_p = get_peft_model(cfm_p, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.3,
                                                target_modules=rx, bias="none",
                                                task_type=None))
        drops_before = [float(m.p) for m in pm_p.modules()
                        if isinstance(m, torch.nn.Dropout)]
        with CL.dropout_off(pm_p) as n_off:
            drops_in = [float(m.p) for m in pm_p.modules()
                        if isinstance(m, torch.nn.Dropout)]
        drops_after = [float(m.p) for m in pm_p.modules()
                       if isinstance(m, torch.nn.Dropout)]
        check("dropout_off 关掉的包括 PEFT 的 lora_dropout",
              n_off == len(drops_before) and 0.3 in drops_before)
        check("dropout_off 块内全部 p=0", all(p == 0.0 for p in drops_in))
        check("dropout_off 退出后逐个还原", drops_after == drops_before)
        del pm_p, cfm_p

        # configure / restore：底座 dropout 与 class_dropout 的账目
        cfm_c = make_tiny_cfm()
        official_drop = [float(m.p) for m in cfm_c.modules()
                         if isinstance(m, torch.nn.Dropout)]
        check("底座里唯一的 Dropout 是 wavenet 的 p=0.2（TODO 钉死的事实）",
              official_drop == [0.2], str(official_drop))
        pv = CL.configure_cfm_for_training(cfm_c, 0.0, -1.0)
        check("configure 后唯一的 Dropout（wavenet）被置 0",
              all(float(m.p) == 0.0 for m in cfm_c.modules()
                  if isinstance(m, torch.nn.Dropout)))
        check("class_dropout=-1 保持官方 0.1",
              float(cfm_c.estimator.class_dropout_prob) == 0.1)
        check("configure 切到 train 模式（mask_content 陷阱的前提）",
              bool(cfm_c.training))
        CL.restore_cfm(cfm_c, pv)
        check("restore 把 wavenet dropout 还原回 0.2",
              [float(m.p) for m in cfm_c.modules()
               if isinstance(m, torch.nn.Dropout)] == [0.2])
        check("restore 后缓存已失效（max_seq_length 归 -1，强制重建）",
              cfm_c.estimator.transformer.max_seq_length == -1)
        pv2 = CL.configure_cfm_for_training(cfm_c, -1.0, 0.0)
        wv_drop = next(iter(m for m in cfm_c.modules()
                            if isinstance(m, torch.nn.Dropout)))
        check("base_dropout=-1 保持官方 0.2；class_dropout=0 被强制",
              float(wv_drop.p) == 0.2
              and float(cfm_c.estimator.class_dropout_prob) == 0.0)
        CL.restore_cfm(cfm_c, pv2)

        # setup_caches 的幂等与只增不减
        FW.setup_cfm_caches(cfm_c, 2, 200)
        mask1 = cfm_c.estimator.transformer.causal_mask
        FW.setup_cfm_caches(cfm_c, 2, 100)          # 更小 → 早退，不缩
        check("setup_caches 只增不减（100 < 200 早退）",
              cfm_c.estimator.transformer.causal_mask is mask1)
        n_kv = FW.invalidate_cfm_caches(cfm_c)
        check("invalidate 后 causal_mask 清空、KV 份数为 0（DiT 硬编码关 KV）",
              cfm_c.estimator.transformer.causal_mask is None and n_kv == 0)
        del cfm_c, tr_cfm, raw, raw2

        # ==============================================================
        head("[5] 注入面：预设 · 死模块 · '*' 展开")
        cfm_s = make_tiny_cfm()
        groups = GD.scan_targets(cfm_s)
        pats = {g.pattern for g in groups}
        check("扫出 DiT 的注意力 wqkv / wo（不是 qkv / out_proj）",
              {"attention/wqkv", "attention/wo"} <= pats)
        check("扫出外层（cond_projection / res_projection）",
              {"estimator/cond_projection", "estimator/res_projection"} <= pats)

        # 【回归】GPT 的名字在 CFM 上一个都匹不上 → 必须报错而不是静默
        cfg_bad = GD.LoRAConfig(rank=4, alpha=8)
        cfg_bad.target_modules = ["attn/c_attn", "attn/c_proj"]
        try:
            TB.inject_lora(cfm_s, cfg_bad)
            check("GPT 的注入面名字用在 CFM 上报「一个都没匹配上」",
                  False, "没报错")
        except ValueError as e:
            check("GPT 的注入面名字用在 CFM 上报「一个都没匹配上」",
                  "一个都没匹配上" in str(e))
        del cfm_s

        # '*' 展开：跳过死模块，且 PEFT 不会把 L1Loss 包进去
        cfm_star = make_tiny_cfm()
        star = GD.resolve_target_patterns(GD.scan_targets(cfm_star), ["*"])
        check("'*' 展开后不含死模块（cond_embedder / content_mask_embedder）",
              not ({"estimator/cond_embedder", "estimator/content_mask_embedder"}
                   & set(star)))
        try:
            pm_star, inj_star = TB.inject_lora(
                cfm_star, GD.LoRAConfig(rank=4, alpha=8,
                                        target_modules=["*"]))
            wrapped_kinds = {type(m.get_base_layer()).__name__
                             for _n, m in pm_star.named_modules()
                             if hasattr(m, "lora_A")}
            check("【回归】'*' 注入成功且没包进 L1Loss",
                  "L1Loss" not in str(wrapped_kinds) and inj_star["lora_layers"] > 0,
                  f"kinds={wrapped_kinds} layers={inj_star['lora_layers']}")
        except ValueError as e:
            check("【回归】'*' 注入成功且没包进 L1Loss",
                  False, str(e)[:60])
        del pm_star, cfm_star

        # attn 预设：梯度连通性（小模型上梯度量级小是冻结随机权重的假象，
        # 真模型实测健康 —— 见文件头。这里只钉「每个张量都拿得到梯度」。）
        cfm_a = make_tiny_cfm()
        cfg_a = GD.LoRAConfig(rank=4, alpha=8, dropout=0.0)
        cfg_a.apply_target_preset("attn", "cfm")
        pm_a, inj_a = TB.inject_lora(cfm_a, cfg_a)
        check("attn 预设注入 4 层（2 层 × wqkv+wo）",
              inj_a["lora_layers"] == 4, f"{inj_a['lora_layers']}")
        pvA = CL.configure_cfm_for_training(pm_a, 0.0, -1.0)
        x1a = torch.randn(1, 80, 120)
        loss_a, _ = FW.cfm_training_forward(
            pm_a, x1a, torch.LongTensor([120]), torch.LongTensor([40]),
            torch.randn(1, 120, DMU), torch.randn(1, 192))
        loss_a.backward()
        grads = {n: p.grad for n, p in pm_a.named_parameters()
                 if p.requires_grad}
        b_ok = all(g is not None and torch.isfinite(g).all()
                   and float(g.abs().sum()) > 0
                   for n, g in grads.items() if ".lora_B." in n)
        a_ok = all(g is not None and torch.isfinite(g).all()
                   for n, g in grads.items() if ".lora_A." in n)
        n_b = sum(1 for n in grads if ".lora_B." in n)
        check("attn 预设的每个 lora_B 都拿到有限非零梯度（梯度真的通了）",
              b_ok, f"{n_b} 个 B 张量")
        check("attn 预设的每个 lora_A 都拿到有限梯度（首步为 0 是数学必然：B=0)",
              a_ok)
        CL.restore_cfm(pm_a, pvA)
        del pm_a, cfm_a

        # ==============================================================
        head("[6] load_base_cfm：好路径 + 三种坏权重")
        from omegaconf import OmegaConf
        real_dir = os.path.join(tmp, "s2mel_dir")
        os.makedirs(real_dir, exist_ok=True)
        # 官方 config.yaml 的根上同时有 s2mel 段和 s2mel_checkpoint 文件名
        root_cfg = OmegaConf.create({"s2mel": TINY_S2MEL,
                                     "s2mel_checkpoint": "s2mel.pth"})
        with open(os.path.join(real_dir, "config.yaml"), "w",
                  encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(root_cfg))
        seed_cfm = make_tiny_cfm("cpu")
        sd_good = seed_cfm.state_dict()

        def write_pth(sd):
            torch.save({"net": {"cfm": sd}},
                       os.path.join(real_dir, "s2mel.pth"))

        # 好路径：missing/unexpected 全 0（结构一致）
        write_pth(sd_good)
        m1, i1 = real_load_base_cfm("cpu", real_dir)
        check("好路径：missing 0 / unexpected 0",
              i1["missing"] == 0 and i1["unexpected"] == 0,
              f"{i1['missing']}/{i1['unexpected']}")
        l1, _y1 = FW.cfm_training_forward(
            m1, torch.randn(1, 80, 120), torch.LongTensor([120]),
            torch.LongTensor([40]), torch.randn(1, 120, DMU),
            torch.randn(1, 192), deterministic=True)
        check("加载出的 CFM 能跑训练前向（loss 有限）",
              torch.isfinite(l1).all().item())
        del m1

        # 坏路径 1：缺关键层
        sd_lack = {k: v for k, v in sd_good.items()
                   if k != "estimator.cond_x_merge_linear.weight"}
        write_pth(sd_lack)
        try:
            real_load_base_cfm("cpu", real_dir)
            check("缺关键权重（cond_x_merge_linear）被拦", False, "没报错")
        except RuntimeError as e:
            check("缺关键权重（cond_x_merge_linear）被拦",
                  "缺少关键权重" in str(e))

        # 坏路径 2：多余的键
        sd_extra = dict(sd_good)
        sd_extra["estimator.from_the_future.weight"] = torch.zeros(4)
        write_pth(sd_extra)
        try:
            real_load_base_cfm("cpu", real_dir)
            check("权重比代码新（unexpected 键）被拦", False, "没报错")
        except RuntimeError as e:
            check("权重比代码新（unexpected 键）被拦", "不认识" in str(e))

        # 坏路径 3：结构不对
        torch.save({"net": {"gpt": sd_good}},
                   os.path.join(real_dir, "s2mel.pth"))
        try:
            real_load_base_cfm("cpu", real_dir)
            check("pth 结构不是 net.cfm 被拦", False, "没报错")
        except RuntimeError as e:
            check("pth 结构不是 net.cfm 被拦", "结构不是预期" in str(e))

        # verify_keys=False 时不拦（供修复流程用），但账要报出来
        write_pth(sd_lack)
        m4, i4 = real_load_base_cfm("cpu", real_dir, verify_keys=False)
        check("verify_keys=False 放行（但报出 missing 的账）",
              i4["missing"] == 1 and i4["unexpected"] == 0,
              f"{i4['missing']}/{i4['unexpected']}")
        del m4, seed_cfm

        # ==============================================================
        head("[7] preflight（不加载模型）")
        cfg = GD.LoRAConfig.preset("balanced")
        cfg.apply_target_preset("attn_mlp", "cfm")     # 预设名只为 est 表
        cfg.target_modules = list(PROBE_TARGETS)       # 实际注入面（文件头说明）
        cfg.rank, cfg.alpha = 16, 32
        cfg.use_rslora = False
        cfg.lr = 2e-2                 # 小模型 + 少量步数，需要大一点的学习率
        cfg.weight_decay = 0.0
        cfg.dropout = 0.0
        cfg.warmup_ratio = 0.05
        cfg.batch_size, cfg.grad_accum = 2, 2
        cfg.epochs, cfg.eval_every = 70, 60
        cfg.val_patience = 0           # 主跑不早停（早停在 [12] 单独验），跑满 max_steps
        cfg.replay_source, cfg.replay_dataset, cfg.replay_ratio = "dataset", RP_NAME, 0.3
        cfg.keep_checkpoints, cfg.bf16 = 3, False
        cfg.max_steps = 560           # 硬刹车（epochs 配得够大，由它刹车）
        opts = CL.CfmTrainOptions(max_prompt_frames=64, max_target_frames=200,
                                  max_total_frames=300, log_every=40)

        tr = CL.CfmTrainer(DS_NAME, cfg=cfg, options=opts, device="cpu",
                           model_dir=model_dir, train_root=guard_root,
                           val_ratio=0.25, run_name="probe_run")
        t0 = time.perf_counter()
        pf = tr.preflight()
        dt = time.perf_counter() - t0
        check("preflight 通过", pf["ok"], str(pf["errors"]))
        check(f"preflight 很快（{dt:.2f}s，没去读权重）", dt < 10.0)
        check("自动划分了 train/val（val_ratio=0.25 → 8 条）",
              pf["n_train"] + pf["n_val"] == N_DS and pf["n_val"] == 8,
              f"train={pf['n_train']} val={pf['n_val']}")
        check("回放池非空", pf["n_replay_pool"] == 6, f"{pf['n_replay_pool']}")
        check("val 配对已固定且非空",
              pf["n_val_pairs"] == pf["n_val"], f"{pf['n_val_pairs']}")
        check("total_steps 被 max_steps 压住（调度器周期才正确）",
              pf["total_steps"] == 560 and
              pf["steps_per_epoch"] * cfg.epochs >= 560,
              f"spe={pf['steps_per_epoch']} total={pf['total_steps']}")
        check("preflight 之后模型还没建", tr.pm is None)
        check("bf16=True 的 warn 不会出现（我们用 False）",
              not any("bf16" in m for m in pf["warnings"]))
        check("batch_size>1 的 warn 出现了（T² 掩码提醒）",
              any("T²" in m or "掩码" in m for m in pf["warnings"]))
        check("回放池信息进了 preflight",
              pf["replay_source"] == "dataset" and pf["replay_dataset"] == RP_NAME)
        check("train/val 划分本身不重叠",
              not (set(tr.train_ids) & set(tr.val_ids)))
        check("val 池与训练池是两个不同的对象",
              tr.val_pools["target"] is not tr.pools["target"])
        check("val 的 batch 划分跨调用稳定（不排序不洗牌）",
              tr.val_batches() == tr.val_batches())
        check("_samples_per_epoch 与回放公式对得上账",
              tr._samples_per_epoch(24, 6) == 34,
              str(tr._samples_per_epoch(24, 6)))

        # 回放集 == 目标集 → 空池警告
        cfg2 = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg2.replay_dataset = DS_NAME
        tr2 = CL.CfmTrainer(DS_NAME, cfg=cfg2, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_run2")
        pf2 = tr2.preflight()
        check("回放集与目标集相同时回放池被判为空",
              pf2["n_replay_pool"] == 0 and
              any("回放数据集与目标数据集相同" in m for m in pf2["warnings"]),
              str(pf2["warnings"])[:120])

        # ==============================================================
        head("[8] prepare：注入 / 冻结 / dtype / 训练态")
        prep = tr.prepare()
        inj = prep["inject"]
        pm = tr.pm
        check("注入面 8 层（外层4 + t_embedder×2 的 mlp.0/mlp.2）",
              inj["lora_layers"] == 8, f"{inj['lora_layers']}")
        check("注入面被 resolve 到的就是 PROBE_TARGETS",
              inj["patterns"] == PROBE_TARGETS, str(inj["patterns"]))
        check("底座全冻结（assert_base_frozen 返回空）",
              GD.assert_base_frozen(pm) == [])
        check("可训练参数 > 0 且小于底座的 1/5",
              0 < inj["adapter_params"] < inj["base_params"] / 5,
              f"{inj['adapter_params']} / {inj['base_params']}")
        check("bf16=False 时底座留在 fp32",
              prep["cast"]["dtype"] == "float32")
        check("adapter 参数是 fp32（AdamW 动量的前提）",
              all(p.dtype == torch.float32 for p in pm.parameters() if p.requires_grad))
        core = FW.unwrap(pm)
        check("模型处于 train 模式（mask_content 陷阱的前提）",
              bool(core.training))
        check("wavenet dropout 被置 0（训推一致）",
              all(float(m.p) == 0.0 for m in core.modules()
                  if isinstance(m, torch.nn.Dropout)))
        check("class_dropout 保持官方 0.1（CFG 分支要见过零条件）",
              float(core.estimator.class_dropout_prob) == 0.1)
        check("run.json 已写入且状态是 running",
              RN.read_run("probe_run").get("status") == RN.STATUS_RUNNING)
        check("run.json 记了 pair_mode 与帧数上限",
              (RN.read_run("probe_run").get("pair_mode") == "other"
               and RN.read_run("probe_run").get("frame_caps", {}).get("total") == 300))

        # ==============================================================
        head("[9] step-0 基线 = 纯底座；val 的确定性")
        # ★ 回归钉：val 必须从 val 池取样（GPT 那边踩过的串池 bug）
        got_ids: list = []
        orig_load = tr.val_pools["target"].load

        def spy_load(uid):
            got_ids.append(uid)
            return orig_load(uid)
        tr.val_pools["target"].load = spy_load
        v_lora0 = tr.evaluate()
        tr.val_pools["target"].load = orig_load
        check("★ evaluate 只碰 val 池的样本（与训练集零交集）",
              got_ids and set(got_ids) <= set(tr.val_ids),
              f"{len(got_ids)} 次 load · 交集 "
              f"{len(set(got_ids) & set(tr.train_ids))}")

        v_b = tr.evaluate()
        check("★ val 确定性：两次 evaluate 逐位一致（rng_guard 定死 t/z）",
              v_lora0 is not None and v_lora0 == v_b,
              f"{v_lora0} vs {v_b}")
        tr.options.val_repeats = 2
        v_r2a = tr.evaluate(); v_r2b = tr.evaluate()
        check("val_repeats=2 依然确定（两组固定噪声取平均）",
              v_r2a == v_r2b)
        check("val_repeats=2 与 =1 的数值不同（多平均了一组噪声）",
              v_r2a != v_lora0, f"{v_r2a} vs {v_lora0}")
        tr.options.val_repeats = 1

        n_off = GD.disable_adapters(pm, True)
        v_base = tr.evaluate()
        GD.disable_adapters(pm, False)
        check("disable_adapters 关掉了 8 层", n_off == 8, f"{n_off}")
        check("★ LoRA 零初始化 → val 与关掉 adapter 完全一致",
              v_base is not None and abs(v_lora0 - v_base) < 1e-9,
              f"lora0={v_lora0} base={v_base}")
        check("评估结束后模型回到 train 模式",
              bool(pm.training))
        check("评估不改变 class_dropout（临时置 0 会还原）",
              float(core.estimator.class_dropout_prob) == 0.1)

        # ==============================================================
        head("[10] 训练：loss 下降 / 底座一字节不变 / adapter 真的动了")
        fp_before = base_fingerprint(pm)
        lora_before = lora_tensors(pm)
        n_lora = inj["lora_layers"]
        b_zero = sum(1 for n, p in pm.named_parameters()
                     if p.requires_grad and ".lora_B." in n
                     and float(p.detach().abs().max()) == 0.0)
        check("训练前所有 lora_B 都是零（PEFT 的标准初始化）",
              b_zero == n_lora, f"{b_zero}/{n_lora}")

        prog = []
        t0 = time.perf_counter()
        rep = tr.run(progress=lambda f, m: prog.append((round(f, 3), m)))
        dt = time.perf_counter() - t0
        print(f"  训练 {rep.steps} 步 / {rep.epochs} 轮 · {dt:.1f}s")
        check("训练正常完成（ok=True，无异常）", rep.ok and not rep.error,
              rep.error or f"steps={rep.steps}")
        check("max_steps 是硬刹车：560 步就停",
              rep.steps == 560, f"{rep.steps}")
        check("max_steps 触发时记录了停止原因",
              "max_steps" in (rep.stop_reason or ""), rep.stop_reason)
        check("进度回调被调用过", len(prog) > 3, f"{len(prog)} 次")
        check("first_val 是基线（step 0）", rep.first_val is not None
              and abs(rep.first_val - v_lora0) < 1e-9,
              f"{rep.first_val} vs {v_lora0}")
        check("best_val 明显低于基线（模型真的学到了东西）",
              rep.best_val is not None and rep.best_val < rep.first_val * 0.90,
              f"{rep.first_val:.4f} → {rep.best_val:.4f}"
              f"（降 {(rep.improved or 0)*100:.1f}%）")
        check("train loss 也降下来了", rep.final_train is not None
              and rep.final_train < rep.first_val,
              f"train={rep.final_train} val={rep.best_val}")
        check("★ 底座权重逐字节未变（防线 1+2 的直接证据）",
              base_fingerprint(pm) == fp_before)
        lora_after = lora_tensors(pm)
        changed = sum(1 for n in lora_before
                      if not torch.equal(lora_before[n], lora_after[n]))
        check("所有 LoRA 参数都被更新过", changed == len(lora_before),
              f"{changed}/{len(lora_before)}")
        b_nonzero = sum(1 for n, p in pm.named_parameters()
                        if p.requires_grad and ".lora_B." in n
                        and float(p.detach().abs().max()) > 0.0)
        check("★ lora_B 不再全零 —— 梯度确实回传了", b_nonzero == n_lora,
              f"{b_nonzero}/{n_lora}")
        check("底座仍然全冻结（训练过程没有解冻任何东西）",
              GD.assert_base_frozen(pm) == [])
        check("回放真的混进了训练（achieved 接近配置的 30%）",
              abs(rep.replay.get("achieved", 0.0) - cfg.replay_ratio) < 0.12,
              f"achieved={rep.replay.get('achieved'):.2%} "
              f"pool={rep.replay.get('pool')} src={rep.replay.get('source')}")
        check("history 里既有 train 也有 val 记录点",
              any(h.get("train") is not None for h in rep.history) and
              sum(1 for h in rep.history if h.get("val") is not None) >= 2)
        check("学习率落在 warmup+余弦退火曲线上（且确实退了）",
              rep.history[0]["lr"] < rep.history[-1]["lr"] < cfg.lr,
              f"{rep.history[0]['lr']:.2e} → {rep.history[-1]['lr']:.2e}")

        # ---- checkpoint / adapter / run.json ----
        ck = RN.vault("probe_run").list()
        check("保险库里存了 checkpoint", len(ck) >= 1, f"{len(ck)} 份")
        check("保险库不超过 keep_checkpoints", len(ck) <= cfg.keep_checkpoints,
              f"{len(ck)} > {cfg.keep_checkpoints}")
        best = RN.best_checkpoint("probe_run")
        check("保险库认定的 best 就是 val 最低的那份",
              best is not None and
              abs(best.metric - min(c.metric for c in ck)) < 1e-12)
        check("adapter/ 里有 PEFT 的文件（每次改善都同步）",
              os.path.isfile(os.path.join(RN.adapter_dir("probe_run"),
                                          "adapter_model.safetensors"))
              or os.path.isfile(os.path.join(RN.adapter_dir("probe_run"),
                                             "adapter_model.bin")))
        rj = RN.read_run("probe_run")
        check("run.json 状态 = done（max_steps 属于正常停）",
              rj.get("status") == RN.STATUS_DONE, str(rj.get("status")))
        check("run.json 的 best_val / steps / epochs 与报告一致",
              abs((rj.get("best_val") or 0) - (rep.best_val or 0)) < 1e-9
              and rj.get("steps") == rep.steps
              and rj.get("epochs") == rep.epochs)
        check("存了优化器状态（可续训）",
              any(os.path.isfile(os.path.join(c.path, "train_state.pt"))
                  for c in ck))
        check("漂移体检算出了非零的 global_rel",
              rep.drift.get("global_rel", 0) > 0, str(rep.drift)[:80])
        check("训练后底座再次校验通过（before + after 两次）",
              rep.base_verify.get("ok") is True and
              rep.base_verify.get("when") == "after")
        check("报告能渲染成 Markdown", len(rep.markdown()) > 400,
              f"{len(rep.markdown())} 字符")
        check("release 后 pm / opt / sched / vault 全部断开",
              tr.pm is None and tr.opt is None and tr.sched is None
              and tr.vault_obj is None)
        check("release 后样本池缓存已清空",
              not tr.pools and not tr.val_pools)

        # ==============================================================
        head("[11] 续训：权重 + 优化器 + 早停 + val 逐位还原")
        tr6 = CL.CfmTrainer(DS_NAME, cfg=cfg, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_resume", resume_from=best.name)
        prep6 = tr6.prepare()
        check("prepare 阶段就把 adapter 权重灌回来了",
              prep6.get("resumed_tensors", 0) >= len(lora_before) // 2,
              str(prep6.get("resumed_tensors")))
        check("恢复了 step 计数（到存盘那一步，不是到训练结束）",
              tr6._step == best.step, f"{tr6._step} vs ckpt {best.step}")
        check("恢复了早停器的 best", tr6.stopper.best is not None
              and abs(tr6.stopper.best - best.metric) < 1e-9)
        check("恢复了优化器动量（state 非空）",
              len(tr6.opt.state) > 0)
        # ★ 续训起点的 val 与 checkpoint 的 metric 完全一致 ——
        # 这只有「权重恢复 + 噪声固定 + 配对固定」三件事同时做对才可能。
        v_resume = tr6.evaluate()
        check("★ 续训起点的 val 与那份 checkpoint 的 metric 完全一致",
              v_resume is not None and abs(v_resume - best.metric) < 1e-9,
              f"{v_resume} vs {best.metric}")
        check("run.json 里记下了续训来源",
              (RN.read_run("probe_resume").get("resume_from") or "").endswith(best.name))
        tr6.release()

        # ==============================================================
        head("[12] 早停 / 手动停 / 轮末兜底评估")
        # 早停：直接喂变差的 val 序列（不依赖真实训练动态）。
        # 主跑的 cfg 关了早停，这里单独开 patience=3。
        cfg_es = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg_es.val_patience = 3
        tr7 = CL.CfmTrainer(DS_NAME, cfg=cfg_es, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_es")
        tr7.prepare()
        v0e = 1.0
        s1 = tr7._handle_eval(v0e, 0)                 # 基线
        s2 = tr7._handle_eval(1.2, 1)                 # 变差 1
        s3 = tr7._handle_eval(1.3, 2)                 # 变差 2
        s4 = tr7._handle_eval(1.4, 3)                 # 变差 3 → patience=3 触发
        check("第一次评估不停（它是基线）", not s1)
        check("变差两次还不停（patience=3）", not s2 and not s3)
        check("连续三次变差 → 早停触发", s4 and tr7.report.stopped_early)
        check("早停原因里写清了最好的那一次在哪",
              "@ epoch 0" in tr7.report.stop_reason, tr7.report.stop_reason)
        check("best_val 留的是最好那一次而不是最后一次",
              abs(tr7.report.best_val - v0e) < 1e-12)
        n_saved = len(RN.vault("probe_es").list())
        check("变差的评估不存 checkpoint（只有改善才存）",
              n_saved == 1, f"{n_saved} 份")
        check("_handle_eval 记下了评估发生在哪一步",
              tr7._last_eval_step == tr7._step)
        tr7.release()

        # 手动停：should_stop 回调
        cfg3 = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg3.max_steps = -1
        cfg3.epochs = 2
        cfg3.eval_every = 1000
        tr3 = CL.CfmTrainer(DS_NAME, cfg=cfg3, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_stop")
        tr3.prepare()
        stop_at = {"n": 0}

        def maybe_stop():
            stop_at["n"] += 1
            return stop_at["n"] > 3
        rep3 = tr3.run(should_stop=maybe_stop)
        check("should_stop 让训练提前结束且状态是 stopped",
              rep3.stop_reason == "用户手动停止"
              and RN.read_run("probe_stop").get("status") == RN.STATUS_STOPPED,
              rep3.stop_reason)
        check("手动停止后仍然做了收尾（底座复校 + 漂移）",
              rep3.base_verify.get("when") == "after"
              and "global_rel" in rep3.drift)
        check("手动停止仍然算成功产出（ok=True，但带着停止原因）",
              rep3.ok and rep3.stop_reason)
        check("停止后模型已释放", tr3.pm is None)

        # eval_every 很大时轮末兜底评估仍然跑（小数据集主路径）
        tr5 = CL.CfmTrainer(DS_NAME, cfg=GD.LoRAConfig.from_dict(cfg3.to_dict()),
                            options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_ckpt")
        tr5.prepare()
        rep5 = tr5.run()
        check("eval_every=1000 时轮末兜底评估仍然跑到了",
              sum(1 for h in rep5.history if h.get("val") is not None) >= 2,
              f"{sum(1 for h in rep5.history if h.get('val') is not None)} 个 val 点")
        check("_train_snapshot 给出了有限的 train loss",
              rep5.final_train is not None and 0 < rep5.final_train < 10)
        tr5.release()

        # ==============================================================
        head("[13] 清理")
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除", not os.path.isdir(tmp))

    except Exception as e:
        import traceback
        traceback.print_exc()
        check(f"未捕获异常：{type(e).__name__}: {e}", False)

    print("\n" + "=" * 70)
    if FAIL == 0:
        print(f"  通过 {PASS} 项 · 失败 0 项")
    else:
        print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
        print("  失败项：")
        for x in FAILS:
            print(f"    FAIL {x}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
