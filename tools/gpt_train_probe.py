"""GPT(T2S) LoRA 训练器的端到端验证。

**为什么不直接用真模型跑**：真 gpt.pth 是 3.2 GB，一次完整训练在 8GB 卡上要
几十分钟，没法当回归测试用。这个脚本改用一个**结构完全相同、尺寸极小**的
UnifiedVoice（2 层 / 64 维 / 4 头 / 词表 256），在 CPU 上十几秒跑完整个
preflight → prepare → run → finalize 闭环。

前向本身的正确性（lang_embedding、mask、交叉熵轴约定、reentrant 检查点）
已经由 `features_probe.py` 在**真实权重**上验证过了，这里不重复。
这个脚本盯的是训练器自己的逻辑：冻结是否真的冻结、基线是否真的等于底座、
早停/checkpoint/续训/回放/收尾释放是否都对得上账。

跑法：  .venv\\Scripts\\python.exe tools\\gpt_train_probe.py
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402

from webui_app.training import dataset as DS           # noqa: E402
from webui_app.training import features as FT          # noqa: E402
from webui_app.training import forward as FW           # noqa: E402
from webui_app.training import gpt_lora as GL          # noqa: E402
from webui_app.training import guard as GD             # noqa: E402
from webui_app.training import runs as RN              # noqa: E402

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

TINY = dict(
    layers=2, model_dim=64, heads=4,
    max_text_tokens=64, max_mel_tokens=256, max_conditioning_inputs=1,
    number_text_tokens=512, number_mel_codes=256,
    start_text_token=0, stop_text_token=1,
    start_mel_token=254, stop_mel_token=255,
    checkpointing=False, types=1,
    # 真配置是 output_size=512 / num_blocks=4 / input_layer=conv2d2。
    # 这两个模块（emo_conditioning_encoder / emo_perceiver_encoder）在
    # UnifiedVoice.__init__ 里是**无条件建**的，即使 spk_cond_mode=campplus；
    # 但训练前向直接拿缓存好的 emo_vec，根本不会调它们。
    # 所以给个极小的同构配置就行，只为让 __init__ 跑得过去。
    emo_condition_module=dict(output_size=32, linear_units=32, attention_heads=4,
                              num_blocks=1, input_layer="linear", perceiver_mult=1),
)
N_MEL = TINY["number_mel_codes"]
START_MEL, STOP_MEL = TINY["start_mel_token"], TINY["stop_mel_token"]
DIM = TINY["model_dim"]


def make_tiny_gpt(device: str = "cpu"):
    """结构与真模型一致、尺寸极小的 UnifiedVoice。

    关键是 `spk_cond_mode="campplus"`：这才会建出 `spk_emb_proj` 与
    `lang_embedding`，也正是训练前向要走的那条分支。
    """
    from indextts.gpt.model_v2 import UnifiedVoice
    torch.manual_seed(0)
    m = UnifiedVoice(**TINY, use_accel=False, spk_cond_mode="campplus")
    return m.to(device)


def base_fingerprint(pm) -> str:
    """底座（不含 adapter）的逐字节指纹。

    只取 requires_grad=False 的参数与浮点 buffer —— PEFT 原地改写之后，
    底座权重挂在 `*.base_layer.weight` 上，adapter 挂在 `*.lora_A/lora_B`，
    用 requires_grad 区分比用名字匹配可靠得多。
    """
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


def gc_uses_non_reentrant(inner) -> bool:
    """确认梯度检查点用的是 use_reentrant=False。

    不能只查 `config.gradient_checkpointing`：transformers 4.52 的
    `gradient_checkpointing_enable()` 改的是**模块属性**与
    `_gradient_checkpointing_func`（一个 partial），config 不一定跟着变。
    而真正决定「冻结底座 + LoRA 能不能回传」的是那个 partial 的关键字。
    """
    fn = getattr(inner, "_gradient_checkpointing_func", None)
    if fn is None:
        return False
    return (getattr(fn, "keywords", {}) or {}).get("use_reentrant") is False


# ---------------------------------------------------------------------------
# 合成数据集
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


def make_wav(path: str, seconds: float, sr: int, seed: int) -> None:
    """一段带谐波的假语音。SNR 要过 dataset.MIN_SNR_DB(12dB) 的体检。"""
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
    noise = rng.standard_normal(n) * 0.002          # 约 50dB SNR
    sf.write(path, (sig + noise).astype(np.float32), sr, subtype="PCM_16")


def build_dataset(name: str, n: int, sr: int, seed: int = 0) -> dict:
    """造一个「特征已提取好」的数据集，返回 id → 特征路径。

    特征全部是合成的，但**形状与 dtype 严格照 FEATURE_SCHEMA**，
    并且刻意做成可学的：codes 由 (说话人, 位置) 确定，
    所以小模型几十步就能把 loss 压到远低于均匀基线 ln(256)=5.545。
    """
    DS.create(name, note="gpt_train_probe 自动生成")
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
        text = TEXTS[spk]                             # 同一说话人 → 同一句
        n_text = 8
        n_codes = 40 + (i % 6) * 12                   # 长度仍然变，好检验 mask/padding
        # 种子用 spk 而不是 i：同一说话人的文本 token 完全一致，
        # 于是整个数据集就是「3 个 (文本, 声纹) → 3 个常量 token 序列」的映射。
        # 刻意做得这么好背，因为这个脚本要验的是**训练器的接线与账目**，
        # 不是小模型的拟合能力 —— 任务难了只会让断言变得不稳定。
        rng = np.random.default_rng(seed * 100 + spk)

        text_tokens = torch.from_numpy(
            rng.integers(2, TINY["number_text_tokens"], size=(n_text,))
        ).to(torch.int64)
        code_val = 10 + spk * 50                      # 每个说话人一个常量 token
        codes = torch.full((n_codes,), code_val, dtype=torch.int64)
        style = torch.zeros(192, dtype=torch.float32)
        style[spk * 40:(spk + 1) * 40] = 1.0        # 每个说话人一个明显不同的声纹
        style += torch.from_numpy(rng.standard_normal(192).astype(np.float32)) * 0.02
        emo_vec = torch.zeros(DIM, dtype=torch.float32)
        emo_vec[spk] = 1.0
        mel_len = int(round(n_codes * FT.MEL_PER_CODE))
        feat = {
            "text_tokens": text_tokens,
            "codes": codes,
            "style": style,
            "emo_vec": emo_vec,
            "mel": torch.zeros(80, mel_len, dtype=torch.float16),
            # mu_* 只有 CFM 训练会用，GPT 路径永远不读它们。
            # is_usable() 只查字段存在与版本，不查形状 —— 所以这里给个
            # 极小的占位就行，否则 32 条 × 2 个 (mel_len,512) 白白写几十 MB。
            "mu_prompt": torch.zeros(4, 512, dtype=torch.float16),
            "mu_target": torch.zeros(4, 512, dtype=torch.float16),
            "n_codes": int(n_codes),
            "n_text_tokens": int(n_text),
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
            "text": text, "has_features": True, "features_at": time.time(),
            "mel_len": int(mel_len), "n_codes": int(n_codes),
            "n_text_tokens": int(n_text), "lang_token": 1,
        }
    FT._apply_meta(name, touched)
    st = DS.refresh_all(name, require_features=True)
    return {"imported": imp, "stats": st, "paths": paths, "dir": ds_dir}


# ===========================================================================
def main() -> int:
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="gpt_train_probe_")
    ds_root = os.path.join(tmp, "datasets")
    run_root = os.path.join(tmp, "runs")
    guard_root = os.path.join(tmp, "guard")
    model_dir = os.path.join(tmp, "checkpoints")
    os.makedirs(ds_root); os.makedirs(run_root)
    os.makedirs(guard_root); os.makedirs(model_dir)

    # 三个模块级路径全部指到临时目录：BaseGuard 的快照会去哈希 3.2 GB 的真底座，
    # 训练记录也会污染用户真实的 training_runs/。
    DS.DATASETS_ROOT = ds_root
    RN.ROOT = run_root
    GL.load_base_gpt = lambda device="cpu", model_dir=None, verify_keys=True: (
        make_tiny_gpt(device), {"missing": 0, "unexpected": 0,
                                "path": "<tiny>", "model_dir": model_dir})

    # 假底座目录：给 BaseGuard 一点东西做快照与校验
    with open(os.path.join(model_dir, "gpt.pth"), "wb") as f:
        f.write(b"\x00" * 4096)
    with open(os.path.join(model_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("gpt_checkpoint: gpt.pth\n")

    SR = 22050
    DS_NAME = "probe_gpt"
    RP_NAME = "probe_replay"
    N_DS = 32            # 必须 ≥27：val_ratio=0.25 拆完还要给训练集留 20 条，
                         # 否则 guard 的「数据量不足」会直接（正确地）把训练拦下来

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
        check("meta 里的长度字段已回写",
              all(u.n_codes > 0 and u.n_text_tokens > 0 for u in items))

        rp = build_dataset(RP_NAME, 6, SR, seed=7)
        check("回放集 6 条全部 ready",
              len([u for u in DS.load_meta(RP_NAME) if u.status == "ready"]) == 6)

        # ==============================================================
        head("[2] 纯逻辑：SamplePool / collate / plan_batches")
        pool = GL.SamplePool(DS_NAME, [u.id for u in items], kind="target")
        check(f"SamplePool 收到 {N_DS} 条，没有 missing",
              len(pool) == N_DS and not pool.missing,
              f"len={len(pool)} missing={pool.missing}")
        ok, bad = pool.usable_ids(max_codes=50, max_text=30)
        check("usable_ids 按长度上限刷掉了超长样本",
              len(bad) > 0 and len(ok) + len(bad) == N_DS,
              f"ok={len(ok)} bad={len(bad)}")
        s = pool.load(pool.ids[0])
        check("load 返回 build_gpt_sample 的字段",
              s is not None and {"text_tokens", "codes", "style", "emo_vec",
                                 "lang_token", "id", "kind"} <= set(s))
        check("重复 load 命中缓存（同一个对象）", pool.load(pool.ids[0]) is s)

        pool.restrict(ok)
        check("restrict 后 ids/meta 一致", len(pool.ids) == len(ok) and
              set(pool.meta) == set(ok), f"{len(pool.ids)} vs {len(ok)}")
        check("restrict 把被剔样本的缓存也清了",
              not (set(pool._cache) - set(ok)))

        b = GL.collate([pool.load(u) for u in pool.ids[:3]])
        check("collate 的 codes 是 (B, Lmax) 且 B=3",
              tuple(b["codes"].shape)[0] == 3 and
              b["codes"].size(1) == max(int(x) for x in b["code_lengths"]))
        check("collate 的 text_lengths 与 code_lengths 与单条一致",
              [int(x) for x in b["code_lengths"]] ==
              [int(pool.meta[u]["n_codes"]) for u in pool.ids[:3]])
        check("style 是 (B,192)、emo_vec 是 (B,%d)" % DIM,
              tuple(b["style"].shape) == (3, 192) and
              tuple(b["emo_vec"].shape) == (3, DIM))

        order = [("target", i) for i in range(N_DS)]
        full = GL.SamplePool(DS_NAME, [u.id for u in items], kind="target")
        bs_sorted = GL.plan_batches(order, {"target": full}, 3, True, seed=1)
        bs_rand = GL.plan_batches(order, {"target": full}, 3, False, seed=1)

        def spread(batches):
            out = []
            for bb in batches:
                L = [full.meta[full.ids[i]]["n_codes"] for _, i in bb]
                out.append(max(L) - min(L))
            return sum(out) / len(out)
        check("按长度排序后，同 batch 的长度差明显更小",
              spread(bs_sorted) < spread(bs_rand),
              f"sorted={spread(bs_sorted):.1f} vs random={spread(bs_rand):.1f}")
        a1 = GL.plan_batches(order, {"target": full}, 3, True, seed=1)
        a2 = GL.plan_batches(order, {"target": full}, 3, True, seed=2)
        check("排序模式下不同 seed 的 batch 顺序不同（不是完全排序）",
              [i for bb in a1 for _, i in bb] != [i for bb in a2 for _, i in bb])
        check("排序不丢样本",
              sorted(i for bb in bs_sorted for _, i in bb) == list(range(N_DS)))

        # ==============================================================
        head("[3] GptTrainOptions.validate 的拦截")
        o = GL.GptTrainOptions()
        check("默认配置没有 error",
              not [x for x in o.validate() if x.level == "error"])
        o2 = GL.GptTrainOptions(max_codes=1817)
        check("max_codes=1817 被拦（mel_pos_embedding 只有 1818）",
              any(x.level == "error" for x in o2.validate()))
        o3 = GL.GptTrainOptions(max_codes=1816, max_text_tokens=600)
        errs = [x.message for x in o3.validate() if x.level == "error"]
        check("3+(600+2)+(1816+2)=2423 > n_positions(2420) 被拦",
              any("2420" in m for m in errs), str(errs))
        o4 = GL.GptTrainOptions(text_loss_weight=0.5)
        check("text 头参与训练时给出 info 提示",
              any(x.level == "info" for x in o4.validate()))
        check("【防线】样本数 <20 时 LoRAConfig 直接报 error（而不是硬跑）",
              any(x.level == "error" and "不足以微调" in x.message
                  for x in GD.LoRAConfig.preset("balanced").validate(n_samples=9)))

        # ==============================================================
        head("[4] preflight（不加载模型）")
        cfg = GD.LoRAConfig.preset("conservative")
        cfg.apply_target_preset("attn", "gpt")
        cfg.rank, cfg.alpha = 8, 16
        cfg.use_rslora = False        # conservative 默认 True，会把增益抬到 5.66 并报 warn
        cfg.lr = 2e-3                 # 小模型 + 少量步数，需要大一点的学习率
        cfg.weight_decay = 0.01       # conservative 的 0.05 在几十步里会把 adapter 拉回零
        cfg.dropout = 0.05
        cfg.warmup_ratio = 0.05
        cfg.batch_size, cfg.grad_accum = 2, 2
        cfg.epochs, cfg.eval_every, cfg.val_patience = 8, 5, 2
        cfg.replay_source, cfg.replay_dataset, cfg.replay_ratio = "dataset", RP_NAME, 0.3
        cfg.keep_checkpoints, cfg.bf16 = 3, False
        cfg.grad_checkpointing = True
        opts = GL.GptTrainOptions(max_codes=120, max_text_tokens=40, log_every=2)

        tr = GL.GptTrainer(DS_NAME, cfg=cfg, options=opts, device="cpu",
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
        check("total_steps 与 steps_per_epoch 对得上账",
              pf["total_steps"] == pf["steps_per_epoch"] * cfg.epochs,
              f"{pf['steps_per_epoch']}×{cfg.epochs}={pf['total_steps']}")
        check("预估 adapter 参数量与 rank 成正比",
              pf["est_adapter_params"] == GL.EST_ADAPTER_PER_RANK["attn"] * cfg.rank)
        check("preflight 之后模型还没建", tr.pm is None)

        # 回归钉：val 必须从 val 池取样。当初 `_gather` 固定读 `self.pools`，
        # 而 evaluate 的下标是在 val 池上算的 —— 于是「验证集」实际是
        # 训练集的前几条。loss 照样下降、曲线照样好看，但早停与
        # 「相对底座的改善」全部失效，而且不会报任何错。
        got = tr._gather([("target", i) for i in range(len(tr.val_ids))],
                         tr.val_pools)
        check("★ evaluate 取的是 val 池的样本（不是训练池的前几条）",
              sorted(s["id"] for s in got) == sorted(tr.val_ids),
              f"{len(got)} 条 · 与训练集交集 "
              f"{len(set(s['id'] for s in got) & set(tr.train_ids))}")
        check("train/val 划分本身不重叠",
              not (set(tr.train_ids) & set(tr.val_ids)),
              f"train={len(tr.train_ids)} val={len(tr.val_ids)}")
        check("val 池与训练池是两个不同的对象",
              tr.val_pools["target"] is not tr.pools["target"])

        # 回放集 == 目标集时必须报空池，不能静默地假装在回放
        cfg2 = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg2.replay_dataset = DS_NAME
        tr2 = GL.GptTrainer(DS_NAME, cfg=cfg2, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_run2")
        pf2 = tr2.preflight()
        check("回放集与目标集相同时回放池被判为空",
              pf2["n_replay_pool"] == 0 and
              any("回放数据集与目标数据集相同" in m for m in pf2["warnings"]),
              str(pf2["warnings"])[:120])

        # ==============================================================
        head("[5] prepare：注入 / 冻结 / dtype / 两个静默失效开关")
        prep = tr.prepare()
        inj = prep["inject"]
        pm = tr.pm
        check("注入面 attn → 4 个 LoRA 层（2 层 × c_attn+c_proj）",
              inj["lora_layers"] == 4, f"{inj['lora_layers']}")
        check("底座全冻结（assert_base_frozen 返回空）",
              GD.assert_base_frozen(pm) == [])
        check("可训练参数 > 0 且远小于底座",
              0 < inj["adapter_params"] < inj["base_params"] / 20,
              f"{inj['adapter_params']} / {inj['base_params']}")
        check("bf16=False 时底座留在 fp32",
              prep["cast"]["dtype"] == "float32")
        check("adapter 参数是 fp32（AdamW 动量的前提）",
              all(p.dtype == torch.float32 for p in pm.parameters() if p.requires_grad))

        inner = FW.unwrap(pm).gpt
        check("底座 attn_pdrop 被置 0", float(inner.config.attn_pdrop) == 0.0)
        check("底座 Dropout 实例的 p 也真的被改了（改 config 不够）",
              all(float(m.p) == 0.0 for m in inner.modules()
                  if isinstance(m, torch.nn.Dropout)))
        check("梯度检查点已开且确实用的 use_reentrant=False",
              bool(getattr(inner, "gradient_checkpointing", False))
              and gc_uses_non_reentrant(inner),
              f"gc={getattr(inner, 'gradient_checkpointing', None)} "
              f"non_reentrant={gc_uses_non_reentrant(inner)}")
        check("预估 vs 实际的交叉校验发现了差异（小模型必然差很多）",
              any("预估" in x["message"] for x in tr.report.notes),
              str([x["message"][:44] for x in tr.report.notes]))
        check("模型处于 train 模式", bool(FW.unwrap(pm).training))

        vr = tr.guard.verify(hashes=False)
        check("底座快照已建立且校验通过", vr.ok and vr.checked == 2,
              f"checked={vr.checked} changed={vr.changed}")
        check("run.json 已写入且状态是 running",
              RN.read_run("probe_run").get("status") == RN.STATUS_RUNNING)
        check("run.json 记的是**解析后**的回放集名字",
              (RN.read_run("probe_run").get("config") or {}).get("replay_dataset") == RP_NAME)

        # ==============================================================
        head("[6] step-0 基线必须严格等于纯底座")
        v_lora0 = tr.evaluate()
        n_off = GD.disable_adapters(pm, True)
        v_base = tr.evaluate()
        GD.disable_adapters(pm, False)
        check("disable_adapters 关掉了 4 层", n_off == 4, f"{n_off}")
        check("LoRA 零初始化 → val 与关掉 adapter 完全一致",
              v_lora0 is not None and v_base is not None
              and abs(v_lora0 - v_base) < 1e-9,
              f"lora0={v_lora0} base={v_base}")
        uniform = float(np.log(N_MEL))
        check(f"底座初始 val 接近均匀基线 ln({N_MEL})={uniform:.3f}",
              abs(v_lora0 - uniform) < 1.2, f"{v_lora0:.4f} vs {uniform:.4f}")
        check("评估结束后模型回到 train 模式（否则 lora_dropout 静默失效）",
              bool(pm.training))

        # ==============================================================
        head("[7] 训练：loss 下降 / 底座一字节不变 / adapter 真的动了")
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
        check("跑到了预期的步数（或被早停提前拦下）",
              rep.steps == pf["total_steps"] or rep.stopped_early,
              f"{rep.steps}/{pf['total_steps']}"
              + (f" · 早停：{rep.stop_reason}" if rep.stopped_early else ""))
        check("进度回调被调用过", len(prog) > 3, f"{len(prog)} 次")

        check("first_val 是基线（step 0）", rep.first_val is not None
              and abs(rep.first_val - v_lora0) < 1e-9,
              f"{rep.first_val} vs {v_lora0}")
        check("best_val 明显低于基线（模型真的学到了东西）",
              rep.best_val is not None and rep.best_val < rep.first_val * 0.85,
              f"{rep.first_val:.4f} → {rep.best_val:.4f}"
              f"（降 {(rep.improved or 0)*100:.1f}%）")
        check("best_val 远低于均匀基线", rep.best_val < uniform * 0.7,
              f"{rep.best_val:.4f} vs {uniform:.4f}")
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
              sum(1 for h in rep.history if h.get("val") is not None) >= 2,
              f"{len(rep.history)} 点")
        vals = [h["val"] for h in rep.history if h.get("val") is not None]
        check("history 的第一个 val 就是基线（runs.info_of 靠它算 first_val）",
              vals and abs(vals[0] - v_lora0) < 1e-4,
              f"{vals[0] if vals else None} vs {v_lora0:.6f}"
              "（_record 存的是 round(·,5)，所以只能比到 1e-4）")
        fn = tr._lr_lambda()
        expect_lr = cfg.lr * fn(rep.steps)
        check("学习率严格落在 warmup+余弦退火曲线上",
              abs(rep.history[-1]["lr"] - expect_lr) < 1e-9,
              f"actual={rep.history[-1]['lr']:.4e} expect={expect_lr:.4e}")
        check("退火确实降下来了（不是停在峰值）",
              expect_lr <= cfg.lr * (opts.min_lr_ratio + 0.35) or rep.stopped_early,
              f"{expect_lr:.3e} vs 峰值 {cfg.lr:.3e}"
              + (" · 早停了所以没跑到地板" if rep.stopped_early else ""))

        # ==============================================================
        head("[8] 产物：adapter / checkpoint / run.json / 漂移")
        ad = RN.adapter_dir("probe_run")
        names = sorted(os.listdir(ad)) if os.path.isdir(ad) else []
        check("adapter/ 里有 PEFT 的两个文件",
              "adapter_config.json" in names and
              any(f.startswith("adapter_model") for f in names), str(names))
        check("adapter/ 里有 provenance.json（事后能查出这份权重从哪来）",
              "provenance.json" in names)
        ck = tr.vault_obj.list() if tr.vault_obj else RN.vault("probe_run").list()
        # release() 会把 vault_obj 置 None，所以从磁盘再读一次
        ck = RN.vault("probe_run").list()
        check("保险库里存了 checkpoint", len(ck) >= 1, f"{len(ck)} 份")
        check("保险库不超过 keep_checkpoints", len(ck) <= cfg.keep_checkpoints,
              f"{len(ck)} ≤ {cfg.keep_checkpoints}")
        best = RN.best_checkpoint("probe_run")
        check("保险库认定的 best 就是 val 最低的那份",
              best is not None and abs(best.metric - rep.best_val) < 1e-6,
              f"{best.metric if best else None} vs {rep.best_val}")
        check("每份 checkpoint 都能被 PEFT 认出来",
              all(os.path.isfile(os.path.join(c.path, "adapter_config.json"))
                  for c in ck))
        check("存了优化器状态（可续训）",
              all(os.path.isfile(os.path.join(c.path, "train_state.pt"))
                  for c in ck))

        rj = RN.read_run("probe_run")
        check("run.json 状态 = done", rj.get("status") == RN.STATUS_DONE)
        check("run.json 的 best_val / steps / epochs 与报告一致",
              rj.get("steps") == rep.steps and rj.get("epochs") == rep.epochs
              and abs((rj.get("best_val") or 0) - rep.best_val) < 1e-9)
        check("run.json 存了 history 与 notes",
              len(rj.get("history") or []) == len(rep.history)
              and isinstance(rj.get("notes"), list))
        ri = RN.info_of("probe_run")
        check("RunInfo 能还原出 improved / has_adapter / arch",
              ri.has_adapter and ri.arch == "gpt" and ri.improved is not None
              and abs(ri.improved - rep.improved) < 1e-6,
              f"improved={ri.improved:.3f} adapter={ri.has_adapter}")
        check("list_runs 能列出这次训练",
              any(r.name == "probe_run" for r in RN.list_runs("gpt")))
        check("日志文件写了内容", len(RN.read_log("probe_run")) > 200)

        check("漂移体检算出了非零的 global_rel",
              rep.drift.get("global_rel", 0.0) > 0.0
              and rep.drift.get("n_layers") == 4,
              f"{rep.drift.get('global_rel'):.5f} · {rep.drift.get('n_layers')} 层")
        check("漂移体检给了等级与建议",
              bool(rep.drift.get("level")) and bool(rep.drift.get("advice")))
        check("训练后底座再次校验通过（before + after 两次）",
              rep.base_verify.get("ok") is True
              and rep.base_verify.get("when") == "after",
              str(rep.base_verify))
        check("报告能渲染成 Markdown", len(rep.markdown()) > 400,
              f"{len(rep.markdown())} 字符")

        # release 之后训练器不该再握着模型
        check("release 后 pm / opt / sched / vault 全部断开",
              tr.pm is None and tr.opt is None and tr.sched is None
              and tr.vault_obj is None)
        check("release 后样本池缓存已清空",
              not tr.pools and tr.val_pool is None)

        # ==============================================================
        head("[9] 防线：手动停止 / max_steps / 重复评估保护")
        cfg3 = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg3.epochs, cfg3.eval_every, cfg3.val_patience = 6, 1, 0
        cfg3.keep_checkpoints = 2
        tr3 = GL.GptTrainer(DS_NAME, cfg=cfg3, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_stop")
        tr3.preflight()
        n3 = [0]

        def stopper():
            n3[0] += 1
            return n3[0] > 4          # 跑 4 个 batch 就叫停
        rep3 = tr3.run(should_stop=stopper)
        check("should_stop 让训练提前结束且状态是 stopped",
              RN.read_run("probe_stop").get("status") == RN.STATUS_STOPPED
              and rep3.stop_reason == "用户手动停止",
              f"steps={rep3.steps} reason={rep3.stop_reason}")
        check("手动停止后仍然做了收尾（底座复校 + 漂移）",
              rep3.base_verify.get("ok") is True and "global_rel" in rep3.drift)
        check("手动停止仍然算成功产出（ok=True，但带着停止原因）",
              rep3.ok and not rep3.error and rep3.stop_reason == "用户手动停止",
              f"ok={rep3.ok} error={rep3.error!r} reason={rep3.stop_reason}")
        check("停止后模型已释放", tr3.pm is None)

        cfg4 = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg4.epochs, cfg4.max_steps, cfg4.eval_every = 8, 3, 1000
        tr4 = GL.GptTrainer(DS_NAME, cfg=cfg4, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_max")
        pf4 = tr4.preflight()
        rep4 = tr4.run()
        check("max_steps 是硬刹车：3 步就停，不跑满 8 轮",
              rep4.steps == 3, f"{rep4.steps} 步")
        check("total_steps 被 max_steps 压住了（调度器周期才正确）",
              pf4["total_steps"] == 3, f"{pf4['total_steps']}")
        check("max_steps 触发时记录了停止原因",
              "max_steps" in (rep4.stop_reason or ""), rep4.stop_reason)
        check("eval_every=1000 时轮末兜底评估仍然跑到了",
              sum(1 for h in rep4.history if h.get("val") is not None) >= 2,
              f"{sum(1 for h in rep4.history if h.get('val') is not None)} 次 val")

        # 重复评估保护：同一个 step 上评两次会让 EarlyStopper 白吞一次耐心
        cfg5 = GD.LoRAConfig.from_dict(cfg.to_dict())
        tr5 = GL.GptTrainer(DS_NAME, cfg=cfg5, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_dup")
        tr5.preflight(); tr5.prepare()
        tr5._step = 7
        tr5._handle_eval(1.0, epoch=0)
        check("_handle_eval 记下了评估发生在哪一步", tr5._last_eval_step == 7)
        check("轮末兜底的条件成立（step 未变）—— run() 会因此跳过重复评估",
              tr5._last_eval_step == tr5._step)
        tr5._step = 9
        check("step 前进后条件不再成立，兜底评估会跑",
              tr5._last_eval_step != tr5._step)
        bad0 = tr5.stopper.bad_count
        tr5._handle_eval(1.0, epoch=0)          # 同样的 val，同一 step，第二次
        check("同样的 val 再评一次会被早停器判为「没改善」（所以必须去重）",
              tr5.stopper.bad_count == bad0 + 1,
              f"bad_count {bad0} → {tr5.stopper.bad_count}")
        check("未改善时不会存 checkpoint（保险库不会被变差的存档洗一遍）",
              len(RN.vault("probe_dup").list()) == 1,
              f"{len(RN.vault('probe_dup').list())} 份")
        tr5.release()

        # 早停的确定性验证：直接喂三个递增的 val，不依赖真实训练曲线
        cfg7 = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg7.val_patience = 2
        tr7 = GL.GptTrainer(DS_NAME, cfg=cfg7, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_es")
        tr7.preflight(); tr7.prepare()
        tr7.stopper = GD.EarlyStopper(patience=2, min_delta=1e-4, mode="min")
        s1 = tr7._handle_eval(1.0, epoch=0); tr7._step = 1
        s2 = tr7._handle_eval(1.1, epoch=0); tr7._step = 2
        s3 = tr7._handle_eval(1.2, epoch=0)
        check("第一次评估不停（它是基线）", not s1)
        check("变差一次还不停（patience=2）", not s2)
        check("连续两次变差 → 早停触发", s3 and tr7.report.stopped_early)
        check("早停原因里写清了最好的那一次在哪",
              "1.00000" in tr7.report.stop_reason and "epoch 0" in tr7.report.stop_reason,
              tr7.report.stop_reason)
        check("best_val 留的是最好那一次而不是最后一次",
              abs(tr7.report.best_val - 1.0) < 1e-9, f"{tr7.report.best_val}")
        check("只有改善那一次存了盘", len(RN.vault("probe_es").list()) == 1,
              f"{len(RN.vault('probe_es').list())} 份")
        tr7.release()

        # ==============================================================
        head("[10] 续训：从 checkpoint 恢复**权重** + 优化器 + 早停状态")
        best_path = best.path if best else ""
        cfg6 = GD.LoRAConfig.from_dict(cfg.to_dict())
        tr6 = GL.GptTrainer(DS_NAME, cfg=cfg6, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_resume", resume_from=best_path)
        tr6.preflight()
        prep6 = tr6.prepare()
        check("prepare 阶段就把 adapter 权重灌回来了",
              prep6.get("resumed_tensors", 0) > 0,
              f"{prep6.get('resumed_tensors')} 个张量 ← {os.path.basename(best_path)}")
        check("恢复了 step 计数（到存盘那一步，不是到训练结束）",
              tr6._step == best.step and tr6._step > 0,
              f"step={tr6._step} best.step={best.step} 总步数={rep.steps}")
        check("恢复了早停器的 best", tr6.stopper.best is not None
              and abs(tr6.stopper.best - rep.best_val) < 1e-6,
              f"{tr6.stopper.best} vs {rep.best_val}")
        check("恢复了优化器动量（state 非空）",
              len(tr6.opt.state) > 0, f"{len(tr6.opt.state)} 个参数有状态")
        check("恢复了学习率调度器的位置",
              tr6._cur_lr() < cfg6.lr, f"lr={tr6._cur_lr():.2e}")
        v6 = tr6.evaluate()
        check("★ 续训起点的 val 与那份 checkpoint 的 metric 完全一致",
              v6 is not None and abs(v6 - rep.best_val) < 1e-6,
              f"{v6:.6f} vs {rep.best_val:.6f}")
        check("run.json 里记下了续训来源",
              os.path.basename(RN.read_run("probe_resume").get("resume_from", ""))
              == os.path.basename(best_path))
        tr6.release()

        # 回归钉：当初的 bug 就是「只恢复优化器状态、忘了恢复权重」。
        # 那种情况下 val 会静默地退回基线（B 矩阵全零 = 等价于纯底座），
        # 而优化器却带着上一轮的动量 —— 不报错，只是续训之后 loss 弹回去。
        cfg6b = GD.LoRAConfig.from_dict(cfg.to_dict())
        tr6b = GL.GptTrainer(DS_NAME, cfg=cfg6b, options=opts, device="cpu",
                             model_dir=model_dir, train_root=guard_root,
                             run_name="probe_resume_b")       # 故意不传 resume_from
        tr6b.preflight(); tr6b.prepare()
        tr6b._load_train_state(best_path)          # 只恢复状态，不恢复权重
        v6b = tr6b.evaluate()
        check("【回归】不恢复权重时 val 退回基线（证明权重这一步不能省）",
              v6b is not None and abs(v6b - v_lora0) < 1e-6
              and abs(v6b - rep.best_val) > 1e-3,
              f"{v6b:.5f} vs 基线 {v_lora0:.5f} vs best {rep.best_val:.5f}")
        check("【回归】但优化器状态仍然恢复成功了 —— 所以它不会报错，只会默默跑偏",
              len(tr6b.opt.state) > 0 and tr6b._step == best.step)
        tr6b.release()

        # _resolve_ckpt_dir 应该能用档位名找到其他 run 下的 checkpoint
        tr8 = GL.GptTrainer(DS_NAME, cfg=cfg6, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_resolve")
        try:
            got = tr8._resolve_ckpt_dir(best.name)
            check("_resolve_ckpt_dir 能跨 run 用档位名找到 checkpoint",
                  os.path.abspath(got) == os.path.abspath(best_path),
                  f"{best.name} → {got}")
        except Exception as e:
            check("_resolve_ckpt_dir 能跨 run 用档位名找到 checkpoint", False,
                  f"{type(e).__name__}: {e}")
        try:
            tr8._resolve_ckpt_dir("ckpt-e999-s999999")
            check("找不到的档位名会报 FileNotFoundError", False)
        except FileNotFoundError:
            check("找不到的档位名会报 FileNotFoundError", True)
        except Exception as e:
            check("找不到的档位名会报 FileNotFoundError", False,
                  f"报了 {type(e).__name__}")

        # ==============================================================
        head("[11] replay.py：底座蒸馏回放集")
        from webui_app.training import replay as RP

        cats = {c for c, _ in RP.GENERIC_TEXTS}
        check("内置语料覆盖了声明的全部类别", cats == set(RP.TEXT_CATEGORIES),
              f"{sorted(cats)}")
        check("语料量足够当回放（≥ 40 条）", len(RP.GENERIC_TEXTS) >= 40,
              f"{len(RP.GENERIC_TEXTS)} 条")
        check("没有重复句子",
              len({t for _c, t in RP.GENERIC_TEXTS}) == len(RP.GENERIC_TEXTS))
        check("generic_texts 的 limit 生效", len(RP.generic_texts(5)) == 5)
        check("generic_texts 的类别筛选生效",
              RP.generic_texts(0, ["number"]) and
              all(x["category"] == "number" for x in RP.generic_texts(0, ["number"])))
        check("coverage_markdown 能渲染", "多音字" in RP.coverage_markdown())

        o = RP.DistillOptions()
        check("没给参考音频时报 error",
              any(x.level == "error" and "参考音频" in x.message for x in o.validate()))
        o2 = RP.DistillOptions(prompt_audio=os.path.join(tmp, "nope.wav"))
        check("参考音频不存在时报 error",
              any(x.level == "error" and "不存在" in x.message for x in o2.validate()))
        prompt_wav = os.path.join(tmp, "prompt_other.wav")
        make_wav(prompt_wav, 2.0, SR, seed=99)
        o3 = RP.DistillOptions(prompt_audio=prompt_wav, temperature=0.3)
        check("温度过低给 warn（回放样本需要多样性）",
              any(x.level == "warn" and "temperature" in x.message for x in o3.validate()))
        o3.temperature = 0.85
        check("正常配置没有 error",
              not [x for x in o3.validate() if x.level == "error"])

        # 排除目标角色音频：直接换掉候选源，把「候选全是目标集音频」这个
        # 最坑的场景造出来 —— 此时必须返回空并说清楚原因，而不是随手挑一个。
        ds_dir = DS.dir_of(DS_NAME)
        tgt_paths = [u.audio_abs(ds_dir) for u in DS.load_meta(DS_NAME)][:3]
        saved_cp = RP.candidate_prompts
        RP.candidate_prompts = lambda: [{"path": p, "source": "t",
                                        "name": os.path.basename(p)} for p in tgt_paths]
        try:
            p_ex, why_ex = RP.auto_pick_prompt(exclude_dataset=DS_NAME)
        finally:
            RP.candidate_prompts = saved_cp
        check("候选全是目标角色的音频时返回空并说明原因",
              p_ex == "" and "目标数据集" in why_ex, why_ex[:60])
        RP.candidate_prompts = lambda: [{"path": prompt_wav, "source": "t",
                                        "name": "prompt_other.wav"}]
        try:
            p_ok, _why = RP.auto_pick_prompt(exclude_dataset=DS_NAME)
        finally:
            RP.candidate_prompts = saved_cp
        check("候选里有非目标音频时能选中它", p_ok == prompt_wav, p_ok)

        # ---- 红线 1：蒸馏期间 adapter 必须被静音，事后必须恢复 ----
        from peft import LoraConfig, get_peft_model
        tiny_for_lora = make_tiny_gpt("cpu")
        GD.freeze_base(tiny_for_lora)
        peft_gpt = get_peft_model(tiny_for_lora, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=GD.build_target_regex(("attn/c_attn", "attn/c_proj")),
            bias="none", task_type=None))
        # 把 lora_B 填成非零，否则强度置 0 与不置 0 的输出一样，测不出区别
        with torch.no_grad():
            for _n, p in peft_gpt.named_parameters():
                if p.requires_grad:
                    p.normal_(0.0, 0.05)
        GD.set_adapter_scale(peft_gpt, 1.0)

        class _FakeDev:
            device_str = "cpu"

        class _FakeCfg:
            device = _FakeDev()

        class _FakeTTS(torch.nn.Module):
            def __init__(self, gpt):
                super().__init__()
                self.gpt = gpt
                self.s2mel = None

        class _FakeEngine:
            """假的 TTSEngine：infer 直接写一段 wav，并**取证**当时的 adapter 强度。"""
            def __init__(self, tts):
                self._tts = tts
                self.cfg = _FakeCfg()
                self.calls: list = []
                self.scale_at_call: list = []

            @property
            def tts(self):
                return self._tts

            def infer(self, **kw):
                self.calls.append(kw)
                # get_adapter_scale 返回的是汇总字典（_min/_max/_mean/_layers），
                # 不是逐层映射 —— 直接取 values() 会把 _layers 当成强度值。
                sc = GD.get_adapter_scale(self._tts.gpt)
                self.scale_at_call.append(float(sc.get("_max", 1.0)))
                make_wav(kw["output_path"], 1.5, SR, seed=len(self.calls))
                return kw["output_path"]

        feng = _FakeEngine(_FakeTTS(peft_gpt))
        muted = RP._adapter_scale_ctx(feng.tts, True)
        sc_off = GD.get_adapter_scale(peft_gpt)
        RP._adapter_scale_ctx(feng.tts, False)
        sc_on = GD.get_adapter_scale(peft_gpt)
        check("_adapter_scale_ctx 找得到 tts.gpt 上的 adapter", muted == ["gpt"],
              str(muted))
        check("★ 静音后全部层的 scaling 都是 0（等价于纯底座）",
              sc_off.get("_layers") == 4 and sc_off.get("_max", 1.0) < 1e-12,
              f"min={sc_off.get('_min')} max={sc_off.get('_max')} "
              f"layers={sc_off.get('_layers')}")
        check("★ 恢复后 scaling 回到 1.0（不会把用户的推理也静音）",
              abs(sc_on.get("_min", 0.0) - 1.0) < 1e-9
              and abs(sc_on.get("_max", 0.0) - 1.0) < 1e-9,
              f"min={sc_on.get('_min')} max={sc_on.get('_max')}")

        # ---- 端到端生成 ----
        GD.set_adapter_scale(peft_gpt, 1.0)          # 故意先置 1，看它会不会被静音
        texts_used = RP.generic_texts(6)
        opts_d = RP.DistillOptions(n_texts=6, prompt_audio=prompt_wav,
                                   extract_features=False, keep_wav=False,
                                   overwrite=True)
        res_d = RP.build_distill_dataset(feng, opts_d, name="probe_distill",
                                         warn_target_prompt=DS_NAME)
        check("生成成功", res_d["ok"], str(res_d["errors"]))
        check("合成了 6 条", res_d["synthesized"] == 6, f"{res_d['synthesized']}")
        check("★ 每次 infer 时 adapter 强度都是 0（红线 1）",
              len(feng.scale_at_call) == 6 and max(feng.scale_at_call) == 0.0,
              f"{feng.scale_at_call}")
        check("★ 生成结束后强度已恢复 1.0",
              abs(GD.get_adapter_scale(peft_gpt).get("_min", 0.0) - 1.0) < 1e-9)
        check("生成时静音了 gpt 的 adapter",
              res_d["adapter_muted"] == ["gpt"], str(res_d["adapter_muted"]))
        check("静音这件事写进了警告（用户得知道发生了什么）",
              any("强度临时置 0" in w for w in res_d["warnings"]),
              str(res_d["warnings"])[:90])

        di = DS.load_meta("probe_distill")
        check("数据集里正好 6 条", len(di) == 6, f"{len(di)}")
        check("★ 每条都有文本", all((u.text or "").strip() for u in di))
        check("★ 文本与音频逐条对应（没有整体错位）",
              [u.text for u in di] == [t["text"] for t in texts_used],
              f"{[u.text[:12] for u in di][:2]} vs "
              f"{[t['text'][:12] for t in texts_used][:2]}")
        check("import_audio 把文件重命名成了 <uid>.wav（所以不能按文件名匹）",
              all(os.path.splitext(os.path.basename(u.audio))[0] == u.id for u in di))
        check("dataset.json 里记下了蒸馏参数与参考音频",
              (DS.info("probe_distill").get("distill_options") or {}).get("n_texts") == 6
              and DS.info("probe_distill").get("prompt_audio", "").endswith(".wav"))
        check("distill_markdown 能渲染", "回放集已就绪" in RP.distill_markdown(res_d))

        # 参考音频误用目标角色 → 必须警告
        res_w = RP.build_distill_dataset(
            feng, RP.DistillOptions(n_texts=2, prompt_audio=tgt_paths[0],
                                    extract_features=False, keep_wav=False,
                                    overwrite=True),
            name="probe_distill2", warn_target_prompt=DS_NAME)
        check("★ 用目标角色的音频当回放提示时会警告",
              any("来自目标数据集" in w for w in res_w["warnings"]),
              str(res_w["warnings"])[:90])

        # 不覆盖 → 追加，并提醒用户
        n_before = len(DS.load_meta("probe_distill"))
        res_a = RP.build_distill_dataset(
            feng, RP.DistillOptions(n_texts=2, prompt_audio=prompt_wav,
                                    extract_features=False, keep_wav=False),
            name="probe_distill")
        n_after = len(DS.load_meta("probe_distill"))
        check("不覆盖时以追加方式写入", n_after == n_before + 2,
              f"{n_before} → {n_after}")
        check("追加时提醒了用户", any("追加" in w for w in res_a["warnings"]),
              str(res_a["warnings"])[:70])
        res_o = RP.build_distill_dataset(
            feng, RP.DistillOptions(n_texts=2, prompt_audio=prompt_wav,
                                    extract_features=False, keep_wav=False,
                                    overwrite=True),
            name="probe_distill")
        check("覆盖模式会重建而不是继续追加",
              len(DS.load_meta("probe_distill")) == 2 and res_o["ok"],
              f"{len(DS.load_meta('probe_distill'))} 条")

        # 没有引擎 / 引擎未加载 时不该崩
        check("engine=None 时给出可读的错误而不是抛异常",
              RP.build_distill_dataset(None, opts_d, name="x")["errors"] != []
              and not RP.build_distill_dataset(None, opts_d, name="x")["ok"])

        # ==============================================================
        head("[12] 清理")
    finally:
        GL.load_base_gpt = None
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除", not os.path.isdir(tmp))

    print("\n" + "=" * 70)
    print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
    if FAILS:
        print("  失败项：")
        for f in FAILS:
            print(f"    - {f}")
    print("=" * 70 + "\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
