"""DPO 偏好对齐（dpo.py）的端到端验证。

与 gpt_train_probe 同一套路：结构同构、尺寸极小的 UnifiedVoice 在 CPU 上
跑完 preflight → prepare → run → finalize 的 DPO 闭环，另加：

  · **ln 2 不变量**：LoRA 零初始化 ⇒ policy 与 ref（adapters 关掉）
    逐位相同 ⇒ logits 全 0 ⇒ step-0 的 DPO loss 必须恰好等于 ln2、
    acc 恰好为 0。这把「参考策略不用另开一份模型」的等价性钉死了。
  · **偏好对构造器**（build_pairs）：注入假的 infer / scorer / extractor，
    只测编排 —— 候选生成、margin 过滤、失败样本处理、入库与追加语义。
    真实合成路径在阶段 2 验收（b9）里端到端跑。

跑法：  .venv\\Scripts\\python.exe tools\\dpo_probe.py
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np
import torch
from typing import Dict

from webui_app.training import dataset as DS                # noqa: E402
from webui_app.training import dpo as DP                    # noqa: E402
from webui_app.training import features as FT               # noqa: E402
from webui_app.training import forward as FW                # noqa: E402
from webui_app.training import gpt_lora as GL               # noqa: E402
from webui_app.training import guard as GD                  # noqa: E402
from webui_app.training import runs as RN                   # noqa: E402

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
# 与 gpt_train_probe 相同的小模型 / 合成数据（拷贝过来，别跨探针 import）
# ---------------------------------------------------------------------------

TINY = dict(
    layers=2, model_dim=64, heads=4,
    max_text_tokens=64, max_mel_tokens=256, max_conditioning_inputs=1,
    number_text_tokens=512, number_mel_codes=256,
    start_text_token=0, stop_text_token=1,
    start_mel_token=254, stop_mel_token=255,
    checkpointing=False, types=1,
    emo_condition_module=dict(output_size=32, linear_units=32, attention_heads=4,
                              num_blocks=1, input_layer="linear", perceiver_mult=1),
)
DIM = TINY["model_dim"]

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


def make_tiny_gpt(device: str = "cpu"):
    from indextts.gpt.model_v2 import UnifiedVoice
    torch.manual_seed(0)
    m = UnifiedVoice(**TINY, use_accel=False, spk_cond_mode="campplus")
    return m.to(device)


def base_fingerprint(pm) -> str:
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


def make_wav(path: str, seconds: float, sr: int, seed: int) -> None:
    import soundfile as sf
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 110.0 + seed * 7
    sig = (np.sin(2 * np.pi * f0 * t)
           + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
           + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    env = 0.5 + 0.5 * np.sin(2 * torch.pi * 1.7 * t + seed) \
        if False else 0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t + seed)
    sig = sig * env
    sig = sig / (np.abs(sig).max() + 1e-9) * 0.6
    sf.write(path, (sig + rng.standard_normal(n) * 0.002).astype(np.float32),
             sr, subtype="PCM_16")


def make_feat(spk: int, n_codes: int, code_val: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_text = 8
    text_tokens = torch.from_numpy(
        rng.integers(2, TINY["number_text_tokens"], size=(n_text,))
    ).to(torch.int64)
    codes = torch.full((n_codes,), code_val, dtype=torch.int64)
    style = torch.zeros(192, dtype=torch.float32)
    style[spk * 40:(spk + 1) * 40] = 1.0
    style += torch.from_numpy(rng.standard_normal(192).astype(np.float32)) * 0.02
    emo_vec = torch.zeros(DIM, dtype=torch.float32)
    emo_vec[spk] = 1.0
    mel_len = int(round(n_codes * FT.MEL_PER_CODE))
    return {
        "text_tokens": text_tokens, "codes": codes, "style": style,
        "emo_vec": emo_vec,
        "mel": torch.zeros(80, mel_len, dtype=torch.float16),
        "mu_prompt": torch.zeros(4, 512, dtype=torch.float16),
        "mu_target": torch.zeros(4, 512, dtype=torch.float16),
        "n_codes": int(n_codes), "n_text_tokens": int(n_text),
        "mel_len": int(mel_len), "lang_token": 1,
        "feature_version": FT.FEATURE_VERSION,
        "extract_seconds": 0.1, "extracted_at": time.time(),
        "source": "synthetic", "warnings": [],
    }


def build_pair_dataset(name: str, n_pairs: int, sr: int, seed: int = 0) -> dict:
    """造一个「偏好对数据集」。

    每对：chosen = 该说话人的真实 token 序列（可学的目标），
    rejected = **另一个**说话人的 token 序列（同条件下更差的回答）。
    两半的 style/emo 完全一致 —— DPO 的语义就是「同一个 x，比两个 y」。
    """
    DS.create(name, note="dpo_probe 自动生成")
    ds_dir = DS.dir_of(name)
    os.makedirs(os.path.join(ds_dir, DS.FEATURE_SUBDIR), exist_ok=True)
    tmpw = os.path.join(ds_dir, "_src")
    os.makedirs(tmpw, exist_ok=True)

    wavs = []
    for i in range(n_pairs):
        p = os.path.join(tmpw, f"s{i:03d}.wav")
        make_wav(p, 1.6 + (i % 4) * 0.35, sr, seed + i)
        wavs.append(p)
    imp = DS.import_audio(name, wavs, copy=True, lang="ZH")
    shutil.rmtree(tmpw, ignore_errors=True)
    items = DS.load_meta(name)

    touched = {}
    rows = []
    for i, u in enumerate(items):
        spk = i % 3
        bad_spk = (spk + 1) % 3                # rejected 用别的说话人的 token
        La = 40 + (i % 6) * 12
        Lb = 44 + (i % 5) * 10                 # 两半长度刻意不同（mask 检验）
        c_id, r_id = f"{u.id}c", f"{u.id}r"
        feat_c = make_feat(spk, La, 10 + spk * 50, seed * 100 + spk)
        feat_r = make_feat(spk, Lb, 10 + bad_spk * 50, seed * 100 + spk)
        torch.save(feat_c, FT.FeatureExtractor.feature_path(ds_dir, c_id))
        torch.save(feat_r, FT.FeatureExtractor.feature_path(ds_dir, r_id))
        # 假音频条目：占住 id（音频文件不存在没关系，训练只读特征）
        touched[c_id] = {"text": TEXTS[spk], "has_features": True,
                         "features_at": time.time(), "n_codes": La,
                         "n_text_tokens": 8, "mel_len": feat_c["mel_len"],
                         "lang_token": 1, "status": "ready"}
        touched[r_id] = {"text": TEXTS[spk], "has_features": True,
                         "features_at": time.time(), "n_codes": Lb,
                         "n_text_tokens": 8, "mel_len": feat_r["mel_len"],
                         "lang_token": 1, "status": "ready"}
        rows.append(DP.PairRow(
            id=f"{u.id}_p{i:03d}", text=TEXTS[spk], chosen=c_id, rejected=r_id,
            prompt_id=u.id, margin=0.3, reward_chosen=0.9,
            reward_rejected=0.6, source="synth", created_at=time.time()))
    # _apply_meta 只更新**已有**行：先把 c/r 两条样本作为独立行追加进 meta
    items = DS.load_meta(name)
    for uid, tv in touched.items():
        items.append(DS.Utterance(
            id=uid, audio="", text=tv["text"], lang="ZH",
            status="ready", has_features=True,
            features_at=time.time(), n_codes=tv["n_codes"],
            n_text_tokens=tv["n_text_tokens"], mel_len=tv["mel_len"],
            lang_token=1))
    DS.save_meta(name, items)
    DP.save_pairs(name, rows)
    DP.split_pairs(name, val_ratio=0.25, seed=7)
    DS.refresh_all(name, require_features=True)
    return {"imported": imp, "rows": rows, "dir": ds_dir}


# ===========================================================================
def main() -> int:
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="dpo_probe_")
    ds_root = os.path.join(tmp, "datasets")
    run_root = os.path.join(tmp, "runs")
    guard_root = os.path.join(tmp, "guard")
    model_dir = os.path.join(tmp, "checkpoints")
    for d in (ds_root, run_root, guard_root, model_dir):
        os.makedirs(d)

    DS.DATASETS_ROOT = ds_root
    RN.ROOT = run_root
    GL.load_base_gpt = lambda device="cpu", model_dir=None, verify_keys=True: (
        make_tiny_gpt(device), {"missing": 0, "unexpected": 0,
                                "path": "<tiny>", "model_dir": model_dir})
    with open(os.path.join(model_dir, "gpt.pth"), "wb") as f:
        f.write(b"\x00" * 4096)
    with open(os.path.join(model_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("gpt_checkpoint: gpt.pth\n")

    DS_NAME = "probe_dpo"
    N_PAIRS = 32
    SR = 22050

    try:
        # ==============================================================
        head("[1] 偏好对的存储与划分（纯逻辑）")
        r1 = DP.PairRow(id="a", text="你好", chosen="c1", rejected="r1",
                        margin=0.2)
        r2 = DP.PairRow(id="b", text="世界", chosen="c2", rejected="r2",
                        margin=0.1)
        DP.save_pairs(DS_NAME, [r1])
        check("save/load 往返一致", [x.id for x in DP.load_pairs(DS_NAME)] == ["a"])
        DP.append_pairs(DS_NAME, [r1, r2])
        check("append 去重（同 id 不会写两遍）",
              [x.id for x in DP.load_pairs(DS_NAME)] == ["a", "b"])
        DP.append_pairs(DS_NAME, [DP.PairRow(id="c", text="!", chosen="c3",
                                             rejected="r3")])
        got = DP.load_pairs(DS_NAME)
        check("append 追加新行",
              [x.id for x in got] == ["a", "b", "c"], str([x.id for x in got]))
        check("PairRow 字段往返不丢",
              all(getattr(got[0], k) == getattr(r1, k)
                  for k in ("text", "chosen", "rejected", "margin")))

        sp = DP.split_pairs(DS_NAME, val_ratio=0.34, seed=3)
        all_ids = {x.id for x in got}
        check("划分覆盖全部对且 train/val 不重叠",
              set(sp["train"]) | set(sp["val"]) == all_ids
              and not (set(sp["train"]) & set(sp["val"])))
        sp2 = DP.load_pair_split(DS_NAME)
        check("划分落盘可读", sp2.get("train") == sp["train"])
        # 同 seed 重跑划分结果一致（续训时不能换了 val 集）
        sp3 = DP.split_pairs(DS_NAME, val_ratio=0.34, seed=3)
        check("同 seed 划分确定", sp3["train"] == sp["train"])

        # ==============================================================
        head("[2] 配置校验")
        errs = lambda vv: [x.message for x in vv.validate() if x.level == "error"]
        warns = lambda vv: [x.message for x in vv.validate() if x.level == "warn"]
        infos = lambda vv: [x.message for x in vv.validate() if x.level == "info"]
        check("默认 DpoTrainOptions 无 error",
              not errs(DP.DpoTrainOptions()))
        check("beta 越界被拦", any("beta" in m for m in
                                   errs(DP.DpoTrainOptions(beta=5.0))))
        check("beta 过小给 warn（梯度奖励无限拉大差距）",
              any("beta" in m for m in warns(DP.DpoTrainOptions(beta=0.02))))
        check("sft_weight 越界被拦",
              any("sft" in m for m in errs(DP.DpoTrainOptions(sft_weight=2.0))))
        check("sft_weight=0 给 info（似然可能崩塌）",
              any("崩塌" in m for m in infos(DP.DpoTrainOptions(sft_weight=0.0))))
        check("max_codes 越界被拦",
              any("max_codes" in m for m in errs(DP.DpoTrainOptions(max_codes=9999))))
        bo = DP.PairBuildOptions(out_dataset="x")
        check("默认 PairBuildOptions 无 error", not errs(bo))
        check("out_dataset 为空被拦",
              any("out_dataset" in m for m in errs(DP.PairBuildOptions())))
        check("n_candidates<2 被拦",
              any("n_candidates" in m for m in
                  errs(DP.PairBuildOptions(out_dataset="x", n_candidates=1))))
        check("min_margin 过小给 warn（学噪声）",
              any("噪声" in m for m in warns(DP.PairBuildOptions(
                  out_dataset="x", min_margin=0.001))))
        check("温度过低给 warn（候选没差异）",
              any("温度" in m for m in warns(DP.PairBuildOptions(
                  out_dataset="x", temperature=0.2))))

        # ==============================================================
        head("[3] mel_seq_logprob：形状 / 归一 / 与手算一致")
        m0 = make_tiny_gpt("cpu")
        GD.freeze_base(m0)
        from peft import LoraConfig, get_peft_model
        rx = GD.build_target_regex(["attn/c_attn", "attn/c_proj"])
        pm0 = get_peft_model(m0, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                            target_modules=rx, bias="none",
                                            task_type=None))
        FW.configure_gpt_for_training(pm0, True, 0.0)
        B, Lt, Lc = 3, 8, 30
        st = torch.randn(B, 192); ev = torch.randn(B, DIM)
        tt = torch.randint(2, 500, (B, Lt)); tl = torch.tensor([Lt, Lt - 2, Lt - 4])
        cd = torch.randint(2, 250, (B, Lc)); cl = torch.tensor([Lc, Lc - 5, Lc - 10])
        lg = torch.LongTensor([1, 1, 1])
        f = FW.gpt_training_forward(pm0, st, ev, lg, tt, tl, cd, cl)
        lp = DP.mel_seq_logprob(f, normalize=False)
        lpn = DP.mel_seq_logprob(f, normalize=True)
        check("输出形状 (B,)", tuple(lp.shape) == (B,))
        check("长度归一后数值变小（除以 token 数）",
              bool((lpn.abs() < lp.abs()).all()),
              f"{float(lp.abs().mean()):.2f} → {float(lpn.abs().mean()):.2f}")
        lp.sum().backward()
        g_b = sum(1 for n, p in pm0.named_parameters()
                  if p.requires_grad and ".lora_B." in n
                  and p.grad is not None and float(p.grad.abs().sum()) > 0)
        g_a = sum(1 for n, p in pm0.named_parameters()
                  if p.requires_grad and ".lora_A." in n
                  and p.grad is not None and torch.isfinite(p.grad).all())
        check("梯度回传到 4 个 lora_B（lora_A 首步为 0 是数学必然：B=0）",
              g_b == 4 and g_a == 4, f"B={g_b} A={g_a}")
        # 手算一条对照
        import torch.nn.functional as F
        with torch.no_grad():
            man = F.log_softmax(f.mel_logits[0].float(), dim=0)     # (V, L)
            m1 = float((man[f.mel_targets[0], torch.arange(f.mel_targets.size(1))]
                        * f.mel_mask[0].float()).sum())
        check("与逐位手算一致", abs(float(lp[0]) - m1) < 1e-4,
              f"{float(lp[0]):.4f} vs {m1:.4f}")
        del pm0, m0

        # ==============================================================
        head("[4] 造偏好对数据集")
        d = build_pair_dataset(DS_NAME, N_PAIRS, SR, seed=1)
        rows = DP.load_pairs(DS_NAME)
        check(f"{N_PAIRS} 对全部落盘", len(rows) == N_PAIRS, f"{len(rows)}")
        check("每对的 chosen/rejected 都有特征文件",
              all(FT.is_usable(FT.FeatureExtractor.feature_path(
                  d["dir"], r.chosen)) and
                  FT.is_usable(FT.FeatureExtractor.feature_path(
                      d["dir"], r.rejected)) for r in rows))
        sp = DP.load_pair_split(DS_NAME)
        check(f"划分 train={len(sp['train'])} / val={len(sp['val'])}",
              len(sp["train"]) + len(sp["val"]) == N_PAIRS)

        # ==============================================================
        head("[5] preflight（不加载模型）")
        cfg = GD.LoRAConfig.preset("balanced")
        cfg.apply_target_preset("attn", "gpt")
        cfg.rank, cfg.alpha = 8, 16
        cfg.use_rslora = False
        cfg.lr = 2e-3
        cfg.weight_decay = 0.01
        cfg.warmup_ratio = 0.05
        cfg.batch_size, cfg.grad_accum = 2, 2
        cfg.epochs, cfg.eval_every = 8, 5
        cfg.val_patience = 0             # 主跑不早停（早停在 [8] 单独验）
        cfg.replay_ratio = 0.3           # 故意给非零：preflight 必须声明忽略
        cfg.keep_checkpoints, cfg.bf16 = 3, False
        cfg.dropout = 0.0
        opts = DP.DpoTrainOptions(log_every=2, sft_weight=0.0)   # 纯 DPO：ln2 不变量才成立

        tr = DP.DpoTrainer(DS_NAME, cfg=cfg, options=opts, device="cpu",
                           model_dir=model_dir, train_root=guard_root,
                           val_ratio=0.25, run_name="probe_dpo_run")
        pf = tr.preflight()
        check("preflight 通过", pf["ok"], str(pf["errors"]))
        check("对数账目对得上", pf.get("n_train", 0) + pf.get("n_val", 0) == N_PAIRS
              and pf.get("n_train", 0) > 0 and pf.get("n_val", 0) > 0,
              f"train={pf.get('n_train')} val={pf.get('n_val')}")
        check("replay_ratio 被声明忽略（DPO 无回放）",
              any("回放" in m for m in pf["infos"]) and cfg.replay_ratio == 0.0)
        check("显存估算比 SFT 放大（4 组前向）",
              pf["est_vram_gb"] > GL.estimate_vram_gb(
                  GL.EST_BASE_PARAMS, pf["est_adapter_params"], False, 2, 900),
              f"{pf['est_vram_gb']}")
        check("avg_margin 进了 preflight",
              abs(pf.get("avg_margin", 0.0) - 0.3) < 1e-9,
              str(pf.get("avg_margin")))
        check("模型还没建", tr.pm is None)
        # val 样本必须取自 val 池（与 SFT 同一根钉）
        vp = tr.val_pools["target"]
        vneed = {x for r in tr.val_pairs for x in (r.chosen, r.rejected)}
        check("val 池覆盖全部 val 对的两侧",
              vneed <= set(vp.ids), f"{len(vneed)} 个样本")

        # ==============================================================
        head("[6] prepare + ★ ln2 不变量")
        prep = tr.prepare()
        inj = prep["inject"]
        pm = tr.pm
        check("注入 4 层 attn（同 SFT）", inj["lora_layers"] == 4,
              str(inj["lora_layers"]))
        check("底座全冻结", GD.assert_base_frozen(pm) == [])
        check("run.json 的 arch=dpo",
              RN.read_run("probe_dpo_run").get("arch") == "dpo")
        check("run.json 记了 beta / sft_weight / length_normalize",
              RN.read_run("probe_dpo_run").get("beta") == opts.beta
              and RN.read_run("probe_dpo_run").get("sft_weight") == opts.sft_weight
              and RN.read_run("probe_dpo_run").get("length_normalize") is True)

        v0 = tr.evaluate()
        check("★ step-0 的 DPO loss 恰好等于 ln2（policy==ref，fp32 精度内）",
              v0 is not None and abs(v0 - DP.LN2) < 1e-6,
              f"{v0} vs {DP.LN2:.9f}")
        check("★ step-0 的 acc=0（logits 全 0，>0 判 False）",
              tr.eval_metrics.get("acc") == 0.0, str(tr.eval_metrics))
        check("★ step-0 的 margin=0", tr.eval_metrics.get("margin") == 0.0)
        v0b = tr.evaluate()
        check("评估确定（两次一致）", v0 == v0b)
        # adapters_off 下逐位等于纯底座 —— 参考策略的等价性：
        # 零初始化时「关掉 adapter」与「开着 adapter」必须给出同一个数
        with DP.adapters_off(pm):
            ref_v = tr.evaluate()
        check("adapters_off 不改变评估结果（零初始化的等价性）",
              ref_v == v0, f"{ref_v} vs {v0}")
        # 锚（sft_weight>0）会把 loss 抬高 chosen 的 NLL×w —— 单独验
        opts.sft_weight = 0.1
        v_anchor = tr.evaluate()
        opts.sft_weight = 0.0
        check("sft_weight=0.1 时 loss 更高（NLL 锚生效）",
              v_anchor > v0, f"{v_anchor:.4f} vs {v0:.4f}"
              f"（+{(v_anchor - v0):.4f}）")

        # ==============================================================
        head("[7] 训练：acc 上升 / margin 拉开 / 底座不变")
        fp_before = base_fingerprint(pm)
        lora_before = lora_tensors(pm)
        prog = []
        t0 = time.perf_counter()
        rep = tr.run(progress=lambda f, m: prog.append((round(f, 3), m)))
        dt = time.perf_counter() - t0
        print(f"  训练 {rep.steps} 步 / {rep.epochs} 轮 · {dt:.1f}s · "
              f"最终 acc={tr.eval_metrics.get('acc')}")
        check("训练正常完成", rep.ok and not rep.error, rep.error or f"steps={rep.steps}")
        check("first_val 是 ln2（基线，fp32 精度内）",
              rep.first_val is not None and abs(rep.first_val - DP.LN2) < 1e-6)
        check("★ best_val 明显低于 ln2（学到了偏好）",
              rep.best_val is not None and rep.best_val < DP.LN2 * 0.9,
              f"{DP.LN2:.4f} → {rep.best_val:.4f}"
              f"（降 {(rep.improved or 0)*100:.1f}%）")
        check("★ acc 高于 0.6（chosen 真的更受偏好）",
              tr.eval_metrics.get("acc", 0) > 0.6, str(tr.eval_metrics))
        check("★ margin 为正且拉开了",
              tr.eval_metrics.get("margin", 0) > 0.05, str(tr.eval_metrics))
        check("底座权重逐字节未变", base_fingerprint(pm) == fp_before)
        lora_after = lora_tensors(pm)
        changed = sum(1 for n in lora_before
                      if not torch.equal(lora_before[n], lora_after[n]))
        check("全部 LoRA 参数被更新", changed == len(lora_before),
              f"{changed}/{len(lora_before)}")
        b_nonzero = sum(1 for n, p in pm.named_parameters()
                        if p.requires_grad and ".lora_B." in n
                        and float(p.detach().abs().max()) > 0.0)
        check("lora_B 不再全零", b_nonzero == inj["lora_layers"],
              f"{b_nonzero}/{inj['lora_layers']}")
        rj = RN.read_run("probe_dpo_run")
        rjd = rj.get("data") or {}
        check("run.json 记了 eval_acc / eval_margin（收尾随 data 落盘）",
              abs((rjd.get("eval_acc") or 0) - tr.eval_metrics["acc"]) < 1e-3
              and abs((rjd.get("eval_margin") or 0) - tr.eval_metrics["margin"]) < 1e-3,
              str(rjd))
        check("报告可渲染", len(rep.markdown()) > 300)
        ck = RN.vault("probe_dpo_run").list()
        check("保险库存了 checkpoint", len(ck) >= 1)
        best = RN.best_checkpoint("probe_dpo_run")
        check("release 断开全部引用",
              tr.pm is None and tr.opt is None and not tr.pools)

        # ==============================================================
        head("[8] 续训逐位还原 + 早停")
        tr2 = DP.DpoTrainer(DS_NAME, cfg=cfg, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_dpo_resume", resume_from=best.name)
        tr2.prepare()
        v_resume = tr2.evaluate()
        check("★ 续训起点的 val 与 checkpoint metric 逐位一致",
              v_resume is not None and abs(v_resume - best.metric) < 1e-9,
              f"{v_resume} vs {best.metric}")
        check("续训恢复了优化器动量", len(tr2.opt.state) > 0)
        tr2.release()

        cfg_es = GD.LoRAConfig.from_dict(cfg.to_dict())
        cfg_es.val_patience = 2
        tr3 = DP.DpoTrainer(DS_NAME, cfg=cfg_es, options=opts, device="cpu",
                            model_dir=model_dir, train_root=guard_root,
                            run_name="probe_dpo_es")
        tr3.prepare()
        s1 = tr3._handle_eval(1.0, 0)
        s2 = tr3._handle_eval(1.2, 1)
        s3 = tr3._handle_eval(1.3, 2)
        check("早停：基线不停、变差两次触发", not s1 and not s2 and s3
              and tr3.report.stopped_early, tr3.report.stop_reason)
        tr3.release()

        # ==============================================================
        head("[9] 偏好对构造器（注入假引擎，只测编排）")
        src = build_src_dataset("probe_src", 4, SR)

        fake_rewards: Dict = {}

        class FakeEngine:
            cfg = None
            tts = object()          # truthy 即可

        def fake_infer(**kw):
            text = kw.get("text", "")
            out = kw.get("output_path", "")
            make_wav(out, 1.0, 16000, hash(os.path.basename(out)) % 1000)
            # 分数按 (文本, 候选序号) 定：好的文本差异大，差的文本几乎没差异
            k = int(os.path.basename(out).split("_c")[-1].split(".")[0])
            uid = os.path.basename(out).split("_c")[0]
            spread = 0.4 if "差异" in text else (0.01 if "模糊" in text else 0.2)
            fake_rewards[os.path.abspath(out)] = 0.8 - spread * k
            return out

        class FakeScorer:
            def score(self, path, text, ref, lang=None):
                r = fake_rewards.get(os.path.abspath(path))
                if r is None:
                    return {"ok": False, "error": "no score"}
                return {"ok": True, "wer": 0.0, "sim": 0.5, "reward": r,
                        "asr_text": text, "seconds": 0.0}
            def unload(self):
                pass

        class FakeExtractor:
            def extract(self, wav, text, lang="ZH"):
                return make_feat(0, 40, 10, 5)

        bo = DP.PairBuildOptions(out_dataset="probe_pairs", min_margin=0.05,
                                 n_candidates=2)
        res = DP.build_pairs(FakeEngine(), "probe_src", bo,
                             infer_fn=fake_infer, scorer=FakeScorer(),
                             extractor=FakeExtractor())
        check("构造报告 ok", res["ok"], str(res["errors"]))
        check("合成了 8 个候选（4 文本 × 2）", res["synthesized"] == 8,
              str(res["synthesized"]))
        check("margin 大的 4 条全部成对", res["kept"] == 4, str(res["kept"]))
        rows = DP.load_pairs("probe_pairs")
        check("偏好对已入库（含 reward 双侧）",
              len(rows) == 4 and all(r.reward_chosen > r.reward_rejected
                                     for r in rows))
        check("成对的 margin ≥ min_margin",
              all(r.margin >= 0.05 for r in rows),
              str([r.margin for r in rows]))
        sp = DP.load_pair_split("probe_pairs")
        check("新数据集自动划分了", bool(sp.get("train") or sp.get("val")))

        # 追加模式：不覆盖旧行
        res2 = DP.build_pairs(FakeEngine(), "probe_src", bo,
                              infer_fn=fake_infer, scorer=FakeScorer(),
                              extractor=FakeExtractor())
        rows2 = DP.load_pairs("probe_pairs")
        check("追加模式不产生重复 id", len(rows2) == len(set(r.id for r in rows2)),
              f"{len(rows2)} 行 / {len(set(r.id for r in rows2))} 个 id")
        # 覆盖模式：清空重写
        bo2 = DP.PairBuildOptions(out_dataset="probe_pairs", min_margin=0.05,
                                  overwrite=True)
        res3 = DP.build_pairs(FakeEngine(), "probe_src", bo2,
                              infer_fn=fake_infer, scorer=FakeScorer(),
                              extractor=FakeExtractor())
        check("覆盖模式重建偏好对",
              len(DP.load_pairs("probe_pairs")) == res3["kept"])

        # 候选差异小于 min_margin → 全部丢弃
        src2 = build_src_dataset("probe_src2", 2, SR, tag="模糊")
        bo3 = DP.PairBuildOptions(out_dataset="probe_pairs2", min_margin=0.05)
        res4 = DP.build_pairs(FakeEngine(), "probe_src2", bo3,
                              infer_fn=fake_infer, scorer=FakeScorer(),
                              extractor=FakeExtractor())
        check("margin 不足的对被丢弃（不学噪声）",
              res4["kept"] == 0 and res4["dropped"] == 2,
              f"kept={res4['kept']} dropped={res4['dropped']}")

        # should_stop 中途停下，已完成的部分保留（追加模式，不动已有对）
        bo4 = DP.PairBuildOptions(out_dataset="probe_pairs", min_margin=0.05)
        res5 = DP.build_pairs(FakeEngine(), "probe_src", bo4,
                              infer_fn=fake_infer, scorer=FakeScorer(),
                              extractor=FakeExtractor(),
                              should_stop=lambda: True)
        check("should_stop 立即停且报告了原因",
              res5["kept"] == 0 and any("停止" in w for w in res5["warnings"]))

        # 引擎没加载 → 可读错误
        class DeadEngine:
            cfg = None
            tts = None
        res6 = DP.build_pairs(DeadEngine(), "probe_src", bo2)
        check("引擎未加载给出可读错误", not res6["ok"]
              and any("加载" in e for e in res6["errors"]))

        md = DP.pairs_markdown("probe_pairs")
        check("pairs_markdown 可渲染", "margin" in md and "|" in md)

        # ==============================================================
        head("[10] 清理")
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
        for x in FAILS:
            print(f"    FAIL {x}")
    print("=" * 70)
    return 1 if FAIL else 0


def build_src_dataset(name: str, n: int, sr: int, seed: int = 0,
                      tag: str = "差异"):
    """构造器的源数据集：n 条带文本与音频的 ready 样本。

    文本里带 tag（差异/模糊），fake_infer 依此决定候选分数差。
    """
    DS.create(name, note="dpo_probe 源数据集")
    ds_dir = DS.dir_of(name)
    tmpw = os.path.join(ds_dir, "_src")
    os.makedirs(tmpw, exist_ok=True)
    wavs = []
    for i in range(n):
        p = os.path.join(tmpw, f"s{i:03d}.wav")
        make_wav(p, 1.5, sr, seed + i)
        wavs.append(p)
    DS.import_audio(name, wavs, copy=True, lang="ZH")
    shutil.rmtree(tmpw, ignore_errors=True)
    touched = {}
    for i, u in enumerate(DS.load_meta(name)):
        touched[u.id] = {"text": f"第{i}句{tag}测试文本。", "status": "ready",
                         "lang": "ZH"}
    FT._apply_meta(name, touched)
    DS.refresh_all(name, require_features=False)
    # refresh 会覆盖 status？把带文本的 ready 状态钉住
    for i, u in enumerate(DS.load_meta(name)):
        if not (u.text or "").strip():
            DS.update(name, u.id, text=f"第{i}句{tag}测试文本。")
    DS.refresh_all(name, require_features=False)
    return {"dir": ds_dir}


if __name__ == "__main__":
    raise SystemExit(main())
