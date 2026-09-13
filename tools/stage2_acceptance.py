"""阶段 2 验收（b9）：真机完整训练闭环。

CPU 探针验证的是**逻辑**；这一步在真实权重 + 真实 GPU（RTX 4060 8GB）
上把整条链走一遍：

    A  数据：引擎合成 28 句（文本已知）→ 建数据集 → 特征提取 → 卸载引擎
    B  GPT LoRA 真权重训练（few steps）→ 底座校验 / 漂移 / adapter 落盘
    C  CFM LoRA 真权重训练（few steps）
    D  DPO 偏好对**真机**构造（引擎 + 挂 adapter + whisper 打分）
    E  A/B 评测（adapter vs 纯底座，真合成 + 真打分 + 试听文件）
    F  挂载/强度旋钮/合并（真权重，产物可被全新底座完整加载）
    G  汇总报告 → outputs/acceptance/stage2_report.md

任何一步失败都会继续往下走（能测多少测多少），最后统一结算。
显存纪律：训练前卸载引擎（脚本自己管），阶段间打印显存账目。

跑法：  .venv\\Scripts\\python.exe tools\\stage2_acceptance.py
预计：  10~15 分钟（首次还会下载 whisper small，约 460 MB）
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import os
import shutil
import time
import traceback

import torch

PROJECT_ROOT = _env.PROJECT_ROOT

PASS, FAIL = [], []
PHASES = []          # (名字, 秒, 明细)


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    mark = "✅" if cond else "✖"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))


def head(t: str):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def vram(tag: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  [显存] {tag}: alloc={alloc:.2f} GB")
        return round(alloc, 2)
    return 0.0


# 28 句训练文本（短句、无数字、无生僻字 —— 让 whisper 的账目干净）
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
    "会议记录已经发到你的邮箱了。",
    "这个周末我们去爬山怎么样。",
    "她笑起来的时候眼睛弯弯的。",
    "记住出门前检查一下门窗。",
    "这道题的答案其实很简单。",
    "图书馆里安静得能听见翻书声。",
    "我们公司的产品远销海外。",
    "孩子们在草地上追逐嬉戏。",
    "技术的进步改变了生活方式。",
    "下雨天记得带伞出门。",
    "这份报告下周一之前要交。",
    "他每天早上都会跑步锻炼。",
    "厨房里飘来阵阵香味。",
    "新学期开始了，大家都很兴奋。",
    "火车缓缓驶离了站台。",
    "老人坐在门前晒太阳。",
]

DS_NAME = "accept_stage2"
GPT_RUN = "accept_gpt"
CFM_RUN = "accept_cfm"
DPO_DS = "accept_pairs"
REF_WAV = os.path.join(PROJECT_ROOT, "examples", "voice_03.wav")


def synth(engine, text, out_path, temperature=0.9, seed=0):
    torch.manual_seed(int(seed))
    engine.infer(
        spk_audio_prompt=REF_WAV, text=text, output_path=out_path,
        lang="ZH", verbose=False, do_sample=True, top_p=0.9, top_k=50,
        temperature=float(temperature), max_mel_tokens=600)
    return os.path.isfile(out_path)


def main() -> int:
    t_all = time.perf_counter()
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "acceptance"),
                exist_ok=True)
    out_root = os.path.join(PROJECT_ROOT, "outputs", "acceptance")

    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    AppContext.reset()
    cfg = config_from_args(["--lazy"])
    ctx = AppContext.get(cfg)
    eng = ctx.engine

    from webui_app.training import cfm_lora as CL
    from webui_app.training import dataset as DS
    from webui_app.training import dpo as DP
    from webui_app.training import evaluate as EV
    from webui_app.training import features as FT
    from webui_app.training import gpt_lora as GL
    from webui_app.training import guard as GD
    from webui_app.training import merge as MG
    from webui_app.training import reward as RW
    from webui_app.training import runs as RN

    # 清掉上次验收的残留（可重复跑）
    for run in (GPT_RUN, CFM_RUN):
        RN.delete_run(run)
    if DS.exists(DS_NAME):
        DS.delete(DS_NAME)
    if DS.exists(DPO_DS):
        DS.delete(DPO_DS)

    # ===================================================================
    tA = time.perf_counter()
    head("[A] 数据准备：引擎合成 → 建数据集 → 特征提取")
    try:
        print("  加载引擎（约 30s）…")
        eng.load()
        vram("引擎加载后")
        check("引擎加载成功", eng.loaded)

        DS.create(DS_NAME, note="阶段2验收：引擎自合成")
        tmpw = os.path.join(DS.dir_of(DS_NAME), "_src")
        os.makedirs(tmpw, exist_ok=True)
        wavs = []
        for i, txt in enumerate(TEXTS):
            p = os.path.join(tmpw, f"a{i:03d}.wav")
            ok = synth(eng, txt, p, seed=100 + i)
            if ok:
                wavs.append((p, txt))
        check(f"合成 {len(TEXTS)} 句全部成功", len(wavs) == len(TEXTS))
        imp = DS.import_audio(DS_NAME, [p for p, _ in wavs], copy=True,
                              lang="ZH")
        check(f"导入 {len(TEXTS)} 条", imp.get("added") == len(TEXTS))
        touched = {}
        for u, (_p, txt) in zip(DS.load_meta(DS_NAME), wavs):
            touched[u.id] = {"text": txt}
        FT._apply_meta(DS_NAME, touched)
        DS.refresh_all(DS_NAME, require_features=False)
        n_ready = len([u for u in DS.load_meta(DS_NAME) if u.status == "ready"])
        check(f"全部 ready（音频体检过）", n_ready == len(TEXTS), f"{n_ready}")

        print("  特征提取（借引擎的 w2v-bert / codec / campplus）…")
        fx = FT.extract_dataset(DS_NAME, only="ready", tts=eng.tts,
                                device=str(getattr(eng.tts, "device", "cuda")))
        check("特征提取 ok", fx.get("ok"), str(fx.get("error", ""))[:60])
        n_feat = len([u for u in DS.load_meta(DS_NAME) if u.has_features])
        check(f"特征缓存 {len(TEXTS)} 条齐了", n_feat == len(TEXTS), f"{n_feat}")
        r = DS.make_split(DS_NAME, val_ratio=0.25, seed=42)
        check(f"划分 train={r.get('train')} / val={r.get('val')}",
              r.get("ok") and r.get("train", 0) >= 20,
              f"{r.get('train')}/{r.get('val')}（≥20 才过数据量防线）")

        print("  卸载引擎（训练要显存）…")
        eng.unload()
        vram("引擎卸载后")
        check("引擎已卸载", not eng.loaded)
        PHASES.append(("A 数据准备", time.perf_counter() - tA, ""))
    except Exception as e:
        traceback.print_exc()
        check(f"A 阶段异常：{e}", False)
        PHASES.append(("A 数据准备", time.perf_counter() - tA, f"异常 {e}"))

    # ===================================================================
    tB = time.perf_counter()
    head("[B] GPT LoRA 真权重训练")
    try:
        cfg_l = GD.LoRAConfig.preset("conservative")
        cfg_l.apply_target_preset("attn", "gpt")
        cfg_l.rank, cfg_l.alpha = 8, 16
        cfg_l.lr = 5e-5
        cfg_l.epochs, cfg_l.max_steps = 8, 6
        cfg_l.eval_every, cfg_l.val_patience = 3, 0
        cfg_l.replay_ratio = 0.0
        cfg_l.warmup_ratio = 0.1
        opts = GL.GptTrainOptions(max_codes=400, log_every=1)
        tr = GL.GptTrainer(DS_NAME, cfg=cfg_l, options=opts,
                           run_name=GPT_RUN, val_ratio=0.25)
        pf = tr.preflight()
        check("GPT 预检通过", pf.get("ok"), str(pf.get("errors"))[:80])
        print(f"  预估显存 {pf.get('est_vram_gb')} GB · "
              f"train={pf.get('n_train')} val={pf.get('n_val')}")
        rep = tr.run(progress=lambda f, m: print(f"    {f:5.0%} {m}")
                     if f % 0.25 < 0.02 else None)
        check("GPT 训练 ok", rep.ok, rep.error or f"steps={rep.steps}")
        check("GPT 底座复校通过（before+after）",
              rep.base_verify.get("ok") is True)
        check("GPT 漂移体检有账", "global_rel" in rep.drift,
              f"全局 {rep.drift.get('global_rel')}")
        check("GPT adapter 已同步",
              os.path.isdir(RN.adapter_dir(GPT_RUN)))
        check("GPT 峰值显存有记录", rep.vram_peak_gb > 0,
              f"{rep.vram_peak_gb} GB")
        vram("GPT 训练后（应已释放）")
        with open(os.path.join(out_root, "gpt_report.md"), "w",
                  encoding="utf-8") as f:
            f.write(rep.markdown())
        PHASES.append(("B GPT 训练", time.perf_counter() - tB,
                       f"steps={rep.steps} drift={rep.drift.get('global_rel')}"))
    except Exception as e:
        traceback.print_exc()
        check(f"B 阶段异常：{e}", False)
        PHASES.append(("B GPT 训练", time.perf_counter() - tB, f"异常 {e}"))

    # ===================================================================
    tC = time.perf_counter()
    head("[C] CFM LoRA 真权重训练")
    try:
        cfg_c = CL.default_config("conservative")
        cfg_c.rank, cfg_c.alpha = 8, 16
        cfg_c.lr = 5e-5
        cfg_c.epochs, cfg_c.max_steps = 8, 4
        cfg_c.eval_every, cfg_c.val_patience = 2, 0
        cfg_c.replay_ratio = 0.0
        cfg_c.warmup_ratio = 0.1
        opts_c = CL.CfmTrainOptions(max_total_frames=1400, max_target_frames=1000,
                                    log_every=1)
        trc = CL.CfmTrainer(DS_NAME, cfg=cfg_c, options=opts_c,
                            run_name=CFM_RUN, val_ratio=0.25)
        pfc = trc.preflight()
        check("CFM 预检通过", pfc.get("ok"), str(pfc.get("errors"))[:80])
        repc = trc.run(progress=lambda f, m: print(f"    {f:5.0%} {m}")
                       if f % 0.25 < 0.02 else None)
        check("CFM 训练 ok", repc.ok, repc.error or f"steps={repc.steps}")
        check("CFM 底座复校通过", repc.base_verify.get("ok") is True)
        check("CFM 漂移体检有账", "global_rel" in repc.drift,
              f"全局 {repc.drift.get('global_rel')}")
        check("CFM adapter 已同步", os.path.isdir(RN.adapter_dir(CFM_RUN)))
        vram("CFM 训练后（应已释放）")
        with open(os.path.join(out_root, "cfm_report.md"), "w",
                  encoding="utf-8") as f:
            f.write(repc.markdown())
        PHASES.append(("C CFM 训练", time.perf_counter() - tC,
                       f"steps={repc.steps} drift={repc.drift.get('global_rel')}"))
    except Exception as e:
        traceback.print_exc()
        check(f"C 阶段异常：{e}", False)
        PHASES.append(("C CFM 训练", time.perf_counter() - tC, f"异常 {e}"))

    # ===================================================================
    tD = time.perf_counter()
    head("[D] DPO 偏好对真机构造（引擎 + adapter + whisper）")
    try:
        print("  重新加载引擎 + 挂 GPT adapter…")
        eng.load()
        tag = MG.mount_run(eng, GPT_RUN, checkpoint="best", scale=1.0)
        check("adapter 挂载成功", tag.startswith("gpt:"), tag)
        bo = DP.PairBuildOptions(
            out_dataset=DPO_DS, n_candidates=2, min_margin=0.03,
            max_pairs=4, temperature=0.95, temperature_jitter=0.2,
            overwrite=True)
        resd = DP.build_pairs(eng, DS_NAME, bo)
        check("偏好对构造 ok", resd.get("ok"), str(resd.get("errors"))[:80])
        check(f"成对 {resd.get('kept')} 对（合成 {resd.get('synthesized')} 个候选）",
              resd.get("kept", 0) >= 1,
              f"kept={resd.get('kept')} dropped={resd.get('dropped')}")
        rows = DP.load_pairs(DPO_DS)
        check("pairs.jsonl 落盘且带 reward 双侧",
              len(rows) >= 1 and all(r.reward_chosen >= r.reward_rejected
                                     for r in rows))
        MG.unmount(eng, "gpt")
        eng.unload()
        vram("D 阶段后（引擎已卸载）")
        PHASES.append(("D DPO 偏好对", time.perf_counter() - tD,
                       f"kept={resd.get('kept')}"))
    except Exception as e:
        traceback.print_exc()
        check(f"D 阶段异常：{e}", False)
        try:
            MG.unmount(eng, "gpt")
            eng.unload()
        except Exception:
            pass
        PHASES.append(("D DPO 偏好对", time.perf_counter() - tD, f"异常 {e}"))

    # ===================================================================
    tE = time.perf_counter()
    head("[E] A/B 评测（adapter vs 纯底座，真合成 + 真打分）")
    try:
        eng.load()
        a = EV.Contender("adapter", run=GPT_RUN, checkpoint="best")
        b = EV.Contender("base")
        opte = EV.EvalOptions(dataset=DS_NAME, n_samples=3, seed=42)
        rese = EV.run_eval(eng, a, b, opte)
        check("A/B 评测 ok", rese.get("ok"), str(rese.get("errors"))[:80])
        check("逐条 3 行", len(rese.get("rows") or []) == 3)
        s = rese.get("summary") or {}
        check("胜负表有账", "win_a" in s,
              f"A胜{s.get('win_a')} B胜{s.get('win_b')} 平{s.get('tie')}")
        check("报告与试听文件落盘",
              os.path.isfile(os.path.join(rese.get("out_dir", ""), "report.md"))
              and len([f for f in os.listdir(rese.get("out_dir", ""))
                       if f.endswith(".wav")]) == 6,
              rese.get("out_dir", ""))
        check("评测后引擎上没有残留 adapter",
              not getattr(eng.stats, "lora_adapters", []))
        # 把报告拷进验收目录
        try:
            shutil.copy(os.path.join(rese["out_dir"], "report.md"),
                        os.path.join(out_root, "ab_report.md"))
        except Exception:
            pass
        eng.unload()
        PHASES.append(("E A/B 评测", time.perf_counter() - tE,
                       f"Δreward={s.get('mean_delta_reward')}"))
    except Exception as e:
        traceback.print_exc()
        check(f"E 阶段异常：{e}", False)
        try:
            eng.unload()
        except Exception:
            pass
        PHASES.append(("E A/B 评测", time.perf_counter() - tE, f"异常 {e}"))

    # ===================================================================
    tF = time.perf_counter()
    head("[F] 挂载强度旋钮 + 合并（真权重）")
    try:
        eng.load()
        MG.mount_run(eng, GPT_RUN, scale=1.0)
        n = MG.set_scale(eng, 0.7, "gpt")
        sc = GD.get_adapter_scale(getattr(eng.tts, "gpt", None))
        check("强度旋钮真机生效（0.7）",
              n > 0 and abs(sc.get("_mean", 0) - 0.7) < 1e-6,
              f"{n} 层 → {sc.get('_mean')}")
        MG.unmount(eng, "gpt")
        eng.unload()

        # 合并 GPT（CPU）
        mg_out = os.path.join(out_root, "gpt_merged.pth")
        repm = MG.merge_lora_to_checkpoint(MG.MergeOptions(
            adapter_dir=RN.adapter_dir(GPT_RUN), arch="gpt",
            out_path=mg_out))
        check("GPT 合并 ok", repm.ok, "; ".join(repm.notes)[:80])
        check("合并产物存在且 >1GB（真权重量级）",
              repm.ok and os.path.getsize(mg_out) > 1e9,
              f"{os.path.getsize(mg_out)/1e9:.2f} GB" if repm.ok else "")
        check("合并校验 missing/unexpected=0",
              repm.missing == 0 and repm.unexpected == 0)

        mg_out2 = os.path.join(out_root, "s2mel_merged.pth")
        repm2 = MG.merge_lora_to_checkpoint(MG.MergeOptions(
            adapter_dir=RN.adapter_dir(CFM_RUN), arch="cfm",
            out_path=mg_out2))
        check("CFM 合并 ok 且校验干净", repm2.ok
              and repm2.missing == 0 and repm2.unexpected == 0,
              "; ".join(repm2.notes)[:80])
        PHASES.append(("F 挂载/合并", time.perf_counter() - tF,
                       f"漂移 {repm.global_rel}"))
    except Exception as e:
        traceback.print_exc()
        check(f"F 阶段异常：{e}", False)
        try:
            eng.unload()
        except Exception:
            pass
        PHASES.append(("F 挂载/合并", time.perf_counter() - tF, f"异常 {e}"))

    # ===================================================================
    head("[G] 汇总")
    total = time.perf_counter() - t_all
    L = ["# 阶段 2 验收报告（真机完整闭环）", "",
         f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')} · 总耗时 {total/60:.1f} 分钟",
         f"- 结论：**{'✅ 通过' if not FAIL else '🔴 有失败项'}**"
         f"（{len(PASS)} 项通过 / {len(FAIL)} 项失败）", "",
         "## 阶段耗时", "", "| 阶段 | 耗时 | 明细 |", "|---|---|---|"]
    for name, sec, note in PHASES:
        L.append(f"| {name} | {sec/60:.1f} 分钟 | {note} |")
    L += ["", "## 全部断言", ""]
    for x in PASS:
        L.append(f"- ✅ {x}")
    for x in FAIL:
        L.append(f"- 🔴 {x}")
    L += ["", "## 产物", "",
          f"- GPT 训练报告：`outputs/acceptance/gpt_report.md`",
          f"- CFM 训练报告：`outputs/acceptance/cfm_report.md`",
          f"- A/B 评测报告：`outputs/acceptance/ab_report.md`",
          f"- 合并权重：`outputs/acceptance/gpt_merged.pth` / "
          f"`s2mel_merged.pth`"]
    report = os.path.join(out_root, "stage2_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
        for x in FAIL:
            print(f"    FAIL {x}")
    else:
        print(f"  通过 {len(PASS)} 项 · 失败 0 项 — 阶段 2 验收通过 ✅")
    print(f"  报告：{report}")
    print(f"  总耗时 {total/60:.1f} 分钟")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
