"""「一键三连」真机端到端验收。

用引擎自己合成一批语音当训练语料，然后把**两种输入形态一起**喂给流水线：

    · 一条长音频（30 句拼成，约 3~4 分钟）→ 必须被自动切片
    · 10 个独立短音频                        → 直接进训练集

然后跑完整七阶段：切片 → 优化 → 逐片 whisper 转写（对齐）→ 筛选划分 →
特征提取 → 自动调参 → GPT+CFM 真训练 → 真机打分择优 → 激活最优。

与其它验收脚本的区别：**本脚本通过 runner 真实提交任务**（不是直接调函数），
所以同时也验证了：
  · require_engine="none" 下流水线能自己 load/unload 引擎；
  · 阶段边界上的 set_engine_req 真的在切换（轮询时采样 engine_req，
    必须同时观测到 loaded 与 unloaded —— 这证明训练期间互斥保护在生效）；
  · runner 的结果契约（ok=False 才判失败）与进度回调。

跑法：  .venv\\Scripts\\python.exe tools\\oneclick_acceptance.py
        加 --keep 保留中间产物（合成音频、数据集、训练记录）便于排查。

退出码：0 = 全过；1 = 有失败项。
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import json
import os
import queue
import shutil
import sys
import time

import numpy as np
import soundfile as sf

from webui_app.config import PROJECT_ROOT, config_from_args    # noqa: E402
from webui_app.services import inference as INF                # noqa: E402
from webui_app.services.engine import TTSEngine                # noqa: E402
from webui_app.training import dataset as DS                   # noqa: E402
from webui_app.training import features as FT                  # noqa: E402
from webui_app.training import oneclick as OC                  # noqa: E402
from webui_app.training import reward as RW                    # noqa: E402
from webui_app.training import runs as RN                      # noqa: E402
from webui_app.training.runner import get_runner               # noqa: E402

PASS = FAIL = 0
FAILS: list = []

REF_AUDIO = os.path.join(PROJECT_ROOT, "examples", "voice_01.wav")
WORK = os.path.join(PROJECT_ROOT, "outputs", "oneclick_accept")
DOC = os.path.join(PROJECT_ROOT, "docs", "verification")

# 合成语料：长短混合，既有完整长句也有短句，便于观察切片行为。
SENTENCES = [
    "大家好，欢迎使用 IndexTTS 二点五，这是一段用于训练的中文语音。",
    "今天天气不错，我们一起去公园散步吧，听说湖边的花都开了。",
    "这本书我已经读完了，里面的故事让我想起了很多小时候的事情。",
    "请把窗户关上，外面的风有点大，我担心桌上的文件会被吹走。",
    "他站在讲台上，用平静而坚定的语气讲述了自己这些年来的经历。",
    "技术的发展速度远超我们的想象，十年前谁能想到手机会变成这样。",
    "厨房里飘来一阵香味，母亲正在准备我们最爱吃的那道家乡菜。",
    "这段音乐让我想起了一场雨，雨点落在屋檐上，声音清脆又温柔。",
    "我们需要在下周之前完成这份报告，所以这两天可能要加班了。",
    "远处传来火车的汽笛声，悠长而低沉，像是从另一个时代传来的。",
    "他笑了笑，没有回答，只是把手中的茶杯轻轻放在了桌子上。",
    "这个故事告诉我们，耐心和坚持往往比天赋更加重要。",
    "秋天到了，树叶慢慢变黄，风一吹就落满了整条小路。",
    "人工智能正在改变我们的工作方式，但有些东西是它替代不了的。",
    "我小的时候特别喜欢在雨天里奔跑，现在想想真是无忧无虑。",
    "会议室里很安静，所有人都等着他开口说第一句话。",
    "这道菜的做法其实很简单，关键在于火候和调料的搭配。",
    "他用了整整三年的时间，才把这个想法变成了真正的产品。",
    "夜晚的城市灯火通明，街道上人来人往，热闹得让人不想回家。",
    "学习一门新的语言需要长期的积累，急不来的。",
    "那天的阳光特别好，照在雪地上，亮得让人睁不开眼睛。",
    "老师说的话我一直记着，做事之前先想清楚为什么要做。",
    "这家小店藏在巷子深处，只有熟客才知道它的位置。",
    "我们的计划是这样的，先收集资料，然后再讨论具体的方案。",
    "音乐响起的时候，整个大厅的人都安静了下来。",
    "他喜欢在清晨写作，那时候思路最清楚，也最不容易被打断。",
    "如果你遇到困难，不要一个人扛着，记得告诉我们。",
    "这本画册记录了这座城市一百年来的变化，非常珍贵。",
    "春天的风带着湿润的气息，吹在脸上让人觉得格外清醒。",
    "他把所有的笔记都整理了一遍，然后才开始动手写第一稿。",
    "时间过得真快，转眼间我们已经认识十年了。",
    "窗外的雨渐渐小了，天边露出了一点淡淡的亮色。",
    "这道题目看起来简单，其实里面藏着好几个陷阱。",
    "我们沿着河岸一直走，直到太阳完全落下去才回家。",
    "他说话语速不快，但每一句都掷地有声。",
    "这个季节的水果特别甜，尤其是刚摘下来的那种。",
    "如果一切顺利，我们下个月就可以开始新的项目了。",
    "她把窗帘拉开，让早晨的阳光洒满了整个房间。",
    "很多人问过这个问题，答案其实一直都很简单。",
    "最后，感谢每一位参与这个项目的同事和朋友。",
]

LONG_COUNT = 30          # 前 30 句拼成一条长音频
SHORT_COUNT = 10         # 后 10 句各自成文件


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


def concat_wavs(paths, out_path, sr=22050, gap_ms=500):
    y = []
    for p in paths:
        d, r = sf.read(p, always_2d=False)
        if d.ndim > 1:
            d = d.mean(axis=1)
        if r != sr:
            import librosa
            d = librosa.resample(d.astype(np.float32), orig_sr=r, target_sr=sr)
        y.append(d.astype(np.float32))
        y.append(np.zeros(int(sr * gap_ms / 1000), dtype=np.float32))
    sf.write(out_path, np.concatenate(y), sr)
    return out_path


def main() -> int:
    keep = "--keep" in sys.argv
    t_start = time.perf_counter()
    print("=" * 70)
    print("  一键三连 · 真机端到端验收")
    print("=" * 70)

    if not os.path.isfile(REF_AUDIO):
        print(f"✖ 缺少参考音频 {REF_AUDIO}，无法进行验收。")
        return 1

    for d in (WORK, DOC):
        os.makedirs(d, exist_ok=True)
    raw_dir = os.path.join(WORK, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 数据集名带时间戳：反复跑不会互相干扰，也便于人工翻看
    ds_name = "accept_oneclick_" + time.strftime("%m%d-%H%M")
    cfg = config_from_args(["--lazy"])
    eng = TTSEngine(cfg)
    runner = get_runner()

    observed_reqs = set()
    poll_stop = queue.Queue()

    # =====================================================================
    head("[A] 用引擎合成训练语料")
    # =====================================================================
    try:
        eng.load()
    except Exception as e:
        print(f"✖ 引擎加载失败：{type(e).__name__}: {e}")
        return 1
    check("引擎加载成功", eng.loaded,
          f"{eng.stats.load_seconds:.1f}s · 显存 {eng.stats.vram_alloc_gb:.2f} GB")

    made = []
    t0 = time.perf_counter()
    reused = 0
    for i, text in enumerate(SENTENCES):
        out = os.path.join(raw_dir, f"sent_{i:02d}.wav")
        # 合成很慢（每句约 5 秒），已经有的就复用 —— 中断后重跑不必重来
        if os.path.isfile(out):
            try:
                d = sf.info(out).duration
                if d > 1.0:
                    made.append((out, text, float(d)))
                    reused += 1
                    continue
            except Exception:
                pass
        req = INF.GenRequest(spk_audio_prompt=REF_AUDIO, text=text, lang="ZH",
                             emo_control_method=0, seed=1000 + i,
                             do_sample=True, temperature=0.8, top_p=0.9,
                             top_k=30, max_mel_tokens=900)
        try:
            res = INF.generate(eng, req, output_path=out)
            made.append((res["path"], text, res["audio_duration"]))
        except Exception as e:
            check(f"合成第 {i} 句", False, f"{type(e).__name__}: {e}")
    total_sec = sum(x[2] for x in made)
    check("合成的句子足够组成训练集", len(made) >= LONG_COUNT + SHORT_COUNT - 2,
          f"{len(made)} 句（复用 {reused}）· 共 {total_sec:.0f} 秒 · "
          f"{time.perf_counter()-t0:.0f}s")
    check("语料总时长足够切出多条片段（真正的要求是筛留 ≥20 条，见 [D]）",
          total_sec > 120, f"{total_sec:.0f}s")
    eng.unload()
    check("合成后引擎已卸载（把显存让给后面的训练）", not eng.loaded)

    # =====================================================================
    head("[B] 备好两种输入：一条长音频 + 若干短音频")
    # =====================================================================
    long_paths = [p for p, _t, _d in made[:LONG_COUNT]]
    short_paths = [p for p, _t, _d in made[LONG_COUNT:LONG_COUNT + SHORT_COUNT]]
    long_wav = concat_wavs(long_paths, os.path.join(WORK, "long_input.wav"))
    long_dur = sf.info(long_wav).duration
    check("长音频已拼好且明显超过切片阈值", long_dur > OC.OneClickOptions().slice_over_sec * 3,
          f"{long_dur:.0f}s")
    check("短音频数量正确", len(short_paths) == SHORT_COUNT, str(len(short_paths)))

    # =====================================================================
    head("[C] 通过 runner 真实提交「一键三连」")
    # =====================================================================
    opts = OC.OneClickOptions(
        dataset_name=ds_name,
        lang="ZH",
        slice_over_sec=20.0,
        slice_target_sec=10.0,
        slice_min_sec=4.0,
        slice_max_pieces=40,
        enhance=True,
        denoise=True,
        denoise_strength=0.6,
        asr=True,
        # 故意不给 whisper_size：验收要用**默认值**跑，否则升到 medium
        # 这类改动就永远测不到（默认值装的是用户实际拿到的那条路径）。
        min_score=0.0,          # 合成音频本身很干净，这里不靠分数筛
        max_text_repeats=3,
        val_ratio=0.15,
        arches="gpt,cfm",
        preset_mode="auto",
        top_k=3,
        rank_eval=True,
        eval_samples=3,
        seed=42,
    )
    check("验收选项自身通过自检",
          not [n for n in opts.validate() if n.level == "error"],
          str([n.message for n in opts.validate() if n.level == "error"]))

    collect = OC.collect_inputs(short_paths, long_wav)
    check("两种输入都被收集到（长音频走路径）",
          len(collect["files"]) == SHORT_COUNT + 1,
          f"{len(collect['files'])} 个文件")

    prog_lines = []

    def fn(progress, should_stop):
        def cb(frac, msg):
            prog_lines.append(f"[{frac:5.0%}] {msg}")
            progress(frac, msg)
        return OC.run_oneclick(eng, opts, paths=short_paths,
                               extra_path=long_wav, progress=cb,
                               should_stop=should_stop, tracker=runner)

    sub = runner.submit("oneclick", f"一键三连验收 · {len(collect['files'])} 个输入",
                        fn, require_engine="none", engine=eng)
    check("任务已提交到 runner", bool(sub.get("ok")), str(sub)[:120])

    # ---- 轮询，同时采样 engine_req（验证动态互斥） ----
    t0 = time.perf_counter()
    last_msg = ""
    while True:
        snap = runner.snapshot()
        observed_reqs.add(runner.engine_req)
        if snap["message"] != last_msg:
            last_msg = snap["message"]
            print(f"    [{snap['progress']:5.0%}] {snap['phase']:9s} {last_msg}")
        if not snap["running"]:
            break
        if time.perf_counter() - t0 > 3600:
            print("✖ 超时（1 小时），终止验收")
            return 1
        time.sleep(3)
    elapsed = time.perf_counter() - t0

    res = runner.snapshot()
    check("runner 报告任务成功结束", res["ok"] is True,
          f"ok={res['ok']} error={res['error'][:200]}")
    check("完成耗时在合理范围内（< 60 分钟）", elapsed < 3600,
          f"{elapsed/60:.1f} 分钟")
    check("进度回调有实际输出（不是空跑）", len(prog_lines) > 40,
          f"{len(prog_lines)} 行进度")
    check("动态引擎互斥在真实路径上生效：同时观测到 loaded 与 unloaded",
          {"loaded", "unloaded"} <= observed_reqs, str(sorted(observed_reqs)))

    result = res.get("result") or {}
    report = result.get("report") or {}
    md = result.get("markdown") or ""

    # =====================================================================
    head("[D] 阶段级断言")
    # =====================================================================
    stages = {s["key"]: s for s in (report.get("stages") or [])}
    for key, title, _w in OC.STAGES:
        s = stages.get(key)
        check(f"阶段「{title}」存在且完成",
              bool(s) and s.get("status") == "done",
              f"{s.get('status') if s else '缺失'} · {s.get('detail', '') if s else ''}")

    # ---- 识别阶段：默认模型 + 提示词 + 用完卸载 ----
    asr_info = (stages.get("asr") or {}).get("info") or {}
    check("识别用的是默认 whisper（medium，产出训练文本）",
          str(asr_info.get("whisper")) == RW.DEFAULT_WHISPER,
          f"实际 {asr_info.get('whisper')} / 默认 {RW.DEFAULT_WHISPER}")
    check("打分的 whisper 比识别小一档（8 GB 卡上引擎与它同时驻留）",
          OC.OneClickOptions().score_whisper_size == "small",
          OC.OneClickOptions().score_whisper_size)
    check("识别带了简体提示词（修繁体字）",
          "普通话" in str(asr_info.get("prompt", "")),
          str(asr_info.get("prompt")))
    check("识别结束后 whisper 已卸载（不留驻显存）",
          asr_info.get("loaded_after_unload") is False,
          str(asr_info.get("loaded_after_unload")))
    check("识别阶段报告了显存释放量",
          asr_info.get("freed_gb") is not None,
          f"before={asr_info.get('vram_before_gb')} "
          f"after={asr_info.get('vram_after_gb')} "
          f"freed={asr_info.get('freed_gb')}")
    check("转写没有大量失败", int(asr_info.get("failed", 0)) == 0,
          f"failed={asr_info.get('failed')} empty={asr_info.get('empty')}")

    inputs = report.get("inputs") or {}
    check("长音频被切片", int(inputs.get("sliced", 0)) >= 1,
          f"sliced={inputs.get('sliced')} pieces={inputs.get('pieces')}")
    check("切片确实产出了多条片段", int(inputs.get("pieces", 0)) >= 10,
          f"pieces={inputs.get('pieces')}")

    data = report.get("data") or {}
    ready = int(data.get("ready_after", 0))
    check("筛留样本数达到训练下限", ready >= OC.MIN_SAMPLES,
          f"ready={ready}（下限 {OC.MIN_SAMPLES}）")
    check("样本总时长被统计", float(data.get("minutes") or 0) > 2.0,
          f"{data.get('minutes')} 分钟")
    check("train/val 已划分", bool(data.get("split")),
          str(data.get("split")))
    check("验证集非空（早停与保险库才有依据）",
          int((data.get("split") or {}).get("val", 0)) >= 1,
          str(data.get("split")))

    # =====================================================================
    head("[E] 自动调参的裁决")
    # =====================================================================
    cands = report.get("candidates") or []
    check("候选配置被逐个真实预检", len(cands) >= 2, f"{len(cands)} 个候选")
    check("有候选被采用", any(c.get("chosen") for c in cands))
    check("被采用的候选带完整参数（rank/alpha/lr/epochs/步数/显存）",
          all(c.get("rank") and c.get("alpha") and c.get("lr")
              and c.get("epochs") and c.get("total_steps")
              and c.get("est_vram_gb") for c in cands if c.get("chosen")),
          str([{k: c.get(k) for k in ("arch", "rank", "alpha", "lr",
                                      "epochs", "total_steps", "eval_every")}
               for c in cands if c.get("chosen")]))
    check("评估间隔按真实步数推导（不是照抄预设的 50/100/200）",
          all(int(c.get("eval_every") or -1) != 100
              for c in cands if c.get("chosen") and c.get("total_steps")),
          str([c.get("eval_every") for c in cands if c.get("chosen")]))
    check("GPT 与 CFM 都选定了一套配置",
          {"gpt", "cfm"} <= set((report.get("chosen") or {}).keys()),
          str(list((report.get("chosen") or {}).keys())))

    # =====================================================================
    head("[F] 训练结果与泛化保护")
    # =====================================================================
    training = report.get("training") or []
    check("两个目标都完成了训练", len(training) == 2, f"{len(training)} 个 run")
    for t in training:
        check(f"{t.get('arch')} 训练 ok 且产出了 adapter",
              bool(t.get("ok")) and int(t.get("n_checkpoints") or 0) >= 1,
              f"run={t.get('run')} steps={t.get('steps')} "
              f"ckpts={t.get('n_checkpoints')} "
              f"val {t.get('first_val')} → {t.get('best_val')}")
    check("val loss 相比底座有下降（学到了东西）",
          any((t.get("improved") or 0) > 0 for t in training),
          str([(t.get("arch"), t.get("improved")) for t in training]))
    check("峰值显存有记录（说明是量化过的真机训练）",
          all(float(t.get("vram_peak_gb") or 0) > 0 for t in training),
          str([(t.get("arch"), t.get("vram_peak_gb")) for t in training]))
    check("峰值显存没有失控（< 8 GB）",
          all(float(t.get("vram_peak_gb") or 0) < 8.0 for t in training))

    # 底座只读快照必须仍然有效 —— 训练不许碰 checkpoints/
    from webui_app.training import guard as GD
    g = GD.BaseGuard()
    vr = g.verify(hashes=False)
    check("泛化保护：训练全程底座逐字节未被改动",
          bool(vr.ok), f"checked={vr.checked} changed={len(vr.changed)} "
                       f"missing={len(vr.missing)}")

    # =====================================================================
    head("[G] 择优与交付")
    # =====================================================================
    ranking = report.get("ranking") or []
    check("候选档位进入了真机打分", len(ranking) >= 2, f"{len(ranking)} 个候选")
    check("每个候选都有 reward 与评价条数",
          all(r.get("reward") is not None and int(r.get("n") or 0) > 0
              for r in ranking),
          str([(r.get("run"), r.get("checkpoint"), r.get("reward"), r.get("n"))
               for r in ranking]))
    check("名次按 reward 从高到低",
          all(float(ranking[i]["reward"]) >= float(ranking[i + 1]["reward"])
              for i in range(len(ranking) - 1)),
          str([r.get("reward") for r in ranking]))
    check("reward 在合理区间（不是全 0 也不是 >1）",
          all(0.0 < float(r["reward"]) <= 1.0 for r in ranking),
          str([r.get("reward") for r in ranking]))
    check("WER 与声纹相似度都有值",
          all(r.get("wer") is not None and r.get("sim") is not None
              for r in ranking))

    best = report.get("best") or {}
    check("选出了推荐模型", bool(best.get("run")), str(best.get("run")))
    check("推荐模型已激活（adapter 同步到位）",
          bool(best.get("activated")), str(best.get("activate_error", "")))
    if best.get("run"):
        adir = RN.adapter_dir(best["run"])
        check("激活后的 adapter 目录存在且含 PEFT 权重",
              os.path.isdir(adir) and any(
                  f.startswith(("adapter_model", "adapter_config"))
                  for f in os.listdir(adir)),
              adir)
        # 推理页要能直接选到它 —— 这是「交付」的实际含义
        runs_with_adapter = [r.name for r in RN.list_runs() if r.has_adapter]
        check("推荐模型出现在推理页的 LoRA 列表里",
              best["run"] in runs_with_adapter, str(runs_with_adapter))

    # 档位数量符合 top_k 的承诺（「三个或一个」）
    per_run = {}
    for r in ranking:
        per_run[r["run"]] = per_run.get(r["run"], 0) + 1
    check(f"每个 run 参评档位数不超过 top_k={opts.top_k}",
          all(v <= opts.top_k for v in per_run.values()), str(per_run))
    check("至少有一个 run 提供了多个候选档位（择优才有意义）",
          any(v >= 2 for v in per_run.values()), str(per_run))

    # =====================================================================
    head("[H] 产物与报告")
    # =====================================================================
    out_dir = result.get("out_dir", "")
    check("报告目录已落盘", bool(out_dir) and os.path.isdir(out_dir), out_dir)
    for name in ("report.json", "report.md"):
        p = os.path.join(out_dir, name)
        check(f"产物 {name} 存在且非空",
              os.path.isfile(p) and os.path.getsize(p) > 200,
              f"{os.path.getsize(p) if os.path.isfile(p) else 0} 字节")
    check("回传的 markdown 非空（UI 面板要靠它渲染）", len(md) > 500,
          f"{len(md)} 字符")
    for token in ("一键三连完成", "自动调参：候选与裁决", "训练结果",
                  "择优（真机合成 + reward 打分）", "推荐使用"):
        check(f"报告含「{token}」", token in md)
    check("报告不含裸露的 None", "None" not in md)

    # 数据集侧最终状态
    st = DS.stats(ds_name)
    check("数据集最终有可训练样本", int(st.get("ready", 0)) >= OC.MIN_SAMPLES,
          f"ready={st['ready']} total={st['total']} {st['by_status']}")
    fstats = FT.stats(ds_name)
    check("离线特征已提取（训练时不再碰大模型）",
          int(fstats.get("ready_with_features", 0)) >= OC.MIN_SAMPLES,
          f"with_features={fstats.get('with_features')}")

    # ---- 择优阶段：打分器只加载一次并最终释放 ----
    rank_info = stages.get("rank") or {}
    check("择优的打分器在结束时已释放",
          rank_info.get("info", {}).get("scorer_released") is not False,
          str(rank_info.get("info", {})))

    # ---- 模型命名：报告里要能看出每个目标叫什么 ----
    names = report.get("names") or {}
    check("报告记录了每个目标的训练记录名", len(names) >= 1, str(names))
    if names:
        check("记录名可在 training_runs 里查到（便于区分管理）",
              all(os.path.isdir(RN.run_dir(n)) for n in names.values()),
              str(list(names.values())))

    # =====================================================================
    print("\n" + "=" * 70)
    print(f"  通过 {PASS} 项 · 失败 {FAIL} 项 · 总耗时 "
          f"{(time.perf_counter()-t_start)/60:.1f} 分钟")
    print("=" * 70)
    if FAILS:
        print("失败项：")
        for f in FAILS:
            print(f"  - {f}")

    # ---- 存档 ----
    try:
        os.makedirs(DOC, exist_ok=True)
        summary = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "passed": PASS, "failed": FAIL, "failures": FAILS,
            "seconds": round(time.perf_counter() - t_start, 1),
            "workflow_seconds": round(elapsed, 1),
            "dataset": ds_name,
            "inputs": inputs,
            "data": data,
            "candidates": cands,
            "chosen": report.get("chosen"),
            "training": training,
            "ranking": ranking,
            "best": best,
            "observed_engine_reqs": sorted(observed_reqs),
            "out_dir": out_dir,
        }
        with open(os.path.join(DOC, "oneclick_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(DOC, "oneclick_report.md"), "w",
                  encoding="utf-8") as f:
            f.write("# 一键三连 · 真机验收报告\n\n")
            f.write(f"- 时间：{summary['time']} · 结论："
                    f"**{'✅ 通过' if not FAIL else '🔴 有失败项'}**"
                    f"（{PASS} 通过 / {FAIL} 失败）\n")
            f.write(f"- 流水线耗时：{elapsed/60:.1f} 分钟"
                    f"（脚本总计 {(time.perf_counter()-t_start)/60:.1f} 分钟）\n")
            f.write(f"- 数据集：`{ds_name}` · 输入 {len(collect['files'])} 个文件"
                    f"（1 条长音频 + {SHORT_COUNT} 条短音频）\n")
            f.write(f"- 留用样本 {ready} 条 · 约 {data.get('minutes')} 分钟\n")
            f.write(f"- 动态引擎互斥观测到：{sorted(observed_reqs)}\n\n")
            f.write("## 自动调参裁决\n\n")
            f.write("| 目标 | 预设 | rank | alpha | lr | epochs | 步数 | "
                    "评估间隔 | 预估显存 | 裁决 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for c in cands:
                v = "✅ 采用" if c.get("chosen") else f"❌ {c.get('reject','')}"
                f.write(f"| {c.get('arch')} | {c.get('preset')} | {c.get('rank')} "
                        f"| {c.get('alpha')} | {c.get('lr')} | {c.get('epochs')} "
                        f"| {c.get('total_steps')} | {c.get('eval_every')} "
                        f"| {c.get('est_vram_gb')} GB | {v} |\n")
            f.write("\n## 训练结果\n\n")
            f.write("| 目标 | run | steps | val（底座 → 最好） | 改善 | 峰值显存 | 档位 |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for t in training:
                f.write(f"| {t.get('arch')} | `{t.get('run')}` | {t.get('steps')} "
                        f"| {t.get('first_val')} → **{t.get('best_val')}** "
                        f"| {t.get('improved')} | {t.get('vram_peak_gb')} GB "
                        f"| {t.get('n_checkpoints')} |\n")
            f.write("\n## 择优排名\n\n")
            f.write("| 名次 | run | 档位 | val loss | reward | WER | 声纹相似 | 条数 |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in ranking:
                f.write(f"| {r.get('place')} | `{r.get('run')}` "
                        f"| {r.get('checkpoint')} | {r.get('val')} "
                        f"| **{r.get('reward')}** | {r.get('wer')} "
                        f"| {r.get('sim')} | {r.get('n')} |\n")
            f.write(f"\n## 推荐模型\n\n```json\n"
                    + json.dumps(best, ensure_ascii=False, indent=2)
                    + "\n```\n")
            if FAILS:
                f.write("\n## 失败项\n\n" + "\n".join(f"- {x}" for x in FAILS)
                        + "\n")
        print(f"\n报告已存档：{os.path.join(DOC, 'oneclick_report.md')}")
    except Exception as e:
        print(f"（报告存档失败：{type(e).__name__}: {e}）")

    # ---- 清理 ----
    if keep:
        print(f"\n--keep：保留全部中间产物\n  {WORK}\n  数据集 {ds_name}\n"
              f"  训练记录 {os.path.join(RN.root())}")
    else:
        # 合成的语料故意留着：重跑时会被复用（每句约 3~5 秒，攒起来很费时间），
        # 想彻底清掉就删 outputs/oneclick_accept/ 整个目录。
        sig = os.path.join(WORK, "long_input.wav")
        if os.path.isfile(sig):
            os.remove(sig)
        print(f"\n长音频输入已清理；合成语料保留在 {raw_dir}（重跑可复用）")
        print(f"数据集 {ds_name} 与训练记录保留，供人工试听复核")

    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
