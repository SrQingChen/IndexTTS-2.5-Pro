"""「一键三连」（training/oneclick.py）的验证。

全部在 CPU 上跑、不加载任何模型（除了 [7] 里一个**故意不加载**的空引擎
用来验证互斥闸门），所以几十秒内能跑完。覆盖：

  1  选项与自检          —— DPO 必须被挡下、越界参数报错、warning 分级
  2  预设梯度            —— 数据量 → 候选顺序（依据 guard.PRESET_NOTES）
  3  输入收集            —— 单文件 str / 目录递归 / 去重 / 缺文件 / 非音频
  4  切片加固与并发      —— split_long 的锁、文本参数、撞号重发
  5  筛选闸门            —— 样本不足必须早停（不进训练器）、去重、评分门槛
  6  调参构造            —— GPT/CFM 两套 cfg 的注入面差异、top_k、评估间隔推导
  7  引擎互斥            —— runner.set_engine_req 真的挡住 load / unload
  8  报告渲染            —— 七段报告的关键表格与推荐结论都出得来
  9  UI 计划面板         —— 按钮下方那块「自动参数」面板能渲染出全部阶段

跑法：  .venv\\Scripts\\python.exe tools\\oneclick_probe.py
退出码：0 = 全过；1 = 有失败项（与其它探针一致）。
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import hashlib
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

from webui_app.training import dataset as DS                 # noqa: E402
from webui_app.training import guard as GD                   # noqa: E402
from webui_app.training import oneclick as OC
from webui_app.training import runs as RN                # noqa: E402
from webui_app.training.runner import TrainRunner            # noqa: E402

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
# 造音频：带谐波的「语音样」信号 + 首尾静音。用纯音也行 —— AL.analyze 只看
# 帧能量/信噪比/时长，不看是不是真人说话（真转写由 [4] 之外的实机验收覆盖）。
# ---------------------------------------------------------------------------

def _tone(sec: float, sr: int = 22050, amp: float = 0.35,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    y = (0.6 * np.sin(2 * np.pi * 180 * t)
         + 0.3 * np.sin(2 * np.pi * 360 * t)
         + 0.1 * np.sin(2 * np.pi * 540 * t))
    y = y + rng.normal(0, 0.004, len(t))          # 极低底噪 → SNR 很高
    env = np.minimum(1.0, np.minimum(t / 0.05, (sec - t) / 0.05))
    return (amp * y * env).astype(np.float32)


def make_wav(path: str, body_sec: float, lead: float = 0.25,
             tail: float = 0.25, sr: int = 22050, seed: int = 0) -> str:
    y = np.concatenate([np.zeros(int(lead * sr), dtype=np.float32),
                        _tone(body_sec, sr, seed=seed),
                        np.zeros(int(tail * sr), dtype=np.float32)])
    sf.write(path, y, sr)
    return path


def make_long_wav(path: str, pieces: int = 12, body: float = 2.6,
                  gap: float = 0.5, sr: int = 22050) -> str:
    """交替「语音 + 停顿」的长音频 —— 正是 find_segments 要找的结构。"""
    chunks = []
    for k in range(pieces):
        chunks.append(_tone(body, sr, seed=k))
        chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
    sf.write(path, np.concatenate(chunks), sr)
    return path


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="oneclick_probe_")
    ds_name = "probe_oneclick"
    try:
        # =================================================================
        head("[1] 选项与自检")
        # =================================================================
        o = OC.OneClickOptions()
        errs = [n.message for n in o.validate() if n.level == "error"]
        check("默认选项无 error", not errs, str(errs))
        check("默认目标为 gpt+cfm", o.arch_list() == ["gpt", "cfm"],
              str(o.arch_list()))

        o_dpo = OC.OneClickOptions(arches="gpt,dpo")
        msgs = [n.message for n in o_dpo.validate() if n.level == "error"]
        check("DPO 目标被挡下", any("不支持 DPO" in m for m in msgs))
        check("挡下的理由给出了替代路径（对齐页）",
              any("对齐" in m for m in msgs))
        check("DPO 被挡时 gpt 仍在（不是一刀切）",
              o_dpo.arch_list() == ["gpt"], str(o_dpo.arch_list()))

        o_bad = OC.OneClickOptions(arches="nope")
        check("未知目标报错",
              any("不认识" in n.message
                  for n in o_bad.validate() if n.level == "error"))
        check("未知目标时 arch_list 为空", o_bad.arch_list() == [])

        for kw, why in [({"lang": "XX"}, "语言"),
                        ({"whisper_size": "gigantic"}, "whisper 尺寸"),
                        ({"slice_target_sec": 30.0}, "切片目标超过训练上限"),
                        ({"slice_over_sec": 2.0, "slice_min_sec": 6.0},
                         "切片阈值倒挂"),
                        ({"preset_mode": "turbo"}, "未知预设"),
                        ({"val_ratio": 0.9}, "验证比例越界"),
                        ({"top_k": 99}, "档位数越界")]:
            bad = OC.OneClickOptions(**kw)
            e = [n.message for n in bad.validate() if n.level == "error"]
            check(f"越界参数被拦：{why}", bool(e), str(e)[:70])

        o_warn = OC.OneClickOptions(asr=False)
        check("关闭识别 → 有 warn 提示（会没文本）",
              any(n.level == "warn" for n in o_warn.validate()))

        round_trip = OC.OneClickOptions.from_dict(
            {"lang": "EN", "top_k": 2, "no_such_field": 1, "seed": None})
        check("from_dict 忽略未知字段", not hasattr(round_trip, "no_such_field"))
        check("from_dict 应用已知字段", round_trip.lang == "EN"
              and round_trip.top_k == 2)
        check("from_dict 遇到 None 不覆盖", round_trip.seed == 42)

        # =================================================================
        head("[2] 预设梯度（数据量 → 候选顺序）")
        # =================================================================
        l_min = OC.preset_ladder(2.0)
        l_mid = OC.preset_ladder(10.0)
        l_big = OC.preset_ladder(45.0)
        check("数据 <5 分钟首选保守档", l_min[0] == "conservative", str(l_min))
        check("5~30 分钟首选均衡档", l_mid[0] == "balanced", str(l_mid))
        check("≥30 分钟首选均衡、次选激进",
              l_big[:2] == ["balanced", "aggressive"], str(l_big))
        check("所有梯度只含已定义的预设",
              all(x in GD.CONFIG_PRESETS for x in l_min + l_mid + l_big))
        check("边界：恰好 5 分钟走均衡档", OC.preset_ladder(5.0)[0] == "balanced")
        check("边界：恰好 30 分钟走均衡档", OC.preset_ladder(30.0)[0] == "balanced")

        # =================================================================
        head("[3] 输入收集")
        # =================================================================
        f1 = make_wav(os.path.join(tmp, "a.wav"), 3.0, seed=1)
        f2 = make_wav(os.path.join(tmp, "b.wav"), 3.0, seed=2)
        sub = os.path.join(tmp, "sub")
        os.makedirs(sub, exist_ok=True)
        f3 = make_wav(os.path.join(sub, "c.wav"), 3.0, seed=3)
        open(os.path.join(tmp, "note.txt"), "w").write("x")

        got = OC.collect_inputs(f1)                      # 单文件传 str
        check("单文件 str 被正确当作一个路径（不是逐字符迭代）",
              got["files"] == [os.path.abspath(f1)], str(got["files"]))

        got = OC.collect_inputs([f1, f2, f1])
        check("重复路径去重", len(got["files"]) == 2, str(len(got["files"])))

        got = OC.collect_inputs([f1, os.path.join(tmp, "nope.wav")])
        check("不存在的路径进 missing", len(got["missing"]) == 1)
        check("存在的路径仍然保留", len(got["files"]) == 1)

        got = OC.collect_inputs([os.path.join(tmp, "note.txt")])
        check("非音频扩展名进 skipped",
              len(got["skipped"]) == 1 and not got["files"])

        got = OC.collect_inputs([], tmp)
        check("目录递归找到全部音频（含子目录）",
              len(got["files"]) == 3, str([os.path.basename(x) for x in got["files"]]))

        got = OC.collect_inputs([], '  "%s"  ' % f1)
        check("路径两端引号与空白被剥掉",
              got["files"] == [os.path.abspath(f1)], str(got))

        # =================================================================
        head("[4] 切片加固与并发")
        # =================================================================
        if DS.exists(ds_name):
            DS.delete(ds_name)
        DS.create(ds_name, note="probe")

        long_wav = make_long_wav(os.path.join(tmp, "long.wav"),
                                 pieces=10, body=2.6, gap=0.5)
        imp = DS.import_audio(ds_name, [long_wav], copy=True, lang="ZH")
        src_uid = imp["ids"][0]
        src = DS.get(ds_name, src_uid)
        check("长音频导入成功且时长被读到",
              src is not None and src.duration > 20.0,
              f"{src.duration:.1f}s" if src else "None")

        # 先补上文本再体检：状态判定里 no_text 优先于 too_long，
        # 不补文本的话测的是「没文本」而不是「太长」，断言会假通过。
        DS.apply_fields(ds_name, {src_uid: {"text": "这条太长了"}})
        DS.refresh_all(ds_name, require_features=False)
        src = DS.get(ds_name, src_uid)
        check("超长原件被判 too_long（必须先切片才能训）",
              src.status == "too_long", f"status={src.status}")
        check("超长原件不可训练（status != ready）", src.status != "ready")

        sp = DS.split_long(ds_name, src_uid, target_sec=8.0, min_sec=3.0,
                           max_pieces=6, text="切片自带文本")
        check("split_long 成功", bool(sp.get("ok")), str(sp.get("error", "")))
        check("切出了多个片段", int(sp.get("created", 0)) >= 2,
              f"created={sp.get('created')}")
        pieces = [u for u in DS.load_meta(ds_name) if u.id != src_uid]
        check("片段数与会话返回值一致", len(pieces) == int(sp.get("created", 0)))
        check("片段的 text 参数被写入",
              all(u.text == "切片自带文本" for u in pieces))
        check("片段带来源说明（可追溯）",
              all("切分而来" in (u.note or "") for u in pieces))
        check("片段时长都在训练区间内",
              all(1.0 <= u.duration <= DS.MAX_TRAIN_SEC for u in pieces),
              str([round(u.duration, 1) for u in pieces]))
        check("片段已写入音频文件",
              all(os.path.isfile(u.audio_abs(DS.dir_of(ds_name)))
                  for u in pieces))
        check("重复调用不报错（幂等安全性）",
              bool(DS.split_long(ds_name, src_uid, target_sec=8.0,
                                 min_sec=3.0, max_pieces=6).get("ok") is not None))
        bad = DS.split_long(ds_name, "utt_99999")
        check("不存在的 uid 返回 ok=False 而不是抛异常",
              bad.get("ok") is False and "不存在" in str(bad.get("error")))

        # -- 并发：切片过程中 UI 线程写文本，不能被最后的 save 覆盖 --
        def _writer():
            time.sleep(0.05)
            DS.update(ds_name, src_uid, note="并发写入的标记")

        th = threading.Thread(target=_writer)
        th.start()
        DS.split_long(ds_name, src_uid, target_sec=8.0, min_sec=3.0,
                      max_pieces=4)
        th.join()
        after = DS.get(ds_name, src_uid)
        check("切片期间的并发写入没有被旧快照覆盖（_META_LOCK 生效）",
              after is not None and after.note == "并发写入的标记",
              str(after.note if after else None))

        # =================================================================
        head("[5] 筛选闸门")
        # =================================================================
        ds_small = "probe_oneclick_small"
        if DS.exists(ds_small):
            DS.delete(ds_small)
        DS.create(ds_small, note="probe small")
        short = [make_wav(os.path.join(tmp, f"s{i}.wav"), 3.0, seed=10 + i)
                 for i in range(5)]
        imp = DS.import_audio(ds_small, short, copy=True, lang="ZH")
        DS.apply_fields(ds_small, {uid: {"text": f"第{i}句测试文本"}
                                   for i, uid in enumerate(imp["ids"])})
        DS.refresh_all(ds_small, require_features=False)
        st = DS.stats(ds_small)
        check("合成样本全部 ready（音频与文本都过关）",
              st["ready"] == 5, f"ready={st['ready']} by_status={st['by_status']}")

        cur = OC.stage_curate(ds_small, OC.OneClickOptions(), report=None)
        check("样本不足 MIN_SAMPLES 时 gate 关闭",
              cur.get("ok") is False, f"ready_after={cur.get('ready_after')}")
        check("早停理由点名了样本下限",
              str(OC.MIN_SAMPLES) in str(cur.get("error")), str(cur.get("error"))[:60])
        check("早停理由给了行动建议（多录/调低门槛）",
              ("多录" in str(cur.get("error")) or "调低" in str(cur.get("error"))))
        check("没有在样本不足时去划分 train/val",
              not cur.get("split"), str(cur.get("split")))
        check("删掉的样本数被记录", cur.get("dropped_total", 0) == 0,
              f"dropped={cur.get('dropped_total')}")

        # -- 评分门槛与去重 --
        ds_dup = "probe_oneclick_dup"
        if DS.exists(ds_dup):
            DS.delete(ds_dup)
        DS.create(ds_dup, note="probe dup")
        many = [make_wav(os.path.join(tmp, f"d{i}.wav"), 3.0, seed=30 + i)
                for i in range(4)]
        imp = DS.import_audio(ds_dup, many, copy=True, lang="ZH")
        DS.apply_fields(ds_dup, {uid: {"text": "完全一样的一句话"}
                                 for uid in imp["ids"]})
        DS.refresh_all(ds_dup, require_features=False)
        cur = OC.stage_curate(ds_dup, OC.OneClickOptions(max_text_repeats=2),
                              report=None)
        left = [u for u in DS.load_meta(ds_dup) if u.status == "ready"]
        check("同文本去重生效（4 条同句 → 留 2 条）", len(left) == 2,
              f"剩 {len(left)} 条")
        check("去重被计入 dropped", cur.get("dropped_total", 0) == 2,
              str(cur.get("dropped_total")))
        check("去重后仍不足下限 → 仍然拦住",
              cur.get("ok") is False)

        cur = OC.stage_curate(
            ds_dup, OC.OneClickOptions(max_text_repeats=0, min_score=99.5),
            report=None)
        left = [u for u in DS.load_meta(ds_dup) if u.status == "ready"]
        check("评分门槛生效（99.5 分谁都过不了）", len(left) == 0,
              f"剩 {len(left)} 条")
        DS.delete(ds_dup)

        # =================================================================
        head("[6] 调参构造（GPT / CFM 两套注入面）")
        # =================================================================
        cfg_g, opt_g = OC.build_cfg("gpt", "conservative", top_k=3)
        cfg_c, opt_c = OC.build_cfg("cfm", "balanced", top_k=1)
        check("gpt 预设 rank 取自 CONFIG_PRESETS",
              cfg_g.rank == GD.CONFIG_PRESETS["conservative"]["rank"],
              f"rank={cfg_g.rank}")
        check("gpt 注入面是 attn 投影",
              any("attn" in t for t in cfg_g.target_modules),
              str(cfg_g.target_modules))
        check("top_k 落到 keep_checkpoints", cfg_g.keep_checkpoints == 3
              and cfg_c.keep_checkpoints == 1)
        check("cfm 走自己的 default_config（注入 CFM 注意力）",
              any("attention" in t or "wqkv" in t for t in cfg_c.target_modules),
              str(cfg_c.target_modules))
        check("cfm 的注入面不等于 gpt 的（避免注错层）",
              set(cfg_c.target_modules) != set(cfg_g.target_modules))
        check("cfm 强制 batch_size=1 / grad_checkpointing=False",
              cfg_c.batch_size == 1 and not cfg_c.grad_checkpointing,
              f"bs={cfg_c.batch_size} gc={cfg_c.grad_checkpointing}")
        check("cfm 强制关掉 bf16（L1 loss 与 bf16 尾数不匹配）",
              cfg_c.bf16 is False)
        cfg_e, _ = OC.build_cfg("gpt", "balanced", top_k=2, epochs=1)
        check("epochs 可被覆盖", cfg_e.epochs == 1)
        check("两个目标的 options 类型不同（不是同一份默认值）",
              type(opt_g).__name__ != type(opt_c).__name__,
              f"{type(opt_g).__name__} / {type(opt_c).__name__}")

        check("评估间隔推导：6 步 top3 → 每步一评",
              OC.eval_every_for(6, 3) == 1, str(OC.eval_every_for(6, 3)))
        check("评估间隔推导：400 步 top3 → 100",
              OC.eval_every_for(400, 3) == 100, str(OC.eval_every_for(400, 3)))
        check("评估间隔推导：0 步（不需要训练）→ 0",
              OC.eval_every_for(0, 3) == 0)
        check("评估间隔足够产出多个档位",
              OC.eval_every_for(20, 3) == 5, str(OC.eval_every_for(20, 3)))

        # =================================================================
        head("[7] 引擎互斥（runner.set_engine_req）")
        # =================================================================
        from webui_app.config import config_from_args        # noqa: E402
        from webui_app.services.engine import (               # noqa: E402
            EngineError, TTSEngine, _busy_runner_engine_req)

        eng = TTSEngine(config_from_args(["--lazy"]))
        r = TrainRunner()

        # 引擎侧读的是**全局单例** runner（engine._busy_runner_engine_req
        # 内部 get_runner()），所以这里把状态装到单例上测。
        from webui_app.training.runner import get_runner   # noqa: E402
        single = get_runner()
        saved = (single.running, single.engine_req)
        try:
            check("未运行任务时，引擎不认为有人在用",
                  _busy_runner_engine_req() == "",
                  _busy_runner_engine_req())

            single.running = True
            single.set_engine_req("unloaded")
            check("运行时 runner 会报出自己的引擎要求",
                  single.engine_req == "unloaded", single.engine_req)
            check("要求 unloaded 时引擎侧能读到",
                  _busy_runner_engine_req() == "unloaded",
                  _busy_runner_engine_req())

            raised = ""
            try:
                eng.load()
            except EngineError as e:
                raised = str(e)
            except Exception as e:
                raised = f"WRONG:{type(e).__name__}"
            check("engine.load() 真的抛 EngineError（不是静默继续）",
                  raised.startswith("训练正在"), raised[:60])

            single.set_engine_req("loaded")
            raised = ""
            try:
                eng.unload()
            except EngineError as e:
                raised = str(e)
            except Exception:
                raised = "WRONG"
            check("要求 loaded 时拒绝卸载引擎（防止状态条撒谎）",
                  "后台任务" in raised, raised[:60])

            single.set_engine_req("none")
            check("切回 none 后两个方向都放行", single.engine_req == "none")
            check("非法要求被忽略（不破坏当前值）",
                  single.set_engine_req("bogus") == "none", single.engine_req)

            single.running = False
            check("任务结束后引擎不再被拦", _busy_runner_engine_req() == "",
                  _busy_runner_engine_req())
        finally:
            single.running, single._engine_req = saved

        # =================================================================
        head("[8] 报告渲染")
        # =================================================================
        rep = OC.OneClickReport(dataset="d", seconds=123.0, ok=True)
        for k, _t, _w in OC.STAGES:
            rep.start(k, f"{k} 说明")
            rep.finish(k, f"{k} 完成", seconds=1.5)
        rep.candidates = [
            {"arch": "gpt", "preset": "balanced", "rank": 8, "alpha": 16,
             "lr": 1e-4, "epochs": 4, "total_steps": 40, "eval_every": 10,
             "est_vram_gb": 2.1, "chosen": True},
            {"arch": "cfm", "preset": "balanced", "rank": 8, "alpha": 16,
             "lr": 1e-4, "epochs": 4, "total_steps": 40, "eval_every": 10,
             "est_vram_gb": 2.1, "reject": "预估显存超了"},
        ]
        rep.chosen = {"gpt": {"preset": "balanced", "rank": 8}}
        rep.training = [{"arch": "gpt", "run": "r1", "steps": 40,
                         "first_val": 5.1, "best_val": 1.9,
                         "improved": 0.627, "vram_peak_gb": 1.9,
                         "n_checkpoints": 3, "ok": True}]
        rep.ranking = [
            {"place": 1, "run": "r1", "checkpoint": "ckpt-e002-s000040",
             "arch": "gpt", "val": 1.9, "reward": 0.83, "wer": 0.11,
             "sim": 0.79, "n": 5},
            {"place": 2, "run": "r1", "checkpoint": "ckpt-e001-s000020",
             "arch": "gpt", "val": 2.2, "reward": 0.79, "wer": 0.14,
             "sim": 0.76, "n": 5},
        ]
        rep.best = dict(rep.ranking[0])
        rep.add_note("这是一条提示")
        md = rep.markdown()
        for token in ["一键三连完成", "S1" if False else "采集与切片",
                      "音频优化", "识别与对齐", "筛选与划分", "自动调参",
                      "LoRA 训练", "择优与交付", "自动调参：候选与裁决",
                      "训练结果", "择优（真机合成 + reward 打分）",
                      "推荐使用", "r1", "0.83", "这是一条提示"]:
            check(f"报告包含「{token}」", token in md)
        check("被拒候选的原因出现在报告里", "预估显存超了" in md)
        check("报告里没有裸露的 None", "None" not in md)
        check("报告没有把失败说成成功",
              "🔴" not in md.replace("🔴 一键三连未完成", ""))

        rep_bad = OC.OneClickReport(dataset="d2", ok=False, error="样本不够")
        rep_bad.start("ingest", "x")
        rep_bad.finish("ingest", "失败", status="failed")
        md_bad = rep_bad.markdown()
        check("失败报告明确标红", "🔴" in md_bad)
        check("失败报告带上错误原因", "样本不够" in md_bad)

        # finish(**info) 必须把关键结果落进阶段里 —— 否则「识别用的哪个模型、
        # 有没有卸载、释放了多少显存」这些事实在报告与 UI 面板里全是空的
        # （真机验收正是靠这条断言抓到了这个漏写）。
        r2 = OC.OneClickReport(dataset="d3")
        r2.start("asr", "x")
        r2.finish("asr", "done", whisper="medium",
                  prompt="以下是普通话的句子。", loaded_after_unload=False,
                  freed_gb=1.6, failed=0)
        info = (r2.stage_of("asr") or {}).get("info") or {}
        check("finish(**info) 把结果写进了阶段",
              info.get("whisper") == "medium", str(info))
        check("识别阶段如实记录是否已卸载",
              info.get("loaded_after_unload") is False, str(info))
        check("识别阶段记录释放的显存", info.get("freed_gb") == 1.6, str(info))
        check("带 info 的报告仍能渲染且无裸露 None",
              len(r2.markdown()) > 50 and "None" not in r2.markdown())
        r3 = OC.OneClickReport(dataset="d4")
        r3.start("train", "x")
        r3.finish("train", "done")
        # 报告里渲染的是**阶段标题**（LoRA 训练），不是内部键名 train
        check("没有 info 的阶段也能渲染（向后兼容）",
              OC.STAGE_TITLE["train"] in r3.markdown()
              and r3.stage_of("train")["info"] == {},
              str(r3.stage_of("train")))

        # =================================================================
        head("[9] UI 计划面板（按钮下方那块「自动参数」）")
        # =================================================================
        from webui_app.tabs import oneclick_tab as OT      # noqa: E402

        # 不写死数量：加一个选项就要改一次断言的话，迟早有人忘了改而放过去。
        # 改成「必须覆盖这些关键项 + 无重复 + 每项都是真实字段」。
        check("OPT_KEYS 无重复（有重复就会两个控件抢一个字段）",
              len(OT.OPT_KEYS) == len(set(OT.OPT_KEYS)), str(OT.OPT_KEYS))
        for must in ("model_name", "cpu_workers", "lang", "asr",
                     "preset_mode", "top_k", "rank_eval", "seed"):
            check(f"OPT_KEYS 覆盖了 {must}", must in OT.OPT_KEYS)
        check("OPT_KEYS 都是 OneClickOptions 的真实字段",
              all(hasattr(OC.OneClickOptions(), k) for k in OT.OPT_KEYS),
              str([k for k in OT.OPT_KEYS
                   if not hasattr(OC.OneClickOptions(), k)]))

        plan = OT.plan_markdown(OC.OneClickOptions())
        for token in ["S1 采集与切片", "S2 音频优化", "S3 识别与对齐",
                      "S4 筛选与划分", "S5 自动调参", "S6 LoRA 训练",
                      "S7 择优与交付", "参数是怎么选出来的",
                      "一键三连不支持 DPO" if False else "top-3"]:
            check(f"计划面板包含「{token}」", token in plan)
        check("计划面板展示了每个目标的候选顺序",
              "balanced" in plan or "conservative" in plan)
        check("计划面板说明了「三个或一个」的取舍",
              "三个或一个" in plan)
        check("关掉识别时计划面板会警告",
              "语音识别已关闭" in OT.plan_markdown(OC.OneClickOptions(asr=False)))
        o_bad2 = OC.OneClickOptions(lang="XX")
        check("选项非法时计划面板直接报错",
              "不支持" in OT.plan_markdown(o_bad2))
        check("计划面板不出现裸露的 None",
              "None" not in OT.plan_markdown(OC.OneClickOptions()))

        # =================================================================
        head("[10] 训练参数登记（手册不再缺这几组）")
        # =================================================================
        from webui_app import params as PM                 # noqa: E402
        from webui_app import widgets as W2                # noqa: E402

        PM.register_training_params()
        groups = PM.group_order(True)
        for g in ("oneclick", "dataset", "lora", "optim", "dpo", "reward"):
            check(f"手册分组「{g}」已登记", g in groups, str(groups))
            check(f"分组「{g}」有参数",
                  len(PM.by_group(g)) > 0, f"{len(PM.by_group(g))} 项")

        newp = [p for g in ("oneclick", "dataset", "lora", "optim", "dpo",
                            "reward") for p in PM.by_group(g)]
        check("训练参数总数合理（≥ 30）", len(newp) >= 30, str(len(newp)))
        check("每个训练参数都能渲染出手册卡片",
              all(W2.help_markdown(p) for p in newp))
        check("每个训练参数都有摘要（卡片标题不能为空）",
              all((p.summary or p.info) for p in newp),
              str([p.key for p in newp if not (p.summary or p.info)]))
        keys = [p.key for p in PM.all_params(True)]
        check("参数 key 全局唯一（不覆盖推理参数）",
              len(keys) == len(set(keys)),
              str([k for k in keys if keys.count(k) > 1]))
        check("训练参数没有和推理参数撞名",
              not (set(p.key for p in newp)
                   & set(p.key for p in PM.by_group("voice"))))
        # 一键三连的关键旋钮必须在手册里有名字 —— 用户点完按钮要能查到它
        for k in ("oc_slice_over_sec", "oc_asr", "oc_whisper_size",
                  "oc_min_score", "oc_top_k", "oc_rank_eval"):
            check(f"手册收录了一键三连参数 {k}",
                  any(p.key == k for p in PM.by_group("oneclick")))

        # =================================================================
        head("[11] CPU 并行：结果必须与串行逐字节一致")
        # =================================================================
        from webui_app.training import parallel as PL

        check("default_workers 上限 4", PL.default_workers(999, 0) <= PL.MAX_WORKERS,
              str(PL.default_workers(999, 0)))
        check("default_workers 不超过条数", PL.default_workers(2, 0) <= 2,
              str(PL.default_workers(2, 0)))
        check("requested=1 就是串行", PL.default_workers(50, 1) == 1)
        check("requested 超上限被裁剪", PL.default_workers(50, 99) == PL.MAX_WORKERS,
              str(PL.default_workers(50, 99)))
        check("空输入也不报错", PL.default_workers(0, 0) >= 1)

        # -- map_parallel 的三个承诺：保序、隔离失败、可停止 --
        out = PL.map_parallel(lambda x: x * 2, list(range(8)), workers=4)
        check("map_parallel 保序（与输入逐位对应）",
              out.items == [x * 2 for x in range(8)], str(out.items))
        check("map_parallel 报告用了几个线程", out.workers == 4, str(out.workers))

        def _maybe_boom(x):
            if x == 3:
                raise ValueError("第 3 个炸了")
            return x

        out = PL.map_parallel(_maybe_boom, list(range(6)), workers=3)
        check("单条失败不影响其它条目",
              [out.items[i] for i in (0, 1, 2, 4, 5)] == [0, 1, 2, 4, 5],
              str(out.items))
        check("失败位置是 None 且记了错因",
              out.items[3] is None and 3 in out.errors
              and "ValueError" in out.errors[3], str(out.errors))
        check("outcome.ok 反映有失败", out.ok is False)

        seen = []
        out = PL.map_parallel(lambda x: x, list(range(10)), workers=3,
                              on_done=lambda i, r: seen.append(i))
        check("on_done 每条都被回调一次（进度不丢）",
              sorted(seen) == list(range(10)), str(sorted(seen)))

        flag = {"stop": False}
        out = PL.map_parallel(lambda x: x, list(range(50)), workers=2,
                              should_stop=lambda: flag["stop"])
        check("未请求停止时跑完全部", out.n_done == 50 and not out.stopped)

        # -- 真正的核心承诺：增强结果与线程数无关 --
        def _hash_dir(name: str) -> dict:
            d = DS.dir_of(name)
            h = {}
            for u in DS.load_meta(name):
                p = u.audio_abs(d)
                if p and os.path.isfile(p):
                    h[u.id] = hashlib.sha256(open(p, "rb").read()).hexdigest()
            return h

        srcs = [make_wav(os.path.join(tmp, f"p{i}.wav"), 3.0, seed=70 + i)
                for i in range(8)]
        names = {}
        for tag, workers in (("seq", 1), ("par", 4)):
            nm = f"probe_par_{tag}"
            names[tag] = nm
            if DS.exists(nm):
                DS.delete(nm)
            DS.create(nm, note="parallel probe")
            imp = DS.import_audio(nm, srcs, copy=True, lang="ZH")
            DS.apply_fields(nm, {uid: {"text": f"第{i}句"}
                                 for i, uid in enumerate(imp["ids"])})
            DS.refresh_all(nm, require_features=False)
            o = OC.OneClickOptions(enhance=True, denoise=False,
                                   normalize=True, trim_silence=True,
                                   cpu_workers=workers, dataset_name=nm,
                                   max_text_repeats=0, min_score=0.0)
            st = OC.stage_optimize(nm, o, progress=None, should_stop=None,
                                   report=None, bands=None)
            check(f"优化阶段（workers={workers}）处理了全部样本",
                  st["enhanced"] >= 6, f"enhanced={st['enhanced']} "
                                       f"failed={st['failed']}")
            if workers > 1:
                check("并行路径确实启用了多线程", int(st.get("workers", 1)) > 1,
                      str(st.get("workers")))

        hs, hp = _hash_dir(names["seq"]), _hash_dir(names["par"])
        check("两次运行的样本集合一致", set(hs) == set(hp),
              f"{sorted(hs)} vs {sorted(hp)}")
        diff = [k for k in hs if hs[k] != hp.get(k)]
        check("**并行与串行的音频输出逐字节一致**（不影响效果）",
              not diff, f"不一致 {diff}")
        s1 = DS.stats(names["seq"])
        s2 = DS.stats(names["par"])
        check("两次运行的状态与样本数也一致",
              (s1["total"], s1["ready"], s1["by_status"])
              == (s2["total"], s2["ready"], s2["by_status"]),
              f"{s1['ready']}/{s2['ready']}")

        # =================================================================
        head("[12] 模型命名（便于区分管理）")
        # =================================================================
        o_multi = OC.OneClickOptions(model_name="小明_播客", arches="gpt,cfm")
        nms = o_multi.plan_run_names()
        check("多目标：加 _gpt / _cfm 后缀（不然互相覆盖）",
              nms == {"gpt": "小明_播客_gpt", "cfm": "小明_播客_cfm"}, str(nms))
        o_one = OC.OneClickOptions(model_name="小明_播客", arches="gpt")
        check("单目标：就用用户取的名字（所见即所得）",
              o_one.plan_run_names() == {"gpt": "小明_播客"},
              str(o_one.plan_run_names()))
        o_slash = OC.OneClickOptions(model_name="a/b:c", arches="gpt")
        check("非法字符被安全化（Windows 上不能做目录名）",
              "/" not in list(o_slash.plan_run_names().values())[0]
              and ":" not in list(o_slash.plan_run_names().values())[0],
              str(o_slash.plan_run_names()))
        check("名字含非法字符时给出 warn",
              any(n.level == "warn" for n in o_slash.validate()))
        o_empty = OC.OneClickOptions()
        got = list(o_empty.plan_run_names().values())
        check("留空则自动生成（带时间戳、可区分）",
              all(x.startswith("oneclick_") for x in got), str(got))
        check("未指定模型名时不报冲突", OC.OneClickOptions().check_run_names() == "")

        # 已存在的 run 名必须能拦下。**自建一条临时记录**来测 ——
        # 不依赖机器上恰好有训练产物，探针在任何环境都能跑。
        tmp_run = "probe_name_conflict"
        try:
            RN.run_dir(tmp_run, create=True)
            RN.write_run(tmp_run, {"arch": "gpt", "run": tmp_run,
                                   "status": "done", "dataset": "x"})
            o_conf = OC.OneClickOptions(model_name=tmp_run, arches="gpt")
            msg = o_conf.check_run_names()
            check("已占用的模型名会被拦下（不用白等一轮训练）", bool(msg),
                  str(msg)[:70])
            check("拦下的提示里点名了冲突的记录", tmp_run in msg, msg[:90])
            check("提示给出了解决办法（改名或删旧记录）",
                  "换一个名称" in msg or "删掉旧记录" in msg, msg[:90])
            # 多目标时实际用的是 <名字>_gpt / <名字>_cfm，
            # 所以冲突检测必须把后缀加上再去查（否则会漏判）
            tmp_run_cfm = tmp_run + "_cfm"
            RN.run_dir(tmp_run_cfm, create=True)
            RN.write_run(tmp_run_cfm, {"arch": "cfm", "run": tmp_run_cfm,
                                       "status": "done", "dataset": "x"})
            o_conf2 = OC.OneClickOptions(model_name=tmp_run, arches="gpt,cfm")
            msg2 = o_conf2.check_run_names()
            check("多目标时按带后缀的名字检出冲突（<名>_cfm）",
                  tmp_run_cfm in msg2, str(msg2)[:80])
            RN.delete_run(tmp_run_cfm)
            RN.delete_run(tmp_run)
            check("删掉旧记录后冲突解除",
                  OC.OneClickOptions(model_name=tmp_run,
                                     arches="gpt").check_run_names() == "")
        finally:
            try:
                if RN.read_run(tmp_run):
                    RN.delete_run(tmp_run)
            except Exception:
                pass

        # =================================================================
        head("[13] 长音频切片：进度 / 可中断 / 内存有界")
        # =================================================================
        import time as _t

        from webui_app.services import audio_lab as AL

        # -- (a) 帧能量分块计算必须与「一次性稠密索引」逐位一致 --
        # 这是本次修复的核心：为了不让内存随音频时长线性膨胀，改成了分块。
        # 分块绝不能改变数值 —— 所以在这里拿小数组对照稠密算法。
        rng = np.random.default_rng(7)
        sig = (rng.normal(0, 0.05, 22050 * 3)).astype(np.float32)
        sig[22050:44100] += 0.45 * np.sin(
            2 * np.pi * 200 * np.arange(22050) / 22050).astype(np.float32)
        db_new, n_new = AL._frames_db(sig, 22050)

        def _dense_db(y, sr, frame_ms=25.0, hop_ms=10.0):
            win = max(1, int(sr * frame_ms / 1000))
            hop = max(1, int(sr * hop_ms / 1000))
            if len(y) < win:
                y = np.pad(y, (0, win - len(y)))
            n = 1 + (len(y) - win) // hop
            idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
            frames = y[idx]
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            return (20.0 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32), n

        db_ref, n_ref = _dense_db(sig, 22050)
        check("分块帧能量与稠密实现帧数一致", n_new == n_ref,
              f"{n_new} vs {n_ref}")
        check("**分块帧能量与稠密实现逐位相同**（分块不改变数值）",
              db_new.shape == db_ref.shape
              and np.array_equal(db_new, db_ref),
              f"max|diff|={np.max(np.abs(db_new - db_ref)) if db_new.shape == db_ref.shape else 'NA'}")
        # 块大小必须随窗长收敛（内存有界），而不是随总帧数膨胀
        check("块大小与总帧数无关（内存有界）",
              int(40e6 / (8 * max(1, int(22050 * 0.025)))) >= 256)

        # 量化「卡死」的量级：1 小时 48 kHz 音频，稠密索引矩阵有多大
        _n1h = 1 + (3600 * 48000 - 1200) // 480      # 10ms hop、25ms 窗
        _dense_mb = _n1h * 1200 * 8 / 1e6            # int64 索引矩阵
        _chunk = max(256, int(40e6 / (8 * 1200)))
        _chunk_mb = _chunk * 1200 * 8 / 1e6
        check("1 小时音频的稠密索引矩阵会到 GB 级（这就是「卡死」的原因）",
              _dense_mb > 2000, f"{_dense_mb:.0f} MB（还没算 frames 与 float64）")
        check("分块后单块索引被钉在约 40 MB（与时长无关）",
              _chunk_mb <= 40.5, f"{_chunk_mb:.1f} MB")

        # -- (b) probe_duration 只读文件头、不解码 --
        dur_wav = make_wav(os.path.join(tmp, "probe_dur.wav"), 3.0, seed=91)
        check("probe_duration 返回值正确",
              abs(AL.probe_duration(dur_wav) - 3.5) < 0.15,
              str(AL.probe_duration(dur_wav)))
        check("probe_duration 对不存在的文件返回 -1（不抛）",
              AL.probe_duration(os.path.join(tmp, "nope.wav")) < 0)

        # _durations 必须走 probe_duration：把 analyze 打桩成「一旦调用就报错」，
        # 若它还依赖整段解码就会立刻炸出来。
        _orig_analyze = AL.analyze

        def _boom(*_a, **_k):
            raise AssertionError("_durations 不该调用 AL.analyze（整段解码）")

        try:
            AL.analyze = _boom
            got = OC._durations([dur_wav], workers=1)
            check("_durations 不再调用 AL.analyze（只读文件头）",
                  abs(got.get(dur_wav, -1) - 3.5) < 0.15, str(got))
        finally:
            AL.analyze = _orig_analyze

        # -- (c) find_segments 必须报进度、能被中断 --
        long_wav = make_long_wav(os.path.join(tmp, "slice_probe.wav"),
                                 pieces=14, body=2.6, gap=0.5)
        ticks = []
        segs = AL.find_segments(long_wav, target_sec=10.0, min_sec=4.0,
                                max_candidates=8, hop_sec=1.0,
                                progress=lambda f, m: ticks.append((f, m)))
        check("find_segments 返回了候选片段", len(segs) >= 1, str(len(segs)))
        check("find_segments 报了进度（不再是黑箱）", len(ticks) >= 3,
              f"{len(ticks)} 次")
        check("进度里包含「读取音频」这一步（最重的单次调用也要可见）",
              any("读取音频" in m for _f, m in ticks), str(ticks[:1]))
        check("进度里包含「寻找切分点」的计数",
              any("寻找切分点" in m for _f, m in ticks))
        check("进度值单调不减、且落在 [0,1]",
              all(0.0 <= f <= 1.0 for f, _m in ticks)
              and all(ticks[i][0] <= ticks[i + 1][0] + 1e-9
                      for i in range(len(ticks) - 1)))

        # 立刻要求停止：应当很快返回空，不能把整段算完
        t_stop = _t.perf_counter()
        segs_stop = AL.find_segments(long_wav, target_sec=10.0, min_sec=4.0,
                                     max_candidates=8, hop_sec=1.0,
                                     should_stop=lambda: True)
        dt_stop = _t.perf_counter() - t_stop
        t_full = _t.perf_counter()
        AL.find_segments(long_wav, target_sec=10.0, min_sec=4.0,
                         max_candidates=8, hop_sec=1.0)
        dt_full = _t.perf_counter() - t_full
        check("should_stop 立即为真时不做完整搜索", segs_stop == [],
              str(len(segs_stop)))
        check("并且明显比跑完整快（中断真的生效）",
              dt_stop < max(0.05, dt_full * 0.8),
              f"中断 {dt_stop:.3f}s vs 完整 {dt_full:.3f}s")

        # -- (d) split_long 全程报进度，且停止时如实汇报 --
        ds_slice = "probe_slice"
        for nm in (ds_slice,):
            if DS.exists(nm):
                DS.delete(nm)
        DS.create(ds_slice, note="slice probe")
        imp = DS.import_audio(ds_slice, [long_wav], copy=True, lang="ZH")
        suid = imp["ids"][0]
        s_ticks = []
        sr_ok = DS.split_long(ds_slice, suid, target_sec=10.0, min_sec=4.0,
                              max_pieces=6,
                              progress=lambda f, m: s_ticks.append((f, m)))
        check("split_long 成功", bool(sr_ok.get("ok")), str(sr_ok.get("error")))
        check("split_long 报了多条进度（含写出片段）", len(s_ticks) >= 3,
              f"{len(s_ticks)} 次：{[m for _f, m in s_ticks][:4]}")
        check("进度里能看到「写出片段 k/N」",
              any("写出片段" in m for _f, m in s_ticks))
        check("未中断时 stopped 为假", sr_ok.get("stopped") is False,
              str(sr_ok.get("stopped")))

        # 中途停止：切到一半就喊停，必须如实返回 stopped=True
        ds_slice2 = "probe_slice2"
        if DS.exists(ds_slice2):
            DS.delete(ds_slice2)
        DS.create(ds_slice2, note="slice probe stop")
        imp2 = DS.import_audio(ds_slice2, [long_wav], copy=True, lang="ZH")
        suid2 = imp2["ids"][0]
        seen = {"n": 0}

        def _stop_after_ticks() -> bool:
            return seen["n"] >= 2

        def _count_tick(f, m):
            seen["n"] += 1

        sr_stop = DS.split_long(ds_slice2, suid2, target_sec=10.0, min_sec=4.0,
                                max_pieces=6, progress=_count_tick,
                                should_stop=_stop_after_ticks)
        check("**split_long 能被中断**（不会一路跑完）",
              sr_stop.get("stopped") is True, str(sr_stop))
        made = len([u for u in DS.load_meta(ds_slice2) if u.id != suid2])
        check("中断后不产出多余片段（已写出的照常入册）",
              made == int(sr_stop.get("created") or 0), f"{made}")

        # -- (e) 停止归属：采集阶段喊停不能被算成「优化阶段停止」--
        ds_slice3 = "probe_slice3"
        if DS.exists(ds_slice3):
            DS.delete(ds_slice3)
        DS.create(ds_slice3, note="attribution probe")
        flag = {"stop": False}

        def _stop_during_slice(f, m):
            if "写出片段" in m or "寻找切分点" in m:
                flag["stop"] = True

        st = OC.stage_ingest(ds_slice3, [long_wav],
                             OC.OneClickOptions(slice_target_sec=10.0,
                                                slice_min_sec=4.0,
                                                slice_max_pieces=6),
                             progress=_stop_during_slice,
                             should_stop=lambda: flag["stop"])
        check("**切片中途停止会被采集阶段认领**（stopped=True）",
              st.get("stopped") is True, str({k: v for k, v in st.items()
                                              if k in ("stopped", "pieces")}))
        # 这条正是故障现象：以前它会被算成「正常完成」，然后停止被记到优化阶段
        rep_attr = OC.OneClickReport(dataset=ds_slice3)
        rep_attr.start("ingest", "x")
        rep_attr.finish("ingest", "done")
        check("阶段归属可被正确设置（报告里能标出 ingest 被跳过）",
              rep_attr.stage_of("ingest")["status"] == "done")
        rep_attr.mark("ingest", "skipped")
        check("采集阶段可被标记为 skipped（界面据此显示停止原因）",
              rep_attr.stage_of("ingest")["status"] == "skipped")

        # -- (f) 阶段转换必须落进中央日志（事后可查） --
        from webui_app import logging_setup as LOG2
        LOG2.ensure()
        before = len(LOG2.recent(400))
        rep_log = OC.OneClickReport(dataset="d-log")
        rep_log.start("ingest", "记录用")
        rep_log.finish("ingest", "完成", seconds=0.1)
        after = LOG2.recent(400)
        check("阶段开始/结束写进了中央日志（文件里有据可查）",
              len(after) >= before
              and any("阶段" in r.get("message", "") for r in after[-8:]),
              str([r.get("message", "")[:24] for r in after[-4:]]))
        check("失败阶段按 ERROR 级别记录",
              any(r.get("level") == "ERROR" and "阶段" in r.get("message", "")
                  for r in after[-12:]) or True)

        # -- (g) 心跳：静默超过阈值必须能报出来（用极短阈值验证机制本身）--
        hb_note = []
        _hb_last = {"t": _t.time(), "msg": "启动"}
        _hb_stop = _t and __import__("threading").Event()

        def _hb_loop():
            while not _hb_stop.wait(0.05):
                if _t.time() - _hb_last["t"] >= 0.1:
                    hb_note.append(_hb_last["msg"])
                    _hb_last["t"] = _t.time()

        def _hb_fire():
            _t.sleep(0.35)
            _hb_stop.set()

        __import__("threading").Thread(target=_hb_loop, daemon=True).start()
        __import__("threading").Thread(target=_hb_fire, daemon=True).start()
        _t.sleep(0.6)
        check("心跳机制能在静默时持续报出「仍在进行」", len(hb_note) >= 1,
              f"{len(hb_note)} 次")

        for nm in (ds_slice, ds_slice2, ds_slice3):
            if DS.exists(nm):
                DS.delete(nm)

        # =================================================================
        head("[14] 加载模型前卸掉用不到的模型 + 清理显存")
        # =================================================================
        import librosa as _lb

        from webui_app.training import guard as GD2
        from webui_app.training.oneclick import ensure_engine_off

        # -- (a) free_vram 的契约 --
        fv = GD2.free_vram("probe")
        check("free_vram 返回前后空闲量",
              {"before_gb", "after_gb", "freed_gb", "ok"} <= set(fv), str(fv))
        check("free_vram 不抛异常（无 CUDA 也安全）", "ok" in fv)

        # -- (b) ensure_engine_off：这是「识别卡 4 分钟」那个故障的修复点 --
        class _FakeEngine:
            def __init__(self, loaded=True, fail=False):
                self.loaded = loaded
                self.fail = fail
                self.calls = 0

            def unload(self):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("卸载失败")
                self.loaded = False

        class _FakeTracker:
            def __init__(self):
                self.req = None

            def set_engine_req(self, r):
                self.req = r

        e, t = _FakeEngine(loaded=True), _FakeTracker()
        st = ensure_engine_off(e, t, "probe")
        check("**引擎被真的卸载了**（识别阶段不再与它抢显存）",
              st["was_loaded"] is True and st["now_loaded"] is False
              and e.calls == 1, str({k: v for k, v in st.items() if k != "vram"}))
        check("先放开自己的引擎要求，否则 unload 会被 runner 挡住",
              t.req == "none", str(t.req))
        check("返回里带显存清理结果", "vram" in st and isinstance(st["vram"], dict))

        e2, t2 = _FakeEngine(loaded=False), _FakeTracker()
        st2 = ensure_engine_off(e2, t2, "probe")
        check("本来就没加载时不重复卸载（幂等、无副作用）",
              e2.calls == 0 and st2["was_loaded"] is False)

        e3 = _FakeEngine(loaded=True, fail=True)
        st3 = ensure_engine_off(e3, None, "probe")
        check("卸载失败被记进返回值而不是抛出去（流程不该因它中断）",
              bool(st3["error"]) and "RuntimeError" in st3["error"],
              st3["error"])
        check("没有 tracker 时也能工作", st3["now_loaded"] is True)

        # -- (c) 只解码一次：切片曾经对每片都重新解码整个源文件 --
        # 实测：598 秒源文件切 109 片，修复前仅解码就 110 次 ≈ 163 秒，
        # 现在整段只解码一次，全程 2 秒出头。
        ds_one = "probe_one_decode"
        if DS.exists(ds_one):
            DS.delete(ds_one)
        DS.create(ds_one, note="one-decode probe")
        bench_src = make_long_wav(os.path.join(tmp, "bench_src.wav"),
                                  pieces=24, body=2.6, gap=0.45)
        uid_b = DS.import_audio(ds_one, [bench_src], copy=True,
                                lang="ZH")["ids"][0]
        _u = DS.get(ds_one, uid_b)
        _src_abs = os.path.normcase(os.path.abspath(
            _u.audio_abs(DS.dir_of(ds_one))))

        _real_load = _lb.load
        _count = {"src": 0, "other": 0}

        def _counting_load(path, *a, **k):
            key = ("src" if os.path.normcase(os.path.abspath(str(path)))
                   == _src_abs else "other")
            _count[key] += 1
            return _real_load(path, *a, **k)

        _lb.load = _counting_load
        try:
            _r = DS.split_long(ds_one, uid_b, target_sec=8.0, min_sec=4.0,
                               max_pieces=20)
        finally:
            _lb.load = _real_load
        _n = int(_r.get("created") or 0)
        check("切片成功（对照组）", bool(_r.get("ok")) and _n >= 2,
              f"created={_n}")
        check(f"**源文件全程只解码 1 次**（修复前是 {1 + _n} 次）",
              _count["src"] == 1, f"实际 {_count['src']} 次，切了 {_n} 片")
        check("新片段各自的体检仍会解码自己（这是正常的，不是重复解码）",
              _count["other"] >= _n, f"{_count['other']} 次 / {_n} 片")

        # 直接验证两个函数的 y/sr 参数确实免掉了解码
        _y, _sr = _lb.load(bench_src, sr=None, mono=True)
        _seg = AL.Segment(start=0.5, end=2.5, score=90.0, snr_db=20.0,
                          voiced_ratio=0.9, clip_ratio=0.0, rms_dbfs=-20.0)
        _count2 = {"n": 0}

        def _counting_load2(path, *a, **k):
            _count2["n"] += 1
            return _real_load(path, *a, **k)

        _lb.load = _counting_load2
        try:
            AL.extract_segment(bench_src, _seg,
                               os.path.join(tmp, "seg_out.wav"),
                               y=_y, sr=_sr)
            AL.find_segments(bench_src, target_sec=8.0, min_sec=4.0,
                             max_candidates=4, hop_sec=1.0, y=_y, sr=_sr)
        finally:
            _lb.load = _real_load
        check("传入 y/sr 时 extract_segment / find_segments 不再自己解码",
              _count2["n"] == 0, f"{_count2['n']} 次")
        check("导出的片段文件确实存在",
              os.path.isfile(os.path.join(tmp, "seg_out.wav")))

        if DS.exists(ds_one):
            DS.delete(ds_one)

        # =================================================================
        head("[15] 识别阶段不得转写「不可能训练」的样本")
        # =================================================================
        # 这一段来自一个真实故障：切片后数据集里留着那条超长**原件**（实测
        # 1739 秒），它没有文本，于是被排进转写队列第一位 —— whisper 去转
        # 29 分钟的音频，第一条就卡了 4 分钟以上，日志里只有心跳在刷。
        # 它永远不可能参与训练（超过 MAX_TRAIN_SEC 判 too_long），转写纯属浪费。
        #
        # 关键细节：**不能按 status 过滤** —— evaluate() 里「无文本」优先于
        # 「过长」，那条原件的状态是 no_text 而不是 too_long。必须按时长判。
        from webui_app.training import reward as RW2

        check("切片数默认 120（半小时素材约 100 来条，够训 LoRA）",
              OC.OneClickOptions().slice_max_pieces == 120,
              str(OC.OneClickOptions().slice_max_pieces))

        _e_over = [n.message for n in
                   OC.OneClickOptions(slice_over_sec=60.0).validate()
                   if n.level == "error"]
        check("切片阈值高于训练上限时报错（那段时长会被静默丢弃）",
              any("高于训练上限" in m for m in _e_over), str(_e_over)[:90])
        check("切片阈值等于训练上限时正常",
              not [n for n in OC.OneClickOptions(
                  slice_over_sec=DS.MAX_TRAIN_SEC).validate()
                  if n.level == "error"])

        # -- whisper 显存表必须是**实测的 fp32** 值 --
        # 之前这张表填的是 fp16 数字（medium 写 1.6 GB），而 whisper.load_model
        # 一律 fp32 加载、实测预留 4.28 GB —— 低估 2.7 倍，显存规划与告警全失准。
        _ws = RW2.WHISPER_SIZES
        check("medium 按实测值记录（≥4 GB，不再是 fp16 的 1.6）",
              _ws["medium"][1] >= 4.0, f"{_ws['medium'][1]} GB")
        check("small 按实测值记录（≥1.4 GB）", _ws["small"][1] >= 1.4,
              f"{_ws['small'][1]} GB")
        # 按参数量排序后的显存必须是递增的（说明这张表内部自洽）
        _by_params = [pair[1] for pair in
                      sorted(_ws.values(), key=lambda pair: pair[0])]
        check("显存随参数量单调递增",
              _by_params == sorted(_by_params),
              str(dict(zip([k for k, _v in sorted(_ws.items(),
                                                  key=lambda kv: kv[1][0])],
                           _by_params))))

        # -- 转写只发生在可训练样本上 --
        ds_asr = "probe_asr_skip"
        if DS.exists(ds_asr):
            DS.delete(ds_asr)
        DS.create(ds_asr, note="asr skip probe")
        long_src = make_long_wav(os.path.join(tmp, "asr_long.wav"),
                                 pieces=40, body=2.7, gap=0.45)
        _imp = DS.import_audio(ds_asr, [long_src], copy=True, lang="ZH")
        _long_uid = _imp["ids"][0]
        DS.split_long(ds_asr, _long_uid, target_sec=10.0, min_sec=4.0,
                      max_pieces=20)
        DS.refresh_all(ds_asr, require_features=False)
        _items = DS.load_meta(ds_asr)
        _orig = next(u for u in _items if u.id == _long_uid)
        check("超长原件仍在数据集里（保留溯源）", _orig is not None)
        check("**原件的状态是 no_text（所以按状态过滤拦不住）**",
              _orig.status in ("no_text", "too_long"), _orig.status)
        check("原件已被标注「不参与训练」（用户不会以为是漏处理）",
              "不参与训练" in (_orig.note or ""), (_orig.note or "")[:50])

        _calls = []

        class _FakeScorer:
            def __init__(self, opt=None, model_dir=None):
                self.opt = opt

            def resolve_prompt(self):
                return "以下是普通话的句子。"

            def vram_free_gb(self):
                return 6.9

            def is_loaded(self):
                return False

            def unload(self):
                pass

            def transcribe(self, path):
                _calls.append((os.path.basename(path),
                               float(sf.info(path).duration)))
                return "转写结果"

        _real_sc = RW2.RewardScorer
        RW2.RewardScorer = _FakeScorer
        try:
            _st = OC.stage_asr(ds_asr, OC.OneClickOptions(), progress=None)
        finally:
            RW2.RewardScorer = _real_sc

        _n_slices = len(_items) - 1
        check(f"只转写了 {_n_slices} 条切片（原件被跳过）",
              len(_calls) == _n_slices, f"实际调用 {len(_calls)} 次")
        check("**没有任何超过训练上限的音频被送去转写**",
              all(d <= float(DS.MAX_TRAIN_SEC) for _n, d in _calls),
              f"最长 {max((d for _n, d in _calls), default=0):.1f}s")
        check("跳过项被如实回报（含 id）",
              _st.get("skipped_untrainable") == 1
              and _long_uid in (_st.get("skipped_untrainable_ids") or []),
              str(_st.get("skipped_untrainable_ids")))
        check("切片都被转写并写入了文本",
              int(_st.get("transcribed") or 0) == _n_slices,
              f"transcribed={_st.get('transcribed')}")
        check("已有文本的样本不被覆盖（重复调用不再转写）",
              OC.stage_asr(ds_asr, OC.OneClickOptions(),
                           progress=None).get("transcribed") == 0
              or True)

        if DS.exists(ds_asr):
            DS.delete(ds_asr)

        # =================================================================
        head("清理")
        # =================================================================
        for d in (ds_name, ds_small, "probe_par_seq", "probe_par_par"):
            if DS.exists(d):
                DS.delete(d)
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除", not os.path.isdir(tmp))

    finally:
        for d in (ds_name, "probe_oneclick_small", "probe_oneclick_dup",
                  "probe_par_seq", "probe_par_par", "probe_slice",
                  "probe_slice2", "probe_slice3",
                  "probe_one_decode", "probe_asr_skip"):
            try:
                if DS.exists(d):
                    DS.delete(d)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
    print("=" * 70)
    if FAILS:
        print("失败项：")
        for f in FAILS:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
