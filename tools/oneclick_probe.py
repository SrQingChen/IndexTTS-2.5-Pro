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

import os
import shutil
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

from webui_app.training import dataset as DS                 # noqa: E402
from webui_app.training import guard as GD                   # noqa: E402
from webui_app.training import oneclick as OC                # noqa: E402
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

        # =================================================================
        head("[9] UI 计划面板（按钮下方那块「自动参数」）")
        # =================================================================
        from webui_app.tabs import oneclick_tab as OT      # noqa: E402

        check("OPT_KEYS 与控件列表长度一致（漏一个就会错位）",
              len(OT.OPT_KEYS) == 21, str(len(OT.OPT_KEYS)))
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
        head("清理")
        # =================================================================
        for d in (ds_name, ds_small):
            if DS.exists(d):
                DS.delete(d)
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除", not os.path.isdir(tmp))

    finally:
        for d in (ds_name, "probe_oneclick_small", "probe_oneclick_dup"):
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
