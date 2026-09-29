"""探针：情感参考库 + 句级编排器。

桩引擎（不加载真模型、不需要 GPU）：验证路由、逐句情感 kwargs、
停顿拼接长度、种子、旁车台本、临时目录清理。
用法：python tools/orchestrator_probe.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from webui_app.services import emotion_bank as EB  # noqa: E402
from webui_app.services import inference as INF  # noqa: E402
from webui_app.services import orchestrator as ORC  # noqa: E402
from webui_app.services.director import DirectorScript, ScriptLine  # noqa: E402

CHECKS = []
LINE_SEC = 0.5          # 桩引擎每句产 0.5s
SR = 22050


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


class _Cfg:
    is_v25 = True
    output_dir = ""
    cache_dir = ""


class _StubEngine:
    """记录每次 infer 的 kwargs，产 0.5s 静音 wav。
    spk_audio_prompt 若指向临时滚动参考（会被 run_dir 清理），
    调用时备份一份到 <cache>/prompts_keep/ 供事后断言。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []
        self.keep_dir = os.path.join(cfg.cache_dir, "prompts_keep")
        os.makedirs(self.keep_dir, exist_ok=True)

    def infer(self, progress=None, **kw):
        import shutil as _sh
        pp = str(kw.get("spk_audio_prompt") or "")
        if "prompt_blk" in pp and os.path.isfile(pp):
            _sh.copy2(pp, os.path.join(self.keep_dir,
                                       os.path.basename(pp)))
        self.calls.append(kw)
        path = kw["output_path"]
        sf.write(path, np.zeros(int(SR * LINE_SEC), dtype=np.int16), SR,
                 subtype="PCM_16")
        return path


def _make_wav(path, sec=1.0):
    sf.write(path, (np.random.RandomState(0).rand(int(SR * sec)) * 2000 - 1000
                    ).astype(np.int16), SR, subtype="PCM_16")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="orc_probe_")
    os.makedirs(os.path.join(tmp, "out"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "cache"), exist_ok=True)

    # ---- 情感参考库：指到临时目录，别动真库 ----
    EB.BANK_DIR = os.path.join(tmp, "emotion_bank")
    EB.INDEX_FILE = os.path.join(EB.BANK_DIR, "index.json")
    spk = os.path.join(tmp, "spk.wav")
    _make_wav(spk)

    print("== 1. 情感参考库 ==")
    e_happy = EB.add("测试者", "happy", _mk(os.path.join(tmp, "happy.wav")))
    EB.add("测试者", "angry", _mk(os.path.join(tmp, "angry.wav")))
    EB.add("别人", "happy", _mk(os.path.join(tmp, "other.wav")))
    check("入库 3 条", len(EB.list_entries()) == 3)
    check("中文情绪标签可归一化", EB.norm_emotion("怒") == "angry")
    check("路由精确命中 happy", EB.pick("测试者", "happy").name == e_happy.name)
    check("未归档情绪回退同角色最高分",
          EB.pick("测试者", "afraid").character == "测试者")
    check("平静句不做情绪借用（回 None）", EB.pick("测试者", "calm") is None)
    check("未归档角色不跨角色借用", EB.pick("第三者", "happy") is None)
    check("非法情绪词报错", _raises(lambda: EB.add("测试者", "兴奋", spk)))

    print("== 2. 编排器：路由与逐句 kwargs ==")
    eng = _StubEngine(_Cfg_populate(tmp))
    req = INF.GenRequest(spk_audio_prompt=spk, text="你好。再见！",
                         emo_alpha=0.65, seed=123)
    sc = DirectorScript(lines=[
        ScriptLine(text="住手！", emotion="angry", intensity=0.9, pause_after_ms=600),
        ScriptLine(text="哈哈，太好了。", emotion="happy", intensity=0.5,
                   pause_after_ms=200),
        ScriptLine(text="走吧。", emotion="calm", intensity=0.3, pause_after_ms=300),
    ])
    res = ORC.perform(eng, req, sc, route=True, character="测试者")

    check("产出文件", os.path.isfile(res["path"]))
    check("调用引擎 3 次", len(eng.calls) == 3)
    routed = [c for c in eng.calls if c.get("emo_audio_prompt")]
    check("2 句路由命中（angry/happy），calm 回退", len(routed) == 2
          and eng.calls[2]["emo_audio_prompt"] is None,
          str([bool(c.get("emo_audio_prompt")) for c in eng.calls]))
    # angry: 0.65 × (0.5+0.9) = 0.91；happy: 0.65 × (0.5+0.5) = 0.65
    check("逐句 alpha = 全局×(0.5+intensity)",
          abs(eng.calls[0]["emo_alpha"] - 0.91) < 1e-6
          and abs(eng.calls[1]["emo_alpha"] - 0.65) < 1e-6,
          f"{[round(c['emo_alpha'], 3) for c in eng.calls[:2]]}")
    check("外推关闭时 alpha ≤1", all(c["emo_alpha"] <= 1.0 for c in eng.calls))
    check("逐句文本正确", [c["text"] for c in eng.calls]
          == ["住手！", "哈哈，太好了。", "走吧。"])
    check("路由参考来自同角色",
          all("测试者" in c["emo_audio_prompt"] for c in routed))

    print("== 3. 编排器：拼接长度与台本 ==")
    expect = 3 * LINE_SEC + (600 + 200) / 1000.0 + 0.16  # +首尾余白 160ms
    dur = sf.info(res["path"]).duration
    check("总时长 = 句长和 + 台本停顿 + 首尾余白", abs(dur - expect) < 0.02,
          f"{dur:.3f}s ≈ {expect:.3f}s")
    side = res["director"]["sidecar"]
    check("旁车台本存在", os.path.isfile(side))
    import json
    data = json.load(open(side, encoding="utf-8"))
    check("台本含逐句种子与路由", len(data["lines"]) == 3
          and data["lines"][0]["seed"] != data["lines"][1]["seed"]
          and data["routed"] == 2 and data["fallback"] == 1)
    check("临时句目录已清理",
          not os.path.exists(os.path.join(eng.cfg.cache_dir, "perform"))
          or not os.listdir(os.path.join(eng.cfg.cache_dir, "perform")))

    print("== 4. 编排器：外推与失败路径 ==")
    eng2 = _StubEngine(_Cfg_populate(tmp))
    res2 = ORC.perform(eng2, req, sc, route=False, extrapolate=True)
    check("route=False 时不路由", all(
        c["emo_audio_prompt"] is None for c in eng2.calls))
    eng3 = _StubEngine(_Cfg_populate(tmp))
    try:
        bad = DirectorScript(lines=[ScriptLine(text="", emotion="calm")])
        ORC.perform(eng3, req, bad)
        check("空台词句报错", False)
    except INF.EngineError:
        check("空台词句报错", True)

    print("== 5. BoN 逐句择优（桩打分器：reward=候选序号） ==")
    import re as _re
    from webui_app.training import reward as RW

    class _RankScorer:
        def __init__(self, opt=None, model_dir=None):
            self.calls = []
        def score(self, path, text, ref, emo_ref_path=None):
            m = _re.search(r"_c(\d+)\.wav$", str(path))
            k = int(m.group(1)) if m else 0
            self.calls.append((os.path.basename(path), k))
            return {"ok": True, "reward": float(k)}
        def unload(self):
            pass

    _real = RW.RewardScorer
    RW.RewardScorer = _RankScorer
    try:
        eng4 = _StubEngine(_Cfg_populate(tmp))
        sc3 = DirectorScript(lines=[
            ScriptLine(text="第一句。", emotion="calm", intensity=0.3,
                       pause_after_ms=200),
            ScriptLine(text="第二句！", emotion="angry", intensity=0.9,
                       pause_after_ms=300),
        ])
        res4 = ORC.perform(eng4, req, sc3, route=False, bon_n=3)
    finally:
        RW.RewardScorer = _real
    check("每句合成 3 个候选", len(eng4.calls) == 6,
          f"calls={len(eng4.calls)}")
    import json as _json
    data4 = _json.load(open(res4["director"]["sidecar"], encoding="utf-8"))
    bon_rows = [l.get("bon") for l in data4["lines"]]
    check("台本记录逐句 BoN 得分与中选",
          all(b and b["n"] == 3 and b["rewards"] == [0, 1, 2]
              and b["chosen"] == 2 for b in bon_rows),
          str(bon_rows))
    check("中选种子 = 候选2 的种子", all(
        l["seed"] == (data4["base_seed"] + 977 * l["idx"] + 131 * 2) % (2**31)
        for l in data4["lines"]), str([l["seed"] for l in data4["lines"]]))
    check("res 摘要带 bon_n", res4["director"].get("bon_n") == 3)

    print("== 6. BoN 保留候选（bon_keep → outputs/bon/） ==")
    from webui_app.training import reward as RW2
    real2 = RW.RewardScorer
    RW.RewardScorer = _RankScorer
    try:
        eng5 = _StubEngine(_Cfg_populate(tmp))
        res5 = ORC.perform(eng5, req, sc3, route=False, bon_n=3, bon_keep=True)
    finally:
        RW.RewardScorer = real2
    data5 = _json.load(open(res5["director"]["sidecar"], encoding="utf-8"))
    kept = [l["bon"] for l in data5["lines"] if (l.get("bon") or {}).get("best_path")]
    check("两句都落了 best/worst 文件", len(kept) == 2)
    check("文件真实存在且在 outputs/bon/ 下",
          all(os.path.isfile(b["best_path"]) and os.path.isfile(b["worst_path"])
              and f"{os.sep}bon{os.sep}" in b["best_path"] for b in kept))
    check("reward 记录成对（best≥worst）",
          all(b["reward_best"] >= b["reward_worst"] for b in kept),
          str([(b["reward_best"], b["reward_worst"]) for b in kept]))
    check("res 摘要 bon_kept=2", res5["director"].get("bon_kept") == 2)
    # bon_keep=False（默认）时不应产生持久文件（先清掉上一段的产物，
    # 否则同秒时间戳会让负例检查吃到正例的文件）
    import shutil as _shutil
    eng6 = _StubEngine(_Cfg_populate(tmp))
    _shutil.rmtree(os.path.join(eng6.cfg.output_dir, "bon"),
                   ignore_errors=True)
    res6 = ORC.perform(eng6, req, sc3, route=False, bon_n=3)
    import glob as _glob
    fresh = [p for p in _glob.glob(os.path.join(eng6.cfg.output_dir, "bon", "**", "*"),
                                   recursive=True) if p.endswith(".wav")]
    check("默认不保留候选", not fresh, str(fresh[:2]))

    print("== 7. 表演块合并（v2 核心） ==")
    eng7 = _StubEngine(_Cfg_populate(tmp))
    sc7 = DirectorScript(lines=[
        ScriptLine(text="我们曾经约定过。", emotion="sad", intensity=0.6,
                   pause_after_ms=300),
        ScriptLine(text="要一起看到最后的结局。", emotion="sad", intensity=0.6,
                   pause_after_ms=300),
        ScriptLine(text="所以,不许你死在这里。", emotion="sad", intensity=0.7,
                   pause_after_ms=400),
        ScriptLine(text="听到了吗?!", emotion="sad", intensity=0.8,
                   pause_after_ms=600),
    ])
    res7 = ORC.perform(eng7, req, sc7, route=False)
    check("同情绪 4 行合并为 1 块（1 次引擎调用）",
          len(eng7.calls) == 1, f"calls={len(eng7.calls)}")
    check("块文本按序拼接", "我们曾经约定过。" in eng7.calls[0]["text"]
          and "听到了吗?!" in eng7.calls[0]["text"])
    check("块内零人工静音（interval_silence 被压到 ≤120ms）",
          eng7.calls[0]["interval_silence"] <= 120,
          str(eng7.calls[0]["interval_silence"]))
    data7 = _json.load(open(res7["director"]["sidecar"], encoding="utf-8"))
    check("台本记录行→块（4 行 1 块）",
          data7.get("lines_in") == 4 and data7.get("blocks") == 1
          and len(data7["lines"][0]["lines"]) == 4)
    check("块后停顿取块末行", data7["lines"][0]["pause_after_ms"] == 600)
    check("摘要 lines_in/blocks", res7["director"].get("lines_in") == 4
          and res7["director"]["n"] == 1)

    # 情绪突变断块 + 块间停顿仍然生效
    eng8 = _StubEngine(_Cfg_populate(tmp))
    sc8 = DirectorScript(lines=[
        ScriptLine(text="平静地说。", emotion="calm", intensity=0.3,
                   pause_after_ms=250),
        ScriptLine(text="突然爆发！", emotion="angry", intensity=0.9,
                   pause_after_ms=280),
    ])
    res8 = ORC.perform(eng8, req, sc8, route=False)
    check("情绪突变断成 2 块", len(eng8.calls) == 2)
    d8 = sf.info(res8["path"]).duration
    expect8 = 2 * LINE_SEC + 250 / 1000.0 + 0.16   # +首尾余白 160ms
    check("块间停顿按等级插入", abs(d8 - expect8) < 0.02,
          f"{d8:.3f}s ≈ {expect8:.3f}s")

    print("== 9. 块 ≤40 字 / 长行守卫 / 滚动参考续合成 ==")
    from webui_app.services.director import ScriptLine as SL
    # 5 句同情绪 × 10 字 → 40 字上限应拆成 ~2 块(而非 1 块 50 字)
    eng10 = _StubEngine(_Cfg_populate(tmp))
    sc10 = DirectorScript(lines=[
        SL(text=f"第{i}句十个字的话呢。", emotion="calm", intensity=0.4,
           pause_after_ms=240) for i in range(1, 6)])
    res10 = ORC.perform(eng10, req, sc10, route=False)
    check("同情绪长内容拆成多块且每块 ≤40 字",
          len(eng10.calls) >= 2
          and all(len(c["text"]) <= 40 for c in eng10.calls),
          f"{len(eng10.calls)} 块 {[len(c['text']) for c in eng10.calls]}")
    check("低显存触发条件(>40字)永不为真",
          all(len(c["text"]) <= 40 for c in eng10.calls))
    # 单条 >40 字的 LLM 行被守卫拆分
    eng11 = _StubEngine(_Cfg_populate(tmp))
    sc11 = DirectorScript(lines=[SL(
        text="这是一段特别长的台词，超过了四十个字符的上限，"
             "所以守卫会在句号处把它拆开，成为多个子行再合并成块。",
        emotion="calm", intensity=0.4, pause_after_ms=300)])
    res11 = ORC.perform(eng11, req, sc11, route=False)
    check(">40 字单行被拆成 ≤40 字的块",
          all(len(c["text"]) <= 40 for c in eng11.calls)
          and len(eng11.calls) >= 2,
          f"{[len(c['text']) for c in eng11.calls]}")
    # 滚动参考:第 2+ 块的 spk_audio_prompt ≠ 原参考,且为临时文件
    eng12 = _StubEngine(_Cfg_populate(tmp))
    sc12 = DirectorScript(lines=[
        SL(text="第一块平静叙述。", emotion="calm", intensity=0.4,
           pause_after_ms=240),
        SL(text="第二块突然爆发！", emotion="angry", intensity=0.8,
           pause_after_ms=240)])   # 情绪突变 → 必为两块
    res12 = ORC.perform(eng12, req, sc12, route=False)
    p1 = eng12.calls[0]["spk_audio_prompt"]
    p2 = eng12.calls[1]["spk_audio_prompt"]
    p2_keep = os.path.join(eng12.keep_dir, os.path.basename(p2))
    check("块1 用原参考", p1 == spk, p1)
    check("块2 参考换成滚动参考(含上一块音频)",
          p2 != spk and "prompt_blk" in p2 and os.path.isfile(p2_keep), p2)
    import soundfile as _sfx
    check("滚动参考 ≤14s(引擎 15s 窗口内)",
          _sfx.info(p2_keep).duration <= 14.5,
          f"{_sfx.info(p2_keep).duration:.1f}s")
    # 无零填充：整条参考的静音占比必须很小(不足段用参考音频补,不是零)
    _ry, _rsr = sf.read(p2_keep, dtype="float32")
    _fr = int(_rsr * 0.02); _n = len(_ry) // _fr
    _rms = np.sqrt(np.mean(_ry[:_n * _fr].reshape(_n, _fr) ** 2, axis=1) + 1e-12)
    _quiet_ratio = float(np.mean(_rms < 1e-4))
    check("滚动参考无零填充段(静音占比<10%)", _quiet_ratio < 0.10,
          f"quiet={_quiet_ratio:.1%}")
    check("滚动参考定长 14.0s(±0.1)", abs(len(_ry) / _rsr - 14.0) < 0.1,
          f"{len(_ry)/_rsr:.2f}s")
    data12 = _json.load(open(res12["director"]["sidecar"], encoding="utf-8"))
    check("台本记录 rolling_ref", data12["lines"][0].get("rolling_ref") is False
          and data12["lines"][1].get("rolling_ref") is True)
    # continuity=False → 全部用原参考
    eng13 = _StubEngine(_Cfg_populate(tmp))
    ORC.perform(eng13, req, sc12, route=False, continuity=False)
    check("关 continuity 时全部用原参考",
          eng13.calls[0]["spk_audio_prompt"] == spk
          and eng13.calls[1]["spk_audio_prompt"] == spk)
    # interval_silence 恒 0(低显存内部分块永不垫音)
    check("interval_silence=0", all(c["interval_silence"] == 0
                                    for c in eng12.calls))

    print("== 8. 省略号强制断块（戏剧性停顿拍） ==")
    eng9 = _StubEngine(_Cfg_populate(tmp))
    sc9 = DirectorScript(lines=[
        ScriptLine(text="遇见我想见的人……", emotion="calm", intensity=0.4,
                   pause_after_ms=600),
        ScriptLine(text="成为一名流浪骑士，是我真正的志向。", emotion="calm",
                   intensity=0.5, pause_after_ms=300),
    ])
    res9 = ORC.perform(eng9, req, sc9, route=False)
    check("同情绪但省略号结尾强制断成 2 块", len(eng9.calls) == 2,
          f"calls={len(eng9.calls)}")
    d9 = sf.info(res9["path"]).duration
    expect9 = 2 * LINE_SEC + 600 / 1000.0 + 0.16  # +首尾余白 160ms
    check("省略号停顿=块间 600ms（不再是模型短停顿）",
          abs(d9 - expect9) < 0.02, f"{d9:.3f}s ≈ {expect9:.3f}s")

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


def _mk(p):
    _make_wav(p)
    return p


def _Cfg_populate(tmp):
    c = _Cfg()
    c.output_dir = os.path.join(tmp, "out")
    c.cache_dir = os.path.join(tmp, "cache")
    return c


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    raise SystemExit(main())
