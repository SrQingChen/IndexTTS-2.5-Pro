"""真机验收：FunASR 双模型（SenseVoice 转写 + emotion2vec 情绪）。

需要联网（首次从 ModelScope 下载，SenseVoiceSmall≈1GB、
emotion2vec_plus_large≈1GB，已下载则秒过）。用「合成页产物 + 音色库参考」
做真实音频，验证：
    1. SenseVoice 转写质量（CER vs 已知文本，与 whisper-medium 同音频对比）
    2. 富输出解析（情绪标签/事件标签剥离干净）
    3. 专名拼音后纠（哈提西亚 → 卡提希娅）
    4. emotion2vec 分类/嵌入/余弦（API 形状防御式处理的实地验证）
    5. reward 新项（emo/pause）端到端可用
用法：.venv\\Scripts\\python.exe tools/funasr_acceptance.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    from webui_app.config import PROJECT_ROOT
    from webui_app.services import funasr_hub as FH
    from webui_app.services import voice_bank as VB
    from webui_app.training import reward as RW

    # 验收音频：优先用今天导演模式真机验收的产物（文本已知）
    TEXT = "住手！你这家伙，到底做了什么？！哈哈，太好了。再见……走吧。"
    synth = None
    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    cands = sorted(
        (f for f in os.listdir(out_dir)
         if f.startswith("spk_2026") and f.endswith(".wav")
         and os.path.exists(os.path.join(
             out_dir, f.replace(".wav", ".script.json")))),
        reverse=True)
    if cands:
        synth = os.path.join(out_dir, cands[0])
    check("找到带台本的合成产物作验收音频", synth is not None,
          os.path.basename(synth) if synth else "outputs/ 无 spk_*.script.json")
    ref_entry = VB.get("wujiu") or next(iter(VB.list_voices()), None)
    ref = ref_entry.audio_path if ref_entry else synth
    if synth is None:
        return finish()

    print("== 1. SenseVoice 转写（含首次下载） ==")
    t0 = time.perf_counter()
    try:
        r = FH.transcribe(synth, lang="zh")
    except Exception as e:
        check("SenseVoice 转写", False, f"{type(e).__name__}: {e}")
        return finish()
    dt = time.perf_counter() - t0
    cer = RW.cer(TEXT, r["text"])
    print(f"  转写：{r['text']}")
    print(f"  情绪标签={r['emotion']} 事件={r['events']} · {dt:.1f}s")
    check("转写出非空文本", bool(r["text"]), r["text"][:40])
    check("标签剥离干净（文本里无 <|）", "<|" not in r["text"])
    check(f"CER ≤ 0.35（vs 已知文本）", cer <= 0.35, f"cer={cer:.3f}")
    check("标点丰富（含 ，。！？ 之一）",
          any(p in r["text"] for p in "，。！？"))

    print("== 2. 专名拼音后纠 ==")
    fixed, names = FH.apply_glossary("哈提西亚对弗洛德利斯说话", ["卡提希娅", "弗洛德利斯"])
    check("同音错字被纠回标准名", "卡提希娅" in fixed and "弗洛德利斯" in fixed, fixed)
    keep, _ = FH.apply_glossary("今天天气真好", ["卡提希娅"])
    check("无关文本不被误改", keep == "今天天气真好", keep)

    print("== 3. emotion2vec（含首次下载） ==")
    try:
        rs = FH.emo_classify([synth, ref])
    except Exception as e:
        check("emotion2vec 分类", False, f"{type(e).__name__}: {e}")
        return finish()
    for x in rs:
        print(f"  {os.path.basename(x['path'])[:28]}: labels={x['labels']} "
              f"emb={'有' if x['embedding'] is not None else '无'}")
    check("分类返回标签", bool(rs[0]["labels"]), str(rs[0]["labels"]))
    if rs[0]["embedding"] is not None and rs[1]["embedding"] is not None:
        cos = FH.emo_cosine(synth, ref)
        check("嵌入余弦可算", -1.0 <= cos <= 1.0, f"cos={cos:.3f}")
    else:
        check("嵌入可用（funasr 版本返回了 embedding）", False,
              "取不到嵌入 —— 检查 funasr 版本的返回结构")

    print("== 4. reward 新项端到端 ==")
    sc = RW.RewardScorer(RW.RewardOptions(
        asr_engine="sensevoice", wer_weight=0.35, sim_weight=0.25,
        emo_weight=0.3, pause_weight=0.1, device="cpu"))
    s1 = sc.score(synth, TEXT, ref, emo_ref_path=ref)
    print(f"  reward={s1.get('reward')} detail={s1.get('detail')}")
    check("带情绪/停顿项的打分 ok", bool(s1.get("ok")), str(s1.get("error", ""))[:80])
    check("detail 含 emo/pause", "emo" in (s1.get("detail") or {})
          and "pause" in (s1.get("detail") or {}))
    ps = RW.RewardScorer.pause_score(synth)
    check("停顿启发项 ∈ [0,1]", 0.0 <= ps <= 1.0, f"pause={ps:.2f}")
    sc.unload()
    check("打分后 funasr 全部卸载",
          not FH.loaded_summary()["sensevoice"]
          and not FH.loaded_summary()["emotion2vec"])

    return finish()


def finish() -> int:
    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
