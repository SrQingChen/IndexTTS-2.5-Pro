"""真机验收：BoN 衔接连续性择优（2026-10-04 块间语感突变治理）。

实测（用户 30 块长文本）：块尾→下块首 F0 中位差 ±2~8 半音，与块内
词组过渡同量级——块独立渲染的起始音区随机；以前 30ms 无缝交叉淡化
掩盖了跳变，句间停顿拉长（阶段35）后每一处边界都成了干净 A/B 对比。
修复：BoN 打分加 _register_jump_penalty——同情绪块间，候选首音区偏
离上一选中块尾音区 >3 半音线性罚（封顶 0.30）。

方法：用户 22:54 台本的 5 个连续 happy 块（blk4-8）原样重跑，保持
temp=1.2 / BoN=8（用户实况——连续性必须在真实温度下有效），对比
基线（用户成品全篇 27 处边界音区差：中位 2.4st / 最大 8.3st）。

用法：.venv\\Scripts\\python.exe tools\\continuity_acceptance.py [--reuse]
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []
SR = 22050
SRC = os.path.join("outputs", "spk_20261003-225415.script.json")


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def f0_med(path_or_y, head: bool) -> float:
    import numpy as np
    import pyworld as pw
    import soundfile as sf
    if isinstance(path_or_y, str):
        y, _ = sf.read(path_or_y, dtype="float32")
    else:
        y = path_or_y
    n = int(min(0.30, len(y) / SR * 0.5) * SR)
    seg = (y[:n] if head else y[-n:]).astype(np.float64)
    f0, _ = pw.harvest(seg, SR, f0_floor=70.0, f0_ceil=600.0)
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) >= 15 else float("nan")


def main() -> int:
    import numpy as np
    import soundfile as sf
    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import inference as INF
    from webui_app.services import orchestrator as ORC
    from webui_app.services.director import DirectorScript, ScriptLine
    from webui_app.training import merge as MG

    side = json.load(open(SRC, encoding="utf-8"))
    lines = side["lines"][3:8]          # blk4-8：连续 happy
    sc = DirectorScript(lines=[
        ScriptLine(text=l["text"], emotion=l["emotion"],
                   intensity=l["intensity"], pause_after_ms=l["pause_after_ms"])
        for l in lines], backend="api", ok=True)

    out = os.path.join("outputs", "continuity_after.wav")
    side_out = os.path.splitext(out)[0] + ".script.json"
    if "--reuse" in sys.argv and os.path.isfile(out):
        print("  --reuse：复用已有成品")
        res = {"director": {"sidecar": side_out}}
    else:
        cfg = config_from_args([])
        ctx = AppContext.get(cfg)
        eng = ctx.engine
        eng.load()
        MG.mount_run(eng, "达妮娅_gpt", "best", 0.9)
        MG.mount_run(eng, "达妮娅_cfm", "best", 1.0)
        req = INF.GenRequest(
            spk_audio_prompt=os.path.join("voice_bank", "audio", "达妮娅.wav"),
            text="".join(l.text for l in sc.lines), emo_alpha=0.65,
            seed=side["base_seed"], temperature=1.2, top_p=0.9,
            max_text_tokens_per_segment=180)
        t0 = time.perf_counter()
        res = ORC.perform(eng, req, sc, route=False, character="达妮娅",
                          lora_run="达妮娅_gpt", pause_scale=1.0,
                          pause_cap_ms=300, bon_n=8, output_path=out)
        print(f"  合成耗时 {time.perf_counter() - t0:.0f}s")

    side2 = json.load(open(res["director"]["sidecar"], encoding="utf-8"))
    y, _ = sf.read(out, dtype="float32")

    acc = 0.08
    spans = []
    for i, l in enumerate(side2["lines"]):
        s0 = acc
        acc += l["samples"] / SR
        if i < len(side2["lines"]) - 1:
            acc += max(side2["joints"][i]["script_gap_ms"], 30) / 1000
        spans.append((s0, min(acc, len(y) / SR)))
    diffs = []
    for i in range(len(spans) - 1):
        a = f0_med(y[int(spans[i][1] * SR) - int(0.3 * SR):
                     int(spans[i][1] * SR)], head=False)
        b = f0_med(y[int(spans[i + 1][0] * SR):
                     int(spans[i + 1][0] * SR) + int(0.3 * SR)], head=True)
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(round(abs(12 * np.log2(b / a)), 1))
    med = float(np.median(diffs)) if diffs else 99.0
    print(f"  块间首尾音区差(半音): {diffs} · 中位 {med:.1f}"
          f"（基线：用户成品中位 2.4 / 最大 8.3）")

    check("同情绪块间音区差中位 ≤2.5 半音（连续性择优生效）",
          med <= 2.5, f"中位 {med:.1f} · {diffs}")
    check("无 >5.5 半音的剧烈边界跳变（基线最大 8.3）",
          all(d <= 5.5 for d in diffs), f"{diffs}")
    check("旁车 bon 记录 continuity（可追溯）",
          all((l.get("bon") or {}).get("continuity") in (True, False)
              for l in side2["lines"][1:]),
          str([(l.get("bon") or {}).get("continuity")
               for l in side2["lines"]]))

    ok = all(c[1] for c in CHECKS)
    print(f"\n{'=' * 58}\n{'✅ 全部通过' if ok else '❌ 存在失败'} · "
          f"{sum(1 for c in CHECKS if c[1])}/{len(CHECKS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
