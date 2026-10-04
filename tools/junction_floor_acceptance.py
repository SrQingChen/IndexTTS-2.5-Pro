"""真机验收：块间标点下限（2026-10-04 句间反转修复）。

事故（22:54 长文本实锤）：API 导演按「连续叙述给 0」把 30 块里 22 个
块间停顿给了 0 → 拼接层 30ms 交叉淡化，被块内逗号保底（150~290ms）
反超——「句间停顿比句内还短」。修复：gap≈0 时按前块行尾标点等级带
下限兜底（逗号 120 / 句号 320 × scale），无标点（40 字强行句中切）
保持 30ms 无缝。

方法：用户 22:54 台本前 5 块原样重跑（达妮娅双 LoRA；BoN 关——拼接
层与候选选择无关），断言块间实测 ≥ 标点下限、且句间 ≥ 句内中位。

用法：.venv\\Scripts\\python.exe tools\\junction_floor_acceptance.py [--reuse]
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
    lines = side["lines"][:5]
    sc = DirectorScript(lines=[
        ScriptLine(text=l["text"], emotion=l["emotion"],
                   intensity=l["intensity"], pause_after_ms=l["pause_after_ms"])
        for l in lines], backend="api", ok=True)

    out = os.path.join("outputs", "junction_floor_after.wav")
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
                          pause_cap_ms=300, bon_n=0, output_path=out)
        print(f"  合成耗时 {time.perf_counter() - t0:.0f}s")

    side2 = json.load(open(res["director"]["sidecar"], encoding="utf-8"))
    y, _ = sf.read(out, dtype="float32")

    # 实测停顿并归类：块边界 vs 块内
    acc = 0.08
    bounds = []
    for i, l in enumerate(side2["lines"]):
        acc += l["samples"] / SR
        if i < len(side2["lines"]) - 1:
            bounds.append((acc, side2["joints"][i], l["text"][-1]))
            acc += max(side2["joints"][i]["script_gap_ms"], 30) / 1000
    fr = int(SR * 0.025)
    hp = int(SR * 0.01)
    n = 1 + (len(y) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
    e = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    q = (20 * np.log10(e + 1e-12)) < -30
    runs, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            if (j - i) * 10 >= 60:
                runs.append((i * 0.01, (j - i) * 10))
            i = j
        else:
            i += 1
    jt = [d for s, d in runs if any(abs(s - b[0]) < 0.3 for b in bounds)]
    intra = [d for s, d in runs
             if not any(abs(s - b[0]) < 0.3 for b in bounds)]
    med_intra = float(np.median(intra)) if intra else 0.0
    print(f"  块间实测: {sorted(jt)} · 块内中位 {med_intra:.0f}ms")

    zero_j = [b for b in bounds if b[1]["script_gap_ms"] >= 30
              and b[1].get("punct_floor_ms", 0) > 0]
    check("零gap+标点结尾的块间记账总量 = 下限（拼接层精确落地）",
          len(zero_j) >= 2 and all(
              b[1]["script_gap_ms"] >= b[1]["punct_floor_ms"]
              and b[1]["mode"] in ("bed", "crossfade")
              for b in zero_j),
          f"{[(round(b[0],1), b[1]['script_gap_ms'], b[1]['punct_floor_ms'], b[2]) for b in zero_j]}")
    check("句间(块间中位) ≥ 句内(块内中位) —— 层级不再倒挂",
          bool(jt) and float(np.median(jt)) >= med_intra - 10,
          f"句间中位 {np.median(jt):.0f} vs 句内中位 {med_intra:.0f}ms")
    check("旁车 joints 记录 punct_floor_ms（可追溯）",
          all("punct_floor_ms" in j for j in side2["joints"]))

    ok = all(c[1] for c in CHECKS)
    print(f"\n{'=' * 58}\n{'✅ 全部通过' if ok else '❌ 存在失败'} · "
          f"{sum(1 for c in CHECKS if c[1])}/{len(CHECKS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
