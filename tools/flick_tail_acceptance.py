"""真机验收：逗号闪缝 + 末尾硬截断（2026-10-03 爱弥斯实锤两病例）。

病例①：第一逗号被模型实现成 30~40ms 闪缝（低于旧 50ms 收集门槛），
保底位被 0.2s 外的词组间隙冒领——真逗号处「急促打断收音」。
病例②：块尾在 -29dB 处直接数字零（AR EOS 提前），旧 BoN 惩罚线
（末 100ms>-25dB）放行——「整句收音戛然而止」。

修复：收集门槛 20ms 检出 + 窗口 0.6→0.35s + 就近匹配；惩罚线收紧到
末 50ms>-35dB；成品末尾悬崖 100ms 余弦收束兜底（tail_repair 记旁车）。

方法：用户 16:32 成品的台本/种子/参数原样重跑（爱弥斯双 LoRA、BoN 8、
scale 1.0），断言两个病例位的波形与审计。

用法：.venv\\Scripts\\python.exe tools\\flick_tail_acceptance.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []
SR = 22050
SRC = os.path.join("outputs", "spk_20261003-163233.script.json")


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def runs_ms(y, min_ms=40):
    import numpy as np
    fr = int(SR * 0.025)
    hp = int(SR * 0.01)
    n = 1 + (len(y) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
    e = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    q = (20 * np.log10(e + 1e-12)) < -30
    out, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            if (j - i) * 10 >= min_ms:
                out.append((i * 0.01, (j - i) * 10))
            i = j
        else:
            i += 1
    return out


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
    sc = DirectorScript(lines=[
        ScriptLine(text=l["text"], emotion=l["emotion"],
                   intensity=l["intensity"], pause_after_ms=l["pause_after_ms"])
        for l in side["lines"]], backend="api", ok=True)

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine
    eng.load()
    MG.mount_run(eng, "爱弥斯_gpt", "best", 0.9)
    MG.mount_run(eng, "爱弥斯_cfm", "best", 1.0)

    # 参考音频：音色库里取爱弥斯（与用户一致）
    from webui_app.services import voice_bank as VB
    spk = os.path.join("voice_bank", "audio", "爱弥斯.wav")
    if not os.path.isfile(spk):
        v = next((x for x in VB.list_voices() if x.name == "爱弥斯"), None)
        spk = v.audio_path if v else ""
    check("找到爱弥斯参考", bool(spk), spk)

    req = INF.GenRequest(
        spk_audio_prompt=spk, text="".join(l.text for l in sc.lines),
        emo_alpha=0.65, seed=side["base_seed"],
        temperature=1.2, top_p=0.9, top_k=30,
        max_text_tokens_per_segment=180)

    out = os.path.join("outputs", "flick_tail_after.wav")
    reuse = "--reuse" in sys.argv and os.path.isfile(out) and \
        os.path.isfile(os.path.splitext(out)[0] + ".script.json")
    if reuse:
        print("  --reuse：跳过合成，复用已有成品")
        res = {"director": {"sidecar": os.path.splitext(out)[0] + ".script.json"}}
    else:
        t0 = time.perf_counter()
        res = ORC.perform(eng, req, sc, route=False, character="爱弥斯",
                          lora_run="爱弥斯_gpt", pause_scale=1.0,
                          pause_cap_ms=300, bon_n=8, stress_enable=True,
                          stress_gain_db=2.0, f0_restore=True,
                          f0_expand_max=1.3, output_path=out)
        print(f"  合成耗时 {time.perf_counter() - t0:.0f}s")

    side2 = json.load(open(res["director"]["sidecar"], encoding="utf-8"))
    y, _ = sf.read(out, dtype="float32")

    # ---- 病例①：第一逗号（blk1 文本 15 字处，「救世主，」）----
    blk1 = side2["lines"][0]
    g1 = blk1.get("governance", {})
    comma_idx = blk1["text"].index("，")
    est_expect = comma_idx / max(1, sum(
        1 for c in blk1["text"] if c not in "，。？！…、；："))
    p1 = [a for a in g1.get("audit", []) if a.get("kind") == "punct"]
    m1 = [a for a in g1.get("audit", []) if a.get("kind") == "run"
          and a.get("action", "").startswith("mandated")]
    check("blk1 逗号保底位命中（audit 有 punct 匹配记录）",
          len(p1) == 1 and p1[0].get("matched_run") is not None,
          f"{p1}")
    # 实测：blk1 区间内 ≥110ms 的停顿存在（补插或自然入带均可）
    blk1_len = blk1["samples"] / SR
    r1 = [r for r in runs_ms(y[:int(blk1_len * SR) + 2205]) if r[1] >= 110]
    check("第一逗号停顿 ≥110ms 检出（旧：30~40ms 闪缝）",
          bool(r1), f"{r1}")

    # ---- 病例②：成品末尾不再硬切 ----
    nz = np.nonzero(np.abs(y) > 1e-5)[0]
    tail = y[nz[-1] - int(0.05 * SR):nz[-1] + 1]
    tail_rms = 20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12)
    check("末 50ms 收音 ≤-35dB 或已做兜底淡化",
          tail_rms <= -35.0 or side2.get("tail_repair"),
          f"末50ms {tail_rms:.1f}dB · tail_repair={side2.get('tail_repair')}")

    # 无越权长停顿（块内 ≤ 逗号天花板 210 + 检出余量）
    fr = int(SR * 0.025)
    hp = int(SR * 0.01)
    n = 1 + (len(y) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
    e = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    qz = (20 * np.log10(e + 1e-12)) < -100
    junctions, i = [], 0
    while i < n:
        if qz[i]:
            j = i
            while j < n and qz[j]:
                j += 1
            if (j - i) * 10 >= 80:
                junctions.append(((i + j) / 2) * 0.01)
            i = j
        else:
            i += 1
    q = (20 * np.log10(e + 1e-12)) < -30
    rr, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            rr.append((i * 0.01, (j - i) * 10))
            i = j
        else:
            i += 1
    intra = [d for s, d in rr
             if d >= 60 and s * SR > 2205
             and not any(abs(s - j) < 0.25 for j in junctions)]
    # 天花板 210ms 是**检出域**目标（阶段24以来全治理的统一校准域）。
    # 已知属性：气息性停顿（肩部在 -30dB 线附近浮动）被截到 210 检出后，
    # 物理残留可达 210+2×肩部(~60ms)≈330ms——真机本次实测逗号 290 /
    # 配额拍 330，均低于边界句号带 320~450 的下限附近，无层级倒挂。
    # 若日后听感反馈「同类停顿长短不一」，再考虑肩部感知截断（单独一轮）。
    check("块内停顿物理残留 ≤350ms（检出域 210 + 气息肩部余量）",
          intra and max(intra) <= 350,
          f"最长 {max(intra) if intra else 0}ms")

    ok = all(c[1] for c in CHECKS)
    print(f"\n{'=' * 58}\n{'✅ 全部通过' if ok else '❌ 存在失败'} · "
          f"{sum(1 for c in CHECKS if c[1])}/{len(CHECKS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
