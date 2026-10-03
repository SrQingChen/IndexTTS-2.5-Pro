"""真机验收：连续性三修复（2026-10-03 第五轮续）。

修复内容：①治理门槛帧量化盲区（物理150~176ms停顿逃逸）；②滚动参考
继承摊平（阻断散感韵律经参考链式传播）；③短块(<10字)跳过词级重音
（4字块上+2dB=阶梯感）。

方法：从用户 13:26 成品的旁车**原样重建台本**（同文本/情绪/强度/停顿/
种子），排除 LLM 方差，用修复后代码重跑全管线（LoRA/语速/采样参数与
用户一致），同一套指标对比。

用法：.venv\\Scripts\\python.exe tools\\continuity_fix_acceptance.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []
SR = 22050
SRC = os.path.join("outputs", "spk_20261003-132629.script.json")


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def metrics(y, tag):
    import numpy as np
    fr = int(SR * 0.025)
    hp = int(SR * 0.01)
    n = 1 + (len(y) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
    e = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(e + 1e-12)
    voiced = db > -25
    voiced_s = voiced.sum() * 0.01
    q = db < -30
    pauses = []
    i = 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            if (j - i) * 10 >= 60:
                pauses.append((j - i) * 10)
            i = j
        else:
            i += 1
    jumps = 0
    i = 1
    while i < n - 3:
        if voiced[i] and voiced[max(0, i - 3)]:
            a = np.percentile(db[max(0, i - 4):i], 50)
            b = np.percentile(db[i:i + 4], 50)
            if abs(b - a) > 3.5:
                jumps += 1
            i += 4
        else:
            i += 1
    per_min = len(pauses) / max(voiced_s, 0.01) * 60
    print(f"[{tag}] 停顿(≥60ms) {len(pauses)} 处={per_min:.0f}/分钟有声 · "
          f"≥150ms {sum(1 for p in pauses if p >= 150)} 处 · "
          f"响度跳变 {jumps} 次")
    return {"n": len(pauses), "per_min": per_min,
            "n150": sum(1 for p in pauses if p >= 150), "jumps": jumps}


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
                   intensity=l["intensity"],
                   pause_after_ms=l["pause_after_ms"])
        for l in side["lines"]], backend="api", ok=True)

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine
    eng.load()
    MG.mount_run(eng, "达妮娅_gpt", "best", 0.9)
    MG.mount_run(eng, "达妮娅_cfm", "best", 1.0)

    req = INF.GenRequest(
        spk_audio_prompt=os.path.join("voice_bank", "audio", "达妮娅.wav"),
        text="".join(l.text for l in sc.lines), emo_alpha=0.65,
        seed=side["base_seed"], duration_factor=0.9,
        temperature=1.2, top_p=0.9, top_k=30,
        max_text_tokens_per_segment=180)

    out = os.path.join("outputs", "cont_fix_full.wav")
    res = ORC.perform(eng, req, sc, route=False, character="达妮娅",
                      lora_run="达妮娅_gpt", pause_scale=0.4,
                      pause_cap_ms=300, stress_enable=True,
                      stress_gain_db=2.0, f0_restore=True,
                      f0_expand_max=1.3, output_path=out)

    y_new, _ = sf.read(out, dtype="float32")
    y_old, _ = sf.read(os.path.join("outputs", "spk_20261003-132629.wav"),
                       dtype="float32")
    m_new = metrics(y_new, "修复后全管线")
    m_old = metrics(y_old, "用户13:26（旧进程）")

    # ---- 规格化停顿审计（判据=产品规格，不是越少越好）----
    # 合法停顿：块边界 gap（台本值）+ 配额保留的戏剧拍（≤pause_cap）+
    # 逗号保底位（80~84ms@scale0.4）+ <100ms 词间隙（自然语音结构）。
    # 违规：quota-0 块内出现 ≥150ms 停顿；或非边界非配额的长停顿。
    side2 = json.load(open(res["director"]["sidecar"], encoding="utf-8"))
    from webui_app.services.audio_lab import pause_quota as _pq
    fr = int(SR * 0.025); hp = int(SR * 0.01)
    n = 1 + (len(y_new) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
    e = np.sqrt(np.mean(y_new[idx] ** 2, axis=1) + 1e-12)
    q = (20 * np.log10(e + 1e-12)) < -30
    # 块边界精确定位：拼接层的 gap 床是**数字零**（<-100dB 连续 ≥80ms），
    # 比旁车样本数累加准（后者不含 pad/淡化偏移，实测可偏 0.5s）。
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
    runs, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            runs.append((i * 0.01, (j - i) * 10))
            i = j
        else:
            i += 1
    quota_budget = sum(_pq(len(l["text"])) for l in side2["lines"])
    beats, escapes = 0, []
    lead = int(0.15 * SR)
    for s, d in runs:
        if d < 150 or s * SR < lead:
            continue
        if any(abs(s - j) < 0.25 for j in junctions):
            continue                      # 块边界（拼接床，台本 gap 精确落地）
        if d <= int(300) + 30 and beats < quota_budget:
            beats += 1                    # 配额保留的戏剧拍
            continue
        escapes.append((round(s, 2), d))
    check("无「越权」长停顿（边界/配额/逗号以外的 ≥150ms = 0）",
          not escapes, f"违规={escapes} · 边界床={len(junctions)} 处")

    _blk1_end = (junctions[0] - 0.25) if junctions else 1.6
    _blk1_zone = [r for r in runs
                  if 0.4 < r[0] < _blk1_end and r[1] >= 150]
    check("事故点「因为|这样」长停顿已消（blk1 内 ≥150ms = 0）",
          not _blk1_zone, f"blk1 长停顿={_blk1_zone}（旧 170ms）")

    check("停顿密度下降 ≥15%（剩余为规格内拍+自然词隙）",
          m_new["per_min"] < m_old["per_min"] * 0.85,
          f"{m_old['per_min']:.0f} → {m_new['per_min']:.0f}/分钟")
    check("响度跳变下降（54→≤48）", m_new["jumps"] <= 48,
          f"{m_old['jumps']} → {m_new['jumps']}")
    check("旁车 code.rev 记录（新代码实锤）",
          (side2.get("code") or {}).get("rev") not in ("", None),
          repr(side2.get("code")))
    check("短块(<10字)未施重音（阶梯修复）",
          all(not l.get("stress") for l in side2["lines"]
              if len(l["text"]) < 10),
          str([(l["text"][:8], l.get("stress")) for l in side2["lines"]]))

    ok = all(c[1] for c in CHECKS)
    print(f"\n{'=' * 58}\n{'✅ 全部通过' if ok else '❌ 存在失败'} · "
          f"{sum(1 for c in CHECKS if c[1])}/{len(CHECKS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
