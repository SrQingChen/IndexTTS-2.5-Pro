"""连续性归因 A/B 探针（2026-10-03 句内散感/语气突变投诉）。

三组对照，同一文本同一种子：
    A. 裸底座（不挂 LoRA、不走导演/编排/任何输出治理）——「原始模型」
    B. 底座 + 达妮娅双 LoRA（gpt 0.9 / cfm 1.0），仍不走编排——模型侧归因
    C. 用户最新全管线成品（spk_20261003-132629.wav）——现网听感

量化：停顿密度（检出 ≥60ms / 分钟有声）、响度跳变（>3.5dB/75ms 频率）、
有声语速（字/秒）、F0 相邻窗口中位漂移（>3 半音记一次「换挡」）。

用法：.venv\\Scripts\\python.exe tools\\continuity_ab_probe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 22050
TEXT = ("因为这样……等她意识到我已经离开的时候，也就不会那么难过了啊……"
        "死掉一个曾经的朋友，总比死掉一个现在的朋友要好受吧？")
N_CHARS = 34          # 非标点字符数（语速分母）
SEED = 20261003
REF = os.path.join("voice_bank", "audio", "达妮娅.wav")


def metrics(path, tag):
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    fr = int(sr * 0.025)
    hp = int(sr * 0.01)
    n = 1 + (len(y) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
    e = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(e + 1e-12)
    voiced = db > -25
    dur = len(y) / sr
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

    # F0 换挡：相邻 300ms 有声窗的 F0 中位漂移 >3 半音
    shifts = 0
    try:
        import pyworld as pw
        f0, _t = pw.harvest(y.astype(np.float64), sr,
                            f0_floor=70.0, f0_ceil=600.0)
        fw = 30                       # 300ms / 10ms 帧
        meds = []
        for s in range(0, max(1, len(f0) - fw), fw):
            w = f0[s:s + fw]
            w = w[w > 0]
            if len(w) >= 10:
                meds.append(float(np.median(w)))
        for a, b in zip(meds, meds[1:]):
            if abs(12 * np.log2(b / a)) > 3.0:
                shifts += 1
    except Exception as ex:
        shifts = -1

    rate = N_CHARS / voiced_s if voiced_s > 0 else 0
    per_min = len(pauses) / max(voiced_s, 0.01) * 60
    print(f"[{tag}] 时长{dur:.2f}s 有声{voiced_s:.2f}s · "
          f"停顿(≥60ms) {len(pauses)} 处 = {per_min:.0f}/分钟有声 · "
          f"其中≥150ms {sum(1 for p in pauses if p >= 150)} 处 · "
          f"响度跳变 {jumps} 次 · F0换挡 {shifts} 次 · "
          f"有声语速 {rate:.2f} 字/秒")
    return {"pauses": len(pauses), "per_min": per_min, "jumps": jumps,
            "shifts": shifts, "rate": rate}


def main() -> int:
    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import inference as INF
    from webui_app.services.engine import EngineError
    from webui_app.training import merge as MG

    # ---- C. 用户成品 ----
    c_path = os.path.join("outputs", "spk_20261003-132629.wav")
    if os.path.isfile(c_path):
        metrics(c_path, "C 全管线·用户13:26")
    c2_path = os.path.join("outputs", "spk_20261003-131406.wav")
    if os.path.isfile(c2_path):
        metrics(c2_path, "C2 全管线·用户13:14")

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine
    eng.load()

    req = INF.GenRequest(spk_audio_prompt=REF, text=TEXT,
                         emo_alpha=0.65, seed=SEED,
                         max_text_tokens_per_segment=180)

    # ---- A. 裸底座 ----
    out_a = os.path.join("outputs", "cont_ab_A_base.wav")
    INF.apply_seed(INF.resolve_seed(SEED))
    INF.generate(eng, req, output_path=out_a)
    metrics(out_a, "A 裸底座·无LoRA无编排")

    # ---- B. 底座 + 双LoRA（用户同参数）----
    MG.mount_run(eng, "达妮娅_gpt", "best", 0.9)
    MG.mount_run(eng, "达妮娅_cfm", "best", 1.0)
    out_b = os.path.join("outputs", "cont_ab_B_lora.wav")
    INF.apply_seed(INF.resolve_seed(SEED))
    INF.generate(eng, req, output_path=out_b)
    metrics(out_b, "B 底座+达妮娅双LoRA")

    print("\n归因：B-A = LoRA(训练素材词组化韵律先验)的贡献；"
          "C-B = 编排/后处理链的贡献。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
