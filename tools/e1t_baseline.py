"""e1t_baseline — 对既有翻唱产物生成 E1-T 基线 JSON（Step 0.3 / r1-7）。

对 `outputs/cover/<时间戳>/` 里的分阶段文件（02 干声 / 03 转换 / 04 F0 /
05 成品）计算逐句 HNR 代理 / 谱平坦度 / F0 / SS，写入
`outputs/e1t_baseline.json`——G1a/G1b 评测的对比锚点（E1-T 母规划 §1 的
诊断口径：HNR 代理 = HPSS 谐波/残余比，同管线内可比，不是 Praat HNR）。

用法：
    python tools/e1t_baseline.py                    # 最新一次 run
    python tools/e1t_baseline.py --run 20261017-154205
    python tools/e1t_baseline.py --ref voice_bank/audio/达妮娅.wav
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.config import PROJECT_ROOT  # noqa: E402

COVER_ROOT = os.path.join(PROJECT_ROOT, "outputs", "cover")
SR = 22050
STAGES = ("02_dry_vocal_22k.wav", "03_converted_22k.wav",
          "04_f0_vocal.wav", "05_vocal_final_22k.wav")


def _spectral_flatness(y: np.ndarray, sr: int) -> float:
    import librosa
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=1024))
    g = np.exp(np.mean(np.log(S + 1e-10), axis=0))
    a = np.mean(S, axis=0)
    return float(np.median(g / (a + 1e-10)))


def analyze_stage(path: str, seg_sec: float = 10.0) -> dict:
    import librosa
    from webui_app.services import svc_f0, svc_singing
    y, _ = librosa.load(path, sr=SR, mono=True)
    f0 = svc_f0.harvest_f0(y, SR)
    rows = []
    seg_n = int(seg_sec / 0.005)
    for s in range(0, len(f0), seg_n):
        seg = f0[s:s + seg_n]
        v = seg[seg > 0]
        if len(v) < 3:
            continue
        a = int(s * 0.005 * SR)
        b = int(min(len(y), (s + seg_n) * 0.005 * SR))
        h = svc_singing.voiced_hnr(y[a:b], SR, seg)
        rows.append({
            "t0": round(s * 0.005, 1),
            "f0_med": round(float(np.median(v)), 1),
            "hnr_db": None if h is None else round(h, 2),
        })
    return {
        "file": os.path.basename(path),
        "f0_median": round(svc_f0.f0_median(f0), 1),
        "spectral_flatness": round(_spectral_flatness(y, SR), 5),
        "hnr_voiced_db": svc_singing.voiced_hnr(y, SR, f0),
        "segments": rows,
    }


def _ss(path: str, ref_path: str) -> float:
    from webui_app.training.reward import RewardScorer, RewardOptions
    sc = RewardScorer(RewardOptions(device="cpu"))
    a, b = sc.embed(path), sc.embed(ref_path)
    v = float(RewardScorer.cosine(a, b))
    sc.unload()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="outputs/cover 下的时间戳目录（空=最新）")
    ap.add_argument("--ref", default=os.path.join(
        PROJECT_ROOT, "voice_bank", "audio", "达妮娅.wav"),
        help="SS 对比的参考音频")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs",
                                                  "e1t_baseline.json"))
    args = ap.parse_args()

    if args.run:
        run_dir = os.path.join(COVER_ROOT, args.run)
    else:
        cands = sorted(glob.glob(os.path.join(COVER_ROOT, "*")))
        cands = [c for c in cands if os.path.isdir(c)]
        if not cands:
            print("[错误] outputs/cover 下没有产物目录")
            return 2
        run_dir = cands[-1]
    if not os.path.isdir(run_dir):
        print(f"[错误] 目录不存在: {run_dir}")
        return 2

    out = {"run": os.path.basename(run_dir), "ref": args.ref,
           "sr": SR, "stage": "baseline", "stages": {}}
    for name in STAGES:
        p = os.path.join(run_dir, name)
        if not os.path.isfile(p):
            continue
        d = analyze_stage(p)
        if os.path.isfile(args.ref):
            try:
                d["ss"] = round(_ss(p, args.ref), 4)
            except Exception as e:
                d["ss_error"] = str(e)
        out["stages"][name.split("_")[0]] = d
        print(f"[{name}] f0中位 {d['f0_median']}Hz HNR {d['hnr_voiced_db']}dB "
              f"平坦度 {d['spectral_flatness']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[基线] → {args.out}\n后续 G1a/G1b 评测与本文件对比"
          "（同口径：HPSS HNR 代理 + 逐句分段）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
