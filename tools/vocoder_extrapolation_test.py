"""vocoder_extrapolation_test — Perrotin 式声码器音高外推测试（E1-T Step 12 入口）。

背景（Perrotin et al., Interspeech 2021, hal-03338483）：神经声码器在
训练分布外的极端 F0 上可能「塌回训练集中位音区」——高音歌声正是这种
场景（22k 语音训练的 BigVGAN 渲染 500-900Hz）。方法：构造**已知 F0** 的
谐波信号 → 过 mel → 声码器重建 → 测重建信号的谐波能量是否落在目标 F0
（而不是塌到训练中位 ~200Hz）。

指标：
- f0_tracking_error：重建信号 harvest F0 与目标 F0 的中位偏差（音分）；
- octave_collapse_rate：|偏差|>600 音分的帧比例（塌中位/塌八度的直接证据）；
- 高音子集（>500Hz）单独统计——整首平均会掩盖塌陷。

用法：
    python tools/vocoder_extrapolation_test.py \
        [--ckpt training_runs/bigvgan_singing/bigvgan_singing.pt]

不传 --ckpt 时测默认 vocoder（对照基线）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.config import PROJECT_ROOT  # noqa: E402

SR = 22050
F0_SET = [98.0, 147.0, 220.0, 330.0, 440.0, 523.0, 659.0, 784.0, 880.0, 1046.0]


def harmonic(f0: float, seconds: float = 2.0, partials: int = 8) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros(len(t))
    for k in range(1, partials + 1):
        if f0 * k > SR / 2:
            break
        y += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    return (0.4 * y / max(np.max(np.abs(y)), 1e-9)).astype(np.float32)


def run(ckpt: str, device: str):
    import torch

    # 引擎同款入口（验收 P1-2）：BigVGAN.from_pretrained 处理 config+权重
    from indextts.s2mel.modules.bigvgan import bigvgan as bv
    ckpt_dir = os.path.dirname(ckpt) if ckpt else os.path.join(
        PROJECT_ROOT, "checkpoints", "hf_cache", "bigvgan")
    gen = bv.BigVGAN.from_pretrained(ckpt_dir).to(device).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("generator", sd) if isinstance(sd, dict) else sd
        r = gen.load_state_dict(sd, strict=False)
        print(f"[加载] {ckpt}（missing={len(r.missing_keys)}, "
              f"unexpected={len(r.unexpected_keys)}）")
    else:
        print("[加载] 默认 vocoder（对照基线）")

    import torchaudio
    from webui_app.services import svc_f0
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, f_min=0.0).to(device)
    # 与引擎 mel_fn 的一致性说明：引擎用 log-mel（libs 转换在 BigVGAN 内做），
    # 这里按 BigVGAN 输入约定给幅度 mel；若有偏差，塌陷指标的**相对比较**
    # （同一前端测两个 ckpt）依然成立。

    rows = []
    with torch.no_grad():
        for f0 in F0_SET:
            y = harmonic(f0)
            wav = torch.tensor(y).unsqueeze(0).to(device)
            mel = mel_fn(wav).unsqueeze(0)                    # (1,80,T)
            rec = gen(mel.float()).squeeze().cpu().numpy()
            rec = rec[:len(y)]
            est = svc_f0.harvest_f0(rec.astype(np.float64), SR)
            tgt = np.full(len(est), f0)
            r, err, nv = svc_f0.f0_correlation(est, tgt)
            both = (est > 0) & (tgt[:len(est)] > 0)
            cents = np.abs(1200 * np.log2(np.where(both, est, 1.0)
                                          / tgt[:len(est)]))
            rows.append({
                "f0_target": f0,
                "f0_est_median": round(float(np.median(est[est > 0])) if
                                       (est > 0).any() else 0.0, 1),
                "median_cents_err": round(float(np.median(cents[both])), 1)
                if both.any() else None,
                "octave_collapse": round(float((cents[both] > 600).mean()), 4)
                if both.any() else None,
                "corr_r": round(r, 3),
            })
            print(f"F0={f0:7.1f}Hz → est中位 {rows[-1]['f0_est_median']:7.1f}Hz "
                  f"偏差 {rows[-1]['median_cents_err']}¢ "
                  f"塌陷率 {rows[-1]['octave_collapse']}")

    hi = [x for x in rows if x["f0_target"] > 500]
    report = {
        "ckpt": ckpt or "(default)",
        "rows": rows,
        "high_subset": {
            "mean_octave_collapse": round(float(np.mean(
                [x["octave_collapse"] for x in hi])), 4) if hi else None,
        },
        "verdict_note": "高音子集塌陷率应显著低于对照基线（Perrotin 式判据）；"
                        "绝对值受 mel 前端一致性影响，以相对比较为准。",
    }
    out = os.path.join(PROJECT_ROOT, "outputs", "vocoder_extrapolation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"[报告] {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="",
                    help="待测 vocoder ckpt（空=默认对照）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.ckpt, args.device)


if __name__ == "__main__":
    main()
