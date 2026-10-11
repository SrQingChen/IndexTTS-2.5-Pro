"""Checkpoint 锦标赛：平台期附近取 N 个 checkpoint 自动评测选优。

用户提案（2026-10-11）：长训到 loss 趋平，取平台期开始的附近的三个
LoRA 作为待评测模型，用统一文本+种子合成，客观打分排名。

流程：
  1. 读 run.json 的 train loss 曲线，检测**平台期起点**（平滑斜率
     连续两点 |slope| < 阈值）；
  2. checkpoint 列表里取平台起点附近的 --top 个（不足则全取）；
  3. 每个候选挂载（GPT@ckpt + CFM@best），固定种子合成 --seeds 次；
  4. 打分三轴：reward（ASR 回转 WER + 声纹相似度，CPU）、
     撕裂率（浊音帧 HF 超语音 +12dB 的占比，越低越好）、
     间隙爆冲（同前），合成综合分 rank；
  5. 表格输出 + 旁车 json（outputs/tournament/<run>.json）。

用法：
  .venv/Scripts/python.exe tools/checkpoint_tournament.py \
      --run 锁暝_gpt --cfm-run 锁暝_cfm --seeds 3 --top 3
可选：--text 自定评测文本；--ref 指定声纹参考（默认音色库同名条目）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import butter, sosfiltfilt  # noqa: E402

SR = 22050
DEFAULT_TEXT = "御者，晚上好，不知近日来是否有睡个好觉？"


def plateau_onset(history: list, eps: float = 0.01) -> tuple:
    """train loss 平滑斜率连续两点 |slope|<eps 的起点。

    返回 (step, still_dropping)。找不到平台（loss 仍陡降）时
    still_dropping=True，调用方应建议续训。
    """
    pts = [(x["step"], x["train"]) for x in history
           if x.get("train") is not None]
    if len(pts) < 4:
        return (pts[-1][0] if pts else 0), True
    slopes = []
    for i in range(2, len(pts)):
        ds = pts[i][0] - pts[i - 2][0]
        slopes.append((pts[i][0], (pts[i][1] - pts[i - 2][1]) / ds if ds else 0))
    for k in range(len(slopes) - 1):
        if abs(slopes[k][1]) < eps and abs(slopes[k + 1][1]) < eps:
            return slopes[k][0], False
    return slopes[-1][0], True


def hf_metrics(path: str) -> dict:
    """撕裂率（浊音帧 HF 超语音 +12dB 占比）与间隙爆冲簇数。"""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    sos_hf = butter(4, [5500 / (sr / 2), 11000 / (sr / 2)],
                    btype="band", output="sos")
    sos_vb = butter(4, [300 / (sr / 2), 3000 / (sr / 2)],
                    btype="band", output="sos")
    hf = sosfiltfilt(sos_hf, y)
    vb = sosfiltfilt(sos_vb, y)
    fr = int(sr * 0.01)
    n = len(y) // fr
    e_hf = np.sqrt(np.mean(hf[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    e_vb = np.sqrt(np.mean(vb[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    ref = float(np.percentile(e_vb, 60))
    both = e_vb > np.percentile(e_vb, 40)
    ratio_db = 20 * np.log10((e_hf + 1e-9) / (e_vb + 1e-9))
    tear = float(np.mean(ratio_db[both] > 12) * 100) if both.any() else 0.0
    wet = (e_vb < ref * 0.35) & (e_hf > ref * 0.45)
    bursts, i = 0, 0
    while i < n:
        if wet[i]:
            j = i
            while j < n and wet[j]:
                j += 1
            if j - i >= 4:
                bursts += 1
            i = j
        else:
            i += 1
    return {"tear_pct": round(tear, 2), "gap_bursts": bursts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GPT run 名")
    ap.add_argument("--cfm-run", default="", help="CFM run 名（空=不挂）")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--ref", default="", help="声纹参考路径（默认音色库同名）")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.9)
    args = ap.parse_args()

    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import audio_lab as AL
    from webui_app.services import inference as INF
    from webui_app.training import merge as MG
    from webui_app.training import reward as RW
    from webui_app.training import runs as RN

    rj = RN.read_run(args.run) or {}
    hist = rj.get("history") or []
    onset_step, still = plateau_onset(hist)
    print(f"loss 平台期检测: 起点附近 step≈{onset_step}"
          + ("（loss 仍在下降——建议先续训再评）" if still else ""))

    ckpts = RN.list_checkpoints(args.run)
    if not ckpts:
        print("没有可用 checkpoint"); return 1
    ckpts = sorted(ckpts, key=lambda c: c.step)
    onset_idx = next((k for k, c in enumerate(ckpts)
                      if c.step >= onset_step), len(ckpts) - 1)
    lo = max(0, onset_idx - 1)
    cands = ckpts[lo:lo + args.top]
    print(f"候选 checkpoint（平台期附近 {len(cands)} 个）: "
          + ", ".join(c.name for c in cands))

    ref = args.ref
    if not ref:
        char = args.run[:-4] if args.run.endswith("_gpt") else args.run
        for cand in (f"voice_bank/audio/{char}.wav",
                     f"voice_bank/audio/{char}_原.wav"):
            if os.path.isfile(cand):
                ref = cand
                break
    if not ref:
        print("找不到声纹参考（--ref 指定）"); return 1

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine
    eng.load()
    if args.cfm_run:
        MG.mount_run(eng, args.cfm_run, "best", 1.0)

    scorer = RW.RewardScorer(RW.RewardOptions(
        asr_engine="sensevoice", device="cpu",
        wer_weight=0.6, sim_weight=0.4, emo_weight=0.0, pause_weight=0.0))

    rows = []
    out_dir = os.path.join("outputs", "tournament")
    os.makedirs(out_dir, exist_ok=True)
    try:
        for ck in cands:
            MG.mount_run(eng, args.run, ck.name, args.scale)
            per_seed = []
            for si in range(args.seeds):
                seed = 900000 + si * 111
                req = INF.GenRequest(
                    spk_audio_prompt=ref, text=args.text,
                    emo_alpha=0.65, seed=seed, temperature=1.0,
                    top_p=0.9, max_text_tokens_per_segment=180)
                INF.apply_seed(seed)
                out = os.path.join(out_dir,
                                   f"{args.run}_{ck.name}_s{si}.wav")
                kw = INF.build_infer_kwargs(eng, req, out)
                eng.infer(progress=None, **kw)
                rr = scorer.score(out, args.text, ref)
                hm = hf_metrics(out)
                per_seed.append({**hm, "reward": rr.get("reward"),
                                 "wer": rr.get("wer"), "sim": rr.get("sim")})
                print(f"  {ck.name} seed{si}: reward={rr.get('reward')} "
                      f"wer={rr.get('wer')} sim={rr.get('sim')} "
                      f"撕裂{hm['tear_pct']}% 爆冲{hm['gap_bursts']}",
                      flush=True)
            tears = [p["tear_pct"] for p in per_seed]
            bursts = [p["gap_bursts"] for p in per_seed]
            rewards = [p["reward"] for p in per_seed if p["reward"] is not None]
            rows.append({
                "ckpt": ck.name, "step": ck.step,
                "reward_mean": round(sum(rewards) / len(rewards), 4)
                    if rewards else None,
                "tear_pct_mean": round(sum(tears) / len(tears), 2),
                "gap_bursts_mean": round(sum(bursts) / len(bursts), 2),
                "per_seed": per_seed,
                "composite": None,
            })
    finally:
        try:
            scorer.unload()
        except Exception:
            pass

    for r in rows:
        if r["reward_mean"] is not None:
            r["composite"] = round(
                r["reward_mean"] - r["tear_pct_mean"] / 100
                - r["gap_bursts_mean"] / 20, 4)
    valid = [r for r in rows if r["composite"] is not None]
    valid.sort(key=lambda r: -r["composite"])

    print("\n" + "=" * 72)
    print(f"{'排名':<4}{'checkpoint':<24}{'reward':<8}{'撕裂%':<8}"
          f"{'爆冲':<6}{'综合':<8}")
    for k, r in enumerate(valid, 1):
        print(f"{k:<4}{r['ckpt']:<24}{r['reward_mean']:<8.4f}"
              f"{r['tear_pct_mean']:<8.2f}{r['gap_bursts_mean']:<6.1f}"
              f"{r['composite']:<8.4f}")
    if valid:
        print(f"\n推荐: {valid[0]['ckpt']} "
              f"（综合分最高；撕裂与爆冲已计入综合分）")
    if still:
        print("⚠️ loss 未进入平台期——建议续训 4-6 epoch（或到平台）后重评。")

    side = os.path.join(out_dir, f"{args.run}.json")
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "text": args.text, "scale": args.scale,
                   "onset_step": onset_step, "still_dropping": still,
                   "rows": rows,
                   "created_at": time.time()}, f, ensure_ascii=False, indent=2)
    print(f"旁车: {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
