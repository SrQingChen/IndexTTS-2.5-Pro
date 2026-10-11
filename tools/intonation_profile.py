"""给已有训练 run 补算语调指纹（无需重训）。

用法：
  .venv/Scripts/python.exe tools/intonation_profile.py --run 新锁暝_gpt
  .venv/Scripts/python.exe tools/intonation_profile.py --dataset oneclick_20261011-1556

从数据集切片实测「问句上翘/陈述下落/感叹」分布 → 写进 run.json 的
intonation_profile（与响度/F0/频谱指纹同一交付模式）。perform() 每次
合成都现读 run.json——写完即生效，无需重启。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="", help="数据集名（默认从 --run 反查）")
    ap.add_argument("--runs", default="", help="逗号分隔的 run 名（默认=该数据集训练的全部 run）")
    args = ap.parse_args()

    from webui_app.services import audio_lab as AL
    from webui_app.training import dataset as DS
    from webui_app.training import runs as RN

    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    dataset = args.dataset
    if not dataset and run_names:
        dataset = (RN.read_run(run_names[0]) or {}).get("dataset") or ""
    if not dataset:
        print("需要 --dataset 或 --run"); return 1
    if not run_names:
        run_names = [ri.name for ri in RN.list_runs()
                     if ri.dataset == dataset]

    metrics = []
    # 与 S2.6 指纹同一口径：只测**可训练时长**的 wav（长原件不参与训练，
    # 卷进指纹是 2026-10-02 两次 C 层崩溃事故的根源）
    cap = 20.0
    ds_dir = DS.dir_of(dataset)
    items = [(u, u.audio_abs(ds_dir)) for u in DS.load_meta(dataset)
             if u.status == "ready"
             and (u.audio_abs(ds_dir) or "").lower().endswith(".wav")
             and os.path.isfile(u.audio_abs(ds_dir))
             and float(u.duration or 0.0) <= cap]
    n_done = 0
    for u, audio in items:
        txt = (u.text or u.asr_text or "").strip()
        if not txt:
            continue
        try:
            import soundfile as sf
            y, sr = sf.read(audio, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            metrics.extend(AL.analyze_intonation_clip(y, sr, txt))
            n_done += 1
        except Exception as e:
            print(f"  跳过 {u.id}: {type(e).__name__}: {e}")
    if not metrics:
        print("没有可测量的短语（检查文本与音频）"); return 1
    prof = AL.aggregate_intonation_profile(metrics)
    prof["clips_analyzed"] = n_done

    qr = prof.get("question_rise") or {}
    sf_ = prof.get("statement_fall") or {}
    print(f"分析 {n_done} 条切片 · {prof['total_phrases']} 个短语：")
    print(f"  问句上翘: n={qr.get('n')} "
          + (f"中位 {qr.get('median_st')}st (p25 {qr.get('p25')} / "
             f"p75 {qr.get('p75')})" if qr.get("n", 0) >= 2 else "（样本不足，推理回退通用默认）"))
    print(f"  陈述下落: n={sf_.get('n')} "
          + (f"中位 {sf_.get('median_st')}st" if sf_.get("n", 0) >= 2 else "（样本不足）"))

    for rn in run_names:
        RN.update_run(rn, intonation_profile=prof)
        print(f"✅ 已写入 {rn} 的 intonation_profile（推理端下次合成即生效，无需重启）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
