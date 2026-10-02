"""真机验收：块内逗号保底位（2026-10-02 第五轮停顿治理）。

事故：参考摊平（阶段28）治好词组化散感后，模型在逗号处的停顿也缩到
75~150ms——低于 150ms 门槛原样放行，≥150ms 的又被短块配额压成 35ms，
听感「逗号几乎没停顿」。修复：normalize_intra_pauses 标点感知模式，
逗号类标点按位置匹配已实现静音并保证落在导演逗号带内。

A/B 基线：outputs/spk_repro_flatten.wav（阶段28同文本同种子同角色的
复现件，当时块内逗号 ≈100ms 检出）。本脚本用完全相同的文本/种子/角色/
LoRA 重跑一次（修复后代码），归因测量块内逗号停顿是否进入设计带。

用法：.venv\\Scripts\\python.exe tools\\comma_pause_acceptance.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


# 与 outputs/spk_repro_flatten.script.json 完全一致（seed 1978276345）
TEXT = "明明可以选择逃避，却偏要把一切都背负起来……\n我就是讨厌这样的生存方式。"
SEED = 1978276345
CHARACTER = "达妮娅"
LORA_RUN = "达妮娅_gpt"
SR = 22050


def _runs_ms(y, min_ms=40):
    """[(start_s, dur_ms)]，与治理同参（25ms 帧/10ms hop/-30dB）。"""
    import numpy as np
    fr = int(SR * 0.025)
    hp = int(SR * 0.01)
    n = 1 + (len(y) - fr) // hp
    if n <= 0:
        return []
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
                out.append((round(i * 0.01, 3), (j - i) * 10))
            i = j
        else:
            i += 1
    return out


def main() -> int:
    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import audio_lab as AL
    from webui_app.services import director as DR
    from webui_app.services import inference as INF
    from webui_app.services import orchestrator as ORC
    from webui_app.training import merge as MG

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine

    print("== 1. 引擎与 LoRA ==")
    t0 = time.perf_counter()
    eng.load()
    MG.mount_run(eng, LORA_RUN, "best", 1.0)
    print(f"  加载+挂载 {time.perf_counter() - t0:.1f}s · 显存 "
          f"{eng.stats.vram_alloc_gb:.2f} GB")

    print("== 2. 合成（修复后管线，其余设置与基线一致） ==")
    sc = DR.direct(TEXT, backend="rules", character=CHARACTER, seed=SEED,
                   pause_scale=1.0)
    req = INF.GenRequest(spk_audio_prompt=os.path.join("voice_bank", "audio",
                                                       "达妮娅.wav"),
                         text=TEXT, emo_alpha=0.65, seed=SEED)
    out_path = os.path.join("outputs", "comma_fix_after.wav")
    res = ORC.perform(eng, req, sc, route=False, character=CHARACTER,
                      lora_run=LORA_RUN, pause_scale=1.0,
                      output_path=out_path)
    side = json.load(open(res["director"]["sidecar"], encoding="utf-8"))
    check("台本旁车：2 块 · 参考已摊平 · 块1文本含逗号",
          side["blocks"] == 2
          and all(x.get("ref_flattened") for x in side["lines"])
          and "，" in side["lines"][0]["text"],
          side["lines"][0]["text"])

    import numpy as np
    import soundfile as sf
    y, sr = sf.read(out_path, dtype="float32")
    check("输出采样率与时长正常", sr == SR and len(y) > SR,
          f"{len(y) / sr:.2f}s")

    print("== 3. 块1逗号归因测量 ==")
    runs = _runs_ms(y)
    # 块1→块2 的台本 gap（……带，~600ms@scale1.0）是文件里首个 ≥450ms 的
    # 长停顿——它之前的所有静音都属于块1（首块不受拼接修剪影响）
    blk1_end = next((s + d / 1000.0 for s, d in runs if d >= 450), None)
    check("找到块间长停顿（……带）定出块1区间", blk1_end is not None,
          f"blk1 ≈ [0, {blk1_end:.2f}s]")
    blk1_runs = [(s, d) for s, d in runs if s < blk1_end - 0.05]
    # 块间 gap（≥450ms 那条）是台本级停顿，不参与「块内」判据
    blk1_inner = [(s, d) for s, d in blk1_runs if d < 450]
    print(f"  块1静音段（≥40ms 检出）：{blk1_runs}")

    blk1_text = side["lines"][0]["text"]
    blk1_slice = y[:int(blk1_end * sr)]
    fr = int(sr * 0.025)
    hp = int(sr * 0.01)
    n_frames = 1 + (len(blk1_slice) - fr) // hp
    idx = np.arange(fr)[None, :] + hp * np.arange(n_frames)[:, None]
    e = np.sqrt(np.mean(blk1_slice[idx] ** 2, axis=1) + 1e-12)
    quiet = (20 * np.log10(e + 1e-12)) < -30
    est = AL._punct_mark_times(blk1_text, quiet, hp / sr)
    print(f"  块1逗号估计时刻：{[round(t, 2) for t in est]}")
    comma = min(blk1_runs, key=lambda r: abs(r[0] + r[1] / 2000 - est[0])) \
        if blk1_runs and est else None
    check("逗号停顿进入设计带（检出 ≥110ms；基线 ≈100ms）",
          comma is not None and comma[1] >= 110,
          f"逗号段 = {comma}ms · 基线 ≈100ms")
    others = [r for r in blk1_inner if r != comma]
    # 判据：除逗号外，其余停顿不超块内封顶（220ms，检出域 ≤240）——
    # 配额保留的戏剧拍合法存在；150~159ms 检出段是旧门槛的既有逃逸
    # （样本级 150ms 门槛 ≈ 检出 157ms 起，历版校准如此），不算回归
    check("其余停顿全部在块内封顶内（散感治理保持）",
          all(d <= 240 for _, d in others),
          f"{others}")

    print("== 4. 基线对照（阶段28复现件同种子） ==")
    base = os.path.join("outputs", "spk_repro_flatten.wav")
    if os.path.isfile(base):
        yb, _ = sf.read(base, dtype="float32")
        base_runs = _runs_ms(yb)
        base_blk1_end = next((s2 + d2 / 1000 for s2, d2 in base_runs
                              if d2 >= 400), 4.6)
        base_blk1 = [(s, d) for s, d in base_runs
                     if s < base_blk1_end - 0.05]
        base_comma = min(
            base_blk1, key=lambda r: abs(r[0] + r[1] / 2000 - 1.79)) \
            if base_blk1 else None
        print(f"  基线块1静音段：{base_blk1}")
        check("修复后逗号停顿显著长于基线（≥110 vs "
              f"{base_comma[1] if base_comma else '?'}ms）",
              comma is not None and base_comma is not None
              and comma[1] > base_comma[1],
              f"{base_comma} → {comma}")

    ok = all(c[1] for c in CHECKS)
    print(f"\n{'=' * 60}\n{'✅ 全部通过' if ok else '❌ 存在失败'} · "
          f"{sum(1 for c in CHECKS if c[1])}/{len(CHECKS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
