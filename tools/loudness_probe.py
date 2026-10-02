"""探针：响度指纹体系（中位数锚定 / 指纹采样 / 逐块增益）+ 呼吸库。

纯 numpy/soundfile，不加载引擎。用法：
python tools/loudness_probe.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from webui_app.services import audio_lab as AL  # noqa: E402

SR = 22050
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def _tone(sec, amp):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="loud_probe_")

    print("== 1. 中位数锚定：相对响度保留 ==")
    # 四条素材原生 -14/-17/-20/-26 dB（表演起伏）
    amps = [10 ** (-14 / 20), 10 ** (-17 / 20), 10 ** (-20 / 20), 10 ** (-26 / 20)]
    levels = [AL.measure_loudness(_tone(1.0, a)) for a in amps]
    anchor = AL.anchor_gain(levels, target_dbfs=-20.0)
    after = []
    for a in amps:
        y2 = AL.apply_anchor(_tone(1.0, a), anchor)
        after.append(AL.rms_dbfs(y2))
    check("全局增益 = 目标-中位（限 ±12dB）",
          abs(anchor["gain_db"] - (-20.0 - anchor["median_db"])) < 0.01,
          f"gain={anchor['gain_db']:+.2f}dB med={anchor['median_db']:.1f}")
    diffs = [round(after[i + 1] - after[i], 2) for i in range(3)]
    src = [round(levels[i + 1] - levels[i], 2) for i in range(3)]
    check("片段间相对差原样保留", diffs == src, f"{src} → {diffs}")
    check("最响片段仍是 -14 一档（+全局增益）",
          abs(after[0] - (levels[0] + anchor["gain_db"])) < 0.15,
          f"{after[0]:.1f} vs {levels[0] + anchor['gain_db']:.1f}")
    peak = float(np.max(np.abs(AL.apply_anchor(
        _tone(1.0, 10 ** (-2 / 20)), anchor))))
    check("峰值保护 ≤ -1dBFS", peak <= 10 ** (-0.99 / 20), f"peak={20*np.log10(peak+1e-12):.2f}dB")
    # 离群钳制：-40dB 的极端素材被拉回边界而不是拖垮全局
    lv2 = levels + [AL.measure_loudness(_tone(1.0, 10 ** (-40 / 20)))]
    a2 = AL.anchor_gain(lv2)
    check("离群不拖垮中位数（±10dB 钳制，容样本数变化的微小移动）",
          abs(a2["median_db"] - AL.anchor_gain(levels)["median_db"]) < 2.0,
          f"{a2['median_db']:.1f}")

    print("== 2. 响度指纹与逐块采样 ==")
    fp = AL.loudness_fingerprint(levels + [-18.5, -21.0])
    check("指纹字段齐全", all(k in fp for k in
          ("n", "median", "p25", "p75", "p10", "p90", "std")), str(fp))
    rng = random.Random(42)
    loud_targets = [AL.sample_target_loudness(fp, 0.9, rng) for _ in range(20)]
    quiet_targets = [AL.sample_target_loudness(fp, 0.2, rng) for _ in range(20)]
    check("爆发块从响端采（≥p60）",
          all(t >= fp["p60"] - 1e-6 for t in loud_targets),
          f"min={min(loud_targets)} ≥ p60={fp['p60']}")
    check("平静块从轻端采（≤p40/p25）",
          all(t <= fp.get("p40", fp["p25"]) + 1e-6 for t in quiet_targets),
          f"max={max(quiet_targets)}")
    check("爆发普遍比平静响", min(loud_targets) > max(quiet_targets),
          f"{min(loud_targets)} vs {max(quiet_targets)}")
    check("无指纹回退 -20（旧行为）",
          AL.sample_target_loudness({}, 0.5, rng) == -20.0)

    print("== 3. 逐块增益与起止整形 ==")
    y = _tone(0.5, 0.5)
    y2 = AL.apply_block_loudness(y, -26.0)
    check("块增益把 RMS 打到目标", abs(AL.rms_dbfs(y2) + 26.0) < 0.1,
          f"{AL.rms_dbfs(y2):.1f}")
    yi = (_tone(0.5, 0.4) * 32767).astype(np.int16)
    ye = AL.shape_edges(yi, SR, pad_ms=80.0, fade_ms=12.0)
    check("int16 起止整形不炸 dtype",
          ye.dtype == np.int16 and len(ye) == len(yi) + 2 * int(SR * 0.08))
    check("首尾被淡化", float(np.max(np.abs(ye[:int(SR*0.012)]))) <
          float(np.max(np.abs(yi))) * 0.5)
    # 中段不受影响
    mid0 = yi[int(len(yi)*0.4): int(len(yi)*0.6)]
    mid1 = ye[int(len(yi)*0.4) + int(SR*0.08): int(len(yi)*0.6) + int(SR*0.08)]
    check("中段原样", np.array_equal(mid0, mid1))

    # ---- 3b. int16 尺度陷阱回归（2026-10-02 成品静音事故）----
    # 编排器把 int16 的块直接传给 apply_stress_gains / apply_block_loudness；
    # 两函数内部的峰值阈值（±1 尺度）曾直接与 ±32767 尺度的峰值比较，
    # 重音段被整体压到数字静音——成品听感「一句只剩一两个残缺声音」。
    _y16 = (_tone(3.0, 0.5) * 32767).astype(np.int16)
    _spans = [(1.0, 2.0)]
    _g16 = AL.apply_stress_gains(_y16, SR, _spans, gain_db=2.0)
    _norm = lambda v: (v.astype(np.float32) / 32768.0
                       if np.issubdtype(v.dtype, np.integer)
                       else v.astype(np.float32))
    _r = lambda v, a, b: 20 * np.log10(np.sqrt(np.mean(
        _norm(v)[int(a*SR):int(b*SR)] ** 2)) + 1e-12)
    check("stress int16：重音区增益正常（非静音）",
          _r(_g16, 1.1, 1.9) - _r(_y16, 1.1, 1.9) > 1.0
          and float(np.max(np.abs(_g16))) > 1000,
          f"重音区 {_r(_y16,1.1,1.9):.1f}→{_r(_g16,1.1,1.9):.1f}dB · "
          f"peak={float(np.max(np.abs(_g16))):.0f}")
    check("stress int16：非重音区原样",
          abs(_r(_g16, 0.1, 0.9) - _r(_y16, 0.1, 0.9)) < 0.01)
    _f32 = AL.apply_stress_gains(_tone(3.0, 0.5), SR, _spans, gain_db=2.0)
    check("stress float32：与 int16 行为一致",
          abs(_r(_f32, 1.1, 1.9) - _r(_g16, 1.1, 1.9)) < 0.1)
    _b16 = AL.apply_block_loudness(_y16, -20.0)
    check("block_loudness int16：RMS 精确到目标",
          _b16.dtype == np.int16 and abs(AL.rms_dbfs(
              _b16.astype(np.float32) / 32768.0) + 20.0) < 0.1,
          f"{AL.rms_dbfs(_b16.astype(np.float32) / 32768.0):.1f}")

    print("== 4. 呼吸库（检测/建库/插入决策） ==")
    from webui_app.services import breath_bank as BB
    # 合成"吸气":低幅高频噪声 0.3s → 语音 0.8s
    t = np.arange(int(0.3 * SR)) / SR
    inh = (0.01 * np.random.default_rng(3).normal(size=len(t))).astype(np.float32)
    speech = _tone(0.8, 0.3)
    probe_wav = os.path.join(tmp, "clip.wav")
    sf.write(probe_wav, np.concatenate([inh, speech]), SR)
    segs = BB.detect_inhales(probe_wav)
    check("检测出语音前的吸气段", bool(segs) and
          abs(segs[0]["start"]) < 0.05, str(segs[:1]))
    # maybe_inhale 决策：gap<200 不插；概率性；无库不插
    rng2 = random.Random(1)
    check("gap<200ms 不插",
          BB.maybe_inhale("无此角色", 150, 0.5, rng2) is None)
    BB.BANK_DIR = os.path.join(tmp, "breath_bank")
    BB.INDEX_FILE = os.path.join(BB.BANK_DIR, "index.json")
    os.makedirs(BB.BANK_DIR, exist_ok=True)
    # 手工放一条采样进库
    os.makedirs(os.path.join(BB.BANK_DIR, "audio"), exist_ok=True)
    sf.write(os.path.join(BB.BANK_DIR, "audio", "t_001.wav"),
             inh, SR)
    BB._save([{"name": "t_001", "character": "测试", "audio": "audio/t_001.wav",
               "duration": 0.3, "score": 0.8, "source": "", "created_at": 0}])
    got = [BB.maybe_inhale("测试", 600, 0.8, random.Random(i)) for i in range(12)]
    n_hit = sum(1 for g in got if g is not None)
    check("有停顿时概率性插入（~50-60%）", 3 <= n_hit <= 11, f"{n_hit}/12")
    # int16 波形先归一化到 ±1 域再测 dBFS
    _lv = [AL.rms_dbfs(g.astype(np.float32) / 32768.0)
           for g in got if g is not None]
    check("返回波形电平在 -24~-30dBFS 邻域",
          all(-32 <= v <= -22 for v in _lv), f"{[round(v,1) for v in _lv][:3]}")

    print("== 5. 频谱画像与包络匹配（不饱满/缺频段根治件） ==")
    # 闷源(强低频+极弱高频——真实"闷人声"的形状;纯正弦无高频可提升,
    # 那是增益不是谐波发生器,测试用例必须含可被抬升的弱高频) vs 亮目标
    _tt = np.arange(int(1.5 * SR)) / SR
    _dull = (0.4 * np.sin(2 * np.pi * 220 * _tt)
             + 0.004 * np.sin(2 * np.pi * 5600 * _tt)).astype(np.float32)
    _bright = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(1.5 * SR)) / SR)
               + 0.15 * np.sin(2 * np.pi * 1320 * np.arange(int(1.5 * SR)) / SR)
               + 0.08 * np.sin(2 * np.pi * 5600 * np.arange(int(1.5 * SR)) / SR)
               ).astype(np.float32)
    _tp = AL.band_profile(_bright)
    _cp0 = AL.band_profile(_dull)
    _y_m = AL.match_band_profile(_dull, SR, _tp)
    _cp1 = AL.band_profile(_y_m)
    check("画像输出 6 频段+质心", len(_cp0.get("bands", [])) == 6
          and "centroid" in _cp0)
    hi0 = sum(_cp0["bands"][3:])      # 3k 以上占比
    hi1 = sum(_cp1["bands"][3:])
    check("闷源匹配后高频占比显著抬升(≥1.5×)", hi1 > hi0 * 1.5,
          f"{hi0:.2f}% → {hi1:.2f}%")
    check("匹配不改时长且限幅生效（质心抬但不越界）",
          len(_y_m) == len(_dull)
          and _cp1["centroid"] > _cp0["centroid"]
          and _cp1["centroid"] < _tp["centroid"] * 1.6,
          f"{_cp0['centroid']} → {_cp1['centroid']} (目标 {_tp['centroid']})")
    _sil = np.concatenate([np.zeros(int(SR * 2), np.float32), _tone(1.0, 0.3)])
    check("画像忽略静音段（含 2s 前导静音仍出有效画像）",
          bool(AL.band_profile(_sil)))
    _pp = AL.dataset_band_profile([_dull, _bright, _tone(1.2, 0.2)])
    check("多素材平均画像", _pp.get("n") == 3 and len(_pp.get("bands", [])) == 6)

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
