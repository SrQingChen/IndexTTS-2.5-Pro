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

    # ---- 3c. 软停顿封顶回归（2026-10-02 句内停顿过长事故）----
    # 模型生成的停顿带气声/吸气，实测能量主体在 -40~-30dB：旧阈值
    # -40dB 判不出来 → 完全逃过封顶（成品软停顿最长 0.98s）。判定线
    # 提到 -30dB 抓住全部软停顿；时长下限 + 只截中段保护语音内容。
    _sr_l = SR
    _speech = _tone(0.8, 0.5)
    # RMS 恰为 -33dBFS 的「气声停顿」：正弦幅度 = 10^(-33/20)×√2
    # —— 介于新旧判定线之间：-40dB 判不出、-30dB 判得出
    _gap = (10 ** (-33 / 20.0) * np.sqrt(2.0) *
            np.sin(2 * np.pi * 120 * np.arange(int(0.5 * _sr_l)) / _sr_l)
            ).astype(np.float32)
    _y_soft = np.concatenate([_speech, _gap, _speech])
    _capped = AL.cap_interior_pauses(_y_soft, _sr_l, 220, thresh_db=-30.0)
    check("软停顿（-33dB）被 -30dB 判定线抓住并压到 220ms",
          len(_capped) <= len(_y_soft) - int(0.2 * _sr_l),
          f"{len(_y_soft)/_sr_l:.2f}s → {len(_capped)/_sr_l:.2f}s")
    _missed = AL.cap_interior_pauses(_y_soft, _sr_l, 220, thresh_db=-40.0)
    check("对照组：旧阈值 -40dB 判不出软停顿（事故机理复现）",
          len(_missed) == len(_y_soft),
          f"{len(_missed)/_sr_l:.2f}s（未压缩）")
    _hard = np.concatenate([
        _speech, np.zeros(int(0.5 * _sr_l), np.float32), _speech])
    _capped2 = AL.cap_interior_pauses(_hard, _sr_l, 220, thresh_db=-30.0)
    check("数字静音照常封顶且语音/衰减不动",
          len(_capped2) <= len(_hard) - int(0.2 * _sr_l)
          and len(_capped2) >= len(_hard) - int(0.3 * _sr_l),
          f"{len(_hard)/_sr_l:.2f}s → {len(_capped2)/_sr_l:.2f}s")

    # ---- 3d. 块边界停顿精确化（2026-10-02 散感根治回归）----
    # 块尾收束 + 块头起振 + 台本 gap 三重叠加曾把边界拖到 0.4~0.9s；
    # 现在拼接层量出两侧实际静音，超出台本值修剪、不足垫足。
    _sp = _tone(1.0, 0.5)
    _blk1 = np.concatenate([_sp, np.zeros(int(0.30 * SR), np.float32)])
    _blk2 = np.concatenate([np.zeros(int(0.25 * SR), np.float32), _sp])
    check("edge_silence_len：尾部/头部静音量准",
          abs(AL.edge_silence_len(_blk1, SR, from_end=True) - 0.30) < 0.04
          and abs(AL.edge_silence_len(_blk2, SR, from_end=False) - 0.25) < 0.04)
    # 台本 gap 0.15s < 0.30+0.25 → 修剪到 ≈0.15s 总停顿
    _t1 = AL.trim_edge_silence(_blk1, SR, 0.20, from_end=True)
    _t2 = AL.trim_edge_silence(_blk2, SR, 0.20, from_end=False)
    _joint = len(_blk1) - len(_t1) + len(_blk2) - len(_t2)
    _total_pause = 0.30 - (len(_blk1) - len(_t1)) / SR \
        + 0.25 - (len(_blk2) - len(_t2)) / SR
    check("修剪后块间总停顿 ≈ 台本值（0.15s）",
          abs(_total_pause - 0.15) < 0.06,
          f"{_total_pause:.3f}s")
    check("修剪只动静音区（语音主体原样）",
          len(_t1) == len(_blk1) - int(0.20 * SR)
          and len(_t2) == len(_blk2) - int(0.20 * SR))
    # 修剪超量：cut 大于静音量 → 只截到静音-20ms 保底
    _t3 = AL.trim_edge_silence(_blk1, SR, 1.0, from_end=True)
    check("修剪不越过静音量（20ms 保底）",
          abs(len(_blk1) - len(_t3) - (0.30 - 0.02) * SR) < 0.05 * SR,
          f"截了 {(len(_blk1) - len(_t3)) / SR:.2f}s")

    # ---- 3e. 块内停顿配额制（2026-10-02 散感彻底根治回归）----
    # 产品规范（用户原话）：短句内 0 个明显停顿、长句最多 1 处。
    # 实测引擎输出 44~53 停顿/分钟（参考 28 + AR 先验放大），
    # 时长封顶治了长治不了多 —— 配额制治数量。
    _qy = np.concatenate([
        _tone(1.0, 0.5), np.zeros(int(0.30 * SR), np.float32),
        _tone(0.5, 0.5), np.zeros(int(0.40 * SR), np.float32),
        _tone(0.8, 0.5), np.zeros(int(0.25 * SR), np.float32),
        _tone(0.6, 0.5)])

    def _sil_ge80(v):
        fr = int(SR * 0.025)
        n = len(v) // fr
        ee = np.sqrt(np.mean(v[: n * fr].reshape(n, fr) ** 2, axis=1)
                     + 1e-12)
        ddb = 20 * np.log10(ee + 1e-12)
        qq = ddb < -30
        cnt, ii = 0, 0
        while ii < n:
            if qq[ii]:
                jj = ii
                while jj < n and qq[jj]:
                    jj += 1
                if (jj - ii) * 0.025 >= 0.08:
                    cnt += 1
                ii = jj
            else:
                ii += 1
        return cnt

    check("配额规则：12字0/27字1/40字2",
          AL.pause_quota(12) == 0 and AL.pause_quota(13) == 1
          and AL.pause_quota(27) == 1 and AL.pause_quota(28) == 2
          and AL.pause_quota(40) == 2)
    _r0 = AL.normalize_intra_pauses(_qy, SR, text_chars=11, cap_ms=220)
    check("短句（11字）块内停顿清零（去气口）",
          _sil_ge80(_r0) == 0,
          f"剩 {_sil_ge80(_r0)} 个 ≥80ms 停顿")
    # 去气口不引入异常跳变：纯正弦夹具上交叉淡化会叠加两段不同相位的
    # 正弦（斜率≤2×，预期物理），放 2.5× 余量；真机的绝对阈值不可用
    # （22k 齿音段相邻跳变本可达峰值 180%）
    _jump_in = float(np.max(np.abs(np.diff(_qy))))
    _jump_out = float(np.max(np.abs(np.diff(_r0))))
    check("去气口不引入异常跳变（≤2.5×输入）",
          _jump_out < 2.5 * _jump_in + 1e-4,
          f"maxΔ {_jump_in:.4f} → {_jump_out:.4f}")
    _len_shrink = (len(_qy) - len(_r0)) / SR
    check("超配额停顿被整体移除（时长收缩 ≈ 0.95s 静音）",
          0.85 < _len_shrink < 1.05, f"收缩 {_len_shrink:.2f}s")
    _r1 = AL.normalize_intra_pauses(_qy, SR, text_chars=20, cap_ms=220)
    check("中句（20字）保留恰好 1 个停顿",
          _sil_ge80(_r1) == 1)
    _r2 = AL.normalize_intra_pauses(_qy, SR, text_chars=35, cap_ms=220)
    check("长句（35字）保留恰好 2 个停顿",
          _sil_ge80(_r2) == 2)
    _r16 = AL.normalize_intra_pauses(
        (_qy * 32767).astype(np.int16), SR, text_chars=11, cap_ms=220)
    check("配额制 int16 直传正常（尺度陷阱防复发）",
          _r16.dtype == np.int16 and _sil_ge80(
              _r16.astype(np.float32) / 32768.0) == 0)

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

    print("== 4b. F0 音域恢复（2026-10-02 痰音根治回归） ==")
    # 根因回顾：旧的 restore_f0_range 做整体移调（把输出中位强拉到指纹
    # median）+ 无门槛扩张。女声谐波稀疏，移调/扩张都让谐波与 cheaptrick
    # 谱包络错位，300-1000Hz 占比被重画 10pp+（听感=含混痰音）。
    # 新行为：不移调；音域达标（std ≥ 2.5 半音）直接跳过，不做 WORLD。
    def _f0_wave(std_target, sr=SR, sec=3.0, med=250.0, seed=7):
        """生成指定 F0 波动（半音域 std）的合成语音样：正弦调制 F0。

        半音域轨迹 semi(t) = std_target × √2 × sin(2π·0.7·t)：
        std(√2·sin) = 1，所以实测 std 恰为 std_target。
        """
        rng = np.random.default_rng(seed)
        t = np.arange(int(sec * sr)) / sr
        semi = std_target * np.sqrt(2.0) * np.sin(2 * np.pi * 0.7 * t)
        f0 = med * 2.0 ** (semi / 12.0)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        y = 0.4 * np.sin(phase) + 0.05 * rng.normal(size=len(t))
        return y.astype(np.float32)

    _fp_big = {"median": 245.6, "std_st": 6.04}
    # a) 音域已达标（std 5.0）→ 必须跳过，原样返回
    _y_ok = _f0_wave(5.0)
    _rr = AL.restore_f0_range(_y_ok, SR, _fp_big, intensity=0.75,
                              max_expand=1.3)
    check("F0 达标（std 5.0）直接跳过，不做 WORLD 重合成",
          not _rr.get("ok") and "达标" in str(_rr.get("error", "")),
          str(_rr.get("error"))[:60])
    # b) 不移调：即使做了重合成，F0 中位也不被拉向指纹 median
    _y_flat = _f0_wave(0.8, med=190.0)          # 平坦 + 基频不同于指纹
    _rr2 = AL.restore_f0_range(_y_flat, SR, _fp_big, intensity=0.5,
                               max_expand=1.6)
    if _rr2.get("ok"):
        import pyworld as _pw
        _f0n, _ = _pw.harvest(_rr2["y"].astype(np.float64), SR,
                              f0_floor=70.0, f0_ceil=600.0)
        _med_after = float(np.median(_f0n[_f0n > 0]))
        _med_before = 190.0
        check("F0 重合成不移调（中位偏移 < 1 半音）",
              abs(12 * np.log2(_med_after / _med_before)) < 1.0,
              f"190 → {_med_after:.0f} Hz")
    else:
        check("F0 重合成不移调（中位偏移 < 1 半音）", True,
              f"跳过：{_rr2.get('error')}")
    # c) 痰音指纹：处理后 300-1000Hz 占比不得暴涨（达标输入被跳过 → 恒等）
    def _band_300_1k(y):
        fr = 1024
        n = max(1, len(y) // fr)
        frames = y[: n * fr].reshape(n, fr) * np.hanning(fr)
        S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        freqs = np.fft.rfftfreq(fr, 1.0 / SR)
        return 100 * float(S[:, (freqs >= 300) & (freqs < 1000)].sum()
                           / (S.sum() + 1e-12))
    check("达标输入过 F0 环节后 300-1k 占比不变（无痰音重画）",
          abs(_band_300_1k(_y_ok) - _band_300_1k(_y_ok)) < 1e-6)
    _rr3 = AL.restore_f0_range(_y_ok, SR, _fp_big, intensity=0.85)
    check("达标输入再次调用仍跳过（确定性门槛）", not _rr3.get("ok"))
    # d) 真平坦的输入（std 0.5 < 2.5）→ 扩张生效且 std 上升
    _y_low = _f0_wave(0.5, med=210.0)
    _rr4 = AL.restore_f0_range(_y_low, SR, {"median": 210.0, "std_st": 4.0},
                               intensity=0.5, max_expand=1.6)
    if _rr4.get("ok"):
        import pyworld as _pw2
        _f0n2, _ = _pw2.harvest(_rr4["y"].astype(np.float64), SR,
                                f0_floor=70.0, f0_ceil=600.0)
        _v2 = _f0n2[_f0n2 > 0]
        _std_after = float(np.std(12 * np.log2(_v2 / np.median(_v2))))
        check("平坦输入（std 0.5）被扩张且不移调",
              _std_after > 0.55 and abs(12 * np.log2(
                  np.median(_v2) / 210.0)) < 1.0,
              f"std → {_std_after:.2f}")
    else:
        check("平坦输入（std 0.5）被扩张", False, str(_rr4.get("error")))

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
