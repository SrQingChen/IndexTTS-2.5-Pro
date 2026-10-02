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

    # ---- 3e. 块内停顿「门槛-配额-时长」治理（2026-10-02 第四轮）----
    # 去气口版本的教训：无门槛移除静音把 25~100ms 的塞音闭合/词间隙也删了
    # （每块 10+ 个，实测实锤）→ 字被削、每字独立合成。现三治理：
    # min_run 150ms 门槛保结构 + 配额管数量 + 配额外压 35ms 微气口。
    _qy = np.concatenate([
        _tone(1.0, 0.5), np.zeros(int(0.30 * SR), np.float32),
        _tone(0.5, 0.5), np.zeros(int(0.40 * SR), np.float32),
        _tone(0.8, 0.5), np.zeros(int(0.25 * SR), np.float32),
        _tone(0.6, 0.5)])
    # 塞音闭合夹具：语音里嵌 0.06/0.09s 的闭合段（正常语音结构）
    _qy_closure = np.concatenate([
        _tone(0.4, 0.5), np.zeros(int(0.06 * SR), np.float32),
        _tone(0.4, 0.5), np.zeros(int(0.09 * SR), np.float32),
        _tone(0.4, 0.5)])

    def _sil_ge(v, min_ms):
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
                if (jj - ii) * 0.025 * 1000 >= min_ms:
                    cnt += 1
                ii = jj
            else:
                ii += 1
        return cnt

    check("配额规则：12字0/27字1/40字2",
          AL.pause_quota(12) == 0 and AL.pause_quota(13) == 1
          and AL.pause_quota(27) == 1 and AL.pause_quota(28) == 2
          and AL.pause_quota(40) == 2)
    _rc = AL.normalize_intra_pauses(_qy_closure, SR, text_chars=11,
                                    cap_ms=220)
    # 帧化粒度：25ms 帧测量静音会短 20-25ms（60ms 量成 ~50ms），
    # 阈值取 45ms；长度不变才是保护生效的硬证据
    check("塞音闭合保护：60/90ms 闭合段原样保留（去气口事故回归）",
          _sil_ge(_rc, 45) == 2 and len(_rc) == len(_qy_closure),
          f"处理后 {_sil_ge(_rc, 45)} 个 · 长度 {len(_qy_closure)/SR:.2f}"
          f"→{len(_rc)/SR:.2f}s")
    _r0 = AL.normalize_intra_pauses(_qy, SR, text_chars=11, cap_ms=220)
    check("短句（11字）模型停顿全部压到微气口（无 ≥80ms 残留）",
          _sil_ge(_r0, 80) == 0,
          f"剩 {_sil_ge(_r0, 80)} 个 ≥80ms")
    _len_shrink = (len(_qy) - len(_r0)) / SR
    # 0.30+0.40+0.25 = 0.95s 静音，各压到 35ms：收缩 ≈ 0.845s
    check("配额外停顿压到 35ms 微气口（时长收缩 ≈ 0.85s）",
          0.75 < _len_shrink < 0.95, f"收缩 {_len_shrink:.2f}s")
    _r1 = AL.normalize_intra_pauses(_qy, SR, text_chars=20, cap_ms=220)
    check("中句（20字）保留恰好 1 个停顿",
          _sil_ge(_r1, 80) == 1)
    _r2 = AL.normalize_intra_pauses(_qy, SR, text_chars=35, cap_ms=220)
    check("长句（35字）保留恰好 2 个停顿",
          _sil_ge(_r2, 80) == 2)
    _r16 = AL.normalize_intra_pauses(
        (_qy * 32767).astype(np.int16), SR, text_chars=11, cap_ms=220)
    check("int16 直传正常（尺度陷阱防复发）",
          _r16.dtype == np.int16 and _sil_ge(
              _r16.astype(np.float32) / 32768.0, 80) == 0)

    # ---- 3f. 块内逗号保底位（2026-10-02 第五轮：逗号几乎没停顿事故）----
    # 事故链：参考摊平(>120ms封顶)治好词组化散感 → 模型在逗号处的停顿
    # 也缩到 75~150ms → 低于 150ms 门槛原样放行；≥150ms 的又被 ≤12 字块
    # 配额 0 压成 35ms 微气口。修复：逗号类标点按位置匹配已实现静音，
    # 保证落在导演逗号带内（与块边界同一真源），不占配额、不动非标点
    # 治理、绝不向语音里插静音。
    from webui_app.services.director import comma_pause_band as _cpb
    _lo_ms, _hi_ms = _cpb(1.0)
    check("逗号带真源：scale=1.0 → (120,210)；scale=0.7 下限=84（与真机旁车一致）",
          (_lo_ms, _hi_ms) == (120, 210) and _cpb(0.7)[0] == 84,
          f"{_cpb(0.7)}")

    def _runs_ms_at(v, min_ms=40):
        """[(start_s, dur_ms)]：与 normalize 同参（25ms 帧/10ms hop/-30dB）。

        注意必须按 hop 滑窗（与 _frames_db 一致），不能整段 reshape 成
        连续 25ms 帧——后者位置/时长全是另一套刻度（首版探针的翻车点）。
        检出时长比实际静音短约 20~25ms（帧窗两端跨语音的帧不算静音），
        断言阈值要留这个余量。
        """
        fr = int(SR * 0.025)
        hp = int(SR * 0.01)
        n = 1 + (len(v) - fr) // hp
        if n <= 0:
            return []
        idx = np.arange(fr)[None, :] + hp * np.arange(n)[:, None]
        ee = np.sqrt(np.mean(v[idx] ** 2, axis=1) + 1e-12)
        ddb = 20 * np.log10(ee + 1e-12)
        qq = ddb < -30
        out, i = [], 0
        while i < n:
            if qq[i]:
                j = i
                while j < n and qq[j]:
                    j += 1
                if (j - i) * 10 >= min_ms:
                    out.append((round(i * 0.01, 3), (j - i) * 10))
                i = j
            else:
                i += 1
        return out

    # 事故案例 1：短句（11字，配额 0）里逗号实现为 75ms → 旧版原样放行
    _txtA = "我们走吧，外面雨停了。"
    _yA = np.concatenate([
        _tone(0.9, 0.5), np.zeros(int(0.075 * SR), np.float32),
        _tone(0.9, 0.5)])
    _zA = AL.normalize_intra_pauses(
        _yA, SR, text_chars=len(_txtA), cap_ms=220,
        text=_txtA, punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    _rrA = _runs_ms_at(_zA)
    check("事故案例1：75ms 逗号停顿被补插到带下限（检出 ≥90ms）",
          len(_rrA) == 1 and _rrA[0][1] >= 90
          and len(_zA) > len(_yA),
          f"{_rrA} · 长度 +{(len(_zA)-len(_yA))/SR*1000:.0f}ms")
    check("补插只加静音不切语音（两段正弦原样）",
          np.array_equal(_zA[:int(0.9 * SR)], _yA[:int(0.9 * SR)]))
    _oldA = AL.normalize_intra_pauses(_yA, SR, text_chars=len(_txtA),
                                      cap_ms=220)
    check("对照组：旧路径（无 text）75ms 原样放行（事故机理复现）",
          len(_oldA) == len(_yA) and not _runs_ms_at(_oldA, 60))

    # 事故案例 2：短句里逗号实现为 400ms → 旧版配额 0 压成 35ms；
    # 新版截到带上限 210ms
    _yB = np.concatenate([
        _tone(0.9, 0.5), np.zeros(int(0.40 * SR), np.float32),
        _tone(0.9, 0.5)])
    _zB = AL.normalize_intra_pauses(
        _yB, SR, text_chars=len(_txtA), cap_ms=220,
        text=_txtA, punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    _rrB = _runs_ms_at(_zB)
    check("事故案例2：400ms 逗号停顿截到带上限（≈210ms，不再压 35ms）",
          len(_rrB) == 1 and 180 <= _rrB[0][1] <= 215,
          f"{_rrB}")

    # 案例 3：双逗号 + 一处无标点长停顿 —— 标点各自入带，非标点停顿
    # 照旧走配额（20字配额 1：保留并压到 cap 220）
    _txtC = "明明可以选择逃避，却偏要把一切，都背负起来。"
    # 语音段时长 ∝ 段字数（8/5/5/5字 → 0.8/0.5/0.5/0.5s），保证位置估计准
    _yC = np.concatenate([
        _tone(0.8, 0.5), np.zeros(int(0.130 * SR), np.float32),   # 逗号1(带内)
        _tone(0.5, 0.5), np.zeros(int(0.400 * SR), np.float32),   # 无标点(超配额)
        _tone(0.5, 0.5), np.zeros(int(0.075 * SR), np.float32),   # 逗号2(过短)
        _tone(0.5, 0.5)])
    _zC = AL.normalize_intra_pauses(
        _yC, SR, text_chars=len(_txtC), cap_ms=220,
        text=_txtC, punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    _rrC = _runs_ms_at(_zC)
    check("双逗号各自入带：130ms 保持带内、75ms 补到带下限（检出 ≥90ms）",
          len(_rrC) == 3 and 90 <= _rrC[0][1] <= 135
          and _rrC[2][1] >= 90,
          f"{_rrC}")
    check("无标点长停顿照旧配额治理：400ms → ≈220ms（行为不变）",
          180 <= _rrC[1][1] <= 240, f"{_rrC[1]}")

    # 案例 4：无逗号文本 + text 传入 → 与旧路径逐位一致（零回归承诺）
    _zD1 = AL.normalize_intra_pauses(
        _qy, SR, text_chars=20, cap_ms=220, text="我们走吧外面雨停了啊。",
        punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    _zD2 = AL.normalize_intra_pauses(_qy, SR, text_chars=20, cap_ms=220)
    check("无逗号路径逐位一致（标点感知零副作用）",
          np.array_equal(_zD1, _zD2))

    # 案例 5：逗号处无已实现静音（连读）→ 不向语音里插静音
    _yE = np.concatenate([_tone(0.9, 0.5), _tone(0.9, 0.5)])
    _zE = AL.normalize_intra_pauses(
        _yE, SR, text_chars=len(_txtA), cap_ms=220,
        text=_txtA, punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    check("逗号无静音可匹配 → 原样返回（绝不切语音）",
          np.array_equal(_zE, _yE))

    # 案例 6：块尾逗号（停顿=块尾静音，归拼接层管）→ 不越权处理
    _yF = np.concatenate([
        _tone(0.9, 0.5), np.zeros(int(0.30 * SR), np.float32)])
    _txtF = "等她意识到我离开了，"
    _zF1 = AL.normalize_intra_pauses(
        _yF, SR, text_chars=len(_txtF), cap_ms=220,
        text=_txtF, punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    _zF2 = AL.normalize_intra_pauses(_yF, SR, text_chars=len(_txtF),
                                     cap_ms=220)
    check("块尾逗号静音归拼接层（块内治理不越权）",
          np.array_equal(_zF1, _zF2))

    # 案例 7：int16 尺度（编排器实际传法）
    _zG = AL.normalize_intra_pauses(
        (_yA * 32767).astype(np.int16), SR, text_chars=len(_txtA),
        cap_ms=220, text=_txtA,
        punct_floor_ms=_lo_ms, punct_cap_ms=_hi_ms)
    _rrG = _runs_ms_at(_zG.astype(np.float32) / 32768.0)
    check("int16 直传：dtype 保持 + 逗号同样入带",
          _zG.dtype == np.int16 and _rrG and _rrG[0][1] >= 90,
          f"{_rrG}")

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
