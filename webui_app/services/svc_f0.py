"""svc_f0 — F0（音高）管线：翻唱旋律保持与实时音高跟随。

背景（docs/dev-plan-ai-cover-vc-2026-10.md D2）：
IndexTTS-2.5 的语义码本（25Hz ResidualVQ）在量化后音高信息大幅丢失，
旋律必须由**外置 F0 管线**保证。本模块提供：

1. 分窗 harvest 提取（30s 窗 + 重叠淡化拼接，规避整段内存护栏）
2. 八度映射（源/目标音区中位数差 → 半音偏移，支持手动微调）
3. F0 轮廓相关系数 r / 八度错误率（auto 模式分支依据 + 质量报告指标）
4. WORLD F0 移植：
   - 全曲移植（restore_f0_range 的保长模式；绝不用 lengthen_abrupt_releases
     的插帧模式——那会破坏与 BGM 的对拍）
   - 局部修复（只对偏差 >40 音分且 >80ms 的浊音段 WORLD 重合成，
     段边界等功率交叉淡化回原波形）
5. F0Tracker：实时块级连续 F0（流式变声用）

工程约定（项目红线）：
- 波形全链 float32 ±1；仅最终落盘才转 PCM_16（int16 尺寸陷阱三连的教训）。
- WORLD 参数 f0_floor=55 必须**同时**传给 harvest 与 cheaptrick（d4c 传 fft_size），
  否则 <71Hz 的男低音帧谱包络直接分析错。
- cheaptrick fft_size=2048 = 2^ceil(log2(3*22050/55))，f0_floor 变了要联动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 常量（D2 / 审查修正后的参数分配）
# ---------------------------------------------------------------------------

F0_FLOOR_HZ = 55.0          # 歌唱男低下探；必须同传 harvest+cheaptrick
F0_CEIL_HZ = 1100.0         # C6=1046.5Hz 留余量
FRAME_PERIOD_MS = 5.0       # pyworld 网格（颤音 5-7Hz 采样密度足够）
FFT_SIZE = 2048             # 2^ceil(log2(3*22050/55))，f0_floor=55 的联动值

HARVEST_WINDOW_SEC = 30.0   # 分窗护栏（与 audio_lab.f0_stats 的 30s 一致）
HARVEST_OVERLAP_SEC = 1.0   # 长曲分窗时的重叠（只取中心帧，不做平均）
HARVEST_EDGE_TRIM_SEC = 1.0  # 长曲分窗丢弃首尾各 1s（窗边缘 F0 不可靠）
HARVEST_FULL_MAX_SEC = 720.0  # ≤12 分钟整段 harvest（常规歌曲）

REPAIR_CENT_THRESHOLD = 40.0    # 局部修复：偏差阈值（音分）
REPAIR_MIN_MS = 80.0             # 局部修复：最短持续时间
REPAIR_PAD_MS = 100.0            # 局部修复：段前后送入 WORLD 的上下文
REPAIR_SEAM_MS = 25.0            # 局部修复：接回原波形的等功率淡化
REPAIR_MAX_SEGMENTS = 40         # 超过则建议全曲移植（调用方决策）
REPAIR_MAX_RATIO = 0.5           # 修复总时长超过全曲该比例 → 建议全曲


def _require_pyworld():
    import pyworld as pw  # noqa: F401
    return pw


# ---------------------------------------------------------------------------
# 1) 分窗 F0 提取
# ---------------------------------------------------------------------------

def frames_of(duration_samples: int, sr: int, frame_period_ms: float = FRAME_PERIOD_MS) -> int:
    """pyworld 帧数公式：floor(N / (sr*fp)) + 1（22050/5ms 实测 2.0s→401 帧）。"""
    return int(duration_samples // int(sr * frame_period_ms / 1000.0)) + 1


def harvest_f0(y: np.ndarray, sr: int,
               f0_floor: float = F0_FLOOR_HZ, f0_ceil: float = F0_CEIL_HZ,
               frame_period: float = FRAME_PERIOD_MS) -> np.ndarray:
    """全曲 F0（5ms 网格，清音=0）。

    **不再对分窗结果做线性平均**（2026-10-17 实锤）：旧实现用 30s 窗+1s
    重叠，在重叠区把两个窗边缘的 F0 相加平均——窗边缘常有八度/半频错误，
    平均后得到 223/264/364 这类中间值，看起来像“F0 跳变/接线错误”。
    现在：
    - ≤12 分钟（常规歌曲）整段一次 harvest（pyworld 内存线性，实测
      300s@22k 67s 可跑；项目历史上 43.7M 样本才 abort）；
    - 更长才分窗，且**丢弃每窗首尾 1s**、只取窗中心帧，按绝对帧号拼接，
      重叠/边缘不做任何数值平均。
    """
    pw = _require_pyworld()
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < int(0.2 * sr):
        return np.zeros(max(frames_of(n, sr, frame_period), 1), dtype=np.float64)

    if n <= int(HARVEST_FULL_MAX_SEC * sr):
        f0, _t = pw.harvest(y, sr, f0_floor=f0_floor, f0_ceil=f0_ceil,
                            frame_period=frame_period)
        return np.asarray(f0, dtype=np.float64)

    total = frames_of(n, sr, frame_period)
    out = np.zeros(total, dtype=np.float64)
    filled = np.zeros(total, dtype=bool)
    win = int(HARVEST_WINDOW_SEC * sr)
    trim = int(HARVEST_EDGE_TRIM_SEC * sr)
    hop = max(1, win - 2 * trim)
    tf = max(1, int(round(trim / (frame_period / 1000.0))))
    start = 0
    first = True
    while start < n:
        seg = y[start:min(start + win, n)]
        f0, _t = pw.harvest(seg, sr, f0_floor=f0_floor, f0_ceil=f0_ceil,
                            frame_period=frame_period)
        f0 = np.asarray(f0, dtype=np.float64)
        base = int(round((start / sr) / (frame_period / 1000.0)))
        a = base if first else base + tf
        b = base + len(f0) - (0 if start + win >= n else tf)
        a = max(0, a); b = min(total, b)
        ra, rb = a - base, b - base
        if rb > ra:
            out[a:b] = f0[ra:rb]
            filled[a:b] = True
        if start + win >= n:
            break
        start += hop
    if not filled.all():
        idx = np.where(filled)[0]
        if len(idx):
            out[~filled] = np.interp(np.where(~filled)[0], idx, out[idx])
    return out


def f0_median(f0: np.ndarray) -> float:
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) >= 3 else 0.0


def f0_stats_st(f0: np.ndarray) -> float:
    """浊音帧 log-F0 标准差（半音）。"""
    v = f0[f0 > 0]
    if len(v) < 3:
        return 0.0
    return float(np.std(12.0 * np.log2(v)))


# ---------------------------------------------------------------------------
# 2) 八度映射
# ---------------------------------------------------------------------------

def octave_shift_semis(src_f0: np.ndarray, ref_f0: np.ndarray,
                       manual: Optional[int] = None) -> int:
    """源→目标音区映射（半音）。

    自动：源/参考浊音中位数比取 12*log2 后**四舍五入到最近的 12 的倍数**
    （同性别 0 / 男→女 +12 / 女→男 -12）；manual 非空时直接用（钳 ±24）。
    """
    if manual is not None:
        return int(max(-24, min(24, int(manual))))
    ms, mr = f0_median(src_f0), f0_median(ref_f0)
    if ms <= 0 or mr <= 0:
        return 0
    # 移调量 = 把**源**音区搬**上/下**到目标音区（shift_f0 作用在源 F0 上）
    cents = 1200.0 * math.log2(mr / ms)
    # 同音区（±8 半音内）保留原调：翻唱首先要保调/保旋律，不能因为
    # “参考说话音高不同”就整首降八度（用户 390Hz 源 vs 257Hz 参考，
    # 旧逻辑选 -12，既改调又踩 WORLD 下行八度的坑）。
    if abs(cents) <= 800.0:
        return 0
    return int(round(cents / 1200.0) * 12)


def shift_f0(f0: np.ndarray, semis: float) -> np.ndarray:
    out = f0.copy()
    v = out > 0
    out[v] = out[v] * (2.0 ** (semis / 12.0))
    return out


def smooth_ratio_curve(f0: np.ndarray, med_win: int = 5, ma_win: int = 5) -> np.ndarray:
    """对移植用比值曲线做中值滤波 + 滑动平均（防逐帧抖动的粗糙感）。"""
    v = f0 > 0
    out = f0.copy()
    if v.sum() < med_win + ma_win:
        return out
    seg = f0[v]
    # 中值滤波
    if med_win >= 3:
        pad = med_win // 2
        padded = np.pad(seg, pad, mode="edge")
        seg = np.array([np.median(padded[i:i + med_win]) for i in range(len(seg))])
    # 滑动平均
    if ma_win >= 2:
        kernel = np.ones(ma_win) / ma_win
        seg = np.convolve(seg, kernel, mode="same")
    out[v] = seg
    return out


# ---------------------------------------------------------------------------
# 3) 轮廓相关系数 / 八度错误率
# ---------------------------------------------------------------------------

def f0_correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> Tuple[float, float, int]:
    """两 F0（同 5ms 网格）的轮廓一致性。

    返回 (r, octave_error_rate, n_voiced)：
    - r：浊音帧 log-F0 Pearson 相关系数（先把 b 全局对齐到 a 的最近八度）
    - octave_error_rate：|cents−最近倍频程| > 600 音分的浊音帧比例
    """
    n = min(len(f0_a), len(f0_b))
    if n < 5:
        return 0.0, 1.0, 0
    a, b = f0_a[:n], f0_b[:n]
    both = (a > 0) & (b > 0)
    n_voiced = int(both.sum())
    if n_voiced < 5:
        return 0.0, 1.0, n_voiced
    la, lb = np.log2(a[both]), np.log2(b[both])
    # 全局对齐到最近八度（消除整体音区差，只看轮廓形状）
    octaves = np.round(np.median(la - lb))
    lb = lb + octaves
    la_c, lb_c = la - la.mean(), lb - lb.mean()
    denom = math.sqrt(float((la_c ** 2).sum()) * float((lb_c ** 2).sum()))
    r = float((la_c * lb_c).sum() / denom) if denom > 1e-9 else 0.0
    # 注意：这里先在 log2 域做**全局**八度对齐，再统计 |cents|>600。
    # 旧写法用 round(cents/1200) 残差，会把局部的 ±12 半音跳变也当成 0
    # （1200 音分的残差为 0），对“缺基频导致的局部高八度”完全失明。
    cents = 1200.0 * (la - lb)
    err = float((np.abs(cents) > 600.0).mean())
    return r, err, n_voiced


def _probe_metrics(y: np.ndarray, sr: int, target_f0: np.ndarray,
                   seconds: float = 20.0) -> dict:
    """移植后的代表段指标（自动选目标浊音最多的 20s 窗）。

    返回 {r, octave_err, within_50c, voiced}。20s harvest 约几秒，
    比整首再跑一遍 harvest 便宜得多，足以反映局部八度跳变/抖动。
    """
    tf = int(seconds * 1000.0 / FRAME_PERIOD_MS)
    if len(target_f0) <= tf or len(y) < int(seconds * sr):
        f = harvest_f0(y, sr)
        r, e, n = f0_correlation(f, target_f0)
        return {"r": round(r, 4), "octave_err": round(e, 4), "voiced": n}
    # 目标浊音密度最高的窗口（步长 5s）
    step = max(1, int(5.0 * 1000.0 / FRAME_PERIOD_MS))
    best_s, best_v = 0, -1
    for st in range(0, len(target_f0) - tf + 1, step):
        v = int((target_f0[st:st + tf] > 0).sum())
        if v > best_v:
            best_v, best_s = v, st
    a = int(best_s * FRAME_PERIOD_MS / 1000.0 * sr)
    b = min(len(y), a + int(seconds * sr))
    f = harvest_f0(y[a:b], sr)
    tgt = target_f0[best_s:best_s + len(f)]
    r, e, n = f0_correlation(f, tgt)
    both = (f > 0) & (tgt > 0)
    within = 0.0
    if both.sum() >= 5:
        la = np.log2(f[both]); lb = np.log2(tgt[both])
        lb = lb + np.round(np.median(la - lb))
        within = float((np.abs(1200.0 * (la - lb)) <= 50.0).mean())
    return {"r": round(r, 4), "octave_err": round(e, 4),
            "within_50c": round(within, 4), "voiced": n}


# ---------------------------------------------------------------------------
# 4) WORLD F0 移植
# ---------------------------------------------------------------------------

@dataclass
class TransplantResult:
    ok: bool
    y: np.ndarray
    reason: str = ""
    mode: str = "none"          # none / full / repair
    repaired_segments: int = 0
    repaired_sec: float = 0.0
    spectral_check: bool = True


def _world_analyze_f0(y: np.ndarray, sr: int):
    """仅 harvest（f0, t）。包络分析推迟到 f0_synth 确定之后——见下。"""
    pw = _require_pyworld()
    y64 = np.asarray(y, dtype=np.float64)
    f0, t = pw.harvest(y64, sr, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ,
                       frame_period=FRAME_PERIOD_MS)
    return np.asarray(f0), np.asarray(t)


def _world_analyze(y: np.ndarray, sr: int, synth_f0: Optional[np.ndarray] = None):
    """WORLD 分析。

    **包络栅格必须用合成 F0**（实验实锤，2026-10-16）：cheaptrick/d4c 若用
    输出侧原 f0 的谐波栅格分析、再用移调后的 f0 合成，输出谱会出现栅格
    失配深谷纹波（220 分析 + 440 激励 → harvest 锁到 103Hz 子谐波）；用
    synth_f0 分析则干净锁定目标值。清音帧栅格无所谓，沿用原 f0 防零。
    """
    pw = _require_pyworld()
    y64 = np.asarray(y, dtype=np.float64)
    f0, t = pw.harvest(y64, sr, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ,
                       frame_period=FRAME_PERIOD_MS)
    f0 = np.asarray(f0)
    if synth_f0 is not None:
        synth_f0 = _interp_to_grid(np.asarray(synth_f0, dtype=np.float64), len(f0))
        f0_for_env = np.where(synth_f0 > 0, synth_f0,
                              np.where(f0 > 0, f0, 100.0))
    else:
        f0_for_env = np.where(f0 > 0, f0, 100.0)
    sp = pw.cheaptrick(y64, f0_for_env, t, sr, fft_size=FFT_SIZE,
                       f0_floor=F0_FLOOR_HZ)
    ap = pw.d4c(y64, f0_for_env, t, sr, fft_size=FFT_SIZE)
    return f0, np.asarray(t), np.asarray(sp), np.asarray(ap)


def _interp_to_grid(f0: np.ndarray, n_frames: int) -> np.ndarray:
    """把（可能来自其他网格的）f0 重采样到当前帧网格长度。"""
    if len(f0) == n_frames:
        return f0
    if len(f0) == 0:
        return np.zeros(n_frames, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(f0))
    xg = np.linspace(0.0, 1.0, n_frames)
    v = f0 > 0
    out = np.interp(xg, xs, f0)
    # 清音段不该被线性插值架桥成假浊音
    if v.sum() > 2:
        vv = np.interp(xg, xs, v.astype(np.float64))
        out[vv < 0.45] = 0.0
    return out


def _probe_voiced_median(y: np.ndarray, sr: int, seconds: float = 4.0) -> float:
    """短片段的浊音中位 F0（用于移植后的快速自检，harvest 4s 很便宜）。"""
    if len(y) < int(0.4 * sr):
        return 0.0
    seg = int(seconds * sr)
    start = min(max(0, int(len(y) * 0.3)), max(0, len(y) - seg))
    f0 = harvest_f0(y[start:start + seg], sr)
    return f0_median(f0)


def world_replace_f0(y: np.ndarray, sr: int, new_f0: np.ndarray,
                     vuv_gate: bool = True) -> TransplantResult:
    """全曲 F0 移植（保长）。

    **包络分析策略是方向相关的**（2026-10-17 实锤，此前固定用合成-F0）：
    - 上行移调（目标中位 ≥ 输出中位）：用**合成 F0**分析包络——用原 F0 分析
      会在目标谐波处落在包络谷，合成后基频缺失、被感知成低八度；
    - 下行移调（目标中位 < 输出中位）：用**输出自身 F0**分析包络——用合成
      F0（更低）分析会把奇次谐波当成 0，合成后基频被抹掉、harvest 锁到原
      八度（用户 372→195 实测：中位只降到 305、r 反降）。
    移植后用 4s 探针核对中位音高，偏差 >0.5 八度则换另一种策略；谱自检
    （RMS±4dB / <1kHz 占比±20pp）不过则回退原波形。
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) < int(0.3 * sr):
        return TransplantResult(False, y, reason="音频过短")
    try:
        f0_out, _t = _world_analyze_f0(y, sr)
    except Exception as exc:  # pyworld 对极端输入偶发 abort
        return TransplantResult(False, y, reason=f"harvest 失败: {exc}")

    new_f0 = _interp_to_grid(np.asarray(new_f0, dtype=np.float64), len(f0_out))
    f0_synth = f0_out.copy()
    gate = (f0_out > 0) if vuv_gate else np.ones(len(f0_out), dtype=bool)
    use = gate & (new_f0 > 0)
    f0_synth[use] = new_f0[use]

    tgt_med = f0_median(new_f0)
    out_med = f0_median(f0_out)
    # 目标明显低于输出 → 下行，用输出 F0 分析包络；否则用合成 F0
    use_synth_env = not (tgt_med > 0 and out_med > 0 and tgt_med < out_med * 0.94)

    pw = _require_pyworld()

    def _render(synth_env: bool) -> np.ndarray:
        _f, _t, sp, ap = _world_analyze(
            y, sr, synth_f0=f0_synth if synth_env else None)
        out = pw.synthesize(f0_synth, sp, ap, sr,
                            frame_period=FRAME_PERIOD_MS)
        out = np.asarray(out, dtype=np.float32)
        if len(out) >= len(y):
            return out[:len(y)]
        return np.pad(out, (0, len(y) - len(out)))

    try:
        y2 = _render(use_synth_env)
    except Exception as exc:
        return TransplantResult(False, y, reason=f"synthesize 失败: {exc}")

    # 探针自检：中位音高应接近目标；否则换另一种包络策略
    if tgt_med > 0:
        p2 = _probe_voiced_median(y2, sr)
        if p2 > 0 and abs(math.log2(p2 / tgt_med)) > 0.5:
            try:
                y_alt = _render(not use_synth_env)
                p_alt = _probe_voiced_median(y_alt, sr)
                if p_alt > 0 and abs(math.log2(p_alt / tgt_med)) <                         abs(math.log2(p2 / tgt_med)):
                    y2 = y_alt
            except Exception:
                pass
        p_final = _probe_voiced_median(y2, sr)
        if p_final > 0 and abs(math.log2(p_final / tgt_med)) > 0.5:
            return TransplantResult(False, y,
                                    reason="F0 移植未收敛（中位音高偏差 >0.5 八度，回退）",
                                    mode="full")

    if not _spectral_guard(y, y2, sr):
        return TransplantResult(False, y, reason="谱自检未过（回退原输出）",
                                mode="full")
    return TransplantResult(True, y2, mode="full")


def _spectral_guard(a: np.ndarray, b: np.ndarray, sr: int) -> bool:
    """移植段谱守卫：RMS ±4dB 且 <1kHz 能量占比 ±20pp（audio_lab 韵尾延长同款）。"""
    def _rms(x):
        return float(np.sqrt(np.mean(np.square(x))) + 1e-9)

    def _low_ratio(x):
        spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
        tot = spec.sum() + 1e-12
        return float(spec[freqs < 1000].sum() / tot)

    ra, rb = _rms(a), _rms(b)
    if rb / ra > 10 ** (4.0 / 20.0) or rb / ra < 10 ** (-4.0 / 20.0):
        return False
    if abs(_low_ratio(a) - _low_ratio(b)) > 0.20:
        return False
    return True


def _equal_power_fade(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """等功率交叉淡化：幅度域 cos/sin（平方和 cos²+sin²=1，能量恒定）。

    注意不要写成 cos²/sin² 幅度版——那版中点能量只有 0.5（掉 3dB）。
    """
    t = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    return (np.cos(t * math.pi / 2)).astype(np.float32), \
           (np.sin(t * math.pi / 2)).astype(np.float32)


def f0_local_repair(y: np.ndarray, sr: int, src_f0: np.ndarray) -> TransplantResult:
    """局部修复：只对 |cents|>40 且 >80ms 的浊音段 WORLD 重合成后淡化接回。

    src_f0 必须已完成八度映射（调 shift_f0）并插值到 y 的 5ms 网格。
    段过多（>REPAIR_MAX_SEGMENTS）或总时长占比 >REPAIR_MAX_RATIO 时返回
    ok=False + reason 提示调用方降级全曲移植。
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) < int(0.3 * sr):
        return TransplantResult(False, y, reason="音频过短")
    try:
        f0_out, _t = _world_analyze_f0(y, sr)
    except Exception as exc:
        return TransplantResult(False, y, reason=f"harvest 失败: {exc}")

    src = _interp_to_grid(np.asarray(src_f0, dtype=np.float64), len(f0_out))
    both = (f0_out > 0) & (src > 0)
    cents = np.zeros(len(f0_out), dtype=np.float64)
    cents[both] = 1200.0 * np.log2(f0_out[both] / src[both])
    # 平滑（中值 5 + 滑动平均 5），尖刺不修
    cents = smooth_ratio_curve(np.where(both, cents, 0.0), 5, 5)
    bad = both & (np.abs(cents) > REPAIR_CENT_THRESHOLD)

    # 连段：>80ms（16 帧 @5ms），间隔 <50ms 合并
    min_frames = int(REPAIR_MIN_MS / FRAME_PERIOD_MS)
    gap_frames = int(50.0 / FRAME_PERIOD_MS)
    segs: List[Tuple[int, int]] = []
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and (bad[j] or (j - i < len(bad) and any(bad[j:j + gap_frames]))):
                j += 1
            if (j - i) >= min_frames:
                segs.append((i, j))
            i = j
        else:
            i += 1

    if not segs:
        return TransplantResult(True, y, mode="repair", repaired_segments=0)
    fpms = FRAME_PERIOD_MS
    total_frames = sum(e - s for s, e in segs)
    if len(segs) > REPAIR_MAX_SEGMENTS or total_frames * fpms / 1000.0 > REPAIR_MAX_RATIO * len(y) / sr:
        return TransplantResult(False, y, reason=f"偏差段过多（{len(segs)} 段/"
                            f"{total_frames * fpms / 1000.0:.1f}s），建议全曲移植",
                                mode="repair")

    out = y.copy()
    pad = int(REPAIR_PAD_MS / 1000.0 * sr)
    seam = int(REPAIR_SEAM_MS / 1000.0 * sr)
    repaired_sec = 0.0
    for s_f, e_f in segs:
        s = max(0, int(s_f * fpms / 1000.0 * sr) - seam)
        e = min(len(y), int(e_f * fpms / 1000.0 * sr) + seam)
        seg = y[max(0, s - pad):min(len(y), e + pad)]
        res = world_replace_f0(seg, sr, src)
        if not res.ok:
            continue
        # 裁回 [s, e)，接缝等功率淡化
        off = s - max(0, s - pad)
        rep = res.y[off:off + (e - s)]
        if len(rep) < e - s:
            rep = np.pad(rep, (0, e - s - len(rep)))
        n = min(seam * 2, (e - s) // 2)
        if n > 8:
            fa, fb = _equal_power_fade(n)
            head, tail = out[s:s + n].copy(), out[e - n:e].copy()
            rep[:n] = rep[:n] * fb + head * fa
            rep[-n:] = rep[-n:] * fa + tail * fb
        out[s:e] = rep
        repaired_sec += (e_f - s_f) * fpms / 1000.0

    if not _spectral_guard(y, out, sr):
        return TransplantResult(False, y, reason="整体谱自检未过（回退）", mode="repair")
    return TransplantResult(True, out.astype(np.float32), mode="repair",
                            repaired_segments=len(segs), repaired_sec=repaired_sec)


def transplant_f0(y: np.ndarray, sr: int, src_f0: np.ndarray, out_f0: Optional[np.ndarray],
                  mode: str = "auto") -> Tuple[TransplantResult, dict]:
    """D2 三模式入口。

    mode: off / auto / force。auto：先测 r，r≥0.75 局部修复（失败/过多段则全曲），
    r<0.75 直接全曲。src_f0 须已八度映射。返回 (结果, 指标 dict)。
    """
    metrics: dict = {"mode_asked": mode}
    if mode == "off":
        return TransplantResult(True, np.asarray(y, dtype=np.float32), mode="none"), metrics
    src = np.asarray(src_f0, dtype=np.float64)
    if out_f0 is None:
        out_f0 = harvest_f0(y, sr)
    r, err, nv = f0_correlation(out_f0, src)
    metrics.update({"f0_r": round(r, 4), "octave_err": round(err, 4), "voiced_frames": nv})
    if mode == "force":
        res = world_replace_f0(y, sr, src)
        metrics["mode_used"] = res.mode
        if res.ok:
            pr = _probe_metrics(res.y, sr, src)
            metrics["f0_r_pre"] = metrics.get("f0_r")
            metrics["f0_r"] = pr.get("r")
            metrics.update({k: v for k, v in pr.items() if k != "r"})
        return res, metrics
    # auto
    if r >= 0.75:
        res = f0_local_repair(y, sr, src)
        if res.ok:
            metrics["mode_used"] = "repair"
            metrics["repaired_segments"] = res.repaired_segments
            metrics["repaired_sec"] = round(res.repaired_sec, 2)
            pr = _probe_metrics(res.y, sr, src)
            metrics["f0_r_pre"] = metrics.get("f0_r")
            metrics["f0_r"] = pr.get("r")
            metrics.update({k: v for k, v in pr.items() if k != "r"})
            return res, metrics
        metrics["repair_fallback"] = res.reason
    res = world_replace_f0(y, sr, src)
    metrics["mode_used"] = res.mode if res.ok else "fallback"
    if res.ok:
        pr = _probe_metrics(res.y, sr, src)
        metrics["f0_r_pre"] = metrics.get("f0_r")
        metrics["f0_r"] = pr.get("r")
        metrics.update({k: v for k, v in pr.items() if k != "r"})
    return res, metrics


# ---------------------------------------------------------------------------
# 5) 实时块级 F0（流式变声）
# ---------------------------------------------------------------------------

class F0Tracker:
    """流式 5ms 网格 F0：吃入设备采样率音频块，产出与块对齐的 f0 段。

    实现口径：以「已累积音频」的固定网格 (n_full_frames = N//(sr*0.005)+1)
    为准，每块只计算新增帧；块内短音频直接 harvest（小输入开销低）。
    """

    def __init__(self, sr: int, f0_floor: float = F0_FLOOR_HZ,
                 f0_ceil: float = F0_CEIL_HZ):
        self.sr = int(sr)
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self._samples = 0
        self._cache: dict = {}

    def reset(self):
        self._samples = 0
        self._cache = {}

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        """返回该 chunk 对应的 5ms f0（清音=0）。连续调用天然同网格。"""
        pw = _require_pyworld()
        chunk = np.asarray(chunk, dtype=np.float64)
        prev_total = frames_of(self._samples, self.sr) - 1  # 上一帧边界
        self._samples += len(chunk)
        total = frames_of(self._samples, self.sr)
        n_new = total - 1 - prev_total  # 新增完整帧数（最后一帧未完成）
        if n_new <= 0:
            return np.zeros(0, dtype=np.float64)
        try:
            f0, _t = pw.harvest(chunk, self.sr, f0_floor=self.f0_floor,
                                f0_ceil=self.f0_ceil, frame_period=FRAME_PERIOD_MS)
            f0 = np.asarray(f0, dtype=np.float64)
        except Exception:
            f0 = np.zeros(n_new, dtype=np.float64)
        if len(f0) == 0:
            return np.zeros(n_new, dtype=np.float64)
        return _interp_to_grid(f0, n_new)
