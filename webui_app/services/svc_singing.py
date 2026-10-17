"""svc_singing — E1-T 歌声推理的指标与信号工具（HNR 代理 / 高音子集 / 高频移植）。

与 svc_f0 的分工：svc_f0 管 F0 轮廓（提取/映射/移植），本模块管
「歌声质量指标」与「零训练的高频应急」：

1. hnr_proxy：HPSS 谐波/残余能量比（dB）——E1-T 诊断实锤的主要掉话指标
   （03 转换阶段 350-450Hz 掉 6-12dB），不是 Praat HNR，但同管线可比。
2. high_subset / per_sentence：高音子集（F0>阈值）单独统计 + 逐句指标，
   供 report.json 与 BoN 择优共用——「整首平均会掩盖高音塌陷」是母规划
   §7.2 的报告红线。
3. transplant_highs：SYKI-SVC（ICASSP 2025, arXiv 2501.02953）式源高频
   移植后处理：转换输出低通 + 源干声高通直接加回（幅度按均值比对齐，
   阈值 ~10kHz）。消融证据：自然度 4.11 vs 3.82。依据是「10kHz 以上几乎
   不含音色信息」而源干声高频可用——22k 引擎下的适用带宽如实收窄
   （>9-10kHz 段），预期收益有限但零训练风险，默认关。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1) HNR 代理（HPSS）
# ---------------------------------------------------------------------------

def hnr_proxy(y: np.ndarray, sr: int, frame_sec: float = 0.04) -> np.ndarray:
    """逐帧 HPSS 谐波/残余能量比（dB）。返回与帧网格对齐的一维数组。"""
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if len(y) < int(0.1 * sr):
        return np.zeros(1, dtype=np.float32)
    H, R = librosa.effects.hpss(y)
    hop = int(frame_sec * sr)
    n = len(y) // hop
    if n < 1:
        return np.zeros(1, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        h = H[i * hop:(i + 1) * hop]
        r = R[i * hop:(i + 1) * hop]
        eh = float(np.mean(h ** 2)) + 1e-12
        er = float(np.mean(r ** 2)) + 1e-12
        out[i] = 10.0 * np.log10(eh / er)
    return out


def voiced_hnr(y: np.ndarray, sr: int, f0: np.ndarray) -> Optional[float]:
    """浊音帧的 HNR 中位（dB）。f0 为 5ms 网格（svc_f0.harvest_f0 口径）。"""
    if f0 is None or len(f0) < 5:
        return None
    h = hnr_proxy(y, sr)
    hop_ratio = sr * 0.04
    idx = (np.arange(len(h)) * hop_ratio / (sr * 0.005)).astype(int)
    idx = idx[idx < len(f0)]
    v = f0[idx] > 0
    if v.sum() < 3:
        return None
    return float(np.median(h[:len(idx)][v]))


# ---------------------------------------------------------------------------
# 2) 高音子集 / 逐句
# ---------------------------------------------------------------------------

def high_subset_stats(y: np.ndarray, sr: int, f0: np.ndarray,
                      threshold_hz: float = 500.0) -> Dict:
    """F0>threshold 帧的 HNR/时长占比。f0 为 5ms 网格。"""
    v = f0 > 0
    hi = v & (f0 > threshold_hz)
    out: Dict = {"threshold_hz": threshold_hz,
                 "voiced_sec": round(float(v.sum()) * 0.005, 2)}
    if hi.sum() < 5:
        out["high_ratio"] = 0.0
        return out
    h = hnr_proxy(y, sr)
    idx = (np.arange(len(h)) * (sr * 0.04) / (sr * 0.005)).astype(int)
    m = idx < len(f0)
    sel = np.zeros(len(h), dtype=bool)
    sel[m] = hi[idx[m]]
    out["high_ratio"] = round(float(hi.sum() / max(v.sum(), 1)), 4)
    out["hnr_db_high"] = round(float(np.median(h[sel])), 2) if sel.any() else None
    return out


def segment_metrics(y: np.ndarray, sr: int, f0: np.ndarray,
                    seg_sec: float = 10.0) -> list:
    """按 seg_sec 切段的 [ {t0, f0_med, hnr} ]——逐句口径（报告红线：
    整首/逐句/探针分开写，不要混用）。"""
    seg_n = int(seg_sec / 0.005)
    out = []
    for s in range(0, len(f0), seg_n):
        seg = f0[s:s + seg_n]
        v = seg[seg > 0]
        a = int(s * 0.005 * sr)
        b = int(min(len(y), (s + seg_n) * 0.005 * sr))
        row = {"t0": round(s * 0.005, 1),
               "f0_med": round(float(np.median(v)), 1) if len(v) >= 3 else None}
        h = voiced_hnr(y[a:b], sr, seg)
        row["hnr_db"] = round(h, 2) if h is not None else None
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 3) SYKI-SVC 式源高频移植（零训练应急，默认关）
# ---------------------------------------------------------------------------

def transplant_highs(converted: np.ndarray, source_dry: np.ndarray, sr: int,
                     cutoff_hz: float = 9000.0,
                     work_sr: int = 44100) -> Tuple[np.ndarray, Dict]:
    """Out = 低通(转换结果) + 高通(源干声)·Diff，在 work_sr 域混合后回采样。

    Diff = mean|src| / mean|conv|（SYKI 原文的幅度均值比对齐）。
    输入两轨需等长（翻唱管线天然满足：块级严格保长）。
    """
    import librosa
    c = np.asarray(converted, dtype=np.float32)
    s = np.asarray(source_dry, dtype=np.float32)
    n = min(len(c), len(s))
    c, s = c[:n], s[:n]
    info: Dict = {"cutoff_hz": cutoff_hz, "work_sr": work_sr}
    if sr < cutoff_hz * 2:
        info["note"] = f"源采样率 {sr}Hz 不足以承载 {cutoff_hz}Hz 截止，收窄"
        cutoff_hz = max(2000.0, sr * 0.45)
        info["cutoff_hz"] = cutoff_hz
    cw = librosa.resample(c, orig_sr=sr, target_sr=work_sr)
    sw = librosa.resample(s, orig_sr=sr, target_sr=work_sr)
    from scipy.signal import butter, sosfiltfilt
    lo = sosfiltfilt(butter(4, cutoff_hz, btype="lowpass", fs=work_sr,
                            output="sos"), cw)
    hi = sosfiltfilt(butter(4, cutoff_hz, btype="highpass", fs=work_sr,
                            output="sos"), sw)
    diff = float(np.mean(np.abs(sw)) + 1e-9) / float(np.mean(np.abs(cw)) + 1e-9)
    diff = float(np.clip(diff, 0.25, 4.0))       # 幅度比对齐，防极端
    mix = lo + hi * diff
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix = mix * (0.98 / peak)
    out = librosa.resample(mix.astype(np.float32), orig_sr=work_sr, target_sr=sr)
    info["diff"] = round(diff, 3)
    return (out.astype(np.float32)[:n] if len(out) >= n
            else np.pad(out, (0, n - len(out))).astype(np.float32)), info


# ---------------------------------------------------------------------------
# 4) BoN 奖励（高音块择优；svc_core 调用，纯函数便于测试）
# ---------------------------------------------------------------------------

def bon_reward(wav: np.ndarray, sr: int, ref_f0: np.ndarray,
               target_f0: np.ndarray, ss: Optional[float] = None) -> Dict:
    """高音候选奖励：HNR + F0 一致性 − 八度错误 + SS。

    ref_f0/target_f0 为 5ms 网格。返回 {reward, hnr, within50, oct_err, ss}，
    供 BoN 择优与报告明细。权重固定（E1-T Step 14 第一版：奖励不稳时
    回退只看 HNR+八度——母规划的回退路径）。
    """
    from webui_app.services import svc_f0
    est = svc_f0.harvest_f0(np.asarray(wav, dtype=np.float64), sr)
    n = min(len(est), len(target_f0))
    est, tgt = est[:n], np.asarray(target_f0)[:n]
    r, err, nv = svc_f0.f0_correlation(est, tgt)
    both = (est > 0) & (tgt > 0)
    within = 0.0
    med_cents = 0.0
    if both.sum() >= 5:
        la = np.log2(est[both])
        lb = np.log2(tgt[both])
        lb_aligned = lb + np.round(np.median(la - lb))
        within = float((np.abs(1200 * (la - lb_aligned)) <= 50).mean())
        # 全局音区差（不做八度对齐）：候选整块掉八度时 f0_correlation 的
        # 对齐口径看不见（它只测轮廓形状），这里单独罚——高音候选掉八度
        # 是 BoN 最要拦的失败模式
        med_cents = float(1200.0 * np.median(la - lb))
    hnr = voiced_hnr(wav, sr, target_f0)
    oct_global_pen = 30.0 if abs(med_cents) > 600.0 else 0.0
    reward = (max(hnr or 0.0, 0.0) * 0.05 + within * 40.0
              - err * 40.0 + (ss or 0.0) * 20.0 - oct_global_pen)
    return {"reward": round(reward, 3), "hnr_db": None if hnr is None else round(hnr, 2),
            "within_50c": round(within, 4), "octave_err": round(err, 4),
            "global_cents": round(med_cents, 1), "ss": ss, "f0_r": round(r, 3)}
