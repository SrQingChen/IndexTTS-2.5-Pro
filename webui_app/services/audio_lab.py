"""L0 参考音频工作台。

IndexTTS-2.5 是零样本 TTS，音色几乎完全由参考音频决定（见 params_inference.py
的 spk_audio_prompt 条目：音频兵分三路 —— CAMPPlus 声纹、w2v-BERT 情感特征、
ref_mel 声学模板）。因此**把参考音频选对、处理好，往往比训练 LoRA 更有效**。

本模块提供：
    analyze()      客观指标分析 + 打分 + 问题清单
    best_segment() 自动挑出最佳的 8~15 秒连续片段
    enhance()      降噪 / 响度归一 / 重采样 / 单声道化
    export()       处理后导出，可直接入库或用作参考音频

全部指标都对应 infer_v2_5.py 里的真实约束，不是泛泛的"音频质量"。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from webui_app.config import OUTPUT_SAMPLE_RATE, REF_AUDIO_IDEAL

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma")


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """读成 float32 单声道。sr=None 时保持原采样率。"""
    import librosa

    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), orig_sr


def save_audio(path: str, y: np.ndarray, sr: int):
    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:            # 防削波
        y = y / peak * 0.999
    sf.write(path, y.astype(np.float32), sr, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# 帧级分析
# ---------------------------------------------------------------------------

def _frames_db(y: np.ndarray, sr: int, frame_ms: float = 25.0,
               hop_ms: float = 10.0) -> Tuple[np.ndarray, int]:
    """逐帧 RMS（dBFS）。返回 (帧dB数组, 帧数)。"""
    win = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return db.astype(np.float32), n


@dataclass
class AudioReport:
    """参考音频的体检报告。"""

    path: str = ""
    ok: bool = False
    error: str = ""

    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    peak_dbfs: float = -120.0
    rms_dbfs: float = -120.0
    clip_ratio: float = 0.0          # 削波样本占比
    silence_ratio: float = 0.0       # 静音帧占比
    snr_db: float = 0.0              # 估算信噪比
    dynamic_range_db: float = 0.0
    spectral_centroid_hz: float = 0.0
    dc_offset: float = 0.0
    leading_silence: float = 0.0     # 开头静音秒数
    trailing_silence: float = 0.0

    score: float = 0.0               # 0~100 综合分
    grade: str = "-"
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze(path: str) -> AudioReport:
    """对一段音频做完整体检。所有阈值来自 REF_AUDIO_IDEAL。"""
    import librosa

    r = AudioReport(path=path)
    if not path or not os.path.isfile(path):
        r.error = f"文件不存在：{path}"
        return r

    try:
        import soundfile as sf
        info = sf.info(path)
        r.channels = int(info.channels)
        r.sample_rate = int(info.samplerate)

        y, sr = librosa.load(path, sr=None, mono=True)
        y = np.asarray(y, dtype=np.float32)
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
        return r

    if len(y) == 0:
        r.error = "音频为空"
        return r

    r.ok = True
    r.duration = float(len(y) / sr)
    r.peak_dbfs = float(20 * np.log10(max(np.max(np.abs(y)), 1e-10)))
    r.rms_dbfs = float(20 * np.log10(max(np.sqrt(np.mean(y.astype(np.float64) ** 2)), 1e-10)))
    r.clip_ratio = float(np.mean(np.abs(y) >= 0.999))
    r.dc_offset = float(np.mean(y))
    r.spectral_centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    db, nframes = _frames_db(y, sr)
    if nframes:
        peak_frame = float(np.max(db))
        # 语音帧：距峰值帧 35dB 以内；其余视为噪声底
        voiced = db >= peak_frame - 35.0
        r.silence_ratio = float(1.0 - np.mean(voiced))
        r.dynamic_range_db = float(peak_frame - np.min(db))
        if voiced.any() and (~voiced).any():
            r.snr_db = float(np.mean(db[voiced]) - np.mean(db[~voiced]))
        elif voiced.all():
            r.snr_db = 60.0        # 全程有声，无从估计噪声底
        else:
            r.snr_db = 0.0

        hop = 0.01                 # _frames_db 的 hop 是 10ms
        nz = np.nonzero(voiced)[0]
        if len(nz):
            r.leading_silence = float(nz[0] * hop)
            r.trailing_silence = float((nframes - 1 - nz[-1]) * hop)

    _score(r)
    return r


def _score(r: AudioReport):
    """按 REF_AUDIO_IDEAL 打分并生成问题清单。

    评分权重刻意向「时长」和「信噪比」倾斜 —— 这两项对最终音色相似度
    的影响最直接（CAMPPlus 需要足够样本统计声纹，w2v-BERT 对噪声敏感）。
    """
    ideal = REF_AUDIO_IDEAL
    score = 100.0
    issues: List[str] = []
    sugg: List[str] = []

    # 时长（权重最高）
    d = r.duration
    if d < ideal["min_sec"]:
        score -= 35
        issues.append(f"时长仅 {d:.1f}s，低于最低要求 {ideal['min_sec']:.0f}s，"
                      "CAMPPlus 声纹统计不稳定")
        sugg.append("换一段更长的录音，或用「智能切片」拼接同一说话人的多段音频")
    elif d < ideal["ideal_min_sec"]:
        score -= 15
        issues.append(f"时长 {d:.1f}s 偏短（理想 {ideal['ideal_min_sec']:.0f}~"
                      f"{ideal['ideal_max_sec']:.0f}s）")
        sugg.append("延长到 8s 以上能明显提升音色稳定性")
    elif d > ideal["max_sec"]:
        issues.append(f"时长 {d:.1f}s 超过 {ideal['max_sec']:.0f}s，"
                      "官方会**硬截断取前 15 秒**，超出部分完全浪费")
        sugg.append("用「智能切片」把最精华的 10~15s 挑出来，别指望模型自己选")
        score -= 5

    # 信噪比
    if r.snr_db < ideal["min_snr_db"]:
        score -= 25
        issues.append(f"估算信噪比仅 {r.snr_db:.1f} dB（理想 ≥{ideal['ideal_snr_db']:.0f} dB），"
                      "背景噪声会被 CAMPPlus 当成声纹的一部分")
        sugg.append("用「增强处理」做一次降噪；有 BGM 的录音建议换素材，降噪救不回音乐")
    elif r.snr_db < ideal["ideal_snr_db"]:
        score -= 10
        issues.append(f"信噪比 {r.snr_db:.1f} dB，尚可但有提升空间")
        sugg.append("轻度降噪可再提几分")

    # 采样率
    if r.sample_rate < ideal["min_sr"]:
        score -= 25
        issues.append(f"采样率 {r.sample_rate} Hz 低于 {ideal['min_sr']} Hz，"
                      "w2v-BERT 与 CAMPPlus 都按 16kHz 设计，低采样率会丢失高频信息")
        sugg.append("换原始高采样率素材。上采样**无法**恢复已丢失的高频")
    elif r.sample_rate < ideal["ideal_sr"]:
        score -= 6
        issues.append(f"采样率 {r.sample_rate} Hz 低于理想的 {ideal['ideal_sr']} Hz")

    # 削波
    if r.clip_ratio > ideal["max_clip_ratio"]:
        score -= 20
        issues.append(f"削波样本占比 {r.clip_ratio*100:.2f}%，"
                      "波形被削平会让 ref_mel 的高频结构失真")
        sugg.append("用更低增益重新录制/导出；已削波的音频无法修复")

    # 响度
    if r.rms_dbfs < -35:
        score -= 12
        issues.append(f"整体响度偏低（RMS {r.rms_dbfs:.1f} dBFS）")
        sugg.append("用「增强处理」做响度归一到 -20 dBFS")
    elif r.rms_dbfs > -6:
        score -= 12
        issues.append(f"整体响度偏高（RMS {r.rms_dbfs:.1f} dBFS），接近削波")
        sugg.append("衰减到 -20 dBFS 左右")

    # 静音占比
    if r.silence_ratio > ideal["max_silence_ratio"]:
        score -= 15
        issues.append(f"静音帧占比 {r.silence_ratio*100:.0f}%，有效语音太少")
        sugg.append("用「智能切片」裁掉静音段")

    # 开头静音（因为官方是"取前15秒"，开头静音直接占用配额）
    if r.leading_silence > 1.0:
        score -= 10
        issues.append(f"开头有 {r.leading_silence:.1f}s 静音 —— 官方截断是「取前 15 秒」，"
                      "这段静音会白白占用配额并污染声纹统计")
        sugg.append("裁掉开头静音，这是最容易被忽略但影响很大的一项")

    # 声道
    if r.channels > 1:
        issues.append(f"{r.channels} 声道，推理时会被混为单声道")
        sugg.append("预先转单声道，避免混音策略带来的意外")

    # DC 偏移
    if abs(r.dc_offset) > 0.01:
        score -= 4
        issues.append(f"存在 DC 偏移（{r.dc_offset:+.4f}）")
        sugg.append("增强处理会自动去除")

    r.score = round(max(0.0, min(100.0, score)), 1)
    r.grade = ("优秀" if r.score >= 88 else
               "良好" if r.score >= 72 else
               "可用" if r.score >= 55 else
               "勉强" if r.score >= 38 else "不建议使用")
    r.issues = issues
    r.suggestions = sugg
    return r


def report_markdown(r: AudioReport) -> str:
    """把报告渲染成 Markdown 表格 + 问题清单。"""
    if not r.ok:
        return f"**分析失败**：{r.error}"

    emoji = {"优秀": "🟢", "良好": "🟢", "可用": "🟡", "勉强": "🟠", "不建议使用": "🔴"}
    lines = [
        f"### {emoji.get(r.grade, '')} 综合评分 **{r.score}** / 100 &nbsp;·&nbsp; {r.grade}",
        "",
        "| 指标 | 实测 | 理想 | 判定 |",
        "|---|---|---|---|",
    ]

    def row(name, got, want, good):
        return f"| {name} | {got} | {want} | {'✅' if good else '⚠️'} |"

    I = REF_AUDIO_IDEAL
    lines += [
        row("时长", f"{r.duration:.2f} s",
            f"{I['ideal_min_sec']:.0f}~{I['ideal_max_sec']:.0f} s",
            I["ideal_min_sec"] <= r.duration <= I["max_sec"]),
        row("采样率", f"{r.sample_rate} Hz", f"≥{I['ideal_sr']} Hz",
            r.sample_rate >= I["ideal_sr"]),
        row("声道", f"{r.channels}", "1", r.channels == 1),
        row("信噪比(估算)", f"{r.snr_db:.1f} dB", f"≥{I['ideal_snr_db']:.0f} dB",
            r.snr_db >= I["ideal_snr_db"]),
        row("RMS 响度", f"{r.rms_dbfs:.1f} dBFS", f"{I['target_lufs']:.0f} dBFS 附近",
            -35 <= r.rms_dbfs <= -6),
        row("峰值", f"{r.peak_dbfs:.1f} dBFS", "≤ -1 dBFS", r.peak_dbfs <= -1.0),
        row("削波占比", f"{r.clip_ratio*100:.3f} %",
            f"≤{I['max_clip_ratio']*100:.1f} %", r.clip_ratio <= I["max_clip_ratio"]),
        row("静音占比", f"{r.silence_ratio*100:.0f} %",
            f"≤{I['max_silence_ratio']*100:.0f} %",
            r.silence_ratio <= I["max_silence_ratio"]),
        row("开头静音", f"{r.leading_silence:.2f} s", "≤ 0.3 s", r.leading_silence <= 0.3),
        row("结尾静音", f"{r.trailing_silence:.2f} s", "≤ 1.0 s", r.trailing_silence <= 1.0),
        row("动态范围", f"{r.dynamic_range_db:.1f} dB", "-", True),
        row("频谱质心", f"{r.spectral_centroid_hz:.0f} Hz", "-", True),
        row("DC 偏移", f"{r.dc_offset:+.4f}", "|x| ≤ 0.01", abs(r.dc_offset) <= 0.01),
    ]

    if r.issues:
        lines += ["", "**发现的问题**", ""]
        lines += [f"- {i}" for i in r.issues]
    if r.suggestions:
        lines += ["", "**建议**", ""]
        lines += [f"- {s}" for s in r.suggestions]
    if not r.issues:
        lines += ["", "🟢 没有发现明显问题，这段音频可以直接作为参考音频使用。"]

    lines += [
        "",
        "> ⚠️ 记住：官方 `_load_and_cut_audio(prompt, 15)` 是**取前 15 秒**，"
        "不是择优。所以音频**开头**的质量最关键。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 智能切片
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    score: float
    snr_db: float
    voiced_ratio: float
    clip_ratio: float
    rms_dbfs: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_segments(
    path: str,
    target_sec: float = 12.0,
    min_sec: float = 6.0,
    max_candidates: int = 8,
    hop_sec: float = 0.5,
) -> List[Segment]:
    """滑动窗口找出评分最高的若干连续片段。

    评分依据（对应三路音色注入的真实需求）：
        · 语音帧占比高   → CAMPPlus 有足够有效样本统计声纹
        · 信噪比高       → w2v-BERT 特征干净
        · 无削波         → ref_mel 高频结构完整
        · RMS 适中       → 不过轻也不过爆
        · 边界不在词中   → 优先让窗口起止落在静音处，避免切断音节
    """
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    dur = len(y) / sr
    if dur < min_sec:
        return []

    db, nframes = _frames_db(y, sr)
    frame_hop_sec = 0.01              # _frames_db 的 hop 固定为 10ms
    peak_frame = float(np.max(db)) if nframes else -120.0
    voiced_all = db >= peak_frame - 35.0

    win = int(target_sec * sr)
    step = max(1, int(hop_sec * sr))
    out: List[Segment] = []

    for s in range(0, max(1, len(y) - win + 1), step):
        e = min(s + win, len(y))
        seg = y[s:e]
        if len(seg) < min_sec * sr:
            continue
        f0 = int(s / (sr * frame_hop_sec))
        f1 = min(nframes, int(e / (sr * frame_hop_sec)) + 1)
        if f1 <= f0:
            continue
        vdb = db[f0:f1]
        vv = voiced_all[f0:f1]

        voiced_ratio = float(np.mean(vv))
        if vv.any() and (~vv).any():
            snr = float(np.mean(vdb[vv]) - np.mean(vdb[~vv]))
        elif vv.all():
            snr = 60.0
        else:
            snr = 0.0
        rms = float(20 * np.log10(max(np.sqrt(np.mean(seg.astype(np.float64) ** 2)), 1e-10)))
        clip = float(np.mean(np.abs(seg) >= 0.999))

        # 打分
        sc = 0.0
        sc += 40 * min(1.0, voiced_ratio / 0.80)          # 语音占比，0.8 封顶
        sc += 30 * min(1.0, max(0.0, snr) / 30.0)          # 信噪比，30dB 封顶
        sc += 12 if clip < 0.0005 else (-25 if clip > 0.002 else -8)
        # 响度：-26 ~ -14 dBFS 之间给满分
        sc += 12 * max(0.0, 1.0 - abs(rms + 20.0) / 14.0)
        # 边界质量：起止点附近能量低 = 落在停顿处，不会切断音节
        edge = 0.02 * sr
        for b in (seg[:int(edge)], seg[-int(edge):]):
            if len(b):
                bd = 20 * np.log10(max(np.sqrt(np.mean(b.astype(np.float64) ** 2)), 1e-10))
                edge_penalty = min(1.0, max(0.0, (bd - (peak_frame - 35)) / 20.0))
                sc += 3 * (1.0 - edge_penalty)

        out.append(Segment(
            start=round(s / sr, 3), end=round(e / sr, 3), score=round(sc, 2),
            snr_db=round(snr, 1), voiced_ratio=round(voiced_ratio, 3),
            clip_ratio=round(clip, 5), rms_dbfs=round(rms, 1),
        ))

    out.sort(key=lambda x: -x.score)
    # 去重：彼此重叠超过 60% 的只留最高分那个
    kept: List[Segment] = []
    for seg in out:
        overlap = False
        for k in kept:
            lo, hi = max(seg.start, k.start), min(seg.end, k.end)
            if hi > lo and (hi - lo) / seg.duration > 0.6:
                overlap = True
                break
        if not overlap:
            kept.append(seg)
        if len(kept) >= max_candidates:
            break
    return kept


def extract_segment(path: str, seg: Segment, out_path: str) -> str:
    """按 Segment 裁切并导出。"""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    a, b = int(seg.start * sr), int(seg.end * sr)
    save_audio(out_path, np.asarray(y[a:b], dtype=np.float32), sr)
    return out_path


def segments_markdown(segs: List[Segment]) -> str:
    if not segs:
        return ("**没有找到合适的片段。** 音频总时长可能短于 6 秒 —— "
                "此时请直接使用原文件，或换更长的素材。")
    lines = [
        "| # | 起点 | 终点 | 时长 | 评分 | 信噪比 | 语音占比 | 削波 | RMS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(segs):
        lines.append(
            f"| {i} | {s.start:.2f}s | {s.end:.2f}s | {s.duration:.2f}s | "
            f"**{s.score:.1f}** | {s.snr_db:.1f} dB | {s.voiced_ratio*100:.0f}% | "
            f"{s.clip_ratio*100:.3f}% | {s.rms_dbfs:.1f} dBFS |"
        )
    lines += [
        "",
        f"🟢 **推荐 #0**（{segs[0].start:.2f}s ~ {segs[0].end:.2f}s）。"
        "评分综合考虑了语音占比、信噪比、削波、响度和边界是否落在停顿处。",
        "",
        "> 选片段时**别只看分数**：#0 可能是内容平淡的一段。"
        "如果目标合成需要某种情绪，优先挑情绪匹配的那一段 —— 因为参考音频的"
        "w2v-BERT 分支会决定默认情感基调。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 增强处理
# ---------------------------------------------------------------------------

@dataclass
class EnhanceResult:
    path: str = ""
    ok: bool = False
    error: str = ""
    steps: List[str] = field(default_factory=list)
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    sample_rate: int = 0


def enhance(
    path: str,
    out_path: str,
    denoise: bool = True,
    denoise_strength: float = 0.6,
    normalize: bool = True,
    target_dbfs: float = -20.0,
    trim_silence: bool = True,
    silence_thresh_db: float = -45.0,
    resample: bool = True,
    target_sr: int = OUTPUT_SAMPLE_RATE,
    max_sec: float = 15.0,
) -> EnhanceResult:
    """一站式增强：去 DC → 裁静音 → 降噪 → 响度归一 → 重采样 → 限长。

    顺序是刻意安排的：先裁静音再降噪，可以避免对纯噪声段做无谓计算；
    先降噪再归一，可以避免把噪声一起放大。
    """
    res = EnhanceResult()
    try:
        before = analyze(path)
        res.before = before.to_dict()
        if not before.ok:
            res.error = before.error
            return res

        y, sr = load_audio(path)
        steps = []

        # 1) 去 DC 偏移
        dc = float(np.mean(y))
        if abs(dc) > 1e-4:
            y = y - dc
            steps.append(f"去除 DC 偏移 {dc:+.4f}")

        # 2) 裁掉首尾静音
        if trim_silence:
            y2 = _trim(y, sr, silence_thresh_db)
            if len(y2) and len(y2) < len(y):
                steps.append(f"裁掉首尾静音 {(len(y)-len(y2))/sr:.2f}s")
                y = y2

        # 3) 降噪
        if denoise and denoise_strength > 0:
            y = _denoise(y, sr, denoise_strength)
            steps.append(f"频谱降噪（强度 {denoise_strength:.2f}）")

        # 4) 响度归一
        if normalize:
            y = _normalize(y, target_dbfs)
            steps.append(f"RMS 响度归一到 {target_dbfs:.0f} dBFS")

        # 5) 重采样
        if resample and sr != target_sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            steps.append(f"重采样 {sr} → {target_sr} Hz")
            sr = target_sr

        # 6) 限长（官方硬截断到前 15s，这里主动裁到目标长度）
        if len(y) > max_sec * sr:
            y = y[:int(max_sec * sr)]
            steps.append(f"裁剪到前 {max_sec:.0f}s（匹配官方硬截断行为）")

        save_audio(out_path, y, sr)
        res.path = out_path
        res.ok = True
        res.steps = steps
        res.duration = float(len(y) / sr)
        res.sample_rate = sr
        res.after = analyze(out_path).to_dict()
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


def _trim(y: np.ndarray, sr: int, thresh_db: float) -> np.ndarray:
    """裁掉首尾低于阈值的静音。"""
    db, n = _frames_db(y, sr, frame_ms=25.0, hop_ms=10.0)
    if not n:
        return y
    active = np.nonzero(db >= thresh_db)[0]
    if not len(active):
        return y
    hop = int(sr * 0.01)
    a = max(0, int(active[0] * hop) - int(0.05 * sr))
    b = min(len(y), int((active[-1] + 1) * hop) + int(0.05 * sr))
    return y[a:b]


def _denoise(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """频谱门限降噪。strength 0~1 映射到 noisereduce 的 prop_decrease。"""
    try:
        import noisereduce as nr
        return np.asarray(
            nr.reduce_noise(
                y=y, sr=sr, prop_decrease=float(np.clip(strength, 0.0, 1.0)),
                stationary=False, n_fft=1024, win_length=1024, hop_length=256,
            ),
            dtype=np.float32,
        )
    except Exception:
        return y


def _normalize(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    """RMS 归一化，带峰值保护（不超过 -1 dBFS）。"""
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    if rms < 1e-10:
        return y
    cur = 20 * np.log10(rms)
    gain = 10 ** ((target_dbfs - cur) / 20.0)
    out = y * gain
    peak = float(np.max(np.abs(out)))
    limit = 10 ** (-1.0 / 20.0)      # -1 dBFS
    if peak > limit:
        out = out * (limit / peak)
    return out.astype(np.float32)


def enhance_markdown(r: EnhanceResult) -> str:
    if not r.ok:
        return f"**处理失败**：{r.error}"
    lines = [f"### ✅ 已导出 `{os.path.basename(r.path)}`", ""]
    if r.before and r.after:
        b, a = r.before, r.after
        lines += [
            "| 指标 | 处理前 | 处理后 |",
            "|---|---|---|",
            f"| 综合评分 | {b['score']} ({b['grade']}) | **{a['score']} ({a['grade']})** |",
            f"| 时长 | {b['duration']:.2f}s | {a['duration']:.2f}s |",
            f"| 采样率 | {b['sample_rate']} Hz | {a['sample_rate']} Hz |",
            f"| 信噪比 | {b['snr_db']:.1f} dB | {a['snr_db']:.1f} dB |",
            f"| RMS 响度 | {b['rms_dbfs']:.1f} dBFS | {a['rms_dbfs']:.1f} dBFS |",
            f"| 削波占比 | {b['clip_ratio']*100:.3f}% | {a['clip_ratio']*100:.3f}% |",
            f"| 开头静音 | {b['leading_silence']:.2f}s | {a['leading_silence']:.2f}s |",
            "",
        ]
        delta = a["score"] - b["score"]
        if delta > 0:
            lines.append(f"🟢 评分提升 **+{delta:.1f}** 分。")
        elif delta < 0:
            lines.append(
                f"🟠 评分反而下降了 **{delta:.1f}** 分。这通常意味着降噪强度过高，"
                "把语音细节也削掉了（听感会发闷、有水声）。建议把降噪强度降到 0.3 以下，"
                "或者干脆关掉降噪只做归一化。"
            )
        else:
            lines.append("· 评分持平。")
    if r.steps:
        lines += ["", "**执行的处理步骤**", ""]
        lines += [f"{i+1}. {s}" for i, s in enumerate(r.steps)]
    if r.after and r.after.get("issues"):
        lines += ["", "**处理后仍存在的问题**", ""]
        lines += [f"- {i}" for i in r.after["issues"]]
    return "\n".join(lines)
