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
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from webui_app.config import OUTPUT_SAMPLE_RATE, REF_AUDIO_IDEAL

# 降噪时**高频段保留原始信号**的频率下限。
#
# noisereduce 是宽带门限：它在所有频段一起估计噪声底，而人声的齿音、气声、
# 咬字细节恰好集中在 7~11 kHz —— 一起压下去，听感就是「闷」「哑」「像蒙了层布」。
# 这里把降噪结果与原始信号**按频率拼回来**：这条线以下用降噪结果（去掉嘶声、
# 嗡声、房间底噪），以上用原始（细节一点不动）。
#
# 8 kHz 是经验值：高于它的能量在语音里占比很小（实测参考素材 8~11 kHz 仅约 5%），
# 但人耳对它的有无非常敏感（决定「通透」还是「发闷」）。
DENOISE_KEEP_HIGH_HZ = 8000.0

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma")


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """读成 float32 单声道。sr=None 时保持原采样率。"""
    import librosa

    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), orig_sr


def save_audio(path: str, y: np.ndarray, sr: int, dither: bool = True):
    """写 WAV（PCM_16）。**拒绝写出 0 个采样点**。

    没有这道闸门时，一次越界切分就会生成一个只有 44 字节文件头的 wav（能写、
    能打开、听不到任何声音），后续体检判「音频为空」，用户看到的就是
    「点了没反应 / 存不进库」—— 排查起来毫无线索。宁可在这里直接抛。
    """
    import soundfile as sf

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(
            "拒绝写出空音频（0 个采样点）：那只会得到一个只有文件头的 wav。"
            "请检查切分范围是否越界、或上游是否传入了空数组。")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:            # 防削波
        y = y / peak * 0.999
    # 写 16 bit 前加 **TPDF 抖动**：直接取整会产生与信号相关的量化失真，
    # 在安静段、尾音、气声处听起来像「砂砾感 / 不干净」。抖动把这些失真
    # 变成不相关的低电平噪声，代价是极低的底噪，听感明显更干净。
    # 固定种子 → 同样的输入永远得到同样的字节，可复现（也保证了并行/
    # 串行输出逐字节一致的那条回归断言仍然成立）。
    if dither:
        lsb = 1.0 / 32768.0
        rng = np.random.default_rng(20240915)
        y = y + (rng.random(y.shape) - rng.random(y.shape)) * (lsb * 0.5)
    sf.write(path, y.astype(np.float32), sr, subtype="PCM_16")
    return path


# ---------------------------------------------------------------------------
# 帧级分析
# ---------------------------------------------------------------------------

def _frames_db(y: np.ndarray, sr: int, frame_ms: float = 25.0,
               hop_ms: float = 10.0,
               progress: Optional[Callable[[float, str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None
               ) -> Tuple[np.ndarray, int]:
    """逐帧 RMS（dBFS）。返回 (帧dB数组, 帧数)。

    **分块计算，不是一次性建索引矩阵。** 这点很关键：早先的写法是

        idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
        frames = y[idx]

    它同时物化「帧数 × 窗长」的索引与切片。10 ms hop、25 ms 窗意味着每帧占
    窗长的 2.5 倍，于是内存 ≈ 时长 × 采样率 × 2.5 × (8 字节索引 + 4 字节样本
    + 8 字节 float64)。一段 **1 小时 48 kHz** 的音频要 ~8.6 GB —— 一趟下来机器
    就在颠簸（表现就是「卡住不动」，日志也没有任何输出）。

    分块后峰值内存与总时长无关（只跟块大小有关），结果是**逐位相同**的：
    每一帧的均值都在自己那一行内累加，分不分块不影响运算顺序。
    """
    win = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    if n <= 0:
        return np.zeros(0, dtype=np.float32), 0

    # 块大小按「索引矩阵不超过约 40 MB」反推，同时给个下限保证向量化效率
    chunk = max(256, int(40e6 / (8 * win)))
    chunk = min(chunk, n)
    db = np.empty(n, dtype=np.float64)
    base = np.arange(win)[None, :]
    for start in range(0, n, chunk):
        if should_stop is not None and should_stop():
            # 被中断：已算出的部分保留（调用方据此决定是否继续）
            db = db[:start]
            n = start
            break
        stop = min(n, start + chunk)
        idx = base + hop * np.arange(start, stop)[:, None]
        frames = y[idx]
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        db[start:stop] = rms
        if progress is not None and n:
            progress(stop / n, f"帧能量 {stop}/{n}")
    return (20.0 * np.log10(np.maximum(db, 1e-10))).astype(np.float32), n


def probe_duration(path: str) -> float:
    """**只读文件头**取时长（秒），不解码整段音频。

    用来回答「这条算不算长音频」这类问题 —— 为此去跑一遍完整 `analyze`
    （整段解码 + 全量帧能量）在长音频上要几十秒甚至更久，纯属浪费。
    读不出来返回 -1。
    """
    try:
        info = sf.info(path)
        return float(info.duration or 0.0)
    except Exception:
        try:
            import librosa
            return float(librosa.get_duration(path=path))
        except Exception:
            return -1.0


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

        # 超长文件只分析前 30s（时长/采样率用 sf.info 的元数据真值）——
        # 990s 原件的 spectral_centroid STFT 单个就要 0.7 GB，4 worker
        # 并行 2.8 GB，实测把 C 运行时堆打爆（0xc0000409 硬崩、无 Python
        # 异常，2026-10-02 事故）。原件本来不可训练，不需要全量频谱。
        true_duration = float(info.duration)
        load_sec = min(true_duration, 30.0)
        y, sr = librosa.load(path, sr=None, mono=True,
                             duration=load_sec)
        y = np.asarray(y, dtype=np.float32)
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
        return r

    if len(y) == 0:
        r.error = "音频为空"
        return r

    r.ok = True
    r.duration = true_duration
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
    progress: Optional[Callable[[float, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> List[Segment]:
    """滑动窗口找出评分最高的若干连续片段。

    评分依据（对应三路音色注入的真实需求）：
        · 语音帧占比高   → CAMPPlus 有足够有效样本统计声纹
        · 信噪比高       → w2v-BERT 特征干净
        · 无削波         → ref_mel 高频结构完整
        · RMS 适中       → 不过轻也不过爆
        · 边界不在词中   → 优先让窗口起止落在静音处，避免切断音节

    progress / should_stop：长音频上这个函数会跑很久（1 小时音频按 1 秒 hop
    就是 3600 次窗口评分），所以必须能报进度、能被中断 —— 否则上层界面上
    看起来就是「点了没反应」，而这恰恰是最难区分「在工作」和「卡住了」的场景。
    被中断时返回**已经评完的那部分**候选（调用方自行决定要不要用）。

    y / sr：调用方若已经解码过整段音频就传进来，省掉这里的重复解码
    （切片时会连调这两个函数，一次解码能省下几百 MB 与数秒）。
    """
    import librosa

    def _tick(frac: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), msg)

    def _stopped() -> bool:
        return bool(should_stop is not None and should_stop())

    if y is None or not sr:
        # 整段解码是这里最重的一步（长音频几秒到几十秒），先报出来
        _tick(0.0, f"读取音频 {os.path.basename(path)}")
        y, sr = librosa.load(path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    dur = len(y) / sr
    if dur < min_sec:
        return []
    _tick(0.05, f"已解码 {dur:.0f} 秒，开始统计帧能量")

    db, nframes = _frames_db(
        y, sr,
        progress=(lambda f, m: _tick(0.05 + 0.25 * f, m)) if progress else None,
        should_stop=should_stop,
    )
    if _stopped():
        return []
    if nframes <= 0:
        return []
    frame_hop_sec = 0.01              # _frames_db 的 hop 固定为 10ms
    peak_frame = float(np.max(db)) if nframes else -120.0
    voiced_all = db >= peak_frame - 35.0

    win = int(target_sec * sr)
    step = max(1, int(hop_sec * sr))
    out: List[Segment] = []

    starts = range(0, max(1, len(y) - win + 1), step)
    n_win = max(1, (max(1, len(y) - win + 1) + step - 1) // step)
    done = 0
    for s in starts:
        if done % 32 == 0 and _stopped():
            break
        done += 1
        if done % 32 == 0 or done == n_win:
            _tick(0.30 + 0.65 * (done / n_win),
                  f"寻找切分点 {done}/{n_win}（{done * hop_sec:.0f}/{dur:.0f} 秒）")
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
    _tick(0.97, f"从 {len(out)} 个候选里去重")
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
    _tick(1.0, f"选定 {len(kept)} 个片段")
    return kept


def extract_segment(path: str, seg: Segment, out_path: str,
                    y: Optional[np.ndarray] = None,
                    sr: Optional[int] = None) -> str:
    """按 Segment 裁切并导出。

    y / sr 可以传入**已经解码好的整段波形**。这不是可有可无的优化：不传的话
    每导出一片就要把整个源文件重新解码一遍，而切片动辄几十上百片 ——
    实测一条长音频切 109 片花了 282 秒，几乎全耗在这 109 次重复解码上。
    调用方（dataset.split_long）现在只解码一次，然后逐片复用。
    """
    if y is None or not sr:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    dur = len(y) / sr if sr else 0.0
    # 端点超出音频长度（超出取整容差）说明这些秒数**不是这个文件的偏移**。
    # 必须报错而不是 clamp：clamp 之后往往还剩几秒音频，于是会静默产出一段
    # 「位置不对」的音频 —— 不出声、不报错，比直接失败更难查。
    tol = 0.05
    if dur <= 0 or seg.start > dur + tol or seg.end > dur + tol:
        raise ValueError(
            f"切分范围 {seg.start:.2f}~{seg.end:.2f}s 超出音频长度 {dur:.2f}s —— "
            "这些秒数是按**另一个**音频算出来的偏移。最常见的情况是：主素材已经"
            "换成了切出来的短片段，而候选片段还是按原长音频扫描的。"
            "请重新扫描候选片段。")
    a = max(0, int(seg.start * sr))
    b = min(len(y), int(seg.end * sr))
    piece = y[a:b]
    if len(piece) < max(1, int(0.05 * sr)):
        raise ValueError(
            f"切出的片段只有 {len(piece) / sr:.3f}s（{seg.start:.2f}~"
            f"{seg.end:.2f}s），太短了。请检查候选片段与音频是否匹配。")
    save_audio(out_path, piece, sr)
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
    highpass_hz: float = 60.0,
    denoise_keep_high_hz: float = DENOISE_KEEP_HIGH_HZ,
    pause_cap_ms: float = 0.0,
) -> EnhanceResult:
    """一站式增强：去 DC → 裁静音 → **句内静音封顶** → 降噪 → 响度归一 → 重采样 → 限长。

    顺序是刻意安排的：先裁静音再降噪，可以避免对纯噪声段做无谓计算；
    先降噪再归一，可以避免把噪声一起放大。

    pause_cap_ms > 0 时启用**句内静音封顶**（2026-09-29 碎裂停顿根治件）：
    片段内部超过该时长的静音一律压缩到该时长。动机（真机实测）：游戏配音
    的真实句间停顿常达 400~800ms，而 ASR 的标点模型把它们标成逗号——
    模型从这样的数据里学到「逗号=长停顿」并在无标点处泛化（实测生成
    一句话 7 个停顿、逗号处 800ms）。封顶后模型听到的句内停顿 ≤400ms
    （人类自然区间），长停顿的「名额」留给句末与省略号（推理端由导演
    模式的块间停顿供给）。
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

        # 1) 去 DC 偏移 + 高通（去掉 60 Hz 以下的隆隆声/空调声/手持噪声）
        dc = float(np.mean(y))
        if abs(dc) > 1e-4:
            y = y - dc
            steps.append(f"去除 DC 偏移 {dc:+.4f}")
        if 20.0 <= float(highpass_hz) < sr / 2 * 0.9 and len(y) > 64:
            try:
                from scipy.signal import butter, sosfiltfilt
                sos = butter(2, float(highpass_hz) / (sr / 2),
                             btype="highpass", output="sos")
                y = np.asarray(sosfiltfilt(sos, y), dtype=np.float32)
                steps.append(f"高通 {highpass_hz:.0f} Hz（去低频隆隆声）")
            except Exception:
                pass    # 高通失败不影响其它步骤

        # 2) 裁掉首尾静音
        if trim_silence:
            y2 = _trim(y, sr, silence_thresh_db)
            if len(y2) and len(y2) < len(y):
                steps.append(f"裁掉首尾静音 {(len(y)-len(y2))/sr:.2f}s")
                y = y2

        # 2.5) 句内静音封顶（pause_cap_ms>0 时）
        if pause_cap_ms and pause_cap_ms > 0:
            y2, n_capped = _cap_pauses(y, sr, silence_thresh_db,
                                       float(pause_cap_ms))
            if n_capped:
                y = y2
                steps.append(f"句内静音封顶 {pause_cap_ms:.0f} ms"
                             f"（压缩 {n_capped} 处超长停顿）")

        # 2.7) 高频过量治理（2026-10-11）：游戏VO分离人声普遍带强齿音/
        # 气声高频（锁暝素材 2.6~10% 浊音帧 HF 超语音 12dB）。训练前压制，
        # LoRA 才不会把撕裂感学进权重。两级：间隙湿爆冲 + 浊音撕裂限幅
        #（+12dB 天花板=已验收基线；正常齿音 +8~12dB 不受影响）。
        try:
            y, n_wet = suppress_wet_hf(y, sr)
            y, n_tear = limit_voiced_hf(y, sr)
            if n_wet or n_tear:
                steps.append(f"高频过量治理（间隙爆冲 {n_wet} 处 · "
                             f"浊音撕裂 {n_tear} 处）")
        except Exception:
            pass

        # 3) 降噪（高频段保留原始信号，避免把齿音/气息细节一起压掉）
        if denoise and denoise_strength > 0:
            y2, did = _denoise(y, sr, denoise_strength, denoise_keep_high_hz)
            if did:
                y = y2
                steps.append(
                    f"分频段降噪（强度 {denoise_strength:.2f}，"
                    f"{denoise_keep_high_hz / 1000:.0f} kHz 以上保留原始细节）")
            else:
                # **不能谎报**：没装 noisereduce 时这一步什么都没做，
                # 却写「频谱降噪」会让用户以为声音已经被处理过了。
                steps.append("降噪已跳过（未安装 noisereduce，其余步骤照常）")

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


def cap_interior_pauses(y: np.ndarray, sr: int, cap_ms: float,
                        thresh_db: float = -40.0) -> np.ndarray:
    """公开封装：把波形**内部**超过 cap_ms 的静音压缩到 cap_ms。

    推理输出侧的对称刀（训练素材封顶的同一技术）：模型在逗号处生成
    的 500~660ms 停顿被压到人味区间，语音与收音衰减一个采样点不动。
    cap_ms<=0 时原样返回。int16/float 均可，返回保持原 dtype。
    """
    if not cap_ms or float(cap_ms) <= 0 or len(y) < sr // 4:
        return y
    was_int = np.issubdtype(y.dtype, np.integer)
    yf = y.astype(np.float32) / 32768.0 if was_int else y
    y2, _n = _cap_pauses(yf, sr, thresh_db, float(cap_ms))
    return (np.clip(y2 * 32767.0, -32767, 32767).astype(np.int16)
            if was_int else y2)


def _cap_pauses(y: np.ndarray, sr: int, thresh_db: float,
                cap_ms: float) -> tuple:
    """把片段**内部**超过 cap_ms 的静音压缩到 cap_ms。返回 (波形, 压缩处数)。

    首尾各保留 50ms 不参与（那是 _trim 的辖区）。压缩方式：从超长静音的
    **中间**截掉超出部分——两端都是静音，不存在波形突变；切口两侧再做
    5ms 线性淡入淡出兜底。帧长 25ms / 帧移 10ms，与 _trim 同一套尺度。
    """
    db, n = _frames_db(y, sr, frame_ms=25.0, hop_ms=10.0)
    if not n:
        return y, 0
    hop = int(sr * 0.01)
    quiet = db < thresh_db
    edge = int(0.05 * sr)                      # 首尾 50ms 保护区
    lo_frame = edge // hop
    hi_frame = max(lo_frame, (len(y) - edge) // hop)

    cap_n = int(sr * cap_ms / 1000.0)
    fade = max(1, int(0.005 * sr))
    runs = []
    i = lo_frame
    while i < hi_frame:
        if quiet[i]:
            j = i
            while j < hi_frame and quiet[j]:
                j += 1
            a = i * hop
            b = min(len(y), j * hop)
            if b - a > cap_n:
                runs.append((a, b))
            i = j
        else:
            i += 1

    if not runs:
        return y, 0
    out = y
    # 从后往前截，前面区间的偏移不受影响
    for a, b in reversed(runs):
        cut = (b - a) - cap_n
        mid_a = a + cap_n // 2
        mid_b = b - (cap_n - cap_n // 2)
        head = out[:mid_a]
        tail = out[mid_b:]
        if len(head) >= fade and len(tail) >= fade:
            head = head.copy()
            tail = tail.copy()
            head[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=head.dtype)
            tail[:fade] *= np.linspace(0.0, 1.0, fade, dtype=tail.dtype)
        out = np.concatenate([head, tail])
    return out, len(runs)


# noisereduce 是**可选依赖**（没写进 pyproject 的基础依赖里）。它缺席时
# `_denoise` 只能原样返回 —— 这是个**静默跳过**：界面说「已降噪」，实际一个
# 采样点都没动。所以这里暴露一个能力探测，让上层（一键三连的参数面板、
# 数据集页、手册）能把「降噪不可用」明确说出来。
_NOISEREDUCE_OK: Optional[bool] = None


def noisereduce_available() -> bool:
    """noisereduce 是否可用。结果缓存（探测要 import，不必反复做）。"""
    global _NOISEREDUCE_OK
    if _NOISEREDUCE_OK is None:
        try:
            import noisereduce  # noqa: F401
            _NOISEREDUCE_OK = True
        except Exception:
            _NOISEREDUCE_OK = False
    return bool(_NOISEREDUCE_OK)


def denoise_note() -> str:
    """给界面用的一句话说明。可用时返回空串。"""
    if noisereduce_available():
        return ""
    return ("未安装 <code>noisereduce</code>，**降噪已被静默跳过**"
            "（其余增强步骤照常执行）。装上它即可生效：<br>"
            "<code>uv pip install --python .venv\Scripts\python.exe "
            "noisereduce</code>")



def _denoise(y: np.ndarray, sr: int, strength: float,
             keep_high_hz: float = DENOISE_KEEP_HIGH_HZ
             ) -> Tuple[np.ndarray, bool]:
    """频谱门限降噪。strength 0~1 映射到 noisereduce 的 prop_decrease。

    返回 (处理后的音频, 是否真的降噪了)。**没有 noisereduce 时原样返回且
    第二个值为 False** —— 调用方必须把「已跳过」如实告知用户，
    不要让这一步看起来像是做过了。

    keep_high_hz 以上保留原始信号，见 DENOISE_KEEP_HIGH_HZ 的说明。
    """
    if not noisereduce_available():
        return y, False
    try:
        import noisereduce as nr
        den = np.asarray(
            nr.reduce_noise(
                y=y, sr=sr, prop_decrease=float(np.clip(strength, 0.0, 1.0)),
                stationary=False, n_fft=1024, win_length=1024, hop_length=256,
            ),
            dtype=np.float32,
        )
    except Exception:
        # 运行期失败（异常参数、NaN 输入等）不该连累整条样本的增强：
        # 原样返回，其余步骤继续。缺包的情况由上面的能力探测单独处理。
        return y, False

    # 频带拼接：把高频细节从降噪结果里换回原始信号
    if keep_high_hz and 0 < keep_high_hz < sr / 2 and len(den) == len(y):
        try:
            import librosa
            n_fft, hop = 1024, 256
            Sd = librosa.stft(den, n_fft=n_fft, hop_length=hop)
            So = librosa.stft(y, n_fft=n_fft, hop_length=hop)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            hi = freqs >= float(keep_high_hz)
            if Sd.shape == So.shape and hi.any():
                Sd[hi] = So[hi]
                den = librosa.istft(Sd, hop_length=hop, length=len(y))
                den = np.asarray(den, dtype=np.float32)
        except Exception:
            pass          # 拼接失败就退回宽带降噪结果，不影响主流程
    return den, True


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


# ---------------------------------------------------------------------------
# 输出后处理：给合成结果「提亮 / 加空气感」
# ---------------------------------------------------------------------------

@dataclass
class PolishResult:
    """输出后处理的结果。ok=False 时带 error。"""

    ok: bool = False
    path: str = ""
    error: str = ""
    steps: List[str] = field(default_factory=list)
    duration: float = 0.0
    sample_rate: int = 0
    centroid_before: float = 0.0     # 谱质心（Hz）—— 「亮」的客观参考
    centroid_after: float = 0.0
    peak_before_dbfs: float = 0.0
    peak_after_dbfs: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _centroid_hz(y: np.ndarray, sr: int) -> float:
    try:
        import librosa
        return round(float(np.mean(
            librosa.feature.spectral_centroid(y=y.astype(np.float32), sr=sr))), 1)
    except Exception:
        return 0.0


def polish(
    path: str,
    out_path: str,
    highpass_hz: float = 60.0,
    presence_hz: float = 4500.0,
    presence_db: float = 0.0,
    exciter: float = 0.0,
    exciter_from_hz: float = 5000.0,
    target_peak_dbfs: float = -1.0,
) -> PolishResult:
    """对**合成输出**做听感提亮。默认中性（只做高通与峰值对齐）。

    为什么需要它：IndexTTS2 的输出固定 22050 Hz，**Nyquist 11 kHz 以上不可能有
    能量** —— 这是模型设计，不是处理链弄丢的（实测增强链前后各频段能量完全一致，
    而且 LoRA 反而让 8~11 kHz 略多）。拿 48 kHz 的原始素材去比，听感上就会觉得
    「发闷、不够通透」。这个函数做的**不是**伪造细节，而是两件也够实在的事：

        1. **presence 提升**：4.5 kHz 以上平滑抬升若干 dB。人耳对这段（齿音、
           咬字清晰度）最敏感，抬一点就明显更「亮」更清楚 —— 是 EQ，不是合成。
        2. **谐波激励（可选）**：从 5 kHz 以上的成分生成谐波并混回，在 11 kHz
           以上补出**属于这段音频自身**的泛音，缓解「高频被切断」的封闭感。
           不引入外来素材，只是把已有高频做非线性延展。

    再加上一个 60 Hz 高通（去掉合成偶发的低频隆隆声）与目标峰值归一
    （避免削波，也避免下游播放器自行衰减）。

    默认 presence_db=0、exciter=0 → **音色不变**，只做高通与峰值对齐。
    想提亮就把 presence_db 调到 2~4 dB（先听后定），需要空气感再加 0.05~0.15 的
    exciter。任何一项都可在合成页开关，不必重训模型。
    """
    res = PolishResult()
    try:
        y, sr = load_audio(path)
        if len(y) == 0:
            res.error = "音频为空"
            return res
        res.duration = float(len(y) / sr)
        res.sample_rate = sr
        res.centroid_before = _centroid_hz(y, sr)
        res.peak_before_dbfs = float(20 * np.log10(max(float(np.max(np.abs(y))), 1e-10)))
        steps: List[str] = []

        # 1) 高通：去掉低频隆隆声。这些能量不参与听感，却会占掉动态余量、
        #    并在峰值归一时逼着整体降电平。
        if 20.0 <= float(highpass_hz) < sr / 2 * 0.9 and len(y) > 64:
            try:
                from scipy.signal import butter, sosfiltfilt
                sos = butter(2, float(highpass_hz) / (sr / 2),
                             btype="highpass", output="sos")
                y = np.asarray(sosfiltfilt(sos, y), dtype=np.float32)
                steps.append(f"高通 {highpass_hz:.0f} Hz")
            except Exception:
                pass

        # 2) presence 提升（FFT 域平滑搁架；用 STFT 避免整段 FFT 的边缘环绕）
        if abs(float(presence_db)) > 0.05 and 500.0 < float(presence_hz) < sr / 2:
            try:
                import librosa
                n_fft, hop = 2048, 512
                S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                # 一个倍频程的升余弦过渡：够平滑，不会在转折频率处产生可闻的"硬边"
                f0 = float(presence_hz)
                gain = np.ones_like(freqs)
                lo, hi = f0 / 2.0, f0
                ramp = np.clip((freqs - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
                smooth = 0.5 * (1.0 - np.cos(np.pi * ramp))      # 升余弦
                gain = 1.0 + smooth * (10 ** (float(presence_db) / 20.0) - 1.0)
                S = S * gain[:, None]
                y = np.asarray(librosa.istft(S, hop_length=hop, length=len(y)),
                               dtype=np.float32)
                steps.append(f"presence +{presence_db:.1f} dB @ {presence_hz:.0f} Hz")
            except Exception:
                pass

        # 3) 谐波激励（可选）：只从高频成分生成，形成属于本段音频的泛音
        amt = float(exciter)
        if 0.0 < amt <= 1.0 and float(exciter_from_hz) < sr / 2:
            try:
                # 用**真正的滤波器**做，不要用 preemphasis/deemphasis 那一对：
                # deemphasis 是 1/(1-0.95z⁻¹)，直流增益高达 20 倍。把它作用在
                # **未预加重的主信号**上会把低频整体抬起来 —— 实测谱质心从
                # 3355 Hz 掉到 788 Hz、99.98% 能量挤进 0-4 kHz，声音直接闷掉。
                # （preemphasis 与 deemphasis 必须成对作用在同一路信号上。）
                from scipy.signal import butter, sosfiltfilt

                def _hp(sig, hz):
                    sos = butter(2, hz / (sr / 2), btype="highpass", output="sos")
                    return np.asarray(sosfiltfilt(sos, sig), dtype=np.float32)

                # ① 取高频成分 ② 非线性生成谐波（去掉线性项，只留新增）
                # ③ 只保留 8 kHz 以上的"空气"：以下已有真实内容，加了只会变糊
                h = _hp(y, max(6000.0, float(exciter_from_hz)))
                h = np.tanh(h * 3.0) - h
                h = _hp(h, min(9000.0, sr / 2 * 0.85))
                pk_h = float(np.max(np.abs(h))) or 1.0
                h = h / pk_h * 0.25
                # 主信号原样保留，只叠加新增的空气感
                y = np.asarray(y + amt * h, dtype=np.float32)
                steps.append(f"谐波激励 {amt:.2f}"
                             f"（从 {exciter_from_hz / 1000:.0f} kHz 生成，"
                             "只叠加 9 kHz 以上的空气感）")
            except Exception:
                pass

        # 4) 峰值对齐（避免削波，也避免下游播放器自行衰减）
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            target = 10 ** (float(target_peak_dbfs) / 20.0)
            if peak > target:
                y = y * (target / peak)
                steps.append(f"峰值对齐到 {target_peak_dbfs:.1f} dBFS")

        res.peak_after_dbfs = float(20 * np.log10(max(float(np.max(np.abs(y))), 1e-10)))
        save_audio(out_path, y, sr)
        res.path = out_path
        res.steps = steps
        res.centroid_after = _centroid_hz(y, sr)
        res.ok = True
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


def polish_markdown(r: PolishResult) -> str:
    """后处理结果的可读说明（含谱质心前后对比，方便判断「亮了没有」）。"""
    if not r.ok:
        return f"**后处理失败**：{r.error}"
    L = [f"### {'✅' if r.ok else '🔴'} 输出后处理完成", ""]
    L += ["| 指标 | 处理前 | 处理后 |", "|---|---|---|",
          f"| 谱质心（越高越亮） | {r.centroid_before:.0f} Hz | "
          f"**{r.centroid_after:.0f} Hz** |",
          f"| 峰值 | {r.peak_before_dbfs:.1f} dBFS | {r.peak_after_dbfs:.1f} dBFS |",
          f"| 时长 | {r.duration:.2f}s | {r.duration:.2f}s |", ""]
    if r.steps:
        L.append("**执行的处理**")
        L += [f"{i+1}. {s}" for i, s in enumerate(r.steps)]
    else:
        L.append("· 未做任何改动（参数都是中性值）。")
    L += ["", "<sub>模型输出固定 22050 Hz（11 kHz 上限），这里做的是"
              "**听感补偿**——把已有的清晰度抬出来、给高频做自身谐波延展，"
              "不伪造外来细节。想更亮就调 presence，想要空气感再加 exciter。</sub>"]
    return "\n".join(L)


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


# ---------------------------------------------------------------------------
# 响度体系（2026-09-29 · 响度指纹迁移，替代逐条归一的"削平"）
# ---------------------------------------------------------------------------

def rms_dbfs(y: np.ndarray) -> float:
    """整段 RMS（dBFS）。静音返回 -120。"""
    r = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    if r < 1e-10:
        return -120.0
    return 20.0 * np.log10(r)


def measure_loudness(path_or_y, sr: Optional[int] = None) -> float:
    """一条音频的响度（dBFS RMS）。接受路径或 (y, sr)。"""
    if isinstance(path_or_y, str):
        y, _sr = load_audio(path_or_y)
        return rms_dbfs(y)
    return rms_dbfs(path_or_y)


def anchor_gain(levels_db: List[float], target_dbfs: float = -20.0,
                max_boost_db: float = 12.0, max_cut_db: float = 15.0,
                clamp_outlier_db: float = 10.0) -> Dict[str, Any]:
    """由**全数据集**响度中位数算一个全局增益（所有人站同一块地板）。

    返回 {gain_db, median_db, lo_db, hi_db, n}：调用方给每条音频乘同一
    个 gain_db；lo/hi 是离群钳制边界（超出中位 ±clamp_outlier_db 的素材
    在数据集里被拉回边界 —— 由 apply_anchor 执行）。
    """
    lv = sorted(float(x) for x in levels_db if x > -100.0)
    if not lv:
        return {"gain_db": 0.0, "median_db": float(target_dbfs),
                "lo_db": float(target_dbfs), "hi_db": float(target_dbfs), "n": 0}
    med = float(np.median(lv))
    gain = float(np.clip(target_dbfs - med, -max_cut_db, max_boost_db))
    return {"gain_db": gain, "median_db": med,
            "lo_db": med - clamp_outlier_db, "hi_db": med + clamp_outlier_db,
            "n": len(lv)}


def apply_anchor(y: np.ndarray, anchor: Dict[str, Any],
                 peak_ceiling_db: float = -1.0) -> np.ndarray:
    """给一条音频施加全局锚定增益（含离群钳制与峰值保护）。

    片段间相对响度保持不变；只有「原始电平离中位数 >10dB 的离群素材」
    被拉回边界（那多半是录音事故，不是表演）。
    """
    cur = rms_dbfs(y)
    if cur <= -100.0:
        return y
    lo, hi = float(anchor["lo_db"]), float(anchor["hi_db"])
    eff = min(max(cur, lo), hi)              # 离群钳制：限的是增益基准
    target = eff + float(anchor["gain_db"])
    gain = 10 ** ((target - cur) / 20.0)
    out = y * gain
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    limit = 10 ** (peak_ceiling_db / 20.0)
    if peak > limit:
        out = out * (limit / peak)
    return out.astype(np.float32)


def loudness_fingerprint(levels_db: List[float]) -> Dict[str, Any]:
    """角色响度指纹：分布统计（写进 run.json，推理端逐块采样用）。"""
    lv = sorted(float(x) for x in levels_db if x > -100.0)
    if len(lv) < 3:
        return {}
    arr = np.asarray(lv)
    return {
        "n": int(len(lv)),
        "median": round(float(np.median(arr)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p40": round(float(np.percentile(arr, 40)), 2),
        "p60": round(float(np.percentile(arr, 60)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p10": round(float(np.percentile(arr, 10)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "std": round(float(np.std(arr)), 2),
    }


def sample_target_loudness(fp: Dict[str, Any], intensity: float,
                           rng: random.Random) -> float:
    """从角色响度指纹里采一个块目标响度（dBFS）。

    intensity（表演强度 0~1）决定从分布的哪一段采：平静块在轻端
    （p10~p40），爆发块在响端（p60~p90）——角色的"这句该响那句该轻"
    由它自己的分布供给，而不是随机抖动。指纹缺失时返回 -20（旧行为）。
    """
    if not fp or fp.get("n", 0) < 3:
        return -20.0
    if intensity >= 0.7:
        lo, hi = fp.get("p60", fp["median"]), fp.get("p90", fp["median"])
    elif intensity <= 0.35:
        lo, hi = fp.get("p10", fp["median"]), fp.get("p40", fp.get("p25"))
    else:
        lo, hi = fp.get("p25", fp["median"]), fp.get("p75", fp["median"])
    if hi < lo:
        lo, hi = hi, lo
    return round(float(rng.uniform(lo, hi)), 2)


def apply_block_loudness(y: np.ndarray, target_dbfs: float,
                         peak_ceiling_db: float = -1.0) -> np.ndarray:
    """把一个合成块调到目标响度（纯增益 + 峰值保护）。

    int16 / float 波形均可（内部按 ±1 浮点尺度运算后还原 dtype）——
    target_dbfs 与峰值阈值都是 ±1 尺度的量，int16 直传会算错增益
    （与 apply_stress_gains 同族的尺度陷阱，2026-10-02 一并加固）。
    """
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    cur = rms_dbfs(yf)
    if cur <= -100.0:
        return y
    gain = 10 ** ((target_dbfs - cur) / 20.0)
    out = yf * gain
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    limit = 10 ** (peak_ceiling_db / 20.0)
    if peak > limit:
        out = out * (limit / peak)
    if was_int:
        return np.clip(out * 32767.0, -32767, 32767).astype(np.int16)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 起止整形（治"被咬掉"感）
# ---------------------------------------------------------------------------

def shape_edges(y: np.ndarray, sr: int, pad_ms: float = 80.0,
                fade_ms: float = 12.0) -> np.ndarray:
    """首尾加垫 + 短淡化：开头淡入更缓（起振不能被切），收尾自然。

    兼容 int16 / float 波形（内部 float 运算后按原 dtype 归还）。
    """
    pad = int(sr * pad_ms / 1000.0)
    f = max(1, int(sr * fade_ms / 1000.0))
    out = np.concatenate([np.zeros(pad, dtype=y.dtype), y,
                          np.zeros(pad, dtype=y.dtype)])
    if len(out) > 2 * f:
        body = out.astype(np.float32)
        body[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
        body[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        out = (np.clip(body, -32767, 32767).astype(np.int16)
               if np.issubdtype(out.dtype, np.integer) else body)
    return out


def ensure_wav(src: str, dst: str) -> str:
    """把任意可解码音频落成 wav（PCM_16+TPDF 抖动）。已是 wav 则直接复制。

    入库即 wav 的统一入口（数据集导入 / 音色库 / 情感库共用）：
    · 引擎侧 torchaudio 对 mp3/m4a 的支持因 libsndfile 版本而异 —— mp3
      参考音频可能**静默读不了**（2026-09-29 响度锚定在 mp3 上崩掉是
      同族问题的显性版）；
    · 转码后全链路（体检/增强/切片/锚定/特征提取）只剩一种格式，
      不再有 sf.write(PCM_16) 到非 wav 的 Invalid combination 特例。
    解码失败或解码结果为空时抛异常，由调用方决定回退。
    """
    import shutil

    ext = os.path.splitext(src)[1].lower()
    if ext in ("", ".wav"):
        shutil.copy2(src, dst)
        return dst
    y, sr = load_audio(src)          # librosa：mp3/m4a/flac/ogg 经 audioread
    save_audio(dst, y, sr)           # 空音频在这里被拒绝
    return dst


# ---------------------------------------------------------------------------
# 频谱画像与包络匹配（2026-09-29 · 不饱满/缺频段的根治件）
# ---------------------------------------------------------------------------

SPEC_BANDS = [(0, 300), (300, 1000), (1000, 3000), (3000, 6000),
              (6000, 8500), (8500, 11000)]


def band_profile(y_or_path, sr: Optional[int] = None) -> Dict[str, Any]:
    # 数组入参未给 sr 时按项目输出采样率(22050)处理
    """长时频谱画像：6 频段能量占比(%) + 谱质心。只统计语音活跃帧。"""
    if isinstance(y_or_path, str):
        y, sr = load_audio(y_or_path)
    else:
        y = np.asarray(y_or_path, dtype=np.float32)
        if sr is None:
            sr = 22050
    y = np.asarray(y, dtype=np.float32)
    if len(y) < sr * 0.3:
        return {}
    fr = int(sr * 0.025)
    n = len(y) // fr
    rms = np.sqrt(np.mean(y[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    thr = float(np.percentile(rms, 70)) * 0.35
    act = rms > thr
    yy = (np.concatenate([y[i * fr:(i + 1) * fr]
                          for i in range(n) if act[i]])
          if act.any() else y)
    S = np.abs(_rfft_stft(yy)) ** 2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    e = S.sum(axis=1)
    tot = float(e.sum()) + 1e-12
    bands = [round(100.0 * float(e[(freqs >= lo) & (freqs < hi)].sum()) / tot, 2)
             for lo, hi in SPEC_BANDS]
    cen = float(np.sum(freqs * e) / tot)
    return {"bands": bands, "centroid": round(cen, 1)}


def _rfft_stft(y: np.ndarray, n_fft: int = 2048, hop: int = 512):
    frames = [y[i:i + n_fft] for i in range(0, max(1, len(y) - n_fft), hop)]
    if not frames:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    return np.stack([np.fft.rfft(f * win)[:n_fft // 2 + 1]
                     for f in frames], axis=1)


def match_band_profile(y: np.ndarray, sr: int, target: Dict[str, Any],
                       max_db: float = 5.0) -> np.ndarray:
    """把波形的 6 频段能量占比向 target 靠拢（STFT 域平滑增益曲线）。

    参考/条件的频谱包络会被克隆链忠实遗传（实测：闷参考→闷输出，
    300-1kHz 占比 47.5% → 输出 33.7%）。这里按「目标占比/当前占比」
    求 dB 增益（逐段钳 ±max_db、段间线性插值成平滑曲线），STFT 幅度
    相乘后 ISTFT —— 相位不变、无振铃。target 缺失时原样返回。
    """
    if not target or not target.get("bands"):
        return y
    cur = band_profile(y, sr)
    if not cur:
        return y
    tb, cb = target["bands"], cur["bands"]
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # 逐段增益(dB)：能量占比的比值 → 对数域
    seg_db = []
    for t, c, (lo, hi) in zip(tb, cb, SPEC_BANDS):
        ratio = max(t, 0.05) / max(c, 0.05)
        seg_db.append(float(np.clip(10.0 * np.log10(ratio), -max_db, max_db)))
    # 段中心频率上的控制点 → 全频段线性插值（dB 域平滑曲线）
    centers = np.array([(lo + hi) / 2.0 for lo, hi in SPEC_BANDS])
    centers[0] = max(80.0, centers[0])
    gain_db = np.interp(freqs, centers, seg_db)
    gain_db[freqs < 70] = gain_db[0]          # 低于首段不再外推
    gain_db[freqs > 11500] = gain_db[-1]      # Nyquist 之上持平
    g = 10.0 ** (gain_db / 20.0)
    # STFT 增益 + ISTFT（overlap-add, hop=n_fft//4）
    hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float32)
    if len(y) < n_fft:
        return y
    out = np.zeros(len(y) + n_fft, dtype=np.float32)
    wsum = np.zeros(len(y) + n_fft, dtype=np.float32)
    for i in range(0, len(y) - n_fft + 1, hop):
        seg = y[i:i + n_fft] * win
        spec = np.fft.rfft(seg)
        seg2 = np.fft.irfft(spec * g[:len(spec)], n=n_fft)
        out[i:i + n_fft] += seg2 * win
        wsum[i:i + n_fft] += win * win
    valid = wsum > 1e-6
    out[valid] /= wsum[valid]
    out = out[:len(y)]
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 10 ** (-1.0 / 20.0):
        out = out * (10 ** (-1.0 / 20.0) / peak)
    return out.astype(np.float32)


def dataset_band_profile(paths: List[str]) -> Dict[str, Any]:
    """一批素材的平均频谱画像（写进 dataset markers → run.json，
    推理端用它把音色参考的包络向角色本人素材靠拢）。"""
    profs = [p for p in (band_profile(x) for x in paths) if p]
    if len(profs) < 3:
        return {}
    bands = [round(float(np.mean([p["bands"][i] for p in profs])), 2)
             for i in range(len(SPEC_BANDS))]
    cen = round(float(np.mean([p["centroid"] for p in profs])), 1)
    return {"bands": bands, "centroid": cen, "n": len(profs)}


def smooth_loudness_sequence(targets: List[float],
                             max_step_db: float = 2.5,
                             anchor_db: Optional[float] = None) -> List[float]:
    """对逐块目标响度序列做**限步平滑**（2026-09-30 用户听感修复）。

    独立采样在宽分布角色上会造成相邻块 ±9dB 的跳变（听感「忽高忽低」，
    混 BGM 后像抽吸/空间感）。规则：
        · 首块钉在分布中位（anchor_db 或序列均值）；
        · 后续块的目标只能相对前一块移动 ≤max_step_db（超出即钳到边界）；
        · 保留采样出来的**相对形状**（爆发块仍最响），只削尖峰跳变。
    """
    if not targets:
        return targets
    base = float(anchor_db if anchor_db is not None
                 else sum(targets) / len(targets))
    out = [base]
    for t in targets[1:]:
        prev = out[-1]
        out.append(float(np.clip(t, prev - max_step_db, prev + max_step_db)))
    return out


# ---------------------------------------------------------------------------
# 词级重音 + F0 音域恢复（2026-09-30 · 超音段四件套的最后两件）
# ---------------------------------------------------------------------------

_PUNCT_CHARS = "。！？…；：，、,.!?;:-—"


def char_time_spans(text: str, duration: float,
                    silences: List[tuple]) -> List[tuple]:
    """文本每字符 → 音频时间区间 (start_s, end_s)。

    模型：相邻静音段之间的音频窗口 = 一个短语的语音；短语按标点切分后
    与窗口**按顺序**对应；短语内按字符数比例分配（中文音节近等时）。
    标点字符自身挂在短语尾部（含其后的静音）。

    静音窗口数与短语数不一致时（ASR/导演合并拆分）按比例贪心对齐：
    多余窗口并入相邻短语。兜底：无静音信息时整段均匀。
    返回与 text 等长的 [(start, end)]，全为秒。
    """
    n = len(text or "")
    if n == 0 or duration <= 0:
        return []
    runs = sorted([(float(a), float(a) + float(d))
                   for a, d in (silences or []) if 0.1 <= float(a)
                   and float(a) + float(d) <= duration])
    # 短语切分（标点粘在前段尾）
    phrases: List[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in _PUNCT_CHARS and len(cur) >= 1:
            phrases.append(cur)
            cur = ""
    if cur:
        phrases.append(cur)
    if not phrases:
        phrases = [text]

    # 窗口 = 相邻静音之间的语音区段
    windows: List[tuple] = []
    prev_end = 0.0
    for a, b in runs:
        if a - prev_end >= 0.15:
            windows.append((prev_end, a))
        prev_end = max(prev_end, b)
    if duration - prev_end >= 0.15:
        windows.append((prev_end, duration))
    if not windows:
        windows = [(0.0, duration)]

    # 短语 ↔ 窗口 贪心顺序对齐：短语总数 > 窗口数时，多个短语共享窗口
    # （按字符占比切窗口）；短语总数 < 窗口数时，一个窗口摊多个短语——
    # 统一按「字符占比 × 窗口总语音」分摊，保证覆盖全部窗口。
    char_spans: List[tuple] = []
    phrase_char_counts = [len(p) for p in phrases]
    total_chars = sum(phrase_char_counts)
    win_total = sum(b - a for a, b in windows)
    win_char_share = win_total / max(1, total_chars)   # 每字符摊的窗口秒数

    pi = 0                                             # 当前窗口索引
    pos_in_win = 0.0                                   # 当前窗口内已消费秒数
    for p in phrases:
        w_start, w_end = windows[min(pi, len(windows) - 1)]
        w_len = w_end - w_start
        p_len = len(p)
        p_share = p_len * win_char_share
        # 短语跨窗口：消费当前窗口剩余+后续窗口
        p_start = w_start + pos_in_win
        consumed = 0.0
        ci = 0
        for ch in p:
            span = win_char_share
            cs = w_start + pos_in_win + consumed
            ce = min(cs + span, w_end)
            char_spans.append((cs, max(ce, cs + 1e-4)))
            consumed += span
            if w_start + pos_in_win + consumed >= w_end - 1e-6:
                pi += 1
                if pi >= len(windows):
                    # 窗口用尽：剩余字符平摊最后窗口的时长
                    rest = (p_len - ci - 1) * win_char_share
                    last_s = char_spans[-1][1]
                    for _ in range(p_len - ci - 1):
                        char_spans.append((last_s, last_s + win_char_share))
                    break
                w_start, w_end = windows[pi]
                w_len = w_end - w_start
                pos_in_win = 0.0
                consumed = 0.0
            ci += 1
        pos_in_win += p_share
        if pos_in_win >= w_len - 1e-6 and pi < len(windows) - 1:
            pi += 1
            pos_in_win = 0.0
    # 补齐长度（防御）
    while len(char_spans) < n:
        last = char_spans[-1] if char_spans else (0.0, duration)
        char_spans.append(last)
    return [(round(a, 4), round(b, 4)) for a, b in char_spans[:n]]


def apply_stress_gains(y: np.ndarray, sr: int,
                       stress_spans: List[tuple],
                       gain_db: float = 2.0,
                       ramp_ms: float = 80.0) -> np.ndarray:
    """对重音词的音频区间施加局部增益（余弦爬坡，防台阶）。

    stress_spans: [(start_s, end_s)]（已按时间排序、去重叠）。
    gain_db ≤ 0 时原样返回。峰值保护 -1dBFS。
    int16 / float 波形均可（内部按 ±1 浮点尺度运算后还原 dtype）——
    峰值阈值 0.891 只对 ±1 尺度有意义；编排器传的是 int16，此前直接
    拿 ±32767 尺度的峰值与 0.891 比较，重音段被整体压到数字静音
    （2026-10-02「成品只剩一两个残缺声音」事故的根因）。
    """
    import numpy as _np
    g = 10 ** (float(gain_db) / 20.0)
    if g <= 1.001 or not stress_spans:
        return y
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    out = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
           else np.asarray(y, dtype=np.float32)).copy()
    ramp = max(1, int(sr * float(ramp_ms) / 1000.0))
    for a_s, b_s in stress_spans:
        a = max(0, int(a_s * sr) - ramp // 2)
        b = min(len(out), int(b_s * sr) + ramp // 2)
        if b - a < ramp * 2:
            continue
        seg = out[a:b]
        env = _np.ones(b - a, dtype=np.float32)
        env[:ramp] *= _np.linspace(1.0, g, ramp)
        env[-ramp:] *= _np.linspace(g, 1.0, ramp)
        env[ramp:-ramp] = g
        seg2 = seg * env
        peak = float(_np.max(_np.abs(seg2))) if len(seg2) else 0.0
        if peak > 10 ** (-1.0 / 20.0):
            seg2 *= (10 ** (-1.0 / 20.0) / peak)
        out[a:b] = seg2
    if was_int:
        return np.clip(out * 32767.0, -32767, 32767).astype(np.int16)
    return out.astype(np.asarray(y).dtype)


def f0_stats(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """一段语音的 F0 统计（半音域）：{median, std_st, voiced_ratio}。

    用 pyworld harvest（快、浊音判定稳）。std 在半音域（对音高尺度
    不变）—— 这是「音域宽窄」的正确度量。浊音不足 0.25s 返回 {}。
    """
    try:
        import pyworld as pw
        y64 = np.asarray(y, dtype=np.float64)
        if len(y64) < sr // 4:
            return {}
        # 超长输入截到前 30s：harvest 的内存随样本数线性膨胀，43.7M 采样
        # （990s 原件）实测直接 C 层 Fatal error: Aborted（2026-10-02 一键
        # 三连事故）。指纹统计要的是分布，30s 已是数万帧，绰绰有余。
        _cap = int(sr) * 30
        if len(y64) > _cap:
            y64 = y64[:_cap]
        f0, _t = pw.harvest(y64, sr, f0_floor=70.0, f0_ceil=600.0)
        voiced = f0[f0 > 0]
        if len(voiced) < max(20, sr // 800):
            return {}
        med = float(np.median(voiced))
        semis = 12.0 * np.log2(voiced / med)
        return {
            "median": round(med, 1),
            "std_st": round(float(np.std(semis)), 2),
            "voiced_ratio": round(float(len(voiced) / len(f0)), 3),
        }
    except Exception:
        return {}


# 逗号类标点（块内停顿「保底位」的触发记号）。句号/问号/叹号不在此列：
# 块内强标点仍走既有配额治理（已被 21~27 轮真机验收的行为，不扩权）。
_COMMA_MARKS = "，,、；;：:"

# 标点保底位认定的最小静音段（名义值）。帧化会吞掉静音两端各约半个
# 帧窗：物理 40ms 的停顿只检出 10~20ms——名义阈值必须低于目标下限留
# 足余量。2026-10-03 两次收紧 50→20ms：真机实锤模型可以把逗号实现成
# 仅 40~60ms 的「闪缝」（爱弥斯 LoRA，第一逗号检出 20~40ms）——收集
# 不到就匹配不上，保底位被 0.2s 外的词组间隙冒领（冒领间隙被保留在带
# 内、真逗号处依旧急促）。20ms 名义（2 帧）≈ 能抓住 ≥35ms 的实际停顿；
# 低于它的静音在帧域已无法与语音内起伏区分，不认定为「逗号已实现」
# ——找不到就跳过该标点，绝不切语音。
_PUNCT_RUN_MS = 20.0

# 标点位置 ↔ 静音段的匹配容差（秒）。按语音时长比例估计标点时刻，误差
# 主要来自音节时长不均；实测（27 字块）估计误差 <0.05s。0.6→0.35 收紧
# （2026-10-03）：过宽的窗口会让真逗号的**下一个词组间隙**（0.2~0.3s
# 外）冒领保底位——真逗号闪缝太短没被收集时尤其如此。
_PUNCT_MATCH_WIN_S = 0.35


def _punct_mark_times(text: str, quiet: np.ndarray,
                      hop_s: float) -> List[float]:
    """块文本中每个逗号类标点的**估计时刻**（秒）。

    模型与 char_time_spans 同一假设：语音时长 ∝ 非标点字符数、标点自身
    不占语音时间。第 k 个标点前有 p_k 比例的语音字符 → 估计时刻落在
    累计语音时长的 p_k 处（在实测帧的浊音分布上查表，静音多的区域自动
    摊薄字符速度，比「时长 × 字符占比」的线性估计准）。
    """
    marks = [i for i, ch in enumerate(text or "") if ch in _COMMA_MARKS]
    if not marks:
        return []
    n_frames = len(quiet)
    if not n_frames:
        return []
    is_speech_char = [ch not in _PUNCT_CHARS for ch in text]
    total_chars = sum(is_speech_char)
    voiced_cum = np.cumsum((~quiet).astype(np.int64))
    total_voiced = int(voiced_cum[-1]) if n_frames else 0
    out: List[float] = []
    for k, m in enumerate(marks):
        if total_chars <= 0 or total_voiced <= 0:
            # 退化输入（全标点/全静音）：按标点序号均摊，宁可粗不能崩
            out.append((k + 1) / (len(marks) + 1) * n_frames * hop_s)
            continue
        chars_before = sum(is_speech_char[:m])
        target_v = chars_before / total_chars * total_voiced
        f = int(np.searchsorted(voiced_cum, max(0, target_v - 0.5),
                                side="left"))
        out.append(min(max(f, 0), n_frames - 1) * hop_s)
    return out


def pause_quota(n_chars: int) -> int:
    """块内允许保留的「明显停顿」名额（产品规范，2026-10-02 用户原话：
    「每一小句内一般甚至都不需要明显断句，只有长句子可能偶然需要一处
    断句」）。

    ≤12 字（短句）：0 个 —— 词间隙以外的停顿全部压到 60ms 语音学水平；
    13~27 字：1 个；28~40 字：2 个（块文本上限就是 40 字）。
    """
    if n_chars <= 12:
        return 0
    if n_chars <= 27:
        return 1
    return 2


def normalize_intra_pauses(y: np.ndarray, sr: int, text_chars: int,
                           cap_ms: float = 220.0,
                           floor_ms: float = 35.0,
                           min_run_ms: float = 150.0,
                           thresh_db: float = -30.0,
                           text: str = "",
                           punct_floor_ms: float = 0.0,
                           punct_cap_ms: float = 0.0,
                           audit_out: Optional[List[Dict[str, Any]]] = None
                           ) -> np.ndarray:
    """块内停顿「门槛-配额-时长」三治理（2026-10-02 第四轮：修去气口误伤）。

    **去气口版本的教训（真机实锤）**：上一版把所有 ≥25ms 的静音段整段
    移除——但 25~100ms 的静音是**塞音闭合**（p/t/k/d/g 闭合期）与自然
    词间隙，是语音结构的一部分（实测每块 10+ 个）。移除它们 = 字的辅音
    被削、词间隙清零 = 「每个字都是独立合成的」。修复：

    · **min_run_ms 门槛**：低于它的静音段完全不碰（保护正常语音结构，
      文献：塞音闭合 30~100ms、词间隙 50~150ms）；
    · **配额**（pause_quota）：配额内的停顿压到 cap_ms（保留的断句）；
    · **配额外的**压到 floor_ms=35ms（防粘连微气口，不归零也不留大洞，
      中段截 + 5ms 切口淡化；调型收束无法在音频域消除，留微隙缓冲）。

    **第五轮（2026-10-02 逗号保底位）**：参考摊平（>120ms 封顶）治好
    词组化散感的同时，把模型在逗号处的停顿也压到了 75~150ms——低于
    门槛原样放行、≥150ms 又会被 ≤12 字块的配额 0 压成 35ms 微气口，
    听感「逗号几乎没有停顿」。text 非空且 punct_floor_ms>0 时启用标点
    感知：块文本里每个逗号类标点（，、；：）按位置匹配到一段已实现的
    静音，**保证它落在 [punct_floor_ms, punct_cap_ms] 带内**——短了在
    静音中段补插（语音一个采样点不动），长了截到带内。这些段**不占**
    配额名额、也不受 cap_ms 约束（标点是有意的停顿，封顶参数管的是无
    标点处的幻觉停顿）。找不到已实现静音（≥50ms 名义）的标点直接跳过，
    **绝不向语音里插静音**；首尾静音（拼接层的辖区）不参与匹配。无标点
    路径与旧版逐位一致。

    背景：参考 28 停顿/分钟 + AR 放大 → 引擎输出 44~53 个/分钟。
    int16/float 均可，返回保持原 dtype。

    **审计接口（2026-10-03 追踪件）**：audit_out 传一个 list 进来，函数
    把每个静音段的处置（动作+原因码）和每个逗号类标点的匹配结果 append
    进去，供旁车落盘。事后归因（「这个停顿为什么还在/为什么没了」）从
    审计记录直接读，不再靠推测。原因码：
        run   edge_owned_by_splice（首尾静音，拼接层辖区）
              below_gate（检出 < 治理门槛，保护塞音闭合/词间隙）
              mandated_in_band / mandated_insert / mandated_trim（标点位）
              quota_keep_in_band / quota_keep_trim / floor_trim（配额制）
              gap_ceiling_trim（词组间隙越过逗号天花板，压回天花板）
        punct matched_run=<下标> / no_realized_silence（无可匹配静音，跳过）
    """
    if not cap_ms or float(cap_ms) <= 0 or len(y) < sr // 4:
        return y
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    quota = pause_quota(int(text_chars))
    punct_mode = bool(text) and punct_floor_ms > 0 and punct_cap_ms > 0

    db, n = _frames_db(yf, sr, frame_ms=25.0, hop_ms=10.0)
    if not n:
        return y
    hop = int(sr * 0.01)
    quiet = db < thresh_db
    edge = int(0.05 * sr)
    lo_frame = edge // hop
    hi_frame = max(lo_frame, (len(yf) - edge) // hop)
    min_run = int(sr * float(min_run_ms) / 1000.0)

    # 静音段收集门槛（2026-10-03 第五轮补丁）：检出时长比物理静音低估
    # 约 25~30ms（帧窗跨界的帧不算静音）——物理 150~176ms 的停顿只检出
    # 140~150ms，卡在样本级 150ms 门槛外逃逸（真机块1「因为|这样」170ms
    # 停顿的漏网路径）。收集与治理统一用 min_run - 3帧 容差把边界移回
    # 物理域 ≈150ms；塞音闭合（≤100ms 物理 → 检出 ≤80ms）与词间隙
    # （≤130ms 物理 → 检出 ≤110ms）仍远低于门槛，保护语义不变。
    # 标点模式另收 ≥5 帧（≈名义 50ms）的短段供逗号保底位匹配。
    punct_min_frames = max(1, int(np.ceil(_PUNCT_RUN_MS / 10.0)))
    collect_min = max(0, min_run - 3 * hop)
    if punct_mode:
        collect_min = min(collect_min, punct_min_frames * hop)
    runs: List[tuple] = []
    i = lo_frame
    while i < hi_frame:
        if quiet[i]:
            j = i
            while j < hi_frame and quiet[j]:
                j += 1
            a, b = i * hop, min(len(yf), j * hop)
            if b - a >= collect_min:
                runs.append((a, b))
            i = j
        else:
            i += 1
    if not runs:
        # 早退也要留审计痕迹（追踪件）：「什么都没做」的原因本身就是归因
        # 要的答案——逗号逐个记 no_realized_silence，再记一条总结性 note。
        if audit_out is not None:
            if punct_mode:
                for t in _punct_mark_times(text, quiet, hop / sr):
                    audit_out.append({
                        "kind": "punct", "est_s": round(t, 3),
                        "matched_run": None, "matched_run_ms": None,
                        "action": "no_realized_silence",
                    })
            audit_out.append({"kind": "note", "action": "no_collected_runs"})
        return y

    # ---- 标点保底位：逗号类标点 ↔ 已实现静音段（按位置贪心匹配）----
    mandated: Dict[int, tuple] = {}          # run 下标 -> (floor_n, cap_n)
    punct_audit: List[Dict[str, Any]] = []
    # 逗号带上限（样本域）：标点模式下它同时是**块内一切静音的天花板**
    #（2026-10-03 层级锚定）：块内皆句内，句内停顿不过逗号级——配额拍与
    # 词组间隙都压在它之下，「词间≤逗号≤句号」在任何 pause_scale 下成立。
    p_cap_n = int(sr * float(punct_cap_ms or 0) / 1000.0) if punct_mode else 0
    if punct_mode:
        hop_s = hop / sr
        # 首尾静音段（连到分析边界）是拼接层的辖区，不参与匹配
        interior = [k for k, (a, b) in enumerate(runs)
                    if a > lo_frame * hop and b < hi_frame * hop]
        used = set()
        est_t = _punct_mark_times(text, quiet, hop_s)
        win = int(_PUNCT_MATCH_WIN_S * sr)
        p_floor = int(sr * float(punct_floor_ms) / 1000.0)
        p_cap = max(p_floor, int(sr * float(punct_cap_ms) / 1000.0))
        for t in est_t:
            best, best_d = -1, None
            for k in interior:
                if k in used:
                    continue
                a, b = runs[k]
                if abs((a + b) / 2 - t * sr) <= win:
                    d = abs((a + b) / 2 - t * sr)
                    if best_d is None or d < best_d:
                        best, best_d = k, d
            if best >= 0:
                used.add(best)
                mandated[best] = (p_floor, p_cap)
            punct_audit.append({
                "kind": "punct", "est_s": round(t, 3),
                "matched_run": best if best >= 0 else None,
                "matched_run_ms": (round((runs[best][1] - runs[best][0])
                                         / sr * 1000) if best >= 0 else None),
                "action": ("matched_run" if best >= 0
                           else "no_realized_silence"),
            })

    # ---- 配额治理（非标点段；治理资格 = 检出 ≥ min_run - 3帧）----
    # 收集门槛更低（供逗号保底位匹配短停顿），但配额治理只碰「够格的
    # 停顿」——塞音闭合/词间隙（检出 <120ms）收集进来也不治理。
    gov_min = max(0, min_run - 3 * hop)
    gov = [k for k in range(len(runs))
           if k not in mandated and runs[k][1] - runs[k][0] >= gov_min]
    keep = set(sorted(gov, key=lambda k: -(runs[k][1] - runs[k][0]))
               [:quota]) if quota else set()

    # ---- 统一成「编辑清单」再从后往前施工（前面区间的偏移不受影响）----
    # edit = (a, b, target_n)：target < 段长 → 中段截；target > 段长 →
    # 静音中段补插。标点段目标来自停顿带，配额段目标来自 cap/floor。
    edits: List[tuple] = []
    for k, (a, b) in enumerate(runs):
        if k in mandated:
            if b - a < mandated[k][0]:
                edits.append((a, b, mandated[k][0]))          # 补插到带下限
            elif b - a > mandated[k][1]:
                edits.append((a, b, mandated[k][1]))          # 截到带上限
        elif k in keep:
            if b - a > int(sr * float(cap_ms) / 1000.0):
                edits.append((a, b, int(sr * float(cap_ms) / 1000.0)))
        elif k in gov:
            if b - a > int(sr * float(floor_ms) / 1000.0):
                edits.append((a, b, int(sr * float(floor_ms) / 1000.0)))
        elif punct_mode and p_cap_n and b - a > p_cap_n:
            # 低于治理门槛的词组间隙越过逗号天花板 → 压回天花板（中段截，
            # 与历版同一把刀；塞音闭合检出 ≤80ms，永远低于 84ms 级天花板，
            # 不受影响）。scale=1.0 时天花板 210ms > 词间隙检出上限 120ms，
            # 此分支天然不触发——默认行为零变化。
            edits.append((a, b, p_cap_n))

    # ---- 审计输出：每段静音的处置与原因（旁车落盘用）----
    if audit_out is not None:
        edit_at = {(a, b): t for a, b, t in edits}
        for k, (a, b) in enumerate(runs):
            det_ms = round((b - a) / sr * 1000)
            if a <= lo_frame * hop or b >= hi_frame * hop:
                action = "edge_owned_by_splice"
            elif k in mandated:
                action = ("mandated_insert" if (a, b) in edit_at
                          and edit_at[(a, b)] > b - a else
                          "mandated_trim" if (a, b) in edit_at else
                          "mandated_in_band")
            elif k in keep:
                action = ("quota_keep_trim" if (a, b) in edit_at
                          else "quota_keep_in_band")
            elif k in gov:
                action = "floor_trim" if (a, b) in edit_at else "untouched"
            elif punct_mode and (a, b) in edit_at:
                action = "gap_ceiling_trim"   # 词组间隙压到逗号天花板
            else:
                action = "below_gate"        # 收集到但不够格治理（受保护）
            audit_out.append({
                "kind": "run", "start_s": round(a / sr, 3),
                "detected_ms": det_ms, "action": action,
            })
        audit_out.extend(punct_audit)

    fade = max(1, int(0.005 * sr))
    out = yf
    for a, b, target in sorted(edits, key=lambda e: -e[0]):
        cur = b - a
        if cur > target:
            # 中段截法（与历版一致）：切口在静音深处，两侧 5ms 淡化防毛刺
            mid_a = a + target // 2
            mid_b = b - (target - target // 2)
            out = np.concatenate([out[:mid_a], out[mid_b:]])
            if len(out) > mid_a + fade:
                out[mid_a:mid_a + fade] *= np.linspace(
                    1.0, 0.0, fade, dtype=np.float32)
            if mid_a - fade >= 0:
                out[mid_a - fade:mid_a] *= np.linspace(
                    0.0, 1.0, fade, dtype=np.float32)
        else:
            # 静音中段补插（只加静音不切语音）：切口两侧 5ms 淡出/淡入，
            # 插入体是数字零——两侧本来就是 ≤-30dB 的静音，听感无跃变
            pad = target - cur
            mid = a + cur // 2
            out = np.concatenate([out[:mid], np.zeros(pad, np.float32),
                                  out[mid:]])
            if mid - fade >= 0:
                out[mid - fade:mid] *= np.linspace(
                    1.0, 0.0, fade, dtype=np.float32)
            if len(out) > mid + pad + fade:
                out[mid + pad:mid + pad + fade] *= np.linspace(
                    0.0, 1.0, fade, dtype=np.float32)

    if was_int:
        return np.clip(out * 32767.0, -32767, 32767).astype(np.int16)
    return out.astype(yf.dtype, copy=False)



def suppress_wet_hf(y: np.ndarray, sr: int,
                    lo_hz: float = 5500.0,
                    voice_hi_hz: float = 3000.0,
                    gap_ratio: float = 0.35,
                    loud_ratio: float = 0.45,
                    max_cut_db: float = 15.0,
                    min_frames: int = 4) -> tuple:
    """抑制音节间隙里的「痰状」高频爆冲（2026-10-10 锁暝取证根治件）。

    实体（真机取证）：6~11kHz 全带平坦、快幅度调制（~600-1500Hz，碎裂
    「痰」感）、电平仅比语音低 3~8dB 的噪声簇，发生在音节间隙——游戏 VO
    源素材（MP3 转码 + 强气声表演）自带，零样本克隆与 LoRA 都忠实复现
    （差分实验：底座/仅GPT/仅CFM/双LoRA 四配置全部出现）。

    判据（三条件同时满足才动，保守）：
      · 语音带（0.3~3kHz）低于全文件语音参考的 gap_ratio（间隙/气口）；
      · 高频带（lo~11kHz）高于语音参考的 loud_ratio（异常响）；
      · 高频带内三子带平坦（平坦度 <0.55；自然 /s/ 擦音有形态峰，方差大；
        浊音区的平坦度分布与爆冲重叠——实测 p50 0.31 vs 0.26——所以
        **只在间隙启用**，浊音区即使响而平也不碰，音色零风险）。
    施工：带通分离出 lo~11kHz 分量，对爆冲段施加限幅增益（≥max_cut_db
    下限，10ms 攻/40ms 放平滑），y_out = y - hf*(1-gain)——只动高频带，
    语音与音色零接触；非爆冲段逐位不变。
    int16/float 均可，返回 (波形, 处理的簇数)。
    """
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    if len(yf) < sr // 2:
        return y, 0
    from scipy.signal import butter, sosfiltfilt, sosfilt
    hi = min(sr / 2 * 0.98, 11000.0)
    if lo_hz >= hi:
        return y, 0
    sos_hf = butter(4, [lo_hz / (sr / 2), hi / (sr / 2)],
                    btype="band", output="sos")
    sos_vb = butter(4, [300.0 / (sr / 2), voice_hi_hz / (sr / 2)],
                    btype="band", output="sos")
    hf = sosfiltfilt(sos_hf, yf)
    vb = sosfiltfilt(sos_vb, yf)
    fr = int(sr * 0.01)
    n = len(yf) // fr
    if n < 20:
        return y, 0
    e_hf = np.sqrt(np.mean(hf[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    e_vb = np.sqrt(np.mean(vb[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    speech_ref = float(np.percentile(e_vb, 60)) + 1e-9
    # 三子带平坦度（lo~mid~hi 等宽）
    import numpy as _np
    edges = _np.linspace(lo_hz, hi, 4)
    sub = []
    for a, b in zip(edges[:-1], edges[1:]):
        sos = butter(4, [a / (sr / 2), b / (sr / 2)],
                     btype="band", output="sos")
        s = sosfiltfilt(sos, yf)
        sub.append(np.sqrt(np.mean(s[:n * fr].reshape(n, fr) ** 2,
                                    axis=1) + 1e-12))
    sub = _np.stack(sub)
    flat = sub.std(axis=0) / (sub.mean(axis=0) + 1e-9)

    wet = ((e_vb < speech_ref * gap_ratio) &
           (e_hf > speech_ref * loud_ratio) &
           (flat < 0.55))
    # 簇化：≥min_frames 连续才算（防孤立毛刺）
    clusters, i = [], 0
    while i < n:
        if wet[i]:
            j = i
            while j < n and wet[j]:
                j += 1
            if j - i >= min_frames:
                clusters.append((i, j))
            i = j
        else:
            i += 1
    if not clusters:
        return y, 0
    # 增益包络：爆冲段把 HF 压到 speech_ref*loud_ratio（限幅 max_cut_db）
    floor = speech_ref * loud_ratio * (10 ** (-max_cut_db / 20.0))
    gain = _np.ones(n, dtype=np.float32)
    for a, b in clusters:
        seg = e_hf[a:b]
        want = _np.clip(speech_ref * loud_ratio / (seg + 1e-9), 0.0, 1.0)
        want = _np.maximum(want, floor / (seg + 1e-9))
        gain[a:b] = _np.minimum(gain[a:b], want)
    # 时域平滑（de-esser 惯例：压入快 10ms / 释放慢 40ms——短簇也来得及
    # 压到位，簇结束不产生台阶）
    atk = max(1, int(0.010 * sr) // fr)      # 增益下降（压入）
    rel = max(1, int(0.040 * sr) // fr)      # 增益回升（释放）
    sm = gain.copy()
    for k in range(1, n):
        if sm[k] < sm[k - 1]:
            sm[k] = max(sm[k], sm[k - 1] - 1.0 / atk)
        else:
            sm[k] = min(sm[k], sm[k - 1] + 1.0 / rel)
    gain_s = _np.interp(_np.arange(len(yf)) / fr, _np.arange(n), sm)
    out = yf - hf * (1.0 - gain_s)
    if was_int:
        return (_np.clip(out * 32767.0, -32767, 32767).astype(np.int16),
                len(clusters))
    return out.astype(np.float32), len(clusters)



def limit_voiced_hf(y: np.ndarray, sr: int,
                    ceiling_db: float = 12.0,
                    max_cut_db: float = 10.0,
                    min_frames: int = 2) -> tuple:
    """浊音区高频「撕裂」限幅器（2026-10-11 锁暝系统撕裂取证根治件）。

    用户实锤：不是个别间隙爆冲，而是**每个稍高音高的浊音段**都有高频
    撕裂感——实体=浊音帧内 5.5~11kHz 带能量超过语音带(0.3-3kHz) 12dB
    以上的帧（真机测量：锁暝合成 2.6~10% 的浊音帧超限，被用户接受的
    达妮娅基线仅 0.3~0.8%；真人参考本身也有 2.4%——角色演法+克隆放大
    的叠加）。 ceiling 以已验收基线校准。

    动作：超限帧把 HF 带压回天花板（限幅 max_cut_db，快压 10ms / 慢放
    40ms，同 suppress_wet_hf 的 de-esser 包络）；正常齿音（+8~12dB 内）
    与浊音谐波零接触。与 suppress_wet_hf（间隙爆冲）互补：一个管间隙、
    一个管浊音，合起来覆盖「痰+撕裂」两类高频过量。
    int16/float 均可，返回 (波形, 处理帧数)。
    """
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    if len(yf) < sr // 2:
        return y, 0
    from scipy.signal import butter, sosfiltfilt
    hi = min(sr / 2 * 0.98, 11000.0)
    sos_hf = butter(4, [5500.0 / (sr / 2), hi / (sr / 2)],
                    btype="band", output="sos")
    sos_vb = butter(4, [300.0 / (sr / 2), 3000.0 / (sr / 2)],
                    btype="band", output="sos")
    hf = sosfiltfilt(sos_hf, yf)
    vb = sosfiltfilt(sos_vb, yf)
    fr = int(sr * 0.01)
    n = len(yf) // fr
    if n < 20:
        return y, 0
    e_hf = np.sqrt(np.mean(hf[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    e_vb = np.sqrt(np.mean(vb[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    voiced = e_vb > np.percentile(e_vb, 40)
    ratio = e_hf / (e_vb + 1e-9)
    ceiling = 10 ** (float(ceiling_db) / 20.0)
    over = voiced & (ratio > ceiling)
    # 簇化：≥min_frames 连续（防孤立帧毛刺）
    clusters, i = [], 0
    while i < n:
        if over[i]:
            j = i
            while j < n and over[j]:
                j += 1
            if j - i >= min_frames:
                clusters.append((i, j))
            i = j
        else:
            i += 1
    if not clusters:
        return y, 0
    max_cut = 10 ** (-float(max_cut_db) / 20.0)
    gain = np.ones(n, dtype=np.float32)
    for a, b in clusters:
        want = np.clip(ceiling * e_vb[a:b] / (e_hf[a:b] + 1e-9),
                       max_cut, 1.0)
        gain[a:b] = np.minimum(gain[a:b], want)
    atk = max(1, int(0.010 * sr) // fr)
    rel = max(1, int(0.040 * sr) // fr)
    sm = gain.copy()
    for k in range(1, n):
        if sm[k] < sm[k - 1]:
            sm[k] = max(sm[k], sm[k - 1] - 1.0 / atk)
        else:
            sm[k] = min(sm[k], sm[k - 1] + 1.0 / rel)
    gain_s = np.interp(np.arange(len(yf)) / fr, np.arange(n), sm)
    out = yf - hf * (1.0 - gain_s)
    if was_int:
        return (np.clip(out * 32767.0, -32767, 32767).astype(np.int16),
                len(clusters))
    return out.astype(np.float32), len(clusters)



# ===========================================================================
# 语调（intonation）三层控制：执行器 / 自动规则 / 手动标记
# 2026-10-11 · 用户核心诉求：问句末尾上翘、随情绪与语言的自然起伏
# 调研对齐：Into-TTS 模板法、Corkey 2023 语调控制、W3C SSML prosody、
# Morrison C-DAR 轮廓编辑——全部落在本项目已有的 WORLD 分析/重合成设施上。
# ===========================================================================

#: 语调偏移的绝对安全边界（半音）。阶段20教训：女声谐波稀疏，大偏移让
#: 谐波与谱包络错位出痰音——±3st 与 restore_f0_range 的逐帧限幅同源。
INTONATION_MAX_ST = 3.0


def edit_f0_contour(y: np.ndarray, sr: int,
                    ramps: List[tuple],
                    max_st: float = INTONATION_MAX_ST) -> Tuple[np.ndarray, int]:
    """F0 轮廓编辑器：按「半音渐变段」重塑浊音区的音高，谱包络不动。

    ramps: [(t_start_s, t_end_s, st_start, st_end)]——段内偏移从 st_start
    线性渐变到 st_end（半音域，正=升调）；段外偏移 0。多段拼接成整条
    偏移曲线。施工：harvest 提取 F0 → 浊音帧乘 2^(offset/12) →
    cheaptrick/d4c 用**原始谱包络**重合成（音色不变，只动音高）。

    安全边界（2026-10-11 两起事故后定稿）：
      · 逐帧偏移钳 ±max_st（默认 ±3 半音，防痰音）；
      · **区域限定**：ramp 区间（±50ms 余量）之外的采样**逐位保真**——
        整块 WORLD 重合成会把全句擦音/气声/瞬态重渲染（真机实测整句
        「高音破碎」，WORLD 的 cheaptrick/d4c 无法复原未建模的高频细节）；
      · **浊音强度门控**：harvest 会在气声/底噪里误检出 F0（真机实测：
        ramp 覆盖到句末静音时，间隙被乘上上翘=长出音高化爆冲，6-11kHz
        与邻区差 64dB）——只有 d4c 低周期性的强浊音帧才采用重合成，
        其余帧用原样本，软掩码 10ms 压入 / 40ms 释放消接缝；
      · 长于 30s 的输入直接跳过（WORLD 内存安全界）。
    int16/float 均可，返回 (波形, 应用的段数)。
    """
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    ramps = [(float(a), float(b), float(x), float(v))
             for a, b, x, v in (ramps or [])
             if float(b) > float(a)]
    if not ramps or len(yf) < sr // 2:
        return y, 0
    if len(yf) > sr * 30:
        return y, -1                          # 超长：调用方自行提示走导演模式
    try:
        import pyworld as pw
    except Exception:
        return y, 0
    y64 = yf.astype(np.float64)
    f0, t = pw.harvest(y64, sr, f0_floor=70.0, f0_ceil=600.0,
                       frame_period=5.0)
    voiced = f0 > 0
    if int(voiced.sum()) < 20:
        return y, 0
    tf = t[:len(f0)]
    # 逐帧偏移曲线：段内线性渐变，段间保持
    off = np.zeros(len(f0), dtype=np.float64)
    for a, b, x0, x1 in ramps:
        m = (tf >= a) & (tf <= b)
        if not m.any():
            continue
        span = max(1e-6, b - a)
        frac = np.clip((tf[m] - a) / span, 0.0, 1.0)
        off[m] = x0 + (x1 - x0) * frac
    off = np.clip(off, -float(max_st), float(max_st))
    if float(np.abs(off[voiced]).max()) < 0.05:
        return y, 0                           # 有效偏移过小：不值得重合成
    new_f0 = f0.copy()
    new_f0[voiced] = f0[voiced] * (2.0 ** (off[voiced] / 12.0))
    sp = pw.cheaptrick(y64, f0, t, sr)
    ap = pw.d4c(y64, f0, t, sr)
    y2 = np.asarray(pw.synthesize(new_f0, sp, ap, sr, frame_period=5.0),
                    dtype=np.float32)
    y2 = y2[:len(yf)]
    if len(y2) < len(yf):
        y2 = np.concatenate([y2, np.zeros(len(yf) - len(y2), np.float32)])

    # ---- 软掩码：区域限定 × 浊音强度门控（事故防线，见 docstring）----
    ff_ap = np.linspace(0, sr / 2, ap.shape[1])
    apb = np.mean(ap[:, (ff_ap > 300) & (ff_ap < 4000)], axis=1)
    strong = voiced & (apb < 0.55)            # 强浊音：气声/底噪误检被排除
    region = np.zeros(len(f0), dtype=bool)
    margin = 0.05
    for a, b, _x0, _x1 in ramps:
        region |= (tf >= max(0.0, a - margin)) & (tf <= b)
    mask_f = (strong & region & (np.abs(off) > 0.05)).astype(np.float32)
    # 帧域平滑（压入 1 帧=5ms，释放 8 帧=40ms），再插值到样本域
    sm = mask_f.copy()
    for k in range(1, len(sm)):
        sm[k] = max(sm[k], sm[k - 1] * 0.92)   # 释放 ~40ms
    mask_s = np.interp(np.arange(len(yf)) / sr, tf, sm).astype(np.float32)
    out = yf + mask_s * (y2 - yf)

    n_applied = int(len(ramps))
    if was_int:
        return (np.clip(out * 32767.0, -32767, 32767).astype(np.int16),
                n_applied)
    return out.astype(np.float32), n_applied


def build_intonation_ramps(text: str, sr: int, duration: float,
                           silences: List[tuple],
                           profile: Optional[Dict[str, Any]] = None
                           ) -> Tuple[List[tuple], str]:
    """自动语调规则（层①）：标点驱动的问句上翘/感叹强调 ramps。

    规则（v1，逐短语）：
      · `？`结尾：短语尾部 ~35%（末2-3音节）从 0 渐变到 rise_st——幅度
        优先取**语调指纹**（该演员自己的问句实测分布，N≥3 才用），
        否则通用默认 +2.5st；
      · `！`结尾：末 1~2 音节 +1.5st 峰值强调；
      · `。`：v1 不动（模型自带的陈述下落已存在，强加反而做作）；
      · 非句末短语（，、）不动——短语中途的语调交给模型。
    ramps 坐标基于 char_time_spans 的字符时间映射（静音感知）。
    返回 (ramps, 说明)。
    """
    n = len(text or "")
    if n == 0 or duration <= 0:
        return [], ""
    prof = profile or {}
    qr = prof.get("question_rise") or {}
    rise_st = 1.8   # 上翘会整体搬高谐波（+1st≈+6%频率），在嘶哑边缘的
                    # 嗓音上 +2.5st 实测越过撕裂线——1.8st 仍清晰可闻且安全
    # 指纹语义（2026-10-11 实测校准）：问句**必须升**是语言规范——指纹只能
    # 在"上翘体制内"调节幅度（演员明显上翘 median>1st 才采纳其值），不能
    # 把方向翻转成不翘/下降。实测锁暝素材问句中位 -0.76st（游戏VO疲惫
    # 演法）——直接采纳会让问句依然不翘，违背用户核心诉求。
    if (int(qr.get("n") or 0) >= 3 and qr.get("median_st") is not None
            and float(qr["median_st"]) > 1.0):
        # 演员明显上翘：采纳其实测幅度（撕裂线约束 ≤2.2st——上翘整体搬高
        # 谐波，+2.5st 实测在嘶哑边缘嗓音上越线）
        try:
            rise_st = float(np.clip(float(qr["median_st"]), 1.0, 2.2))
        except (TypeError, ValueError):
            pass
    # 字符 → 时间
    spans = char_time_spans(text, duration, silences)
    if len(spans) < n:
        return [], ""
    # 短语切分（标点粘在前短语尾）
    phrases: List[tuple] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _PUNCT_CHARS:
            phrases.append((start, i, ch))
            start = i + 1
    if start < n:
        phrases.append((start, n - 1, ""))
    ramps: List[tuple] = []
    applied: List[str] = []
    for p0, p1, punct in phrases:
        if punct not in ("？", "!"):
            continue
        tail_chars = max(1, min(3, (p1 - p0) // 3))
        c0 = max(p0, p1 - tail_chars)
        t0 = spans[c0][0]
        t1 = spans[p1][1]
        if punct == "？":
            st = rise_st
        else:
            st = 1.5
        if t1 - t0 < 0.08:
            continue
        ramps.append((t0, t1, 0.0, st))
        applied.append(f"{punct}@{t0:.2f}s+{st:.1f}st")
    return ramps, "; ".join(applied)


_MARKUP_RE = None


def parse_intonation_markup(text: str) -> Tuple[str, List[tuple]]:
    """手动标记解析（层②）：从文本剥离 [+2.5st]/[-1st]/[升]/[降] 标记，
    返回 (净文本, [(净文本字符下标, st)])。

    语义（控制点模型）：标记出现在哪个字前面，就定义该字位置上的 F0
    偏移；相邻控制点之间线性渐变，最后一个控制点之后保持到短语尾。
    示例：`今天[+1st]去哪[+3st]儿？` → 「去哪儿」一路爬升到 +3st。
    [升] = +2st、[降] = -2st 的中文糖。标记不进合成文本（否则会被读出）。
    """
    global _MARKUP_RE
    if _MARKUP_RE is None:
        import re as _re
        _MARKUP_RE = _re.compile(r"\[([+-]?\d+(?:\.\d+)?st|升|降)\]",
                                 _re.IGNORECASE)
    edits: List[tuple] = []
    out: List[str] = []
    pos = 0
    for m in _MARKUP_RE.finditer(text or ""):
        out.append(text[pos:m.start()])
        pos = m.end()
        tok = m.group(1).lower()
        if tok == "升":
            st = 2.0
        elif tok == "降":
            st = -2.0
        else:
            try:
                st = float(tok[:-2])
            except ValueError:
                continue
        st = float(np.clip(st, -INTONATION_MAX_ST, INTONATION_MAX_ST))
        edits.append((len("".join(out)), st))
    out.append(text[pos:])
    clean = "".join(out)
    return clean, edits


def markup_edits_to_ramps(edits: List[tuple],
                          spans: List[tuple],
                          phrase_end_s: float) -> List[tuple]:
    """标记控制点 → ramps：控制点间线性渐变，末点保持到短语尾。"""
    pts = [(spans[min(idx, len(spans) - 1)][0], st)
           for idx, st in edits if idx < len(spans)] if spans else []
    if not pts:
        return []
    ramps: List[tuple] = []
    if pts[0][1] != 0.0:
        ramps.append((max(0.0, pts[0][0] - 0.25), pts[0][0], 0.0, pts[0][1]))
    for (ta, sa), (tb, sb) in zip(pts[:-1], pts[1:]):
        if sb != sa and tb > ta:
            ramps.append((ta, tb, sa, sb))
    last_t, last_s = pts[-1]
    if phrase_end_s > last_t and last_s != 0.0:
        ramps.append((last_t, phrase_end_s, last_s, last_s))
    return ramps



def analyze_intonation_clip(y: np.ndarray, sr: int,
                            text: str) -> List[Dict[str, Any]]:
    """单条素材的逐短语语调测量（训练时语调指纹的原料）。

    按标点切短语 → char_time_spans 对齐 → 每个短语测「末音节窗口」的
    F0 首尾变化（半音）：问句（？）记 rise，句号记 fall。f0=0 的无声
    帧自动跳过。返回 [{punct, rise_st}, ...]（每短语一条）。
    """
    if not text or len(y) < sr // 2:
        return []
    try:
        import pyworld as pw
    except Exception:
        return []
    y64 = np.asarray(y, dtype=np.float64)
    f0, t = pw.harvest(y64, sr, f0_floor=70.0, f0_ceil=600.0,
                       frame_period=5.0)
    if len(f0) < 20 or not (f0 > 0).any():
        return []
    fr_s = 0.005

    def _win_st(a: float, b: float) -> Optional[float]:
        """[a,b]s 窗口内首1/3 vs 末1/3 的浊音中位 F0 差（半音，正=升）。"""
        ia, ib = int(a / fr_s), min(len(f0), int(b / fr_s))
        if ib - ia < 6:
            return None
        w = f0[ia:ib]
        k = max(2, (ib - ia) // 3)
        head = w[:k]; head = head[head > 0]
        tail = w[-k:]; tail = tail[tail > 0]
        if len(head) < 3 or len(tail) < 3:
            return None
        mh, mt = float(np.median(head)), float(np.median(tail))
        if mh <= 0 or mt <= 0:
            return None
        return 12.0 * np.log2(mt / mh)

    dur = len(y) / sr
    spans = char_time_spans(text, dur, [])
    if len(spans) < len(text):
        return []
    out: List[Dict[str, Any]] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _PUNCT_CHARS and ch in "？！。":
            p0, p1 = start, i
            if p1 > p0:
                tail_chars = max(1, min(3, (p1 - p0) // 3))
                c0 = max(p0, p1 - tail_chars)
                a = spans[c0][0]
                b = spans[p1][1]
                d = _win_st(a, b)
                if d is not None and abs(d) < 8.0:
                    out.append({"punct": ch, "rise_st": round(d, 2),
                                "n_chars": p1 - p0})
            start = i + 1
    return out


def aggregate_intonation_profile(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐短语测量 → 语调指纹。N<2 的项只记条数（推理端据此回退通用默认）。"""
    q = [m["rise_st"] for m in metrics if m["punct"] == "？"]
    st = [m["rise_st"] for m in metrics if m["punct"] == "。"]
    ex = [m["rise_st"] for m in metrics if m["punct"] == "！"]

    def _stat(v: List[float]) -> Dict[str, Any]:
        if len(v) < 2:
            return {"n": len(v)}
        return {"n": len(v),
                "median_st": round(float(np.median(v)), 2),
                "p25": round(float(np.percentile(v, 25)), 2),
                "p75": round(float(np.percentile(v, 75)), 2)}

    return {"question_rise": _stat(q), "statement_fall": _stat(st),
            "exclaim": _stat(ex), "total_phrases": len(metrics)}


def repair_truncated_tail(y: np.ndarray, sr: int,




                          thresh_db: float = -35.0,
                          min_tail_s: float = 0.03,
                          fade_ms: float = 100.0) -> np.ndarray:
    """成品末尾被硬截时的兜底修复：给尾部一段余弦淡出。

    背景（2026-10-03 爱弥斯实锤）：AR 偶发 EOS 提前，末字衰减在 -29dB
    处直接接数字零——听感「收音戛然而止」。BoN 的截断惩罚能挡住大部分
    候选，但当**全部**候选都被截（LoRA 素材本身是急收尾的游戏配音）时
    择优无米下锅。这里不伪造内容，只把硬悬崖变成 100ms 的快速收束：
    尾部实测静音（-35dB 判定）≥30ms 视为自然收尾，原样返回（一次都不
    动）；短于 30ms（还有语音能量就断了）才淡化。int16/float 均可，
    返回保持原 dtype。
    """
    yf = (np.asarray(y, dtype=np.float32) / 32768.0
          if np.issubdtype(np.asarray(y).dtype, np.integer)
          else np.asarray(y, dtype=np.float32))
    tail = edge_silence_len(yf, sr, from_end=True, thresh_db=thresh_db,
                            max_s=0.5)
    if tail >= min_tail_s:
        return y
    n = min(int(sr * fade_ms / 1000.0), len(yf) // 2)
    if n <= 1:
        return y
    out = yf.copy()
    out[-n:] *= (0.5 * (1.0 + np.cos(
        np.linspace(0.0, np.pi, n, dtype=np.float32)))).astype(np.float32)
    if np.issubdtype(np.asarray(y).dtype, np.integer):
        return np.clip(out * 32767.0, -32767, 32767).astype(np.int16)
    return out


def lengthen_abrupt_releases(y: np.ndarray, sr: int,
                             min_run_ms: float = 100.0,
                             slope_max_ms: float = 80.0,
                             take_ms: float = 150.0,
                             factor: float = 2.0,
                             seam_ms: float = 15.0,
                             max_releases: int = 4
                             ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """对「急收尾」做韵尾延长（final lengthening 的输出侧合成）。

    背景（2026-10-03 爱弥斯实锤）：该 LoRA 把标点前的收音渲染成 10~50ms
    的陡坡（8 个 BoN 候选全部如此——游戏配音素材先验，选择侧无解），
    听感「戛然而止/收音超级快」。语音学上真人句内/句末停顿的实现是
    **末音节延长**而非静音切段（阶段25已引）。这里对停顿前的尾段做
    WORLD 时长拉伸（音高/谱不变），把陡坡拖成可感知的收音。

    触发条件（全部满足才动，保守）：
      · 该停顿检出 ≥min_run_ms（词间隙级别的小缝不碰）；
      · 峰相对坡长 <slope_max_ms（「语音主体线」帧峰-8dB 滑到停顿的
        时长；自然收音 100~300ms 的坡不触发。不能用绝对 dB 线——素材
        电平低时永不触发，真机实锤）；
      · 尾段有足够浊音帧（harvest 守卫：纯气息/无谐波不做 WORLD）；
      · 每块最多 max_releases 处（防整段拖沓）。
    施工：WORLD 分析 [停顿前 take_ms]，谱帧线性插值到 factor 倍长后重
    合成，首接缝 seam_ms 等功率交叉淡化、末端 20ms 淡出（落在停顿里）；
    从后往前处理，偏移不互相影响。int16/float 均可，
    返回 (波形, [{at_s, added_ms}, ...])。
    """
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    if len(yf) < sr // 2:
        return y, []

    db, n = _frames_db(yf, sr, frame_ms=25.0, hop_ms=10.0)
    if not n:
        return y, []
    hop = int(sr * 0.01)
    quiet = db < -30.0
    edge = int(0.05 * sr)
    lo_frame = edge // hop
    hi_frame = max(lo_frame, (len(yf) - edge) // hop)
    min_run = int(sr * float(min_run_ms) / 1000.0)

    # 候选停顿（含块尾：末段有声→静音的转折也算）
    cands: List[int] = []          # 停顿起始帧
    i = lo_frame
    while i < hi_frame:
        if quiet[i]:
            j = i
            while j < hi_frame and quiet[j]:
                j += 1
            if j - i >= max(1, min_run // hop):
                cands.append(i)
            i = j
        else:
            i += 1

    # 陡坡判据（2026-10-03 三次校准后定稿）：**主体相对坡长**——从
    # 「语音主体线」（非静音帧的 p75，电平不变）滑到停顿的时长
    # <slope_max_ms 才算急收。不能用帧峰-8dB 之类的线：块响度归一前后
    # 峰-体距离会漂（raw 块正文离峰仅 2dB 时线掉进停顿肩部，坡长全是
    # 假 10ms；真机实锤两轮）。用真机校准：自然收音 150~800ms 不触发，
    # 真急收点（30ms，如「对不起，」）触发。
    line = float(np.percentile(db[~quiet], 75)) if (~quiet).any() \
        else float(np.max(db)) - 8.0
    targets = []
    for a in cands:
        if a * hop < int(sr * 0.15):
            continue
        k = a - 1
        while k >= max(0, a - 80) and db[k] < line:
            k -= 1
        slope_ms = (a - 1 - k) * 10 if k >= 0 else 10_000
        if slope_ms < float(slope_max_ms):
            targets.append(a)
    targets = targets[-max_releases:]
    if not targets:
        return y, []

    try:
        import pyworld as pw
    except Exception:
        return y, []

    out = yf.copy()
    releases: List[Dict[str, Any]] = []
    take = int(sr * float(take_ms) / 1000.0)
    seam = max(1, int(sr * float(seam_ms) / 1000.0))
    for a in reversed(targets):
        s0 = a * hop - take
        if s0 < seam or a * hop + hop >= len(out):
            continue
        seg = out[s0:a * hop].astype(np.float64)
        if len(seg) < sr // 10:
            continue
        try:
            f0, t = pw.harvest(seg, sr, f0_floor=70.0, f0_ceil=600.0,
                               frame_period=5.0)
            # 浊音守卫（宽）：harvest 对无谐波信号（纯正弦/纯噪声）判全
            # 无声带（f0=0），cheaptrick 用默认 f0 分析、重合成出频率错乱
            # 的垃圾（实锤：220Hz 正弦被合成成 ~330Hz）。真实尾段多为
            # 混合型（16/30 帧浊音），只要「有结构」（≥4 帧）就做，合成
            # 后另有谱自检兜底，不达标整刀放弃。
            if int((f0 > 0).sum()) < 4:
                continue
            sp = pw.cheaptrick(seg, f0, t, sr)
            ap = pw.d4c(seg, f0, t, sr)
            n_fr = max(2, len(f0))
            idx = np.linspace(0.0, n_fr - 1.0,
                              max(2, int(n_fr * float(factor))))

            def _interp_frames(arr):
                # 谱/非周期性按帧线性插值（nearest-hold 会让帧间包络
                # 跳台阶，合成出来是可闻的帧率抖动）
                i0 = np.floor(idx).astype(int)
                i1 = np.clip(i0 + 1, 0, arr.shape[0] - 1)
                w = (idx - i0)[:, None].astype(np.float64)
                return arr[i0] * (1.0 - w) + arr[i1] * w

            seg2 = pw.synthesize(
                np.interp(idx, np.arange(n_fr), f0),
                _interp_frames(sp), _interp_frames(ap), sr,
                frame_period=5.0)
        except Exception:
            continue
        seg2 = np.asarray(seg2, dtype=np.float32)
        want = int(take * float(factor))
        seg2 = seg2[:want + sr // 20]      # WORLD 尾部多合成一点再裁
        if len(seg2) < want:
            seg2 = np.concatenate(
                [seg2, np.zeros(want - len(seg2), np.float32)])
        seg2 = seg2[:want]
        # 谱自检（兜底守卫）：WORLD 重合成的电平与频谱形状必须贴近原段
        # （RMS ±4dB、<1kHz 占比 ±20pp）——拉伸让能量略向低频移是自然
        # 收音方向，允许；但整体变闷/变亮/变响说明重合成失真，整刀放弃
        # （真机气声尾实测：-30.3→-29.4dB、低频段 63.6→78.4pp，通过）。
        def _rms_db(x):
            return 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)

        def _low_share(x):
            S = np.abs(np.fft.rfft(x)) ** 2
            ff = np.fft.rfftfreq(len(x), 1.0 / sr)
            tot = float(S.sum()) + 1e-12
            return 100.0 * float(S[ff < 1000.0].sum()) / tot

        if (abs(_rms_db(seg2) - _rms_db(seg)) > 4.0
                or abs(_low_share(seg2) - _low_share(seg)) > 20.0):
            continue
        m = min(seam, len(seg2))
        seg2[:m] = (seg2[:m] * np.linspace(0.0, 1.0, m, dtype=np.float32)
                    + out[s0:s0 + m]
                    * np.linspace(1.0, 0.0, m, dtype=np.float32))
        # 末端淡出：seg2 结束处落在停顿静音里，WORLD 重合成与原相位
        # 无关、末尾未必归零——不淡化会在这里留一个可闻台阶
        f_out = min(int(0.02 * sr), len(seg2) // 3)
        if f_out > 1:
            seg2[-f_out:] *= np.linspace(1.0, 0.0, f_out,
                                         dtype=np.float32)
        out = np.concatenate([out[:s0], seg2, out[s0 + take:]])
        releases.append({
            "at_s": round(a * hop / sr, 3),
            "added_ms": round((len(seg2) - take) / sr * 1000),
        })
    if not releases:
        return y, []
    if was_int:
        return (np.clip(out * 32767.0, -32767, 32767).astype(np.int16),
                releases)
    return out.astype(np.float32), releases


def edge_silence_len(y: np.ndarray, sr: int, from_end: bool = True,                     thresh_db: float = -30.0, max_s: float = 1.0) -> float:
    """量波形尾（from_end=True）或头部的连续静音时长（秒，封顶 max_s）。

    块边界停顿管理用：模型的块尾收束静音 + 块头起振静音 + 台本 gap 三者
    叠加是「长句拆块后每一小句都很散」的主因（2026-10-02），拼接层要
    拿实际静音量来修剪/垫足，让块间总停顿精确等于台本值。
    int16/float 均可（阈值是 ±1 尺度量——int16 直传会整体偏 +90dB、
    判定全废，与 apply_stress_gains 同族的尺度陷阱）。
    """
    yf = (np.asarray(y, dtype=np.float32) / 32768.0
          if np.issubdtype(np.asarray(y).dtype, np.integer)
          else np.asarray(y, dtype=np.float32))
    fr = int(sr * 0.025)
    if len(yf) < fr:
        return 0.0
    frames = yf[: max(1, len(yf) // fr) * fr].reshape(-1, fr)
    e = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(e + 1e-12)
    quiet = db < thresh_db
    n = len(db)
    max_frames = int(max_s * sr) // fr
    count = 0
    if from_end:
        for i in range(n - 1, -1, -1):
            if not quiet[i] or count >= max_frames:
                break
            count += 1
    else:
        for i in range(n):
            if not quiet[i] or count >= max_frames:
                break
            count += 1
    return count * fr / sr


def trim_edge_silence(y: np.ndarray, sr: int, cut_s: float,
                      from_end: bool = True, thresh_db: float = -30.0
                      ) -> np.ndarray:
    """把尾（from_end=True）/头部静音**从外侧**截掉 cut_s 秒。

    与 cap_interior_pauses 的中段截法同款保底：只动静音区（按 -30dB 判
    定，且不超过实测静音量 - 20ms），保留贴语音的衰减/起振端；切口做
    5ms 淡化防 click。int16/float 均可，返回保持原 dtype。
    """
    if cut_s <= 0:
        return y
    was_int = np.issubdtype(np.asarray(y).dtype, np.integer)
    avail = max(0.0, edge_silence_len(y, sr, from_end=from_end,
                                      thresh_db=thresh_db) - 0.02)
    n = int(min(cut_s, avail) * sr)
    if n <= 0:
        return y
    yf = (np.asarray(y, dtype=np.float32) / 32768.0 if was_int
          else np.asarray(y, dtype=np.float32))
    if from_end:
        out = yf[:-n].copy()
        f = max(1, int(0.005 * sr))
        if len(out) > f:
            out[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    else:
        out = yf[n:].copy()
        f = max(1, int(0.005 * sr))
        if len(out) > f:
            out[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    if was_int:
        return np.clip(out * 32767.0, -32767, 32767).astype(np.int16)
    return out.astype(np.asarray(y).dtype, copy=False)


def restore_f0_range(y: np.ndarray, sr: int, fp: Dict[str, Any],
                     intensity: float = 0.5, max_shift_st: float = 3.0,
                     max_expand: float = 1.6,
                     min_expand_std: float = 2.5) -> Dict[str, Any]:
    """把一段语音的 F0 波动向角色指纹适度扩张（WORLD 重合成）。

    2026-10-02「痰音」根治（实测谱取证 + 变量隔离实验）：
      · **不再整体移调**（shift_med 恒为 0）。原先把输出中位强拉到指纹
        median，是越权——音高由模型/情绪表演决定，不是要"恢复"的对象；
        且女声谐波稀疏，移调 2~3 半音就让谐波与 cheaptrick 谱包络错位，
        300-1000Hz 能量占比被重画 10pp+（听感=含混的痰音）。
      · **音域已达标就不动**。WORLD 扩张本身也有谐波涂抹代价（k=1.3
        实测 +11pp），只有输出真的平坦（std < min_expand_std 半音）才
        值得付这个代价；达标还硬扩是纯损伤（实测达标块被强扩后
        300-1k +12pp）。达标/跳过时原样返回 ok=False 并带原因。

    扩张：new_f0_i = 原中位 × (原F0_i/原中位)^(k-1)+1，k = 目标std/原std
    （只扩不缩，限 max_expand）。指纹缺失/浊音不足时也原样返回。
    """
    import numpy as _np
    try:
        import pyworld as pw
    except Exception as e:
        return {"ok": False, "error": f"pyworld 未安装：{e}"}
    if not fp or float(fp.get("median", 0)) <= 0:
        return {"ok": False, "error": "无指纹"}
    y64 = np.asarray(y, dtype=np.float64)
    if len(y64) < sr // 4:
        return {"ok": False, "error": "音频太短"}
    f0, _t = pw.harvest(y64, sr, f0_floor=70.0, f0_ceil=600.0)
    voiced = f0 > 0
    if int(voiced.sum()) < max(20, sr // 800):
        return {"ok": False, "error": "浊音帧不足"}
    med = float(np.median(f0[voiced]))
    semis = 12.0 * _np.log2(f0[voiced] / med)
    orig_std = float(_np.std(semis))
    t_std = float(fp.get("std_st", orig_std)) * (
        1.0 + 0.3 * float(np.clip(intensity, 0.0, 1.0)))
    if orig_std >= float(min_expand_std):
        return {"ok": False,
                "error": (f"音域已达标（std {orig_std:.2f} ≥ "
                          f"{min_expand_std:.1f} 半音，跳过重合成）"),
                "orig_std": round(orig_std, 2),
                "target_std": round(t_std, 2)}
    if orig_std < 0.3:
        k = 1.0
    else:
        k = float(_np.clip(t_std / orig_std, 1.0, float(max_expand)))
    if k <= 1.01:
        return {"ok": False, "error": "目标波动不高于原波动，跳过",
                "orig_std": round(orig_std, 2),
                "target_std": round(t_std, 2)}

    shift_med = 0.0
    new_f0 = f0.copy()
    frame_shift = _np.clip((k - 1.0) * semis,          # 只扩张，不移调
                           -float(max_shift_st), float(max_shift_st))
    new_f0[voiced] = f0[voiced] * 2.0 ** (frame_shift / 12.0)

    sp, ap = pw.cheaptrick(y64, f0, _t, sr), pw.d4c(y64, f0, _t, sr)
    y2 = pw.synthesize(new_f0, sp, ap, sr, frame_period=5.0)
    y2 = _np.asarray(y2, dtype=np.float32)[:len(y)]
    if len(y2) < len(y):
        y2 = _np.concatenate([y2, _np.zeros(len(y) - len(y2), np.float32)])
    return {"ok": True, "k": round(k, 3),
            "shift_med_st": 0.0,
            "orig_std": round(orig_std, 2), "target_std": round(t_std, 2),
            "y": y2}
