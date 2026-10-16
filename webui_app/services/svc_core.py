"""svc_core — 原生 Voice Conversion 核心管线（绕过 GPT）。

通路（docs/dev-plan-ai-cover-vc-2026-10.md §3.1，逐行对齐 indextts/infer_v2_5.py）：

    源音频 16k → extract_features → get_emb(归一化 50Hz 连续特征, (1,T,1024))
      → [变体A: semantic_codec.quantize→decode | 变体B: 连续特征直通]
      → length_regulator(ylens=目标 mel 帧数) → cond (1,T,512)
      → cat([prompt_condition, cond], dim=1)
      → cfm.inference(cat, [len], ref_mel, style, None, steps, cfg)
      → mel[:, :, ref_mel_len:] → bigvgan(mel.float()) → float32 ±1 波形

关键事实（审查核验）：
- CFM 完全无状态可重入（setup_caches(use_kv_cache=False)，DiT 纯 SDPA 全注意力，
  input_pos 每次切片），逐块调用无需重置。
- 实际上限是 16384 帧位置缓冲 + T² 全注意力算力 → 离线安全块 60s。
- prompt_condition 槽 = 参考的**原始连续特征**（infer_v2_5.py:650-655），
  生成槽（TTS 路径）= decode 重构 → 变体 B 与 prompt 槽同构，变体 A 与 TTS 同域。
- 波形全链 float32 ±1（int16 尺度陷阱红线）；mel 帧数 = (N22k-256)//256+1
  （mel_fn center=False + reflect pad 384）；w2v 帧数 = floor((N16k-400)/320)+1。
- 量化 squeeze 由 num_quantizers=1 触发（dim0 是 N 非 batch）——**禁止 B>1 批处理**。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from webui_app.services import svc_f0

# 引擎采样率（与 indextts 一致）
ENGINE_SR = 22050
FEAT_SR = 16000
# 50Hz 特征帧 ↔ 86.13 mel 帧换算
MEL_PER_FEAT = 1.7227  # 86.133/50
# mel 帧公式（mel_fn: center=False + F.pad(reflect,(384,384)) + hop 256）
HOP = 256
# w2v-BERT 帧步长（16k，kaldi 400 窗/320 移位）
FEAT_STEP = 320
# 流式边缘丢弃（覆盖双向注意力边界 + 重采样瞬态 + decode off-by-one）
EDGE_DROP_FRAMES = 3
# BigVGAN mel 域 pad（非因果感受野；用边缘帧复制填充，log 域禁 zeros）
VOC_PAD_FRAMES = 5
# 离线块间输入重叠（= 交叉淡化时长，对拍红线：不重叠=每块净丢淡化时长）
DEFAULT_OVERLAP_MS = 40.0
# 低能量切点搜索半径
CUT_SEARCH_SEC = 5.0


def mel_frames(n_samples_22k: int) -> int:
    """与 engine.mel_fn 一致的帧数公式。"""
    if n_samples_22k < HOP:
        return 1
    return (n_samples_22k - HOP) // HOP + 1


# ---------------------------------------------------------------------------
# 参考上下文（音色锚定三路：ref_mel / campplus style / prompt_condition）
# ---------------------------------------------------------------------------

@dataclass
class ReferenceContext:
    key: str
    ref_mel: torch.Tensor            # (1, 80, T_ref)
    style: torch.Tensor              # (1, 192)
    prompt_condition: torch.Tensor   # (1, T_ref, 512)
    ref_frames: int

    def to(self, device):
        self.ref_mel = self.ref_mel.to(device)
        self.style = self.style.to(device)
        self.prompt_condition = self.prompt_condition.to(device)
        return self


_REF_CACHE: Dict[str, ReferenceContext] = {}


def build_reference(engine, ref_path: str, ref_seconds: float = 15.0,
                    use_cache: bool = True) -> ReferenceContext:
    """构建音色参考三路条件（复刻 infer_v2_5.py:626-655，支持任意参考时长）。

    LoRA：调用方先 engine.attach_lora(dir, "cfm") 再进本模块；CFM LoRA 包在
    s2mel.models["cfm"] 上对 inference 透明生效。length_regulator 不在注入面。
    """
    tts = engine.tts
    key = f"{os.path.abspath(ref_path)}|{os.path.getmtime(ref_path)}|{ref_seconds}"
    if use_cache and key in _REF_CACHE:
        return _REF_CACHE[key].to(tts.device)

    import librosa
    import torchaudio
    audio, sr = librosa.load(ref_path, sr=None, mono=True)
    max_n = int(ref_seconds * sr)
    if len(audio) > max_n:
        audio = audio[:max_n]
    # soxr 高质量重采样（torchaudio 默认滤波对 ≥3:1 降采样混叠，见 convert_file 注）
    audio = np.asarray(audio, dtype=np.float32)
    audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=ENGINE_SR)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=FEAT_SR)
    audio_22k = torch.tensor(audio_22k, dtype=torch.float32).unsqueeze(0)
    audio_16k = torch.tensor(audio_16k, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        inputs = tts.extract_features(audio_16k, sampling_rate=FEAT_SR,
                                       return_tensors="pt")
        feats = inputs["input_features"].to(tts.device)
        mask = inputs["attention_mask"].to(tts.device)
        spk_cond_emb = tts.get_emb(feats, mask)          # (1, T, 1024)

        ref_mel = tts.mel_fn(audio_22k.to(tts.device).float())   # (1, 80, T)
        ref_frames = int(ref_mel.size(2))

        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device), num_mel_bins=80, dither=0,
            sample_frequency=FEAT_SR)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = tts.campplus_model(feat.unsqueeze(0))    # (1, 192)

        prompt_condition = tts.s2mel.models["length_regulator"](
            spk_cond_emb,
            ylens=torch.LongTensor([ref_frames]).to(ref_mel.device),
            n_quantizers=3, f0=None)[0]

    ctx = ReferenceContext(key=key, ref_mel=ref_mel, style=style,
                           prompt_condition=prompt_condition, ref_frames=ref_frames)
    _REF_CACHE.clear()   # 只留一份（音色切换场景足够；防显存累积）
    _REF_CACHE[key] = ctx
    return ctx


# ---------------------------------------------------------------------------
# 单窗转换
# ---------------------------------------------------------------------------

def _extract_feats(engine, wav16k: torch.Tensor) -> torch.Tensor:
    """16k 波形 [1,N] → 归一化连续特征 (1, T, 1024)。"""
    tts = engine.tts
    inputs = tts.extract_features(wav16k.cpu(), sampling_rate=FEAT_SR,
                                  return_tensors="pt")
    feats = inputs["input_features"].to(tts.device)
    mask = inputs["attention_mask"].to(tts.device)
    return tts.get_emb(feats, mask)


def _content_codes(engine, feats: torch.Tensor, variant: str) -> torch.Tensor:
    """特征 → 50Hz 内容表示 (1, T', 1024)。

    变体 A：quantize→decode（与 TTS 生成路径同域，默认）。
    变体 B：连续特征直通（与 prompt_condition 槽同构）。
    """
    tts = engine.tts
    if variant == "B":
        return feats
    with torch.no_grad():
        codes, _q = tts.semantic_codec.quantize(feats)     # (1, T//2)
        S = tts.semantic_codec.decode(codes)               # (1, ~T, 1024)
    return S


def _vocode(engine, mel: torch.Tensor) -> torch.Tensor:
    """BigVGAN：mel 域首尾各复制 VOC_PAD_FRAMES 帧填充（log 域禁 zeros），
    声码后裁掉对应样本，抑制非因果边缘伪影。返回 (1, N) float32。"""
    pad = VOC_PAD_FRAMES
    if pad > 0 and mel.size(2) > pad * 2:
        head = mel[:, :, :1].expand(-1, -1, pad)
        tail = mel[:, :, -1:].expand(-1, -1, pad)
        mel_in = torch.cat([head, mel, tail], dim=2)
    else:
        mel_in = mel
        pad = 0
    wav = engine.tts.bigvgan(mel_in.float())
    wav = wav.squeeze(1)                                    # (B, N)
    if pad > 0:
        cut = pad * HOP
        if wav.size(1) > 2 * cut:
            wav = wav[:, cut:-cut]
    return wav


def convert_window(engine, wav16k: np.ndarray, wav22k: np.ndarray,
                   ref: ReferenceContext, *, variant: str = "A",
                   steps: int = 25, cfg_rate: float = 0.7,
                   keep_range: Optional[Tuple[int, int]] = None,
                   target_frames: Optional[int] = None) -> np.ndarray:
    """转换单窗。

    离线：keep_range=None，target_frames=mel_frames(len(wav22k))，返回整窗波形。
    流式：keep_range=(k0,k1) 指定新块 mel 帧范围（用于裁特征与对齐上下文），
          target_frames=k1-k0。
    """
    tts = engine.tts
    dev = tts.device
    t16 = torch.tensor(np.asarray(wav16k, dtype=np.float32)).unsqueeze(0)

    with torch.no_grad(), torch.inference_mode():
        feats = _extract_feats(engine, t16)                # (1, T50, 1024)

        if keep_range is not None:
            # 特征帧 j 中心 ≈ 320j+200（16k 样本）；mel 帧 k 中心 ≈ 256k+128
            # 由 keep 的 mel 帧范围反推保留的特征帧范围（含边缘丢弃）
            k0, k1 = keep_range
            s16 = k0 * HOP * FEAT_SR / ENGINE_SR          # 新块起点（16k 坐标近似）
            e16 = k1 * HOP * FEAT_SR / ENGINE_SR
            j0 = int(max(0, np.ceil((s16 + EDGE_DROP_FRAMES * FEAT_STEP - 200) / FEAT_STEP)))
            j1 = int(min(feats.size(1), np.floor((e16 - 200) / FEAT_STEP) + 1))
            if j1 - j0 < 2:
                j0, j1 = 0, feats.size(1)
            feats = feats[:, j0:j1, :]

        S = _content_codes(engine, feats, variant)
        if target_frames is None:
            target_frames = mel_frames(len(wav22k))
        S = S.to(ref.prompt_condition.dtype)

        cond = tts.s2mel.models["length_regulator"](
            S, ylens=torch.LongTensor([int(target_frames)]).to(dev),
            n_quantizers=3, f0=None)[0]

        cat = torch.cat([ref.prompt_condition, cond], dim=1)
        mel = tts.s2mel.models["cfm"].inference(
            cat,
            torch.LongTensor([cat.size(1)]).to(dev),
            ref.ref_mel, ref.style, None, int(steps),
            inference_cfg_rate=float(cfg_rate))
        mel = mel[:, :, ref.ref_mel.size(-1):]

        wav = _vocode(engine, mel)                          # (1, N)

    out = wav.squeeze(0).float().cpu().numpy().astype(np.float32)
    out = np.clip(out, -1.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 离线：分块 + 低能量切点 + 输入重叠 + 波形域交叉淡化
# ---------------------------------------------------------------------------

def _energy_valleys(y: np.ndarray, sr: int, hop_ms: float = 20.0) -> np.ndarray:
    """20ms 帧能量。"""
    hop = max(1, int(sr * hop_ms / 1000.0))
    n = len(y) // hop
    if n < 2:
        return np.array([float(np.sqrt(np.mean(y ** 2) + 1e-12))])
    frames = y[:n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)


def plan_chunks(y: np.ndarray, sr: int, chunk_sec: float,
                overlap_ms: float = DEFAULT_OVERLAP_MS) -> List[Tuple[int, int]]:
    """分块计划：目标块界 ±CUT_SEARCH_SEC 内找能量谷切点（找不到硬切），
    相邻块输入重叠 = overlap_ms（对拍红线：输出淡化不丢时长）。"""
    n = len(y)
    target = int(chunk_sec * sr)
    if n <= target:
        return [(0, n)]
    overlap = int(overlap_ms / 1000.0 * sr)
    energies = _energy_valleys(y, sr)
    hop_n = max(1, int(sr * 0.02))
    med = float(np.median(energies)) + 1e-9

    bounds = [0]
    start = 0
    while n - start > target * 1.5:
        center = start + target
        lo = max(start + int(sr * 2.0), center - int(CUT_SEARCH_SEC * sr))
        hi = min(n - int(sr * 2.0), center + int(CUT_SEARCH_SEC * sr))
        if hi <= lo:
            cut = center
        else:
            f_lo, f_hi = lo // hop_n, hi // hop_n
            seg = energies[f_lo:f_hi]
            if len(seg) > 0:
                rel = int(np.argmin(seg))
                # 谷判定：能量 < 30% 中位数才认（歌唱气口），否则硬切
                if seg[rel] < 0.3 * med:
                    cut = (f_lo + rel) * hop_n
                else:
                    cut = center
            else:
                cut = center
        cut = int(np.clip(cut, start + int(sr * 2.0), n - int(sr * 2.0)))
        bounds.append(cut)
        start = cut - overlap
    bounds.append(n)

    chunks = []
    for i in range(len(bounds) - 1):
        a = bounds[i] - overlap if i > 0 else bounds[i]
        b = bounds[i + 1]
        if b - a > int(sr * 1.0):
            chunks.append((int(max(0, a)), int(min(n, b))))
    return chunks


def _crossfade_append(out: np.ndarray, wav: np.ndarray, overlap: int) -> np.ndarray:
    """把新块拼进输出流：前 overlap 样本与 out 尾部等功率淡化。"""
    if len(out) == 0:
        return wav.copy()
    if overlap <= 0 or overlap >= len(wav) or overlap > len(out):
        return np.concatenate([out, wav])
    fa, fb = svc_f0._equal_power_fade(overlap)
    out[-overlap:] = out[-overlap:] * fa + wav[:overlap] * fb
    return np.concatenate([out, wav[overlap:]])


@dataclass
class ConvertResult:
    ok: bool
    wav: Optional[np.ndarray] = None
    sr: int = ENGINE_SR
    reason: str = ""
    chunks: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    chunk_times: List[float] = field(default_factory=list)


def convert_file(engine, src_path: str, ref: ReferenceContext, *,
                 variant: str = "A", steps: int = 25, cfg_rate: float = 0.7,
                 chunk_sec: float = 60.0, overlap_ms: float = DEFAULT_OVERLAP_MS,
                 progress: Optional[Callable] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> ConvertResult:
    """离线整曲转换：分块转换 → 输入重叠 + 波形域交叉淡化 → 保长输出。"""
    import librosa
    t_all = time.perf_counter()
    y, sr = librosa.load(src_path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if len(y) < int(0.5 * sr):
        return ConvertResult(False, reason="源音频过短（<0.5s）")

    # 高质量重采样（soxr 后端）。torchaudio.transforms.Resample 默认滤波宽度
    # 对 ≥3:1 降采样（44.1/48k 源）抗混叠不足——GPU 验收实锤：48k 源直接进
    # 该链会把混叠特征喂进码本，输出高频嘶声且声纹崩坏（SS≈0）。
    y22 = np.asarray(librosa.resample(y, orig_sr=sr, target_sr=ENGINE_SR),
                     dtype=np.float32)
    y16 = np.asarray(librosa.resample(y, orig_sr=sr, target_sr=FEAT_SR),
                     dtype=np.float32)

    chunks = plan_chunks(y22, ENGINE_SR, chunk_sec, overlap_ms)
    overlap = int(overlap_ms / 1000.0 * ENGINE_SR)
    out: List[np.ndarray] = []
    total_out = 0
    chunk_times: List[float] = []
    r16 = FEAT_SR / ENGINE_SR

    for i, (a, b) in enumerate(chunks):
        if should_stop is not None and should_stop():
            return ConvertResult(False, reason="用户停止", chunks=i,
                                 timing={"elapsed": time.perf_counter() - t_all})
        t0 = time.perf_counter()
        seg22 = y22[a:b]
        a16, b16 = int(round(a * r16)), int(round(b * r16))
        seg16 = y16[a16:b16]
        want = b - a
        wav = convert_window(engine, seg16, seg22, ref, variant=variant,
                             steps=steps, cfg_rate=cfg_rate)
        # 块级严格保长（输出 trim/pad 到源窗样本数——对拍保证）
        if len(wav) >= want:
            wav = wav[:want]
        else:
            wav = np.pad(wav, (0, want - len(wav)))
        chunk_times.append(time.perf_counter() - t0)
        total_out += len(wav) - (overlap if i > 0 else 0)
        merged = _crossfade_append(np.concatenate(out) if out else np.zeros(0, dtype=np.float32),
                                   wav, overlap if i > 0 else 0)
        out = [merged]
        if progress is not None:
            try:
                progress((i + 1) / len(chunks),
                         f"VC 转换 {i + 1}/{len(chunks)} 块 "
                         f"({chunk_times[-1]:.1f}s/块)")
            except Exception:
                pass

    wav = out[0] if out else np.zeros(0, dtype=np.float32)
    elapsed = time.perf_counter() - t_all
    dur = len(wav) / ENGINE_SR
    timing = {"elapsed": elapsed, "audio_sec": dur,
              "rtf": round(elapsed / max(dur, 1e-6), 3),
              "per_chunk_avg": round(float(np.mean(chunk_times)), 2) if chunk_times else 0.0}
    return ConvertResult(True, wav=wav, sr=ENGINE_SR, chunks=len(chunks),
                         timing=timing, chunk_times=chunk_times)


# ---------------------------------------------------------------------------
# 流式转换器（AI 变声）
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    block_frames: int = 31        # 新块 mel 帧数（21/31/63 → D6 档位）
    steps: int = 10
    cfg_rate: float = 0.0
    left_ctx_sec: float = 2.0
    right_ctx_sec: float = 0.02
    variant: str = "A"
    overlap_ms: float = DEFAULT_OVERLAP_MS


class StreamConverter:
    """流式 VC：feed(设备采样率块) → 新块转换波形（自带与上块的重叠段）。

    架构（D5）：
    - 设备采样率原样入环形缓冲；**重采样发生在窗口拼接之后**（FIR 瞬态
      落在被丢弃的上下文里）。
    - 窗口 = 左上下文 + 新块 + 右上下文，整窗过特征/码本，只保留新块帧
      （左缘多丢 EDGE_DROP_FRAMES 帧覆盖注意力边界效应）。
    - 返回 (wav, overlap_samples)：wav 前 overlap 与已输出流尾部做等功率
      淡化（调用方执行），不丢时长。
    """

    def __init__(self, engine, ref: ReferenceContext, dev_sr: int,
                 cfg: Optional[StreamConfig] = None):
        import torchaudio
        self.engine = engine
        self.ref = ref
        self.cfg = cfg or StreamConfig()
        self.dev_sr = int(dev_sr)
        self._res22 = torchaudio.transforms.Resample(self.dev_sr, ENGINE_SR)
        self._res16 = torchaudio.transforms.Resample(self.dev_sr, FEAT_SR)
        # 设备采样率缓冲（只保留左上下文 + 在途块）。
        # **坐标约定**：_consumed/_buf_offset 都是绝对样本号（从会话起点累计），
        # 缓冲裁剪后窗口索引必须减去 _buf_offset 换算到相对坐标——否则取错
        # 段、同一段数据被重复消费（mock 测试实锤过这个 bug）。
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._buf_offset = 0        # self._buf[0] 的绝对样本号
        self._consumed = 0          # 已发射的设备样本边界（绝对坐标）
        self._first = True
        self._prev_tail: Optional[np.ndarray] = None
        self.block_dev = int(self.cfg.block_frames * HOP * self.dev_sr / ENGINE_SR)
        self.overlap_out = int(self.cfg.overlap_ms / 1000.0 * ENGINE_SR)
        self.last_infer_ms = 0.0
        self.reset()

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_offset = 0
        self._consumed = 0
        self._first = True
        self._prev_tail = None

    def feed(self, chunk: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
        """喂入设备采样率块；块满时返回 (新块波形, overlap_samples)，否则 None。

        返回的波形为 22k；前 overlap_samples 与**上一次返回的尾部**重叠，
        调用方等功率淡化后追加到输出流。
        """
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32)])
        have = self._buf_offset + len(self._buf)      # 缓冲覆盖到的绝对位置
        right_dev = int(self.cfg.right_ctx_sec * self.dev_sr)
        need = (self._consumed + self.block_dev + right_dev) if not self._first \
            else (self.block_dev + right_dev)
        if have < need:
            return None

        block_end = self._consumed + self.block_dev
        left_dev = int(self.cfg.left_ctx_sec * self.dev_sr) if not self._first else 0
        win_a = max(self._buf_offset, block_end - self.block_dev - left_dev)
        win_b = min(have, block_end + right_dev)
        window = torch.tensor(self._buf[win_a - self._buf_offset:win_b - self._buf_offset],
                              dtype=torch.float32).unsqueeze(0)
        win22 = self._res22(window).squeeze(0).numpy().astype(np.float32)
        win16 = self._res16(window).squeeze(0).numpy().astype(np.float32)

        # 新块在窗口内的 mel 帧范围（窗口起点对齐）
        scale22 = ENGINE_SR / self.dev_sr
        blk_a_dev = block_end - self.block_dev - win_a
        k0 = max(0, int(round(blk_a_dev * scale22 / HOP)))
        k1 = k0 + self.cfg.block_frames
        win_frames = mel_frames(len(win22))
        if k1 > win_frames:
            k1 = win_frames
        if k1 - k0 < 4:
            self._advance(block_end)
            return None

        t0 = time.perf_counter()
        wav = convert_window(self.engine, win16, win22, self.ref,
                             variant=self.cfg.variant,
                             steps=self.cfg.steps, cfg_rate=self.cfg.cfg_rate,
                             keep_range=(k0, k1), target_frames=k1 - k0)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        overlap = 0 if self._first else self.overlap_out
        if overlap > 0 and self._prev_tail is not None:
            wav = np.concatenate([self._prev_tail, wav])
        elif overlap > 0:
            overlap = 0    # 无上块尾部（理论不达，防御）
        # 无条件保存尾部，下一块淡化用（曾因首块 overlap=0 不保存而死锁为全 0）
        self._prev_tail = wav[-self.overlap_out:].copy() \
            if len(wav) >= self.overlap_out else None
        self._advance(block_end)
        self._first = False
        return wav, overlap

    def _advance(self, block_end: int):
        keep = int(self.cfg.left_ctx_sec * self.dev_sr)
        cut = max(0, block_end - keep)
        if cut > self._buf_offset:
            drop = cut - self._buf_offset
            if 0 < drop <= len(self._buf):
                self._buf = self._buf[drop:]
                self._buf_offset = cut
        self._consumed = max(self._consumed, block_end)
