"""vc_realtime — AI 变声实时运行时（sounddevice WASAPI 双流 + 分块推理）。

架构（dev-plan D5/D6）：
    麦克风 callback → in_q ──推理线程──> StreamConverter.feed（窗口拼接后
    才重采样）→ 输出交叉淡化 → out_ring → 扬声器 callback

- 输入调理：去DC / 轻量 AGC（慢 attack，防模型吃不到信号）。
- VAD 静音直通（Seed-VC 同款）：不说话时零输出，减少拼接毛刺与算力。
- F0：off / smooth（语调趋势移植，中值滤波窗大）/ follow（完整跟随，质量档）。
  映射系数由参考与首块预计算一次，全程恒定（跨块连续性红线）。
- 延迟诚实显示：算法延迟（2×块+右上下文）+ 实测回环（脉冲互相关）。
- WebUI 无法向浏览器流式送音频——输出走本机声卡（耳机监听或 VB-Cable
  虚拟声卡供 OBS/Discord 采集），UI 是控制面板。

sounddevice 用法（审查修正）：默认共享模式 + WasapiSettings(auto_convert=True)
（exclusive 与 auto_convert 互斥）；独占模式仅在采样率匹配时提供。
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from webui_app.services import svc_core, svc_f0

# 档位表（D6：帧数按 mel 帧公式精确计算，以帧为准）
PRESETS = {
    "快速（低延迟）": dict(block_frames=21, steps=6, cfg_rate=0.0,
                           left_ctx_sec=1.0, ref_seconds=5.0, f0_mode="off",
                           algo_latency_ms=2 * 21 * 256 / 22050 * 1000 + 20),
    "均衡（推荐）": dict(block_frames=31, steps=10, cfg_rate=0.0,
                         left_ctx_sec=2.0, ref_seconds=6.0, f0_mode="smooth",
                         algo_latency_ms=2 * 31 * 256 / 22050 * 1000 + 20),
    "质量（更像）": dict(block_frames=63, steps=16, cfg_rate=0.3,
                         left_ctx_sec=2.5, ref_seconds=8.0, f0_mode="follow",
                         algo_latency_ms=2 * 63 * 256 / 22050 * 1000 + 20),
}

AGC_TARGET_RMS = 10 ** (-20.0 / 20.0)   # -20dBFS
AGC_MAX_GAIN = 8.0
SILENCE_RMS = 10 ** (-42.0 / 20.0)      # VAD 门限


def sd_available() -> bool:
    try:
        from importlib.util import find_spec
        return find_spec("sounddevice") is not None
    except Exception:
        return False


def list_devices() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """(输入设备, 输出设备)，元素=(显示名, 索引)。sounddevice 未装/失败返回空。"""
    if not sd_available():
        return [], []
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        ins, outs = [], []
        for i, d in enumerate(devs):
            name = f"{d['name']} ({int(d.get('default_samplerate', 48000))}Hz)"
            if d.get("max_input_channels", 0) > 0:
                ins.append((name, i))
            if d.get("max_output_channels", 0) > 0:
                outs.append((name, i))
        return ins, outs
    except Exception:
        return [], []


class _RingBuffer:
    """单生产单消费的样本环形队列（np 拼接版，块级粒度足够）。"""

    def __init__(self):
        self._parts: List[np.ndarray] = []
        self._len = 0
        self._lock = threading.Lock()

    def put(self, chunk: np.ndarray):
        with self._lock:
            self._parts.append(np.asarray(chunk, dtype=np.float32))
            self._len += len(chunk)

    def get(self, n: int) -> Tuple[np.ndarray, bool]:
        """取 n 个样本；不足时补零并标记 underrun。"""
        with self._lock:
            out = np.zeros(n, dtype=np.float32)
            got = 0
            while got < n and self._parts:
                head = self._parts[0]
                take = min(len(head), n - got)
                out[got:got + take] = head[:take]
                got += take
                if take < len(head):
                    self._parts[0] = head[take:]
                else:
                    self._parts.pop(0)
            self._len -= got
            return out, got < n

    def __len__(self):
        return self._len


@dataclass
class SessionStatus:
    running: bool = False
    error: str = ""
    level_in_db: float = -120.0
    level_out_db: float = -120.0
    blocks: int = 0
    underruns: int = 0
    last_infer_ms: float = 0.0
    algo_latency_ms: float = 0.0
    measured_latency_ms: float = -1.0
    silent_ratio: float = 0.0
    dropped: int = 0           # 输入积压丢块计数（推理慢于实时时）


class RealtimeVC:
    """一次实时变声会话。start() 成功后 status() 轮询，stop() 收尾。"""

    def __init__(self, engine, ref_path: str):
        self.engine = engine
        self.ref_path = ref_path
        self._status = SessionStatus()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._in_q: "queue.SimpleQueue" = None
        self._out_ring = _RingBuffer()
        self._conv: Optional[svc_core.StreamConverter] = None
        self._f0: Optional[svc_f0.F0Tracker] = None
        self._f0_semis: int = 0
        self._agc_gain = 1.0
        self._record_buf: List[np.ndarray] = []
        self._preset: dict = {}
        self._dev_sr = 22050
        self._last_out_tail: Optional[np.ndarray] = None
        self.record_enabled = False

    # ------------------------------------------------------------------
    def start(self, in_dev: Optional[int], out_dev: Optional[int], preset_label: str,
              *, variant: str = "A", agc: bool = True, silence_bypass: bool = True,
              f0_mode: Optional[str] = None, record: bool = False) -> SessionStatus:
        import queue as _queue
        import sounddevice as sd

        if self._running:
            self._status.error = "会话已在运行"
            return self._status
        self._preset = dict(PRESETS.get(preset_label, PRESETS["均衡（推荐）"]))
        if f0_mode:
            self._preset["f0_mode"] = f0_mode
        self.record_enabled = record
        self._status = SessionStatus(algo_latency_ms=self._preset.get("algo_latency_ms", 0.0))

        # 引擎必须已加载（由 UI 保证），开始前挂租约
        if not getattr(self.engine, "loaded", False):
            self._status.error = "推理引擎未加载——请先在合成页加载模型"
            return self._status
        self.engine.begin_svc("AI变声实时")

        wasapi = None
        try:
            if hasattr(sd, "WasapiSettings"):
                wasapi = sd.WasapiSettings(auto_convert=True)   # 共享模式（审查修正：exclusive 互斥）
        except Exception:
            wasapi = None

        kwargs = dict(samplerate=22050, blocksize=512, dtype="float32",
                      channels=1, callback=self._audio_cb)
        if wasapi is not None:
            kwargs["extra_settings"] = wasapi

        try:
            self._in_q = _queue.SimpleQueue()
            self._out_ring = _RingBuffer()
            # 参考与档位
            ref = svc_core.build_reference(self.engine, self.ref_path,
                                           float(self._preset.get("ref_seconds", 6.0)))
            scfg = svc_core.StreamConfig(
                block_frames=int(self._preset.get("block_frames", 31)),
                steps=int(self._preset.get("steps", 10)),
                cfg_rate=float(self._preset.get("cfg_rate", 0.0)),
                left_ctx_sec=float(self._preset.get("left_ctx_sec", 2.0)),
                variant=variant)
            self._conv = svc_core.StreamConverter(self.engine, ref, 22050, scfg)
            self._f0 = svc_f0.F0Tracker(22050)
            self._f0_semis = 0
            self._preset["_agc"] = agc
            self._preset["_silence"] = silence_bypass

            if in_dev is not None:
                kwargs["device"] = (in_dev, out_dev)
            elif out_dev is not None:
                kwargs["device"] = (sd.default.device[0], out_dev)
            self._stream = sd.Stream(**kwargs)
            self._stream.start()
            self._dev_sr = 22050
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="vc-realtime")
            self._thread.start()
            self._status.running = True
        except Exception as exc:
            self._cleanup()
            self._status.error = f"打开音频流失败: {exc}"
            self._status.running = False
        return self._status

    def stop(self) -> Optional[str]:
        """停止会话。返回录制文件路径（record 开启且非空），否则 None。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        path = None
        if self.record_enabled and self._record_buf:
            try:
                import soundfile as sf
                import os
                root = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "outputs", "vc"))
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, time.strftime("vc-%Y%m%d-%H%M%S.wav"))
                wav = np.concatenate(self._record_buf)
                sf.write(path, wav, 22050, subtype="PCM_16")
            except Exception:
                path = None
        self._cleanup()
        return path

    def status(self) -> SessionStatus:
        st = SessionStatus(**{k: v for k, v in self._status.__dict__.items()})
        st.running = self._running
        if self._conv is not None:
            st.last_infer_ms = getattr(self._conv, "last_infer_ms", 0.0)
        return st

    def _cleanup(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            self.engine.end_svc("AI变声实时")
        except Exception:
            pass

    # ------------------------------------------------------------------
    _IN_Q_MAX = 200   # ≈4.6s @512/22050；推理慢于实时时丢最旧块防延迟无界

    def _audio_cb(self, indata, outdata, frames, time_info, status_flags):
        """PortAudio 线程：入队输入 / 出队输出（这里不做任何重活）。"""
        chunk = indata[:, 0].copy()
        while self._in_q.qsize() >= self._IN_Q_MAX:
            try:
                self._in_q.get_nowait()
                self._status.dropped += 1
            except Exception:
                break
        self._in_q.put(chunk)
        out, underrun = self._out_ring.get(frames)
        if underrun:
            self._status.underruns += 1
        rms = float(np.sqrt(np.mean(np.square(out)) + 1e-12))
        self._status.level_out_db = 20 * math.log10(rms + 1e-9)
        outdata[:, 0] = out

    def _loop(self):
        """推理线程：in_q → AGC/VAD → StreamConverter → F0 → 淡化 → out_ring。"""
        import queue as _queue
        pending: List[np.ndarray] = []
        f0_pending: List[np.ndarray] = []
        silent_count = 0
        total_count = 0
        try:
            while self._running:
                try:
                    chunk = self._in_q.get(timeout=0.2)
                except _queue.Empty:
                    continue
                total_count += 1
                # 输入调理
                dc = float(np.mean(chunk))
                if abs(dc) > 0.002:
                    chunk = chunk - dc
                rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
                self._status.level_in_db = 20 * math.log10(rms + 1e-9)

                # 轻量 AGC（慢 attack / 慢 release）
                if self._preset.get("_agc", True) and rms > 1e-6:
                    want = AGC_TARGET_RMS / rms
                    self._agc_gain += (min(want, AGC_MAX_GAIN) - self._agc_gain) * 0.05
                    gain = float(np.clip(self._agc_gain, 1.0, AGC_MAX_GAIN))
                    if gain > 1.01:
                        chunk = np.clip(chunk * gain, -0.99, 0.99)

                # VAD 静音直通
                if self._preset.get("_silence", True) and rms < SILENCE_RMS:
                    silent_count += 1
                    self._status.silent_ratio = silent_count / max(total_count, 1)
                    # 静音期也要喂 F0 网格（保持连续）——harvest 全零快，直接清零
                    if self._f0 is not None:
                        f0_pending.append(np.zeros(len(chunk) // 110, dtype=np.float64))
                    continue
                self._status.silent_ratio = silent_count / max(total_count, 1)

                # F0 跟随（与输入同步累积）
                if self._f0 is not None and self._preset.get("f0_mode", "off") != "off":
                    f0_pending.append(self._f0.feed(chunk))

                result = self._conv.feed(chunk)
                if result is None:
                    continue
                wav, overlap = result
                self._status.blocks += 1

                # F0 移植（块级；smooth=语调趋势）
                mode = self._preset.get("f0_mode", "off")
                if mode != "off" and len(f0_pending) and self._f0 is not None:
                    f0_all = np.concatenate(f0_pending)
                    f0_pending = []
                    # 供给帧数≈块+右上下文（不含 overlap 尾巴）——不必强等
                    # 输出长度，world_replace_f0 内部 _interp_to_grid 会补齐网格
                    need = max(len(wav) // 110, 8)
                    seg = f0_all[-need:]
                    if len(seg) >= 8 and np.any(seg > 0):
                        if self._f0_semis == 0 and np.any(seg > 0):
                            # 首个浊音块预计算八度映射（之后恒定）
                            ref_y, ref_sr = _load_quick(self.ref_path)
                            if ref_y is not None:
                                ref_f0 = svc_f0.harvest_f0(
                                    _resample_quick(ref_y, ref_sr, 22050), 22050)
                                self._f0_semis = svc_f0.octave_shift_semis(seg, ref_f0)
                        seg = svc_f0.shift_f0(seg, self._f0_semis)
                        if mode == "smooth":
                            seg = svc_f0.smooth_ratio_curve(seg, 41, 21)  # ~200ms 窗
                        res = svc_f0.world_replace_f0(wav, 22050, seg)
                        if res.ok:
                            wav = res.y

                # 与上块输出交叉淡化（overlap 区等功率混合），写入 out_ring
                if overlap > 0 and self._last_out_tail is not None:
                    fa, fb = svc_f0._equal_power_fade(overlap)
                    if len(wav) >= overlap:
                        wav[:overlap] = wav[:overlap] * fb + self._last_out_tail * fa
                uniq = wav[overlap:] if overlap > 0 else wav
                self._last_out_tail = wav[-overlap:].copy() if overlap > 0 else None
                self._out_ring.put(uniq)
                if self.record_enabled:
                    self._record_buf.append(uniq.copy())
        except Exception as exc:
            self._status.error = f"{type(exc).__name__}: {exc}"
            self._running = False


# ---------------------------------------------------------------------------
# 回环延迟测量（脉冲互相关）
# ---------------------------------------------------------------------------

def measure_roundtrip(in_dev: Optional[int], out_dev: Optional[int],
                      duration: float = 1.2) -> dict:
    """播放脉冲 → 录音 → 互相关求往返延迟。

    需要外部回路（扬声器外放被麦克风收到，或 VB-Cable 直连）。
    返回 {ok, ms, note}。
    """
    if not sd_available():
        return {"ok": False, "error": "sounddevice 未安装"}
    try:
        import sounddevice as sd
        sr = 48000
        n = int(duration * sr)
        # 脉冲序列：每 0.3s 一个 5ms 单极性脉冲
        sig = np.zeros(n, dtype=np.float32)
        for pos in range(0, n - int(0.005 * sr), int(0.3 * sr)):
            k = int(0.005 * sr)
            sig[pos:pos + k] = np.linspace(0.8, -0.8, k, dtype=np.float32)
        rec = np.zeros(n, dtype=np.float32)
        wasapi = None
        try:
            if hasattr(sd, "WasapiSettings"):
                wasapi = sd.WasapiSettings(auto_convert=True)
        except Exception:
            pass
        kwargs = dict(samplerate=sr, blocksize=1024, dtype="float32", channels=1)
        if wasapi is not None:
            kwargs["extra_settings"] = wasapi
        devices = None
        if in_dev is not None or out_dev is not None:
            devices = (in_dev if in_dev is not None else sd.default.device[0],
                       out_dev if out_dev is not None else sd.default.device[1])
        if devices is not None:
            kwargs["device"] = devices

        def _cb(indata, outdata, frames, t, flags):
            i = _cb.i
            take = min(frames, n - i)
            outdata[:take, 0] = sig[i:i + take]
            rec[i:i + take] = indata[:take, 0]
            _cb.i += take
        _cb.i = 0

        with sd.Stream(**kwargs, callback=_cb):
            sd.sleep(int(duration * 1000) + 300)
        # 互相关（限制滞后窗 1.5s）
        corr = np.correlate(rec, sig[:int(0.005 * sr)], mode="valid")
        peak = int(np.argmax(np.abs(corr)))
        ms = peak / sr * 1000.0
        conf = float(np.abs(corr[peak]) / (np.abs(corr).std() + 1e-9))
        if conf < 3.0:
            return {"ok": False, "error":
                    "未检出回环信号（置信度低）——请确认扬声器→麦克风回路/虚拟声卡直连"}
        return {"ok": True, "ms": round(ms, 1), "confidence": round(conf, 1),
                "note": "往返（播放+采集）延迟；单程约一半"}
    except Exception as exc:
        return {"ok": False, "error": f"测量失败: {exc}"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _load_quick(path: str):
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        return np.asarray(y, dtype=np.float32), int(sr)
    except Exception:
        return None, 0


def _resample_quick(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    import librosa
    return np.asarray(librosa.resample(y, orig_sr=sr, target_sr=target), dtype=np.float32)
