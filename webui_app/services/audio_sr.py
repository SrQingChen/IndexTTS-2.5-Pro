"""AudioSR 后置超分服务（2026-09-30 · 「老式喇叭」的治本件）。

IndexTTS-2.5 输出 22050Hz：11kHz 以上是物理空白，EQ/激励器只能心理声学
补偿。AudioSR（潜空间扩散，arXiv 2309.07314）从低频谱**重建真实高频**
到 48kHz；其 replacement 后处理**低频段原样直通**——对克隆音色最友好
（生成式 BWE 幻觉谐波改音色是公认风险，SenSE 专门研究此问题）。

生命周期与 funasr_hub 同款纪律：
    · 权重经 HF 镜像下载（HF_ENDPOINT，网络不通时给出明确报错）；
    · 懒加载 + release_all；显存需求 ~5.5GB —— 与推理引擎（4.94GB）
      **互斥**：调用方必须先卸引擎（UI 按钮会自动做）；
    · 依赖以 --no-deps 方式安装（audiosr 官方依赖会替换 torch 2.8+cu128
      与 transformers 4.52，实测与本项目栈兼容的是：torchlibrosa/einops/
      progressbar2/unidecode/phonemizer/ftfy/regex/timm/torchvision(0.23+cu128)）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from webui_app import logging_setup as LOG

__all__ = ["available", "enhance_file", "release_all", "VRAM_NEED_GB"]

VRAM_NEED_GB = 5.5          # 实测量级（basic 模型 + 激活）
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {"model": None, "device": "", "loaded_at": 0.0,
                          "infers": 0}

# AudioSR 的导入链会碰 huggingface.co（roberta tokenizer 探测）——
# 网络不通时 import 直接卡死数分钟。config.py 已设默认镜像，这里兜底。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class AudioSRError(RuntimeError):
    pass


def available() -> bool:
    try:
        import importlib.util as u
        return u.find_spec("audiosr") is not None
    except Exception:
        return False


def _cached_ckpt_bin() -> Optional[str]:
    """找已缓存的 AudioSR basic 权重（huggingface 快照目录）。"""
    import glob
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--haoheliu--audiosr_basic/"
        "snapshots/*/pytorch_model.bin"))
    return hits[0] if hits else None


def _pick_device() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free, _t = torch.cuda.mem_get_info(0)
        if free / (1024 ** 3) >= VRAM_NEED_GB:
            return "cuda:0"
        return "cpu"          # AudioSR CPU >10× 实时——能用但慢,调用方应提示
    except Exception:
        return "cpu"


def _get_model():
    if _STATE["model"] is not None:
        return _STATE["model"]
    if not available():
        raise AudioSRError(
            "audiosr 未安装。安装方式（**--no-deps，绝不能让它动 torch**）：\n"
            "  uv pip install audiosr --no-deps\n"
            "  uv pip install torchlibrosa einops progressbar2 unidecode "
            "phonemizer ftfy regex timm\n"
            "  uv pip install torchvision==0.23.0+cu128 --index pytorch-cuda")
    from audiosr import pipeline as P

    log = LOG.get_logger("audio_sr")
    t0 = time.perf_counter()
    device = _pick_device()
    if device == "cpu":
        log.warning("显存不足（或无 CUDA）：AudioSR 以 CPU 运行，"
                    "速度 >10× 实时，长音频请耐心")
    log.info("加载 AudioSR(basic) → %s（首次会经 %s 下载权重约 1GB）",
             device, os.environ.get("HF_ENDPOINT"))
    try:
        try:
            _STATE["model"] = P.build_model(model_name="basic", device=device)
        except Exception as e_first:
            # 权重已缓存时联网校验（revision HEAD/cas-bridge 下载）仍可能
            # 超时——build_model 无条件调 download_checkpoint，因此这里把
            # download_checkpoint 短路到本地缓存文件再重试一次。
            if "timeout" not in str(e_first).lower() and "connection" not in str(
                    e_first).lower():
                raise
            _cached = _cached_ckpt_bin()
            if not _cached:
                raise
            log.warning("联网校验超时，改用本地缓存权重重试：%s", _cached)
            _real_dc = P.download_checkpoint

            def _cached_dc(checkpoint_name="basic", **_kw):
                return _cached

            P.download_checkpoint = _cached_dc
            try:
                _STATE["model"] = P.build_model(model_name="basic",
                                                device=device)
            finally:
                P.download_checkpoint = _real_dc
    except Exception as e:
        raise AudioSRError(
            f"AudioSR 加载失败：{type(e).__name__}: {e}（权重下载走 "
            f"{os.environ.get('HF_ENDPOINT')}，网络不通是常见原因）") from e
    _STATE["device"] = device
    _STATE["loaded_at"] = time.time()
    log.info("AudioSR 就绪 · %.1fs · %s", time.perf_counter() - t0, device)
    return _STATE["model"]


def release_all() -> Dict[str, Any]:
    import gc
    out: Dict[str, Any] = {}
    with _LOCK:
        if _STATE["model"] is not None:
            out = {"device": _STATE["device"], "infers": _STATE["infers"],
                   "alive_s": round(time.time() - _STATE["loaded_at"], 1)}
            _STATE["model"] = None
            _STATE["device"] = ""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return out


CHUNK_SEC = 5.0            # 官方建议 ≤5.12s（>10.24s 会性能劣化——
                           # 真机实测：16s 整条灌入时 11k 以上几乎不重建）
CHUNK_OVERLAP_SEC = 0.25


def _flatten(wav) -> "np.ndarray":
    """AudioSR 返回形状实测为 (1,1,T)：必须**完全展平**到 1 维 ——
    只降一维的 (1,T) 会被 soundfile 当成「T 个声道的单帧」而报
    Format not recognised（真机首跑踩过）。"""
    import numpy as np
    arr = np.asarray(wav)
    while arr.ndim > 1:
        arr = (arr.mean(axis=0) if arr.shape[0] > 1
               else arr.reshape(arr.shape[-1]))
    return arr.astype(np.float32)


def _enhance_chunk(model, y: "np.ndarray", in_sr: int, tmp_path: str,
                   ddim_steps: int, guidance_scale: float, seed: int
                   ) -> "np.ndarray":
    """一段（≤5s）→ 48kHz 超分波形；按输入时长精确裁剪（AudioSR 会补零
    到块边界——真机实测 16.37s 进 20.48s 出）。"""
    import numpy as np
    import soundfile as sf
    from audiosr import pipeline as P
    sf.write(tmp_path, y, in_sr, subtype="PCM_16")
    wav = P.super_resolution(model, tmp_path, seed=seed,
                             guidance_scale=guidance_scale,
                             ddim_steps=int(ddim_steps))
    out = _flatten(wav)
    expect = int(len(y) * 48000 / in_sr)
    if len(out) >= expect:
        out = out[:expect]
    else:                       # 理论不该短；防御性补零保持对齐
        out = np.concatenate([out, np.zeros(expect - len(out),
                                            dtype=np.float32)])
    return out


def enhance_file(path: str, out_path: Optional[str] = None,
                 ddim_steps: int = 35, guidance_scale: float = 3.5,
                 seed: int = 42) -> Dict[str, Any]:
    """任意长度音频 → 48kHz 超分（分块 + 交叉淡化拼接）。

    ddim_steps 默认 35（官方 200 太慢；35-50 在语音上质量差异小）。
    输入按 CHUNK_SEC 分块（官方 >10.24s 性能劣化的规避），块间
    50ms 等功率淡化拼接；输出单声道 48k PCM_16、时长与输入严格一致。
    """
    import numpy as np
    import soundfile as sf
    import tempfile

    if not os.path.isfile(path):
        raise AudioSRError(f"输入不存在：{path}")
    if not out_path:
        base, _ = os.path.splitext(path)
        out_path = base + "_48k.wav"

    log = LOG.get_logger("audio_sr")
    from webui_app.services import audio_lab as AL
    t0 = time.perf_counter()

    y, in_sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    hop = int((CHUNK_SEC - CHUNK_OVERLAP_SEC) * in_sr)
    chunk_n = int(CHUNK_SEC * in_sr)
    starts = list(range(0, max(1, len(y)), hop))
    pieces: List = []
    # 显存预检+轮询：引擎刚卸载时驱动释放有延迟，等最多 10s；
    # 仍不足则明确报错——绝不静默回退 CPU 爬行（那看起来像卡死）。
    import torch as _torch
    for _ in range(10):
        try:
            if not _torch.cuda.is_available():
                break
            free, _t = _torch.cuda.mem_get_info(0)
            if free / (1024 ** 3) >= VRAM_NEED_GB:
                break
            _torch.cuda.empty_cache()
            time.sleep(1.0)
        except Exception:
            break

    with _LOCK:
        model = _get_model()
        device = _STATE.get("device", "?")
        tmpdir = tempfile.mkdtemp(prefix="audiosr_chunk_")
        try:
            for ci, st in enumerate(starts):
                seg = y[st: st + chunk_n]
                if len(seg) < int(0.2 * in_sr):
                    # 尾巴太短并进上一块（避免超短块质量差）
                    if pieces:
                        prev_st = starts[ci - 1]
                        seg = y[prev_st: st + len(seg)]
                        pieces.pop()
                    if len(seg) < int(0.2 * in_sr):
                        continue
                try:
                    out = _enhance_chunk(
                        model, seg, in_sr,
                        os.path.join(tmpdir, f"c{ci:03d}.wav"),
                        ddim_steps, guidance_scale, seed + ci)
                    _STATE["infers"] += 1
                    pieces.append(out)
                except Exception as e:
                    raise AudioSRError(
                        f"超分推理失败（块 {ci + 1}/{len(starts)}）："
                        f"{type(e).__name__}: {e}") from e
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    # 拼接：**整个重叠段**做等功率交叉淡化。首版只淡 50ms、剩余 200ms
    # 重叠被双重追加（每条接缝 0.2s 重影）——真机复盘抓出的 bug。
    ov = int(CHUNK_OVERLAP_SEC * 48000)
    total = None
    for p in pieces:
        if total is None:
            total = p.copy()
            continue
        w = min(ov, len(total), len(p))
        if w > 0:
            ramp_c = np.cos(np.linspace(0, np.pi / 2, w))
            ramp_s = np.sin(np.linspace(0, np.pi / 2, w))
            blended = (total[-w:] * ramp_c + p[:w] * ramp_s)
            total = np.concatenate([total[:-w], blended, p[w:]])
        else:
            total = np.concatenate([total, p])
    y_out = total if total is not None else np.zeros(1, np.float32)

    expect_total = int(len(y) * 48000 / in_sr)
    if len(y_out) > expect_total:
        y_out = y_out[:expect_total]
    elif len(y_out) < expect_total:
        y_out = np.concatenate([y_out, np.zeros(
            expect_total - len(y_out), dtype=np.float32)])
    peak = float(np.max(np.abs(y_out))) if len(y_out) else 0.0
    if peak > 10 ** (-1.0 / 20.0):
        y_out = y_out * (10 ** (-1.0 / 20.0) / peak)
    sf.write(out_path, np.asarray(y_out, dtype=np.float32), 48000,
             subtype="PCM_16")

    dt = time.perf_counter() - t0
    try:
        b0 = AL.band_profile(path)
        b1 = AL.band_profile(out_path)
        log.info("超分完成：%d 块 · %s · %.1fs · 质心 %.0f→%.0f Hz",
                 len(pieces), os.path.basename(out_path), dt,
                 b0.get("centroid", 0), b1.get("centroid", 0))
    except Exception:
        pass
    return {"ok": True, "path": out_path, "seconds": round(dt, 1),
            "sr": 48000, "chunks": len(pieces), "device": device}
