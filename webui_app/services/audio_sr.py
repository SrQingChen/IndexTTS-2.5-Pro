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
        _STATE["model"] = P.build_model(model_name="basic", device=device)
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


def enhance_file(path: str, out_path: Optional[str] = None,
                 ddim_steps: int = 35, guidance_scale: float = 3.5,
                 seed: int = 42) -> Dict[str, Any]:
    """一条 22.05k 音频 → 48kHz 超分。返回 {ok, path, seconds, sr}。

    ddim_steps 默认 35（官方 200 太慢；35-50 在语音上质量差异小、
    速度快 4-6 倍）。输出单声道 48k PCM_16。
    """
    import numpy as np
    import soundfile as sf

    if not os.path.isfile(path):
        raise AudioSRError(f"输入不存在：{path}")
    if not out_path:
        base, _ = os.path.splitext(path)
        out_path = base + "_48k.wav"

    log = LOG.get_logger("audio_sr")
    from webui_app.services import audio_lab as AL
    t0 = time.perf_counter()
    with _LOCK:
        model = _get_model()
        try:
            from audiosr import pipeline as P
            wav = P.super_resolution(
                model, path, seed=seed, guidance_scale=guidance_scale,
                ddim_steps=int(ddim_steps))
            _STATE["infers"] += 1
        except Exception as e:
            raise AudioSRError(
                f"超分推理失败：{type(e).__name__}: {e}") from e

    arr = np.asarray(wav)
    if arr.ndim == 1:
        y = arr.astype(np.float32)
    else:
        y = arr.mean(axis=0).astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 10 ** (-1.0 / 20.0):
        y = y * (10 ** (-1.0 / 20.0) / peak)
    sf.write(out_path, y, 48000, subtype="PCM_16")

    dt = time.perf_counter() - t0
    try:
        b0 = AL.band_profile(path)
        b1 = AL.band_profile(out_path)
        log.info("超分完成：%s · %.1fs · 质心 %.0f→%.0f Hz · 3k+ %.1f%%→%.1f%%",
                 os.path.basename(out_path), dt,
                 b0.get("centroid", 0), b1.get("centroid", 0),
                 sum(b0.get("bands", [])[3:]), sum(b1.get("bands", [])[3:]))
    except Exception:
        pass
    return {"ok": True, "path": out_path, "seconds": round(dt, 1),
            "sr": 48000}
