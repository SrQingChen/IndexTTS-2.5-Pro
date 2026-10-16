"""separation — 人声分离 / 去混响（audio-separator 封装，可选 extra）。

设计（dev-plan D8）：
- `audio-separator` 为可选依赖（pip install "audio-separator[gpu]"），未装时
  available()=False，UI 优雅降级（与 audiosr/noisereduce 同模式）。
- 权重自动下载到 checkpoints/separators/（checkpoints 是 artifacts 盘点的
  明文排除区——模型资源归模型页，不进清理页产物盘点）。
- 分离不改时长（与源曲同时间轴），是混音对齐的前提。
- 显存：roformer 2-4GB，与引擎互斥——调用方负责先卸引擎（cover_pipeline
  统一编排），本模块镜像 audio_sr 的显式释放模式。
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "separators")

# 模型选择（调研报告 §3.1/3.2：社区默认 + UVR DeEcho 三件套之一）
VOCALS_MODELS: Dict[str, str] = {
    "标准（BS-Roformer ViperX，SDR 12.98）": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "轻量（Mel-Band Roformer，SDR 11.44）": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "人声专用（vocals_mel_band_roformer）": "vocals_mel_band_roformer.ckpt",
}
DEREVERB_MODELS: Dict[str, str] = {
    "标准（UVR-DeEcho-DeReverb）": "UVR-DeEcho-DeReverb.pth",
    "轻柔（UVR-DeEcho-Normal）": "UVR-DeEcho-Normal.pth",
}
DEFAULT_VOCALS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_DEREVERB = "UVR-DeEcho-DeReverb.pth"

_LOCK = threading.Lock()
_active: List = []          # 持有 Separator 实例，release_all 用


def available() -> bool:
    try:
        from importlib.util import find_spec
        return find_spec("audio_separator") is not None
    except Exception:
        return False


def _vram_free_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def release_all():
    """显式释放分离模型显存（镜像 audio_sr.release_all）。"""
    global _active
    with _LOCK:
        _active.clear()
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _make_separator(out_dir: str):
    from audio_separator.separator import Separator
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return Separator(
        model_file_dir=MODEL_DIR,
        output_dir=out_dir,
        output_format="WAV",
        output_bitdepth="PCM_16",
    )


def _stem_path(files: List[str], keys: List[str]) -> Optional[str]:
    """从 separate() 返回的输出文件名里按关键字找 stem。"""
    for f in files:
        base = os.path.basename(f).lower()
        if any(k.lower() in base for k in keys):
            return f
    return None


def separate_vocals(src_path: str, out_dir: str,
                    model_label: str = "标准（BS-Roformer ViperX，SDR 12.98）",
                    progress=None) -> Dict:
    """人声/伴奏分离。返回 {ok, vocals, instrumental, seconds, model}。"""
    model_name = VOCALS_MODELS.get(model_label, DEFAULT_VOCALS)
    return _run_sep(src_path, out_dir, model_name,
                    vocal_keys=["vocals", "vocal"], inst_keys=["instrumental", "accompaniment", "no_vocals", "other"],
                    progress=progress, both=True)


def dereverb(src_path: str, out_dir: str,
             model_label: str = "标准（UVR-DeEcho-DeReverb）",
             progress=None) -> Dict:
    """去混响/去回声（干声化）。返回 {ok, dry, ...}。"""
    model_name = DEREVERB_MODELS.get(model_label, DEFAULT_DEREVERB)
    return _run_sep(src_path, out_dir, model_name,
                    vocal_keys=["no_reverb", "noreverb", "dry", "vocals", "vocal"],
                    inst_keys=["reverb"],
                    progress=progress, both=False)


def _run_sep(src_path: str, out_dir: str, model_name: str,
             vocal_keys: List[str], inst_keys: List[str],
             progress=None, both: bool = True) -> Dict:
    if not available():
        return {"ok": False, "error": "audio-separator 未安装（pip install \"audio-separator[gpu]\"）"}
    if not os.path.isfile(src_path):
        return {"ok": False, "error": f"文件不存在: {src_path}"}
    # 引擎互斥由调用方保证；这里再做一次显存预检（<1.5GB 提示但继续——audio-separator 会自己回退 CPU）
    free_gb = _vram_free_gb()
    t0 = time.perf_counter()
    try:
        with _LOCK:
            if progress is not None:
                try:
                    progress(0.05, f"加载分离模型 {model_name}（显存空闲 {free_gb:.1f}GB）")
                except Exception:
                    pass
            sep = _make_separator(out_dir)
            _active.append(sep)
            sep.load_model(model_filename=model_name)
            files = sep.separate(src_path)
    except Exception as exc:
        release_all()
        return {"ok": False, "error": f"分离失败: {exc}", "model": model_name}

    seconds = time.perf_counter() - t0
    vocals = _stem_path(files, vocal_keys)
    instrumental = _stem_path(files, inst_keys) if both else None
    ok_v = vocals is not None and os.path.isfile(vocals)
    if not ok_v:
        release_all()
        return {"ok": False, "error": f"未找到人声输出（产物: {files}）",
                "model": model_name, "seconds": round(seconds, 1)}
    return {"ok": True, "vocals": vocals, "instrumental": instrumental,
            "seconds": round(seconds, 1), "model": model_name,
            "all": files}


def model_dir_size_mb() -> float:
    """已下载分离权重的体积（模型页/清理提示用）。"""
    if not os.path.isdir(MODEL_DIR):
        return 0.0
    total = 0
    for root, _dirs, names in os.walk(MODEL_DIR):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total / (1024 ** 2)
