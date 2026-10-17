"""svc_state — 翻唱/变声参数记忆（独立文件，复用 synth_state 的原子写/钳制模式）。

刻意不并入 synthesis_state.json：那份文件有 56 控件位置解包硬校验，
新增键会破坏合成页的自检（阶段37 设计）。
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STATE_DIR = os.path.join(PROJECT_ROOT, "outputs", "state")

COVER_STATE = os.path.join(STATE_DIR, "cover_state.json")
VC_STATE = os.path.join(STATE_DIR, "vc_state.json")

_LOCK = threading.Lock()

# 白名单（键 -> 类型）；数值钳制
_COVER_KEYS: Dict[str, str] = {
    "input_state": "str", "separate_model": "str", "dereverb": "bool",
    "denoise": "float", "variant": "str", "steps": "int", "cfg": "float",
    "ref_seconds": "float", "chunk_sec": "float", "f0_mode": "str",
    "transpose_auto": "bool", "transpose_semis": "int",
    "key_mode": "str", "key_semitones": "int", "key_output": "str",
    "key_ceiling": "int",
    "hf_deess": "bool", "hf_limit": "bool", "spectral_match": "bool",
    "breath_soften": "bool", "highpass": "int", "sr_enable": "bool",
    "sr_steps": "int", "auto_mix": "bool", "vocal_gain": "float",
    "loudnorm": "bool", "lora_run": "str", "lora_ckpt": "str",
    "lora_scale": "float", "ref_voice": "str", "bgm_path": "str",
    # E1-T 歌声模型（阶段 47）。教训同阶段37的「白名单静默丢键」：cover_tab
    # 在保存这些键，白名单没有它们 = 重启后全部回到默认、用户以为没存上。
    "singing_enable": "bool", "technique_mode": "str",
    "technique_manual": "int", "bon_high": "bool", "hf_transplant": "bool",
    "register_mode": "str", "octave_fix": "bool",
}
_VC_KEYS: Dict[str, str] = {
    "preset": "str", "agc": "bool", "silence_bypass": "bool", "record": "bool",
    "in_device": "int", "out_device": "int", "ref_voice": "str",
    "lora_run": "str", "lora_ckpt": "str", "lora_scale": "float",
    "variant": "str",
}

_CLAMP = {
    "denoise": (0.0, 1.0), "cfg": (0.0, 1.0), "ref_seconds": (5.0, 15.0),
    "chunk_sec": (10.0, 60.0), "transpose_semis": (-24, 24),
    "key_semitones": (-12, 0), "key_ceiling": (450, 800),
    "highpass": (0, 150), "sr_steps": (10, 100), "vocal_gain": (-6.0, 9.0),
    "steps": (5, 50), "lora_scale": (0.0, 2.0),
}


def sanitize(values: Dict[str, Any], keys: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, kind in keys.items():
        if k not in values:
            continue
        v = values[k]
        try:
            if kind == "bool":
                out[k] = bool(v)
            elif kind == "int":
                out[k] = int(v)
            elif kind == "float":
                out[k] = float(v)
            else:
                out[k] = str(v or "")
            if k in _CLAMP:
                lo, hi = _CLAMP[k]
                cur = out[k]
                out[k] = max(lo, min(hi, cur)) if isinstance(cur, (int, float)) else cur
        except (TypeError, ValueError):
            continue
    return out


def _atomic_write(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save(path: str, values: Dict[str, Any], keys: Dict[str, str]):
    with _LOCK:
        _atomic_write(path, sanitize(values, keys))


def load(path: str, keys: Dict[str, str]) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sanitize(json.load(f), keys)
    except Exception:
        return {}


def save_cover(values: Dict[str, Any]):
    save(COVER_STATE, values, _COVER_KEYS)


def load_cover() -> Dict[str, Any]:
    return load(COVER_STATE, _COVER_KEYS)


def save_vc(values: Dict[str, Any]):
    save(VC_STATE, values, _VC_KEYS)


def load_vc() -> Dict[str, Any]:
    return load(VC_STATE, _VC_KEYS)
