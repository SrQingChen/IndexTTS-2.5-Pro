"""推理参数装配。

把界面上的散值组装成 `IndexTTS2.infer()` 的 kwargs。
这里复刻了官方 webui.py `gen_single()` 的情感分支逻辑，
但把模式 3（情感文本）改成**串行**执行以规避 8GB 显存 OOM。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from webui_app import params as P
from webui_app.config import AppConfig
from webui_app.services.engine import EngineError, TTSEngine

EMO_VECTOR_KEYS = [f"emo_vec_{i}" for i in range(8)]


@dataclass
class GenRequest:
    """一次合成请求的全部输入。字段名与 params.REGISTRY 的 key 一致。"""

    # 音色 / 文本
    spk_audio_prompt: Optional[str] = None
    text: str = ""
    lang: str = "ZH"

    # 情感
    emo_control_method: int = 0
    emo_audio_prompt: Optional[str] = None
    emo_alpha: float = 0.65
    emo_vector: List[float] = field(default_factory=lambda: [0.0] * 8)
    emo_text: str = ""
    use_random: bool = False

    # 分句与时长
    max_text_tokens_per_segment: int = 120
    duration_factor: float = 1.0
    interval_silence: int = 200
    text_normalization: bool = True
    seed: int = -1

    # GPT 采样
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    num_beams: int = 3
    repetition_penalty: float = 10.0
    length_penalty: float = 0.0
    max_mel_tokens: int = 1500

    # 输出
    output_path: Optional[str] = None
    verbose: bool = False

    @classmethod
    def from_ui(cls, values: Dict[str, Any], cfg: Optional[AppConfig] = None) -> "GenRequest":
        """从 {key: value} 字典构造，忽略未知键。

        界面上的 8 维情感向量是 emo_vec_0..7 八个独立控件，而 dataclass
        里是一个 emo_vector 列表字段 —— 这里负责把前者归并成后者。
        注意 emo_vec_* 不是 dataclass 字段名，必须先放行再归并，
        否则会被「只保留已知字段」的过滤器提前丢掉。
        """
        known = set(cls.__dataclass_fields__)
        vec_keys = set(EMO_VECTOR_KEYS)
        kw: Dict[str, Any] = {}
        for k, v in values.items():
            if v is None:
                continue
            if k in known or k in vec_keys:
                kw[k] = v

        if "emo_vector" in kw:
            for k in EMO_VECTOR_KEYS:
                kw.pop(k, None)
        else:
            kw["emo_vector"] = [float(kw.pop(k, 0.0) or 0.0) for k in EMO_VECTOR_KEYS]

        if cfg is not None and "verbose" not in kw:
            kw["verbose"] = cfg.verbose
        return cls(**kw)


def resolve_seed(seed: int) -> int:
    """seed=-1 时生成一个新种子。返回实际使用的种子。"""
    import random

    if seed is None or int(seed) < 0:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def apply_seed(seed: int):
    """把种子应用到所有随机源。"""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 情感分支（严格复刻官方语义）
# ---------------------------------------------------------------------------

def resolve_emotion(engine: TTSEngine, req: GenRequest) -> Dict[str, Any]:
    """把界面上的情感设置解析成 infer() 需要的参数。

    官方 `gen_single()` 的四种模式：
        0 → emo_audio_prompt=None，由 infer 内部回退到音色音频，emo_alpha 被强制为 1.0
        1 → emo_audio_prompt=上传的情感音频，emo_alpha 生效（线性插值系数）
        2 → emo_vector=normalize_emo_vec(8维, apply_bias=True)，emo_audio_prompt=None
        3 → QwenEmotion(emo_text) → emo_vector（**不**再套 normalize_emo_vec）

    本实现对模式 3 做了改造：先串行算出向量并立即卸载 QwenEmotion，
    再走与模式 2 相同的下游路径。因为代码里模式 3 本质就是
    「文本 → 8维向量 → emo_vector 路径」，两者下游完全等价，
    但避免了 QwenEmotion 与主推理共存导致的 OOM。
    """
    mode = int(req.emo_control_method or 0)
    out: Dict[str, Any] = {
        "emo_audio_prompt": None,
        "emo_alpha": float(req.emo_alpha),
        "emo_vector": None,
        "use_emo_text": False,
        "emo_text": None,
        "use_random": bool(req.use_random),
    }

    if mode == 0:
        # 跟随音色音频。infer 内部会 emo_audio_prompt = spk_audio_prompt 且 alpha=1.0
        return out

    if mode == 1:
        if not req.emo_audio_prompt:
            raise EngineError(
                "情感控制方式为「使用情感参考音频」，但没有上传情感参考音频。"
            )
        out["emo_audio_prompt"] = req.emo_audio_prompt
        return out

    if mode == 2:
        vec = [float(v or 0.0) for v in (req.emo_vector or [0.0] * 8)]
        if len(vec) < 8:
            vec += [0.0] * (8 - len(vec))
        vec = vec[:8]
        out["emo_vector"] = engine.normalize_emo_vec(vec, apply_bias=True)
        return out

    if mode == 3:
        text = (req.emo_text or "").strip() or (req.text or "").strip()
        if not text:
            raise EngineError("情感描述文本和目标文本都为空，无法推断情感。")
        vec = engine.text_to_emo_vector(text)
        # 官方 mode-3 直接用 QwenEmotion 原始输出，不套 normalize_emo_vec
        out["emo_vector"] = [float(v) for v in vec]
        return out

    raise EngineError(f"未知的情感控制方式: {mode}")


# ---------------------------------------------------------------------------
# 装配 kwargs
# ---------------------------------------------------------------------------

def build_infer_kwargs(
    engine: TTSEngine, req: GenRequest, output_path: str
) -> Dict[str, Any]:
    """组装 IndexTTS2.infer() 的完整 kwargs。"""
    if not req.spk_audio_prompt:
        raise EngineError("请先上传音色参考音频。")
    if not os.path.isfile(req.spk_audio_prompt):
        raise EngineError(f"音色参考音频文件不存在：{req.spk_audio_prompt}")
    if not (req.text or "").strip():
        raise EngineError("目标文本为空。")

    emo = resolve_emotion(engine, req)

    kwargs: Dict[str, Any] = dict(
        spk_audio_prompt=req.spk_audio_prompt,
        text=req.text,
        output_path=output_path,
        verbose=bool(req.verbose),
        max_text_tokens_per_segment=int(req.max_text_tokens_per_segment),
        duration_factor=float(req.duration_factor),
        interval_silence=int(req.interval_silence),
        text_normalization=bool(req.text_normalization),
        # GPT 采样
        do_sample=bool(req.do_sample),
        top_p=float(req.top_p),
        # 官方语义：top_k=0 → None → 禁用该过滤器
        top_k=int(req.top_k) if int(req.top_k) > 0 else None,
        temperature=float(req.temperature),
        num_beams=int(req.num_beams),
        repetition_penalty=float(req.repetition_penalty),
        length_penalty=float(req.length_penalty),
        max_mel_tokens=int(req.max_mel_tokens),
        **emo,
    )
    if engine.cfg.is_v25:
        kwargs["lang"] = (req.lang or "ZH").upper()
    return kwargs


def output_filename(cfg: AppConfig, tag: str = "spk") -> str:
    """生成一个不冲突的输出文件路径。"""
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(cfg.output_dir, f"{tag}_{ts}")
    path = base + ".wav"
    n = 1
    while os.path.exists(path):
        path = f"{base}_{n}.wav"
        n += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def generate(
    engine: TTSEngine,
    req: GenRequest,
    progress=None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """执行一次合成。返回 {path, seconds, seed, kwargs}。

    Raises:
        EngineError: 输入不合法或引擎未加载
    """
    cfg = engine.cfg
    seed = resolve_seed(req.seed)
    apply_seed(seed)

    out = output_path or output_filename(cfg)
    kwargs = build_infer_kwargs(engine, req, out)

    t0 = time.perf_counter()
    result = engine.infer(progress=progress, **kwargs)
    seconds = time.perf_counter() - t0

    path = result if isinstance(result, str) else out
    if not path or not os.path.isfile(path):
        raise EngineError(
            "推理完成但没有产出音频文件。可能是文本为空或被模型提前终止，"
            "试着提高 max_mel_tokens 或减小 max_text_tokens_per_segment。"
        )

    dur = audio_duration(path)
    return {
        "path": path,
        "seconds": seconds,
        "audio_duration": dur,
        "rtf": (seconds / dur) if dur else None,
        "seed": seed,
        "kwargs": kwargs,
    }


def audio_duration(path: str) -> float:
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 预设（复用官方 presets 模块，但字段名对齐 GenRequest）
# ---------------------------------------------------------------------------

PRESET_FIELDS = [
    "emo_control_method", "emo_alpha", "emo_vector", "emo_text", "use_random",
    "max_text_tokens_per_segment", "duration_factor", "interval_silence",
    "text_normalization", "lang", "seed",
    "do_sample", "top_p", "top_k", "temperature", "num_beams",
    "repetition_penalty", "length_penalty", "max_mel_tokens",
]


def preset_from_request(req: GenRequest) -> Dict[str, Any]:
    """把 GenRequest 序列化成可存盘的预设字典。"""
    data = {}
    for f in PRESET_FIELDS:
        v = getattr(req, f, None)
        if f == "emo_vector":
            v = [float(x) for x in (v or [0.0] * 8)]
        data[f] = v
    return data


def request_from_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    """把预设字典还原成 GenRequest 的构造参数（忽略缺失字段）。"""
    out = {}
    for f in PRESET_FIELDS:
        if f in data and data[f] is not None:
            out[f] = data[f]
    vec = out.get("emo_vector")
    if isinstance(vec, list):
        if len(vec) < 8:
            vec = vec + [0.0] * (8 - len(vec))
        out["emo_vector"] = [float(x) for x in vec[:8]]
    return out
