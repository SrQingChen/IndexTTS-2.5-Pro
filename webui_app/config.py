"""全局配置与路径管理。

所有路径、运行时开关、设备探测集中在此，其他模块只读不写，
避免官方 webui.py 里那种散落各处的模块级全局变量。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 官方代码依赖 cwd 为项目根目录（i18n 用 os.path.relpath、infer_v2_5 写死
# './checkpoints/hf_cache'），这里统一兜底。
if os.getcwd() != PROJECT_ROOT:
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass

for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "indextts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows 控制台中文输出兜底
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _d(*parts) -> str:
    p = os.path.join(PROJECT_ROOT, *parts)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SUPPORTED_VERSIONS = ("2", "2.5")

# v2.5 支持的语言（顺序即下拉框顺序）
LANGUAGES_V25 = ["ZH", "EN", "JA", "AR", "ES"]
LANGUAGES_V2 = ["ZH", "EN"]

LANGUAGE_LABELS = {
    "ZH": "中文 (ZH)",
    "EN": "英语 (EN)",
    "JA": "日语 (JA)",
    "AR": "阿拉伯语 (AR)",
    "ES": "西班牙语 (ES)",
}

# 情感控制方式（索引与官方 webui.py 严格一致，勿改顺序）
EMO_MODE_SPEAKER = 0
EMO_MODE_AUDIO = 1
EMO_MODE_VECTOR = 2
EMO_MODE_TEXT = 3

# 8 维情感向量的语义顺序（与 feat1.pt/feat2.pt 的 emo_num 切分对应）
EMO_VECTOR_LABELS = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]
EMO_VECTOR_KEYS = [
    "happy", "angry", "sad", "afraid",
    "disgusted", "melancholic", "surprised", "calm",
]
# config.yaml 中 emo_num: [3, 17, 2, 8, 4, 5, 10, 24]
EMO_MATRIX_SPLITS = [3, 17, 2, 8, 4, 5, 10, 24]

# 官方 infer_v2_5.normalize_emo_vec 里的情感偏置系数
EMO_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
EMO_SUM_LIMIT = 0.8

# 低于该显存自动进入省显存模式（与官方 webui.py 阈值一致）
LOW_VRAM_THRESHOLD_GB = 10.0

OUTPUT_SAMPLE_RATE = 22050

# 参考音频的推荐规格（L0 工作台据此打分）
REF_AUDIO_IDEAL = {
    "min_sec": 3.0,
    "ideal_min_sec": 8.0,
    "ideal_max_sec": 15.0,
    "max_sec": 15.0,        # 官方 _load_and_cut_audio 硬截断到 15s
    "min_sr": 16000,
    "ideal_sr": 22050,
    "min_snr_db": 15.0,
    "ideal_snr_db": 25.0,
    "target_lufs": -20.0,
    "max_clip_ratio": 0.001,
    "max_silence_ratio": 0.30,
}


# ---------------------------------------------------------------------------
# 设备探测
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """GPU / 设备探测结果，只读。"""

    backend: str = "cpu"            # cuda / xpu / mps / cpu
    device_str: str = "cpu"         # 传给 IndexTTS2 的 device
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cuda_version: str = ""
    torch_version: str = ""
    bf16_supported: bool = False
    low_vram: bool = False

    @property
    def is_cuda(self) -> bool:
        return self.backend == "cuda"

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "device": self.device_str,
            "gpu": self.gpu_name,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_free_gb": round(self.vram_free_gb, 2),
            "cuda": self.cuda_version,
            "torch": self.torch_version,
            "bf16": self.bf16_supported,
            "low_vram": self.low_vram,
        }


def detect_device() -> DeviceInfo:
    """探测可用加速设备与显存，不加载任何模型。"""
    import torch

    info = DeviceInfo(torch_version=torch.__version__)
    if torch.cuda.is_available():
        info.backend = "cuda"
        info.device_str = "cuda:0"
        props = torch.cuda.get_device_properties(0)
        info.gpu_name = props.name
        info.vram_total_gb = props.total_memory / (1024 ** 3)
        free, _total = torch.cuda.mem_get_info(0)
        info.vram_free_gb = free / (1024 ** 3)
        info.cuda_version = torch.version.cuda or ""
        info.bf16_supported = torch.cuda.is_bf16_supported()
        info.low_vram = info.vram_total_gb < LOW_VRAM_THRESHOLD_GB
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        info.backend = "xpu"
        info.device_str = "xpu"
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        info.backend = "mps"
        info.device_str = "mps"
    else:
        info.backend = "cpu"
        info.device_str = "cpu"
    return info


def refresh_vram_free(info: DeviceInfo) -> float:
    """刷新并返回当前空闲显存（GB）。非 CUDA 返回 0。"""
    if not info.is_cuda:
        return 0.0
    import torch

    try:
        free, _total = torch.cuda.mem_get_info(0)
        info.vram_free_gb = free / (1024 ** 3)
    except Exception:
        pass
    return info.vram_free_gb


# ---------------------------------------------------------------------------
# 应用配置
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """WebUI 的完整配置。由 CLI 参数或默认值构造，运行期只读。"""

    # 服务
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    concurrency: int = 20

    # 模型
    version: str = "2.5"
    model_dir: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, "checkpoints"))

    # 精度与加速
    fp16: bool = False            # v2 用 fp16；v2.5 会自动转成 bf16
    deepspeed: bool = False
    cuda_kernel: bool = False
    accel: bool = False
    torch_compile: bool = False

    # 显存策略
    force_qwen_emo: bool = False  # 低显存下仍强制加载 QwenEmotion
    autoload_engine: bool = True  # 启动即加载模型（False 则由 UI 手动加载）

    # 生成默认值
    gui_seg_tokens: int = 120
    verbose: bool = False

    # 训练（阶段 2 使用）
    dataset_dir: str = field(default_factory=lambda: _d("datasets"))
    lora_dir: str = field(default_factory=lambda: _d("outputs", "lora"))
    train_dir: str = field(default_factory=lambda: _d("outputs", "train"))

    # 派生（构造后填充）
    device: DeviceInfo = field(default_factory=DeviceInfo)

    def __post_init__(self):
        self.device = detect_device()
        self.output_dir = _d("outputs")
        self.tasks_dir = _d("outputs", "tasks")
        self.voice_bank_dir = _d("voice_bank")
        self.cache_dir = _d("outputs", "cache")
        self.examples_dir = os.path.join(PROJECT_ROOT, "examples")

        self.is_v25 = self.version == "2.5"
        self.languages = LANGUAGES_V25 if self.is_v25 else LANGUAGES_V2

        # 半精度：显式开启 或 低显存自动开启
        self.half_precision = self.fp16 or self.device.low_vram
        # QwenEmotion：显式开启 或 非低显存
        self.load_qwen_emo = self.force_qwen_emo or not self.device.low_vram

        self.cfg_path = os.path.join(self.model_dir, "config.yaml")

    # -- 便捷查询 ----------------------------------------------------------

    @property
    def use_bf16(self) -> bool:
        """v2.5 走 bf16（需硬件支持），v2 走 fp16。"""
        if not self.half_precision:
            return False
        return self.is_v25 and self.device.bf16_supported

    @property
    def use_fp16(self) -> bool:
        return self.half_precision and not (self.is_v25 and self.device.bf16_supported)

    def engine_kwargs(self) -> dict:
        """构造 IndexTTS2 的关键字参数。"""
        kwargs = dict(
            model_dir=self.model_dir,
            cfg_path=self.cfg_path,
            use_deepspeed=self.deepspeed,
            use_cuda_kernel=self.cuda_kernel,
            use_accel=self.accel,
            use_torch_compile=self.torch_compile,
            use_qwen_emo=False,   # QwenEmotion 一律按需挂载，见 services/engine.py
        )
        if self.is_v25:
            kwargs["use_bf16"] = self.use_bf16
        else:
            kwargs["use_fp16"] = self.use_fp16
        return kwargs

    def startup_notes(self) -> List[str]:
        """启动时的环境提示，会显示在 UI 的系统页。"""
        notes = []
        d = self.device
        if d.backend == "cpu":
            notes.append("未检测到可用 GPU，将以 CPU 模式运行，速度会很慢。")
        elif d.low_vram:
            notes.append(
                f"检测到 {d.vram_total_gb:.1f} GB 显存（< {LOW_VRAM_THRESHOLD_GB:.0f} GB），"
                "已自动启用半精度并开启长文本分块。"
            )
        if self.half_precision and self.is_v25 and not d.bf16_supported:
            notes.append("当前 GPU 不支持 BF16，已回退到全精度推理。")
        if not self.load_qwen_emo:
            notes.append(
                "QwenEmotion 未常驻加载（低显存策略）。情感文本控制会在首次使用时"
                "按需加载，用完自动释放。"
            )
        if self.accel:
            notes.append("已启用 GPT2 加速引擎（需 flash-attn）。")
        if self.torch_compile:
            notes.append("已启用 torch.compile 优化 s2mel（首次推理会额外编译耗时）。")
        return notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webui_pro.py",
        description="IndexTTS 模块化 WebUI（推理 + 参考音频工作台 + LoRA/DPO 训练 + 评测）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("服务")
    g.add_argument("--host", default="127.0.0.1", help="监听地址；0.0.0.0 可局域网访问")
    g.add_argument("--port", type=int, default=7860, help="监听端口")
    g.add_argument("--share", action="store_true", help="生成 gradio.live 公网临时链接")
    g.add_argument("--concurrency", type=int, default=20, help="任务队列并发上限")

    g = p.add_argument_group("模型")
    g.add_argument("--model_dir", default=None, help="模型目录，默认 ./checkpoints")
    g.add_argument("--version", default="2.5", choices=list(SUPPORTED_VERSIONS),
                   help="模型版本")

    g = p.add_argument_group("精度与加速")
    g.add_argument("--fp16", action="store_true",
                   help="强制半精度（v2.5 上若硬件支持会自动用 bf16）")
    g.add_argument("--deepspeed", action="store_true", help="启用 DeepSpeed 推理加速")
    g.add_argument("--cuda_kernel", action="store_true",
                   help="启用 BigVGAN 融合激活 CUDA kernel")
    g.add_argument("--accel", action="store_true",
                   help="启用 GPT2 加速引擎（需 flash-attn）")
    g.add_argument("--torch_compile", action="store_true",
                   help="对 s2mel 启用 torch.compile（需 triton）")

    g = p.add_argument_group("显存策略")
    g.add_argument("--qwen_emo", action="store_true", dest="force_qwen_emo",
                   help="低显存下也允许情感文本控制（按需加载，非常驻）")
    g.add_argument("--no-qwen-emo", action="store_true",
                   help="彻底禁用情感文本控制入口")
    g.add_argument("--lazy", action="store_true",
                   help="启动时不加载模型，由 UI 手动点击加载")

    g = p.add_argument_group("生成")
    g.add_argument("--gui_seg_tokens", type=int, default=120,
                   help="分句最大 Token 数的界面默认值")
    g.add_argument("--verbose", action="store_true", help="打印详细推理日志")

    return p


def config_from_args(argv: Optional[List[str]] = None) -> AppConfig:
    args = build_arg_parser().parse_args(argv)
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    if kwargs.get("no_qwen_emo"):
        kwargs["force_qwen_emo"] = False
    kwargs.pop("no_qwen_emo", None)
    if kwargs.get("lazy"):
        kwargs["autoload_engine"] = False
    kwargs.pop("lazy", None)
    if not kwargs.get("model_dir"):
        kwargs["model_dir"] = os.path.join(PROJECT_ROOT, "checkpoints")
    return AppConfig(**kwargs)
