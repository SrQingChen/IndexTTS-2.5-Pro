"""TTS 引擎生命周期管理。

职责：
    · 懒加载 / 卸载主模型，显存可控
    · QwenEmotion 的**串行**挂载策略（不与主推理共存，规避 8GB OOM）
    · LoRA adapter 的挂载 / 卸载 / 合并（阶段 2）
    · 线程安全（Gradio 并发请求共享同一个引擎实例）
    · 运行时统计（加载耗时、推理次数、显存峰值）

关键实测数据（RTX 4060 Laptop 8GB, BF16）：
    常驻显存 4.94 GB，推理峰值 5.24 GB，跑完仅剩 1.34 GB 空闲。
    QwenEmotion(fp16) 需 1.2 GB —— 与主推理**共存必然 OOM**，
    因此这里采用「挂载 → 算向量 → 立即卸载 → 走 emo_vector 路径推理」。
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from webui_app import logging_setup as LOG
from webui_app.config import AppConfig, refresh_vram_free


# ---------------------------------------------------------------------------
# QwenEmotion 的安全加载器
# ---------------------------------------------------------------------------

class QwenEmotionSafe:
    """官方 `QwenEmotion` 的替身，复用其全部解析逻辑，只修掉加载方式的三个坑。

    官方实现的问题（见 params_inference.py 的 emo_text 条目）：
      1. `device_map="auto"` —— 显存不足时 accelerate 会**静默**把层 offload
         到 CPU，推理慢几十倍且不报错。这里改为显式指定 device。
      2. `torch_dtype="float16"` —— 新版 transformers 已改名 `dtype`，会告警。
      3. `max_new_tokens=32768` —— 模型跑飞会无限生成，没有上限保护。
    """

    # 情感 JSON 很短，64 token 绰绰有余；留到 256 以防模型输出冗余文本
    MAX_NEW_TOKENS = 256

    def __init__(self, model_dir: str, device: str = "cuda:0", dtype: str = "float16"):
        import torch
        from modelscope import AutoModelForCausalLM
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.device = device
        self._torch = torch
        self.last_latency = 0.0
        self.last_raw: Any = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # 先载到 CPU 再显式 .to(device)，完全绕开 accelerate 的 device_map 分派。
        # 官方用 device_map="auto"，显存不足时它会静默把部分层 offload 到 CPU，
        # 推理慢几十倍且不报错。这里用确定性的两步加载代替。
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
        except TypeError:
            # 老版 transformers 仍用 torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=dtype,
            )
        self.model = self.model.to(device)
        self.model.eval()

        # 以下常量与官方 QwenEmotion 完全一致，保证输出语义相同
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid",
            "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = [
            "高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然",
        ]
        self.melancholic_words = {
            "低落", "melancholy", "melancholic", "depression", "depressed", "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    # -- 以下三个方法逐字复用官方逻辑 -------------------------------------

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def normalize_content(self, content):
        if isinstance(content, dict):
            normalized = dict(content)
        else:
            normalized = {}

        def label_to_cn_key(value):
            if not isinstance(value, str):
                return None
            value = value.strip()
            if value in self.cn_key_to_en:
                return value
            value_lower = value.lower()
            for cn_key, en_key in self.cn_key_to_en.items():
                if value_lower == en_key:
                    return cn_key
            return None

        detected_key = label_to_cn_key(content) if isinstance(content, str) else None
        if detected_key is None:
            for alias in ("emotion", "emotion_label", "label", "情感", "情绪"):
                detected_key = label_to_cn_key(normalized.get(alias))
                if detected_key is not None:
                    break
        if detected_key is not None and all(
            key not in normalized for key in self.desired_vector_order
        ):
            normalized[detected_key] = 1.0

        for cn_key in self.desired_vector_order:
            detected_key = label_to_cn_key(normalized.get(cn_key))
            if detected_key is not None:
                normalized[cn_key] = 1.0 if detected_key == cn_key else 0.0
                if detected_key != cn_key:
                    normalized[detected_key] = 1.0
        return normalized

    def convert(self, content):
        content = self.normalize_content(content)
        emotion_dict = {
            self.cn_key_to_en[k]: self.clamp_score(content.get(k, 0.0))
            for k in self.desired_vector_order
        }
        if all(v <= 0.0 for v in emotion_dict.values()):
            emotion_dict["calm"] = 1.0
        return emotion_dict

    # -- 推理（加了长度上限保护）-------------------------------------------

    def inference(self, text_input: str) -> Dict[str, float]:
        import json
        import re

        torch = self._torch
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"{text_input}"},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 某些 tokenizer 版本不接受 enable_thinking
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0][len(inputs.input_ids[0]):].tolist()

        # 官方用 rindex 找 151668 (</think>)；找不到就从头开始
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }

        # 「低落 / 悲伤」辨析的官方 workaround
        low = text_input.lower()
        if any(w in low for w in self.melancholic_words):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

        result = self.convert(content)
        self.last_latency = time.perf_counter() - start
        self.last_raw = content
        return result


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

@dataclass
class EngineStats:
    loaded: bool = False
    load_seconds: float = 0.0
    loaded_at: float = 0.0
    infer_count: int = 0
    infer_total_seconds: float = 0.0
    last_infer_seconds: float = 0.0
    vram_alloc_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    vram_peak_gb: float = 0.0
    vram_free_gb: float = 0.0
    qwen_mounted: bool = False
    qwen_load_seconds: float = 0.0
    qwen_infer_count: int = 0
    lora_adapters: List[str] = field(default_factory=list)
    error: str = ""
    notes: List[str] = field(default_factory=list)


class EngineError(RuntimeError):
    """引擎状态错误（未加载、显存不足等），UI 应捕获并友好提示。"""


def _busy_runner_engine_req() -> str:
    """有训练类后台任务在跑时，返回它对引擎的要求（loaded/unloaded）。

    没有任务在跑（或 runner 不可用）返回 ""。这是 runner ↔ engine
    的双向互斥：runner 在 submit 门口检查引擎状态，engine 在
    load/unload 门口反查 runner —— 只拦一边，训练跑着的几十分钟里
    用户仍能从合成页把 5GB 的引擎加载回来，WDDM 下显存互踩静默降速
    20~30 倍且不报错。
    """
    try:
        from webui_app.training.runner import get_runner  # 延迟导入避免环
        r = get_runner()
    except Exception:
        return ""
    if getattr(r, "running", False):
        return str(getattr(r, "engine_req", "") or "")
    return ""


class TTSEngine:
    """线程安全的引擎封装。全应用只应存在一个实例（由 AppContext 持有）。"""

    # QwenEmotion fp16 权重 1.14GB + KV cache/激活 ~0.2GB
    QWEN_VRAM_REQUIRE_GB = 1.4

    def __init__(self, config: AppConfig):
        self.cfg = config
        self._lock = threading.RLock()
        self._tts = None
        self._qwen: Optional[QwenEmotionSafe] = None
        self.stats = EngineStats()
        self._on_event: Optional[Callable[[str, str], None]] = None
        # emo_text -> 8维向量。QwenEmotion 每次挂卸要 1.7s + 1.2GB 显存，
        # 同一段情感描述反复用（调参、批量合成）时缓存收益很大。
        self._emo_cache: Dict[str, List[float]] = {}
        self.qwen_device_pref = "auto"   # auto | cuda | cpu

    # -- 事件回调（供 UI 显示进度）-----------------------------------------

    def set_event_handler(self, fn: Optional[Callable[[str, str], None]]):
        self._on_event = fn

    def _emit(self, event: str, message: str = ""):
        if self._on_event:
            try:
                self._on_event(event, message)
            except Exception:
                pass

    # -- 基础状态 ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    @property
    def tts(self):
        if self._tts is None:
            raise EngineError(
                "模型尚未加载。请到「引擎控制」或页面顶部点击「加载模型」。"
            )
        return self._tts

    def require(self):
        """显式断言已加载，返回 tts 实例。"""
        return self.tts

    # -- 加载 / 卸载 -------------------------------------------------------

    def load(self, force: bool = False) -> EngineStats:
        """加载主模型。已加载时直接返回（除非 force）。"""
        with self._lock:
            req = _busy_runner_engine_req()
            if req == "unloaded":
                raise EngineError(
                    "训练正在进行中，引擎不能加载（8 GB 显存放不下训练器"
                    "加引擎两份）。请先停止训练或等它结束。")
            if req == "loaded" and force:
                raise EngineError(
                    "后台任务（特征提取/偏好对构造/评测）正在使用引擎，"
                    "不能强制重载。")
            if self._tts is not None and not force:
                return self.stats
            if self._tts is not None and force:
                self._unload_locked()

            self._emit("loading", "正在加载模型…")
            self.stats.error = ""
            self.stats.notes = []

            # 加载前先清一次显存：只归还**已释放**的块，所以对「上一次卸载没
            # 彻底」或「别处残留的 IPC 句柄」有帮助。实测价值在于：引擎加载
            # 前若还留着 whisper/训练器的残影，8 GB 卡上很容易直接失败或
            # 静默降速，而这一步几乎不花时间。
            try:
                from webui_app.training import guard as _GD
                _GD.free_vram("加载引擎前", LOG.get_logger("engine"))
            except Exception:
                pass

            missing = self.audit_missing()
            if missing:
                msg = "模型文件缺失：" + ", ".join(missing)
                self.stats.error = msg
                self._emit("error", msg)
                raise EngineError(msg + "。请到「模型资源」页下载。")

            t0 = time.perf_counter()
            try:
                if self.cfg.is_v25:
                    from indextts.infer_v2_5 import IndexTTS2
                else:
                    from indextts.infer_v2 import IndexTTS2

                # QwenEmotion 一律不在这里加载（走串行策略）
                kwargs = self.cfg.engine_kwargs()
                kwargs["use_qwen_emo"] = False
                self._tts = IndexTTS2(**kwargs)
            except Exception as e:
                self.stats.error = f"{type(e).__name__}: {e}"
                self._emit("error", self.stats.error)
                self._tts = None
                raise

            self.stats.load_seconds = time.perf_counter() - t0
            self.stats.loaded = True
            self.stats.loaded_at = time.time()
            self._refresh_vram()

            if getattr(self._tts, "low_vram", False):
                self.stats.notes.append(
                    "低显存模式已激活：超过 40 字的文本会被自动按标点切块，"
                    "块间韵律不接续。可在「参数手册 → 显存策略」查看缓解办法。"
                )
            self._emit(
                "loaded",
                f"模型加载完成，耗时 {self.stats.load_seconds:.1f}s，"
                f"占用显存 {self.stats.vram_alloc_gb:.2f} GB",
            )
            return self.stats

    def unload(self) -> EngineStats:
        """卸载主模型，归还显存。"""
        with self._lock:
            if _busy_runner_engine_req() == "loaded":
                # 任务线程手里还攥着 tts 的引用，这里删自己的引用并不会
                # 释放显存 —— 状态条却会说已卸载。用户接着加载就得到
                # GPU 上的第二份模型。所以直接拒绝，让任务先跑完。
                raise EngineError(
                    "后台任务（特征提取/偏好对构造/评测）正在使用引擎，"
                    "完成或停止任务后再卸载。")
            self._unload_locked()
            self._emit("unloaded", "模型已卸载，显存已释放")
            return self.stats

    def _unload_locked(self):
        self.release_qwen_locked()
        if self._tts is not None:
            del self._tts
            self._tts = None
        gc.collect()
        self._empty_cache()
        self.stats.loaded = False
        self.stats.lora_adapters = []
        self._refresh_vram()

    def _empty_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _refresh_vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                s = self.stats
                s.vram_alloc_gb = torch.cuda.memory_allocated(0) / 1024**3
                s.vram_reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
                s.vram_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
                s.vram_free_gb = refresh_vram_free(self.cfg.device)
        except Exception:
            pass

    def empty_cache(self) -> tuple[float, float]:
        """归还 PyTorch 的未引用缓存块，返回 (清理前, 清理后) 的已分配 GB。

        注意：empty_cache() 只归还**缓存块**，不会释放仍被张量引用的显存。
        所以模型常驻时调用它通常几乎没效果 —— 调用方应该把这一点告知用户，
        而不是让他以为“清一下就能腾出 4 GB”。
        """
        self._refresh_vram()
        before = self.stats.vram_alloc_gb
        self._empty_cache()
        self._refresh_vram()
        return before, self.stats.vram_alloc_gb

    # -- 资源审计 ----------------------------------------------------------

    def audit_missing(self) -> List[str]:
        """返回缺失的必需文件列表。"""
        from tools.model_fetcher import MAIN_FILES

        need = MAIN_FILES.get(self.cfg.version, [])
        missing = []
        for name, required, _size in need:
            if not required:
                continue
            if not os.path.isfile(os.path.join(self.cfg.model_dir, name)):
                missing.append(name)
        # 辅助模型
        hf = os.path.join(self.cfg.model_dir, "hf_cache")
        for rel in (
            os.path.join("w2v-bert-2.0", "config.json"),
            "campplus_cn_common.bin",
            "semantic_codec_model.safetensors",
            os.path.join("bigvgan", "config.json"),
            os.path.join("bigvgan", "bigvgan_generator.pt"),
        ):
            if not os.path.isfile(os.path.join(hf, rel)):
                missing.append(f"hf_cache/{rel}")
        # w2v-bert 权重（safetensors 或 bin 二者其一）
        w2v = os.path.join(hf, "w2v-bert-2.0")
        if not any(
            os.path.isfile(os.path.join(w2v, f))
            for f in ("model.safetensors", "pytorch_model.bin")
        ):
            missing.append("hf_cache/w2v-bert-2.0/model.safetensors")
        return missing

    def qwen_available(self) -> bool:
        """QwenEmotion 权重是否已下载。"""
        if not self.cfg.is_v25:
            return False
        d = os.path.join(self.cfg.model_dir, "qwen0.6bemo4-merge")
        return os.path.isfile(os.path.join(d, "model.safetensors")) and \
            os.path.isfile(os.path.join(d, "config.json"))

    # -- QwenEmotion 串行策略 ----------------------------------------------

    def mount_qwen(self) -> QwenEmotionSafe:
        """挂载 QwenEmotion（带显存预检 + CPU 回退）。用完应尽快 release_qwen()。

        device 选择策略（qwen_device_pref）：
            auto → 显存够就上 CUDA（1.7s），不够则回退 CPU fp32（约 2.4GB 内存，
                   每次几秒到十几秒，但**零显存风险**）
            cuda → 强制 CUDA，不够直接报错
            cpu  → 强制 CPU
        """
        with self._lock:
            if self._qwen is not None:
                return self._qwen
            if not self.qwen_available():
                raise EngineError(
                    "QwenEmotion 权重未下载（checkpoints/qwen0.6bemo4-merge/）。"
                    "请到「模型资源」页勾选下载。"
                )

            path = os.path.join(self.cfg.model_dir, "qwen0.6bemo4-merge")
            device, dtype = self._pick_qwen_device()

            t0 = time.perf_counter()
            self._emit("qwen_loading", f"正在加载 QwenEmotion 到 {device} …")
            try:
                self._qwen = QwenEmotionSafe(path, device=device, dtype=dtype)
            except Exception as e:
                self._qwen = None
                raise EngineError(f"QwenEmotion 加载失败：{type(e).__name__}: {e}")

            self.stats.qwen_load_seconds = time.perf_counter() - t0
            self.stats.qwen_mounted = True
            self._refresh_vram()
            self._emit(
                "qwen_loaded",
                f"QwenEmotion 已加载到 {device}（{self.stats.qwen_load_seconds:.1f}s）",
            )
            return self._qwen

    def _pick_qwen_device(self) -> tuple[str, str]:
        """选定 QwenEmotion 的 device 与 dtype，必要时回退到 CPU。

        这里不像官方那样用 device_map="auto"：显存不够时它会静默 offload
        部分层到 CPU，结果是“既不报错又慢得离谱”。我们要么全上 GPU，
        要么全上 CPU，行为可预测。
        """
        pref = self.qwen_device_pref
        cuda_dev = self.cfg.device.device_str
        has_cuda = cuda_dev.startswith("cuda")

        if pref == "cpu" or not has_cuda:
            # CPU 上用 fp16 会命中大量不支持的算子，必须用 fp32
            return "cpu", "float32"

        gc.collect()
        self._empty_cache()
        free = refresh_vram_free(self.cfg.device)
        if free >= self.QWEN_VRAM_REQUIRE_GB:
            return cuda_dev, "float16"

        if pref == "cuda":
            raise EngineError(
                f"显存不足：回收缓存后仍只有 {free:.2f} GB 空闲，"
                f"加载 QwenEmotion 需约 {self.QWEN_VRAM_REQUIRE_GB:.1f} GB。"
            )

        # auto：静默回退到 CPU，但明确告知用户（不默默降速）
        self._emit(
            "qwen_fallback",
            f"显存仅剩 {free:.2f} GB，不足以安全加载 QwenEmotion；"
            "已自动回退到 CPU 推理（零显存占用，但会慢几秒）。"
            "若想避开这个等待，可改用「8 维情感向量」模式。",
        )
        self.stats.notes.append(
            "QwenEmotion 已回退到 CPU 推理（显存不足）。"
        )
        return "cpu", "float32"

    def release_qwen_locked(self):
        if self._qwen is not None:
            del self._qwen
            self._qwen = None
            self.stats.qwen_mounted = False
            gc.collect()
            self._empty_cache()
            self._refresh_vram()

    def release_qwen(self):
        with self._lock:
            self.release_qwen_locked()
            self._emit("qwen_released", "QwenEmotion 已卸载，显存已归还")

    def text_to_emo_vector(self, emo_text: str, use_cache: bool = True) -> List[float]:
        """文本 → 8 维情感向量。**串行执行**：挂载 → 推理 → 立即卸载。

        返回的向量顺序与官方 `desired_vector_order` 一致：
            [高兴, 愤怒, 悲伤, 恐惧, 反感, 低落, 惊讶, 自然]
        即 UI 上的 [喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静]。

        注意：**不要**对这个向量再调用 `normalize_emo_vec(apply_bias=True)`，
        官方 mode-3 路径直接使用 QwenEmotion 的原始输出（clamp 到 [0, 1.2]）。
        再套一层偏置会与官方行为不一致。
        """
        key = (emo_text or "").strip()
        if use_cache and key in self._emo_cache:
            self._emit("qwen_cache_hit", f"情感向量命中缓存：{key[:24]}")
            return list(self._emo_cache[key])

        with self._lock:
            qwen = self.mount_qwen()
            try:
                d = qwen.inference(emo_text)
            finally:
                # 无论成功失败都立即释放，不让它与主推理共存
                self.release_qwen_locked()
            self.stats.qwen_infer_count += 1

        vec = list(d.values())
        if use_cache and key:
            self._emo_cache[key] = list(vec)
        return vec

    def clear_emo_cache(self):
        """清空情感向量缓存，返回被清除的条数。"""
        with self._lock:
            n = len(self._emo_cache)
            self._emo_cache.clear()
        return n

    # -- LoRA（阶段 2 使用，此处仅提供挂载点）------------------------------

    # -- LoRA 挂载 ---------------------------------------------------------

    def _lora_module(self, target: str):
        """取该目标的当前模块（可能是 PeftModel，也可能是纯底座）。"""
        tts = self.tts
        if target == "gpt":
            return tts.gpt
        if target == "cfm":
            return tts.s2mel.models["cfm"]
        raise EngineError(f"未知 LoRA 目标: {target}")

    def _lora_set_module(self, target: str, mod) -> None:
        tts = self.tts
        if target == "gpt":
            tts.gpt = mod
        else:
            tts.s2mel.models["cfm"] = mod

    @staticmethod
    def _is_wrapped(mod) -> bool:
        """模块是否已被 PEFT 包过一层。

        **不能只信 `stats.lora_adapters` 这个标签列表** —— 它会和真实状态
        脱节：`from_pretrained` 中途抛异常时，LoRA 层可能已经注进了底座，
        却既没被包装也没打上标签。只按标签判断就会在 PeftModel 上再包一层，
        之后每次挂载都报错，且持续到引擎重载 —— 正是「偶现后持续报错」
        这种故障形态。所以判断以**模块自身状态**为准。
        """
        try:
            from peft import PeftModel
            if isinstance(mod, PeftModel):
                return True
        except Exception:
            pass
        # 兜底：PeftModel 会把底座挂在 .base_model 上
        return getattr(mod, "base_model", None) is not None

    @staticmethod
    def _lora_hint(mod) -> str:
        """给日志用的一句话模块状态描述。"""
        try:
            n = sum(1 for m in mod.modules() if hasattr(m, "lora_A"))
        except Exception:
            n = -1
        return (f"{type(mod).__name__}(wrapped="
                f"{TTSEngine._is_wrapped(mod)}, lora_layers={n})")

    def attach_lora(self, adapter_dir: str, target: str = "gpt") -> str:
        """把 LoRA adapter 挂到当前引擎上（已挂着就先卸掉，替换语义）。

        target: gpt | cfm。不先卸载的话 PEFT 会把已包装的模块再包一层，
        而 detach 只解一层 —— UI 显示「纯底座」但内层 LoRA 仍在生效，
        这比挂不上严重得多。

        判断「是否已挂着」用**模块真实状态 ∪ 标签**：任何一方与实际脱节
        都不会造成二次包装（打包后失败的残留也能被清掉）。
        """
        log = LOG.get_logger("engine.lora")
        t0 = time.perf_counter()
        # 先把"请求"记下来：引擎没加载时 self.tts 会直接抛，
        # 那样就什么痕迹都没有了 —— 而"点了没反应"恰恰最难查。
        log.info("挂载请求 target=%s dir=%s（引擎已加载=%s）",
                 target, adapter_dir, self.loaded)
        with self._lock:
            tts = self.tts
            from peft import PeftModel

            if target not in ("gpt", "cfm"):
                log.error("挂载失败：未知目标 %r", target)
                raise EngineError(f"未知 LoRA 目标: {target}")

            if not os.path.isdir(adapter_dir):
                log.error("挂载失败：adapter 目录不存在 %s", adapter_dir)
                raise EngineError(f"adapter 目录不存在: {adapter_dir}")

            before = self._lora_module(target)
            tags = [t for t in self.stats.lora_adapters
                    if t.startswith(target + ":")]
            log.info("挂载前状态 target=%s · 当前 %s · 标签 %s",
                     target, self._lora_hint(before), tags or "无")

            if self._is_wrapped(before) or tags:
                if not tags:
                    log.warning(
                        "目标 %s 已被包装但没有对应标签 —— 状态曾经脱节，"
                        "先强制卸载再挂。请把这段日志连同报错一起反馈", target)
                self.detach_lora(target)   # _lock 是 RLock，重入安全

            base = self._lora_module(target)
            try:
                wrapped = PeftModel.from_pretrained(base, adapter_dir)
            except Exception:
                # 失败可能已经把 LoRA 层注进底座却没包装起来。立刻清理：
                # 否则底座会带着一层随机初始化的旁路继续跑，输出变成垃圾，
                # 而且看起来像「模型坏了」而不是「挂载失败」。
                log.error(
                    "PeftModel.from_pretrained 失败（target=%s dir=%s）：\n%s",
                    target, adapter_dir, exc_info=True)
                try:
                    self.detach_lora(target)
                    log.warning("已在失败后清理 target=%s 的残留", target)
                except Exception:
                    log.error("失败后清理 target=%s 也失败了", target,
                              exc_info=True)
                raise EngineError(
                    f"加载 adapter 失败：{adapter_dir}（详见 error.log）") from None

            self._lora_set_module(target, wrapped)

            name = os.path.basename(adapter_dir.rstrip("/\\"))
            tag = f"{target}:{name}"
            if tag not in self.stats.lora_adapters:
                self.stats.lora_adapters.append(tag)
            log.info("挂载完成 %s · %.2fs · 之后 %s · 标签 %s",
                     tag, time.perf_counter() - t0,
                     self._lora_hint(self._lora_module(target)),
                     list(self.stats.lora_adapters))
            self._emit("lora_attached", f"已挂载 LoRA {tag}")
            return tag

    def detach_lora(self, target: str = "gpt"):
        """卸载 LoRA，恢复 base 模型。"""
        log = LOG.get_logger("engine.lora")
        t0 = time.perf_counter()
        log.info("卸载请求 target=%s（引擎已加载=%s）", target, self.loaded)
        with self._lock:
            tts = self.tts
            mod = self._lora_module(target)
            before = self._lora_hint(mod)
            tags = [t for t in self.stats.lora_adapters
                    if t.startswith(target + ":")]

            base = getattr(mod, "base_model", None)
            if base is not None and hasattr(mod, "unload"):
                try:
                    unwrapped = mod.unload()
                    self._lora_set_module(target, unwrapped)
                except Exception:
                    log.error("卸载 %s 失败（模块状态 %s）", target, before,
                              exc_info=True)
                    raise
            elif tags:
                # 不是 PeftModel 却带着标签 —— 状态脱节。记下来：
                # 这类脱节正是一批「持续报错」的源头。
                log.warning("卸载 %s：模块并非 PeftModel（%s）却带着标签 %s"
                            " —— 状态脱节，仅清理标签", target, before, tags)
            else:
                log.debug("卸载 %s：本来就没挂载（%s）", target, before)

            self.stats.lora_adapters = [
                t for t in self.stats.lora_adapters if not t.startswith(target + ":")
            ]
            self._empty_cache()
            log.info("卸载完成 %s · %.2fs · %s → %s · 标签 %s",
                     target, time.perf_counter() - t0, before,
                     self._lora_hint(self._lora_module(target)),
                     list(self.stats.lora_adapters) or "无")
            self._emit("lora_detached", f"已卸载 {target} 的 LoRA")

    def lora_status(self) -> List[Dict[str, Any]]:
        """引擎上 LoRA 的**真实**状态（不信标签，直接看模块）。

        推理页的 LoRA 面板用它显示 —— 标签与模块脱节时立刻看得出来，
        而不是等用户发现「选了纯底座却还是那个声音」。
        """
        out: List[Dict[str, Any]] = []
        if not self.loaded:
            return out
        with self._lock:
            for target in ("gpt", "cfm"):
                try:
                    mod = self._lora_module(target)
                except Exception as e:
                    out.append({"target": target, "error": str(e)})
                    continue
                wrapped = self._is_wrapped(mod)
                tags = [t for t in self.stats.lora_adapters
                        if t.startswith(target + ":")]
                out.append({"target": target, "wrapped": wrapped, "tags": tags,
                            "module": type(mod).__name__,
                            "consistent": bool(wrapped) == bool(tags)})
        return out

    # -- 推理 --------------------------------------------------------------

    def infer(self, progress=None, **kwargs) -> Optional[str]:
        """执行一次推理。kwargs 直接透传给 IndexTTS2.infer()。"""
        with self._lock:
            tts = self.tts
            tts.gr_progress = progress
            self._refresh_vram()
            t0 = time.perf_counter()
            try:
                out = tts.infer(**kwargs)
            except Exception as e:
                self.stats.error = f"{type(e).__name__}: {e}"
                self._emit("infer_error", self.stats.error)
                raise
            finally:
                tts.gr_progress = None

            dt = time.perf_counter() - t0
            self.stats.infer_count += 1
            self.stats.last_infer_seconds = dt
            self.stats.infer_total_seconds += dt
            self._refresh_vram()
            self._emit(
                "inferred",
                f"推理完成，耗时 {dt:.2f}s，峰值显存 {self.stats.vram_peak_gb:.2f} GB",
            )
            return out

    def count_tokens(self, text: str) -> int:
        """统计文本的 tiktoken token 数（v2.5）。"""
        tts = self.tts
        try:
            return len(tts.tokenizer.encode(text, allowed_special="all"))
        except Exception:
            return 0

    def preview_segments(self, text: str, max_tokens: int, lang: str = "ZH") -> List[dict]:
        """预览 v2.5 的实际分句结果（调用官方的 split_text_by_tokens）。"""
        tts = self.tts
        if not text:
            return []
        lang_prefix = f"<|{(lang or 'ZH').lower()}|> "
        try:
            segs = tts.split_text_by_tokens(text, int(max_tokens), lang_prefix)
        except Exception:
            segs = [text]
        rows = []
        for i, s in enumerate(segs):
            rows.append({
                "index": i,
                "text": s,
                "chars": len(s),
                "tokens": self.count_tokens(lang_prefix + s),
            })
        return rows

    def clear_reference_cache(self):
        """清空参考音频缓存。换音色时官方会自动清，这个是手动强制清。"""
        with self._lock:
            tts = self.tts
            for attr in (
                "cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt",
                "cache_spk_audio_prompt", "cache_emo_cond",
                "cache_emo_audio_prompt", "cache_mel",
            ):
                if hasattr(tts, attr):
                    setattr(tts, attr, None)
            self._empty_cache()
            self._refresh_vram()
            self._emit("cache_cleared", "参考音频缓存已清空")

    def normalize_emo_vec(self, vec: List[float], apply_bias: bool = True) -> List[float]:
        return list(self.tts.normalize_emo_vec(list(vec), apply_bias=apply_bias))
