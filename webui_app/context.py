"""AppContext —— 跨 Tab 共享的运行时上下文。

Gradio 的每个 Tab 是独立函数，但它们需要共享同一个引擎实例、同一份配置、
同一个事件日志。用 context 单例传递，避免官方 webui.py 那种模块级全局变量
+ 函数间靠闭包引用的写法（那种写法在拆文件后无法维护）。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from webui_app.config import AppConfig, refresh_vram_free
from webui_app.services.engine import TTSEngine


@dataclass
class EventLog:
    """引擎事件环形日志，UI 顶部状态条和「系统」页都从这里读。"""

    maxlen: int = 300
    _items: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=300))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, event: str, message: str = "", level: str = "info"):
        with self._lock:
            self._items.append({
                "ts": time.time(),
                "event": event,
                "message": message,
                "level": level,
            })

    def tail(self, n: int = 40) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._items)[-n:]

    def last(self, event: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            items = list(self._items)
        for it in reversed(items):
            if event is None or it["event"] == event:
                return it
        return None

    def markdown(self, n: int = 60) -> str:
        rows = self.tail(n)
        if not rows:
            return "_暂无事件_"
        lines = ["| 时间 | 事件 | 说明 |", "|---|---|---|"]
        icon = {"info": "·", "ok": "✅", "warn": "⚠️", "error": "✖"}
        for it in rows:
            t = time.strftime("%H:%M:%S", time.localtime(it["ts"]))
            lines.append(
                f"| `{t}` | {icon.get(it['level'], '·')} `{it['event']}` "
                f"| {it['message'] or '-'} |"
            )
        return "\n".join(lines)


_LEVEL_BY_EVENT = {
    "loaded": "ok", "inferred": "ok", "qwen_loaded": "ok",
    "lora_attached": "ok", "cache_cleared": "ok",
    "unloaded": "info", "qwen_released": "info", "lora_detached": "info",
    "loading": "info", "qwen_loading": "info",
    "error": "error", "infer_error": "error",
}


class AppContext:
    """全局上下文。由 app.py 创建一次，通过 gr.State 或模块级单例传给各 Tab。"""

    _instance: Optional["AppContext"] = None
    _inst_lock = threading.Lock()

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.engine = TTSEngine(config)
        self.log = EventLog()
        self.started_at = time.time()
        self.engine.set_event_handler(self._on_engine_event)
        # 跨 Tab 共享的 Gradio 组件槽位。
        # app.py 在渲染任何 Tab **之前**先把顶栏状态条放进去，
        # 各 Tab 再引用它作为 callback 的 output —— 避开了
        # “在 callback 里临时新建组件” 这种会产生幽灵控件的写法。
        self.shared: Dict[str, Any] = {}

    def component(self, name: str):
        """取一个共享组件；不存在时报错而不是静默新建。"""
        if name not in self.shared:
            raise KeyError(
                f"共享组件 {name!r} 尚未注册。app.py 必须在渲染 Tab 之前先创建它。"
            )
        return self.shared[name]

    # -- 单例 --------------------------------------------------------------

    @classmethod
    def get(cls, config: Optional[AppConfig] = None) -> "AppContext":
        with cls._inst_lock:
            if cls._instance is None:
                if config is None:
                    config = AppConfig()
                cls._instance = AppContext(config)
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._inst_lock:
            cls._instance = None

    # -- 事件 --------------------------------------------------------------

    def _on_engine_event(self, event: str, message: str):
        self.log.push(event, message, _LEVEL_BY_EVENT.get(event, "info"))

    # -- 状态快照 ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """给状态条用的实时快照。"""
        s = self.engine.stats
        refresh_vram_free(self.cfg.device)
        d = self.cfg.device
        return {
            "engine_loaded": s.loaded,
            "load_seconds": round(s.load_seconds, 1),
            "infer_count": s.infer_count,
            "last_infer_seconds": round(s.last_infer_seconds, 2),
            "vram_alloc_gb": round(s.vram_alloc_gb, 2),
            "vram_peak_gb": round(s.vram_peak_gb, 2),
            "vram_free_gb": round(d.vram_free_gb, 2),
            "vram_total_gb": round(d.vram_total_gb, 2),
            "qwen_mounted": s.qwen_mounted,
            "qwen_available": self.engine.qwen_available(),
            "qwen_infer_count": s.qwen_infer_count,
            "lora_adapters": list(s.lora_adapters),
            "gpu": d.gpu_name,
            "backend": d.backend,
            "bf16": d.bf16_supported,
            "low_vram": d.low_vram,
            "version": self.cfg.version,
            "error": s.error,
            "notes": list(s.notes),
            "uptime": time.time() - self.started_at,
        }

    def status_chips(self) -> List[str]:
        """渲染顶部状态条的 HTML chip 列表。"""
        from webui_app import theme as T

        snap = self.snapshot()
        chips = []

        if snap["backend"] == "cpu":
            chips.append(T.chip("设备", "CPU", "warn"))
        else:
            chips.append(T.chip("GPU", snap["gpu"].replace("NVIDIA ", ""), "ok"))

        if snap["engine_loaded"]:
            chips.append(T.chip("引擎", f'已加载 {snap["load_seconds"]}s', "ok"))
        else:
            chips.append(T.chip("引擎", "未加载", "idle"))

        vram_txt = (f'{snap["vram_alloc_gb"]:.2f} / {snap["vram_total_gb"]:.1f} GB'
                    if snap["backend"] == "cuda" else "N/A")
        state = "ok"
        if snap["backend"] == "cuda":
            used_ratio = snap["vram_alloc_gb"] / max(snap["vram_total_gb"], 0.1)
            state = "err" if used_ratio > 0.9 else ("warn" if used_ratio > 0.7 else "ok")
        chips.append(T.chip("显存", vram_txt, state))

        if snap["low_vram"]:
            chips.append(T.chip("低显存模式", "已激活", "warn"))

        if snap["qwen_mounted"]:
            chips.append(T.chip("QwenEmotion", "已挂载", "load"))
        elif snap["qwen_available"]:
            chips.append(T.chip("QwenEmotion", "按需加载", "idle"))
        else:
            chips.append(T.chip("QwenEmotion", "未下载", "warn"))

        if snap["lora_adapters"]:
            chips.append(T.chip("LoRA", ", ".join(snap["lora_adapters"]), "ok"))

        if snap["infer_count"]:
            chips.append(T.chip(
                "已合成", f'{snap["infer_count"]} 次 / 最近 {snap["last_infer_seconds"]}s',
                "idle",
            ))

        if snap["error"]:
            chips.append(T.chip("错误", snap["error"][:40], "err"))

        return chips

    def status_html(self) -> str:
        from webui_app import theme as T
        return T.statusbar(self.status_chips())
