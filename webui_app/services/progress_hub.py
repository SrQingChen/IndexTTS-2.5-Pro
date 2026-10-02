"""进度日志包装：把 progress(frac, msg) 回调变成「文件日志 + 静默心跳」。

为什么需要它
============
长任务（一键三连、导演编排、BoN 择优）动辄几分钟，而界面上的 Gradio
进度条在浏览器切走/最小化后看不到，进程又在实打实干活 —— 用户没法
区分「在工作」和「卡死了」，事后翻文件日志也只有任务的开始和结束。
一键三连（training/oneclick.py）最早为此内联了这套机制，本模块把它
提炼成通用件给所有长任务复用：

    · 每条进度都写进文件日志（INFO 级限流防刷屏，其余降 DEBUG）；
    · 一条心跳线程：超过 heartbeat_idle 秒没有新进度，就按 INFO 报
      「仍在进行 · 已静默多久 · 最后一条进度」。静默本身就是信息 ——
      它能把「长音频在算」和「真的卡住」分开。

用法
====
    lp = LoggedProgress(LOG.get_logger("synth"), user_progress=progress)
    lp.start()
    try:
        ...长任务（把 lp 当 progress 传下去）...
    finally:
        lp.stop()          # 心跳必须停，否则任务结束后它还继续写日志
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from webui_app import logging_setup as LOG

ProgressFn = Callable[[float, str], None]


class LoggedProgress:
    """progress 回调包装器：限流入日志 + 心跳线程。

    与 oneclick 内联版的差异：阶段标题由调用方在消息里自带（如
    "[导演] 块 1/3"），本类不再维护 stage 映射。
    """

    def __init__(self, log: Any, user_progress: Optional[ProgressFn] = None,
                 heartbeat_idle: float = 15.0, period: float = 5.0,
                 info_every: float = 5.0, sink: Optional[Any] = None):
        self._log = log
        self._user = user_progress
        self._idle = float(heartbeat_idle)
        self._period = float(period)
        self._info_every = float(info_every)
        # sink：可选择的行缓冲（如 collections.deque(maxlen=N)），每条进度
        # 与心跳同步写入 —— 给 UI 的实时日志窗（gr.Timer 轮询）供料。
        # append 必须线程安全（AnyIO 回调线程 + 心跳线程都会写）。
        self._sink = sink
        self._last = {"t": time.time(), "msg": "启动", "frac": 0.0,
                      "info_t": 0.0}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _push(self, line: str) -> None:
        if self._sink is not None:
            try:
                self._sink.append(line)
            except Exception:
                pass

    # -- 作为 progress 回调使用 -------------------------------------------
    def __call__(self, frac: float, msg: str = "", **kw) -> None:
        # 兼容两种调用形态：orchestrator 用位置参数，官方引擎用
        # `gr_progress(value, desc=...)` 关键字（indextts/infer_v2.py）。
        msg = str(msg or kw.get("desc") or "")
        now = time.time()
        self._last.update(t=now, msg=msg, frac=float(frac))
        # 进度行限流升到 INFO：每条都写会在 BoN/切片时刷屏；全压 DEBUG
        # 又会让常规级别下「整整几分钟一条日志都没有」。折中：最多每
        # info_every 秒留一条 INFO，其余走 DEBUG（排查时开 DEBUG 拿全量）。
        if now - self._last["info_t"] >= self._info_every:
            self._last["info_t"] = now
            self._log.info("[%3.0f%%] %s", float(frac) * 100, msg)
        else:
            self._log.debug("[%3.0f%%] %s", float(frac) * 100, msg)
        self._push(f"[{float(frac) * 100:3.0f}%] {msg}")
        if self._user is not None:
            try:
                if kw:
                    self._user(frac, **kw)
                else:
                    self._user(frac, msg)
            except Exception:
                pass

    @property
    def last_msg(self) -> str:
        return str(self._last.get("msg", ""))

    @property
    def last_frac(self) -> float:
        return float(self._last.get("frac", 0.0))

    # -- 生命周期 -----------------------------------------------------------
    def start(self) -> "LoggedProgress":
        if self._thread is None:
            self._thread = threading.Thread(target=self._heartbeat,
                                            daemon=True,
                                            name="progress-heartbeat")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._period):
            idle = time.time() - self._last["t"]
            if idle < self._idle:
                continue
            line = (f"…仍在进行 · 已静默 {idle:.0f} 秒 · "
                    f"最后进度：{self._last['msg']}")
            self._log.info("…仍在进行 · 已静默 %.0f 秒 · 最后进度：%s",
                           idle, self._last["msg"])
            self._push(line)
