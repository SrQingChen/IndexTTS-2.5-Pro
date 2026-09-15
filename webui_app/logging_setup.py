"""统一日志：落盘 + 控制台 + 内存环形缓冲，一个开关控制级别。

为什么要有这个模块
==================

排查故障时「没有日志」是最大的障碍。此前只有 `context.EventLog` —— 内存里
300 条事件、进程一退就没，而且**不记异常堆栈**。于是遇到「切换 LoRA 时偶现
持续报错」这类问题，手上只有一句用户描述，没有任何 traceback 可查。

所以这里把日志做成四路输出，各有分工：

    outputs/logs/indextts.log   全量（按级别），5 MB × 5 轮转，排查主战场
    outputs/logs/error.log      只记 WARNING 及以上，一眼看出出过什么事
    内存环形缓冲                供 UI「系统」页实时查看，不必读盘
    控制台                       开发时直接看（可 --quiet 关掉）

另外两件容易被忽略但很关键的事：

  · **未捕获异常也要落盘**。主线程 `sys.excepthook`、子线程
    `threading.excepthook` 都挂上 —— runner 的后台线程崩了通常只剩一句
    「任务失败」，有了它就能拿到完整堆栈。
  · **Gradio 会吞掉回调里的异常**（转成界面上的红条），异常本身不会经过
    我们的代码。所以提供了一个 `@ui_guard("名字")` 装饰器，把它套在关心
    的回调上，异常先落盘再原样抛出 —— 界面行为不变，但日志里留了证据。

用法
====

    from webui_app import logging_setup as LOG
    LOG.setup(level="INFO")                  # 进程入口调一次
    log = LOG.get_logger("engine")
    log.info("加载完成 %.1fs", sec)
    LOG.install_excepthooks()                # 挂全局兜底

级别可在运行期改（UI 上有开关）：`LOG.set_level("DEBUG")`。
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# 所有日志都挂在这个根名字下，便于一次配置、一次改级别。
ROOT_NAME = "ix"

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_LEVEL = "INFO"

_MAX_BYTES = 5 * 1024 * 1024          # 单文件 5 MB
_BACKUPS = 5                          # 保留 5 份
_RING_MAX = 800                       # 内存里保留最近 800 条供 UI 读

_FMT = ("%(asctime)s.%(msecs)03d | %(levelname)-7s | %(threadName)-14s | "
        "%(name)s | %(message)s")
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_lock = threading.RLock()
_configured = False
_log_dir: str = ""
_log_path: str = ""
_err_path: str = ""
_level: str = DEFAULT_LEVEL
_ring: Deque[Dict[str, Any]] = deque(maxlen=_RING_MAX)
_hooks_installed = False


# ---------------------------------------------------------------------------
# 内存环形缓冲
# ---------------------------------------------------------------------------

class _RingHandler(logging.Handler):
    """把最近若干条记录留在内存里，供 UI 直接读（不碰磁盘）。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _ring.append({
                "ts": record.created,
                "level": record.levelname,
                "name": record.name,
                "thread": record.threadName,
                "message": record.getMessage(),
                "exc": (self.format_exception(record)
                        if record.exc_info else ""),
            })
        except Exception:
            pass          # 日志本身绝不能再抛

    @staticmethod
    def format_exception(record: logging.LogRecord) -> str:
        try:
            return "".join(traceback.format_exception(*record.exc_info))
        except Exception:
            return "<无法格式化异常>"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _default_dir() -> str:
    try:
        from webui_app.config import PROJECT_ROOT
        return os.path.join(PROJECT_ROOT, "outputs", "logs")
    except Exception:
        return os.path.join(os.getcwd(), "outputs", "logs")


def setup(level: str = DEFAULT_LEVEL, log_dir: str = "",
          console: bool = True, force: bool = False) -> str:
    """配置日志。可重复调用（默认已配置就只改级别）。返回日志文件路径。"""
    global _configured, _log_dir, _log_path, _err_path, _level

    with _lock:
        _level = _norm_level(level)
        root = logging.getLogger(ROOT_NAME)

        if _configured and not force:
            root.setLevel(getattr(logging, _level))
            for h in root.handlers:
                h.setLevel(getattr(logging, _level))
            return _log_path

        # 先清掉旧 handler，避免重复配置时同一条日志写两遍
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                pass

        root.setLevel(getattr(logging, _level))
        root.propagate = False        # 不要冒泡到根 logger，免得被别处的配置接管
        fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

        _log_dir = log_dir or _default_dir()
        try:
            os.makedirs(_log_dir, exist_ok=True)
            _log_path = os.path.join(_log_dir, "indextts.log")
            _err_path = os.path.join(_log_dir, "error.log")

            fh = logging.handlers.RotatingFileHandler(
                _log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS,
                encoding="utf-8", delay=True)
            fh.setFormatter(fmt)
            fh.setLevel(getattr(logging, _level))
            root.addHandler(fh)

            # 单独一份「只记问题」的文件：排查时先看它，不用在几千行里翻
            eh = logging.handlers.RotatingFileHandler(
                _err_path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS,
                encoding="utf-8", delay=True)
            eh.setFormatter(fmt)
            eh.setLevel(logging.WARNING)
            root.addHandler(eh)
        except Exception as e:                      # 目录不可写也不能拖垮启动
            _log_path = _err_path = ""
            print(f"[日志] 文件日志不可用（{type(e).__name__}: {e}），"
                  "只输出到控制台。", file=sys.stderr)

        if console:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setFormatter(fmt)
            ch.setLevel(getattr(logging, _level))
            root.addHandler(ch)

        root.addHandler(_RingHandler())

        _configured = True
        root.info("日志已就绪：级别 %s · 文件 %s",
                  _level, _log_path or "(无)")
        return _log_path


def _norm_level(level: str) -> str:
    lv = str(level or "").strip().upper()
    if lv not in LEVELS:
        return DEFAULT_LEVEL
    return lv


def _bootstrap_silent() -> None:
    """在 `setup()` 之前先别输出，只挂一个 NullHandler 占位。

    为什么不能在这里直接装文件/控制台处理器：`get_logger` 会在 import 期被
    各模块调用，而入口（webui_pro.py）的 `setup(level=..., quiet=..., 
    log_dir=...)` 可能还没执行。若惰性路径抢先配好了，入口那次就会因为
    「已配置」而只改级别 —— `--quiet`、`--log-dir`、`--log-level` 全部失效。
    所以真正的配置只有 `setup()` 能做，这里保证「不报错、不输出」即可。
    """
    global _configured
    with _lock:
        if _configured:
            return
        root = logging.getLogger(ROOT_NAME)
        root.propagate = False
        if not root.handlers:
            root.addHandler(logging.NullHandler())


def get_logger(name: str = "") -> logging.Logger:
    """取一个子 logger。名字建议用「模块/功能」，如 engine、merge、oneclick。

    `setup()` 之前调用是安全的：只拿到一个不输出的 logger，
    等入口配置好后自动开始记录（handler 装在根 logger 上）。
    """
    if not _configured:
        _bootstrap_silent()
    return logging.getLogger(f"{ROOT_NAME}.{name}" if name else ROOT_NAME)


def ensure() -> None:
    """确保日志系统已按默认值配置（幂等）。供不确定入口调过没的独立脚本用。"""
    if not _configured:
        setup()


# ---------------------------------------------------------------------------
# 运行期开关与查询
# ---------------------------------------------------------------------------

def set_level(level: str) -> str:
    """改级别（DEBUG 会记下每次回调与每步训练，排查完记得调回 INFO）。"""
    global _level
    with _lock:
        _level = _norm_level(level)
        root = logging.getLogger(ROOT_NAME)
        root.setLevel(getattr(logging, _level))
        for h in root.handlers:
            # 错误文件固定 WARNING 起，别被 DEBUG 冲淡
            if getattr(h, "baseFilename", "").endswith("error.log"):
                h.setLevel(logging.WARNING)
            else:
                h.setLevel(getattr(logging, _level))
        get_logger("logging").info("日志级别已切换为 %s", _level)
    return _level


def level() -> str:
    return _level


def log_dir() -> str:
    return _log_dir


def log_path() -> str:
    return _log_path


def error_log_path() -> str:
    return _err_path


def recent(n: int = 200, min_level: str = "") -> List[Dict[str, Any]]:
    """最近的日志记录（内存里读，快）。min_level 可过滤。"""
    want = _norm_level(min_level) if min_level else ""
    if want:
        order = {lv: i for i, lv in enumerate(LEVELS)}
        floor = order.get(want, 0)
        items = [r for r in list(_ring)
                 if order.get(str(r.get("level", "INFO")), 0) >= floor]
    else:
        items = list(_ring)
    return items[-n:]


def recent_markdown(n: int = 200, min_level: str = "") -> str:
    """给 UI 用的 markdown 表格。"""
    rows = recent(n, min_level)
    if not rows:
        return ("_还没有日志。日志文件在 "
                f"`{_log_path or '(未启用文件日志)'}`_")
    icon = {"DEBUG": "🔍", "INFO": "·", "WARNING": "⚠️", "ERROR": "✖"}
    L = [f"| 时间 | 级别 | 来源 | 内容 |", "|---|---|---|---|"]
    for r in rows:
        t = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        msg = str(r["message"]).replace("|", "\\|").replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:220] + "…"
        if r.get("exc"):
            msg += " ⚠️ 含堆栈，见文件日志"
        L.append(f"| `{t}` | {icon.get(r['level'], '·')} {r['level']} "
                 f"| `{r['name']}` | {msg} |")
    return "\n".join(L)


def file_tail(n: int = 300, path: str = "") -> str:
    """读日志文件尾部（跨轮转也在）。读失败返回错误说明而不是抛异常。"""
    p = path or _log_path
    if not p or not os.path.isfile(p):
        return f"_日志文件不存在：{p or '(未启用)'}_"
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-int(n):]) or "_日志文件是空的_"
    except Exception as e:
        return f"_读取日志失败：{type(e).__name__}: {e}_"


def stats() -> Dict[str, Any]:
    """文件大小等信息，给 UI 显示。"""
    def _sz(p: str) -> int:
        try:
            return os.path.getsize(p) if p and os.path.isfile(p) else 0
        except Exception:
            return 0
    return {"level": _level, "dir": _log_dir, "log": _log_path,
            "error_log": _err_path, "log_bytes": _sz(_log_path),
            "error_bytes": _sz(_err_path), "ring": len(_ring)}


# ---------------------------------------------------------------------------
# 全局兜底：没被捕获的异常也要落盘
# ---------------------------------------------------------------------------

def install_excepthooks() -> None:
    """把主线程与子线程的未捕获异常都写进日志（幂等）。"""
    global _hooks_installed
    with _lock:
        if _hooks_installed:
            return
        log = get_logger("uncaught")

        prev_hook = sys.excepthook

        def _hook(exc_type, exc, tb):
            if issubclass(exc_type, KeyboardInterrupt):
                prev_hook(exc_type, exc, tb)
                return
            log.error("主线程未捕获异常：%s", exc, exc_info=(exc_type, exc, tb))
            prev_hook(exc_type, exc, tb)

        sys.excepthook = _hook

        def _thread_hook(args):
            # 后台线程崩了以前只会静默消失 —— 这里留下完整堆栈
            if issubclass(args.exc_type, SystemExit):
                return
            log.error("线程 %s 未捕获异常：%s",
                      getattr(args.thread, "name", "?"), args.exc_value,
                      exc_info=(args.exc_type, args.exc_value,
                                args.exc_traceback))

        threading.excepthook = _thread_hook
        _hooks_installed = True


# ---------------------------------------------------------------------------
# 装饰器
# ---------------------------------------------------------------------------

def _brief(args: tuple, kwargs: dict, limit: int = 90) -> str:
    """把回调参数压成一行短描述 —— 绝不把整个 dict / ndarray 打进日志。"""
    def one(v: Any) -> str:
        s = "" if v is None else str(v)
        s = s.replace("\n", " ")
        if len(s) > limit:
            s = s[:limit] + "…"
        return f"{type(v).__name__}({s})"
    parts = [one(a) for a in args[:4]]
    parts += [f"{k}={one(v)}" for k, v in list(kwargs.items())[:3]]
    if len(args) > 4:
        parts.append(f"…共 {len(args)} 个参数")
    return ", ".join(parts)


def ui_guard(name: str, slow_sec: float = 1.0):
    """套在 Gradio 回调上：异常先落盘再原样抛出，慢调用也记一笔。

    Gradio 会把回调里的异常转成界面上的红条，异常不会经过我们的代码 ——
    没有这个装饰器就永远拿不到堆栈。界面行为完全不变（异常照旧抛出）。
    """
    log = get_logger("ui")

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            log.debug("→ %s(%s)", name, _brief(args, kwargs))
            try:
                r = fn(*args, **kwargs)
            except Exception:
                # 用 exc_info=True 而不是把 format_exc() 拼进消息：堆栈走
                # 结构化通道，文件与内存缓冲都能拿到完整内容（UI 也能标记
                # 「含堆栈」）。拼字符串的话内存缓冲里就只剩一行摘要。
                log.error("%s 抛出异常（参数：%s）", name,
                          _brief(args, kwargs), exc_info=True)
                raise
            dt = time.perf_counter() - t0
            if dt >= slow_sec:
                log.info("%s 完成，用时 %.2fs", name, dt)
            else:
                log.debug("← %s ok %.3fs", name, dt)
            return r
        return wrapper
    return deco


def log_call(name: str = "", level: str = "DEBUG"):
    """给普通函数/方法记调用与耗时（默认 DEBUG，不刷屏）。"""
    log = get_logger("call")
    lv = getattr(logging, _norm_level(level))

    def deco(fn):
        label = name or getattr(fn, "__qualname__", str(fn))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not log.isEnabledFor(lv):
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            log.log(lv, "→ %s(%s)", label, _brief(args, kwargs))
            try:
                r = fn(*args, **kwargs)
            except Exception:
                log.error("%s 抛出异常", label, exc_info=True)
                raise
            log.log(lv, "← %s %.3fs", label, time.perf_counter() - t0)
            return r
        return wrapper
    return deco


def shutdown() -> None:
    """关闭并摘掉全部 handler，并复位配置状态。

    主要给测试与工具用：Windows 上只要日志文件还有打开的句柄，所在目录就
    删不掉（探针跑完清理临时目录会失败）。应用运行期不需要调它。
    """
    global _configured, _log_path, _err_path, _log_dir
    with _lock:
        root = logging.getLogger(ROOT_NAME)
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                pass
        _configured = False
        _log_path = _err_path = _log_dir = ""


def fmt_exc() -> str:
    """当前异常的堆栈字符串（在 except 块里调用）。"""
    return traceback.format_exc()
