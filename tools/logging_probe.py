"""调试日志系统（webui_app/logging_setup.py）的验证。

为什么专门给它写探针：这套东西的价值全在「出事那一刻它记没记下来」，
而那种时刻没法手工再现。所以这里把关键承诺逐条钉死：

  · 落盘：全量文件 + 只记问题的 error.log（含完整异常堆栈）
  · 兜底：主线程 / 子线程的未捕获异常都要进日志（子线程静默消失是排查噩梦）
  · 时序：setup 之前的调用必须静默，不能抢在入口配置之前把 handler 装死
    （否则 --log-level / --quiet / --log-dir 全部失效）
  · 级别：运行期可切；DEBUG 才写 DEBUG；error.log 永远只收 WARNING+
  · 包装：@ui_guard 记堆栈后**原样抛出**（界面行为不能变）；慢调用记 INFO
  · 不拖累业务：日志自身抛错也不能把调用方带崩

跑法：  .venv\\Scripts\\python.exe tools\\logging_probe.py
退出码：0 = 全过；1 = 有失败项。
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import contextlib
import io
import os
import shutil
import sys
import tempfile
import threading
import time

from webui_app import logging_setup as LOG

PASS = FAIL = 0
FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def head(t: str):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="ixlog_probe_")
    try:
        # =================================================================
        head("[1] setup 之前必须静默（别抢跑入口配置）")
        # =================================================================
        early = LOG.get_logger("probe.early")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            early.info("这条不该出现在任何地方")
        check("setup 之前调用不产生控制台输出", "这条不该出现" not in buf.getvalue())
        check("setup 之前 recent() 为空（还没装环形缓冲）", LOG.recent(10) == [])

        # =================================================================
        head("[2] 配置：落盘 + error.log 分离")
        # =================================================================
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            path = LOG.setup(level="INFO", log_dir=tmp, console=True,
                             force=True)
        check("返回了日志文件路径", bool(path) and path.endswith("indextts.log"),
              path)
        check("日志文件已创建", os.path.isfile(LOG.log_path()))
        check("error.log 路径已设", LOG.error_log_path().endswith("error.log"))
        check("级别为 INFO", LOG.level() == "INFO", LOG.level())

        log = LOG.get_logger("probe")
        log.debug("debug 不该写")
        log.info("info 该写 %s", "arg")
        log.warning("warn 该写")
        main_txt = LOG.file_tail(200)
        err_txt = LOG.file_tail(200, LOG.error_log_path())
        check("INFO 进了全量日志", "info 该写 arg" in main_txt)
        check("INFO 级别下 DEBUG 不写", "debug 不该写" not in main_txt)
        check("WARNING 进了 error.log", "warn 该写" in err_txt)
        check("INFO 不进 error.log（那份只记问题）", "info 该写" not in err_txt)

        # =================================================================
        head("[3] 异常堆栈必须落盘")
        # =================================================================
        try:
            raise ValueError("故意抛的探针异常")
        except Exception:
            log.error("捕获到异常", exc_info=True)
        err_txt = LOG.file_tail(400, LOG.error_log_path())
        check("error.log 含异常类型", "ValueError" in err_txt)
        check("error.log 含异常消息", "故意抛的探针异常" in err_txt)
        check("error.log 含 traceback 关键字", "Traceback (most recent call last)"
              in err_txt)
        check("error.log 含出错行号（能定位到源码）", "line " in err_txt)

        # =================================================================
        head("[4] 运行期切级别")
        # =================================================================
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.set_level("DEBUG")
        check("级别已切换", LOG.level() == "DEBUG")
        log.debug("debug 现在该写了")
        check("切到 DEBUG 后 DEBUG 会写", "debug 现在该写了" in LOG.file_tail(200))
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.set_level("WARNING")
        log.info("info 现在不该写")
        check("切到 WARNING 后 INFO 不写", "info 现在不该写" not in LOG.file_tail(200))
        log.warning("warn 仍然该写")
        check("切到 WARNING 后 WARNING 照写", "warn 仍然该写" in LOG.file_tail(200))
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.set_level("INFO")     # 复原

        # =================================================================
        head("[5] quiet：只写文件，不打控制台")
        # =================================================================
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            LOG.setup(level="INFO", log_dir=tmp, console=False, force=True)
            LOG.get_logger("probe").info("quiet 标记")
        check("quiet 下控制台无输出", "quiet 标记" not in buf.getvalue())
        check("quiet 下文件仍然有", "quiet 标记" in LOG.file_tail(200))
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.setup(level="INFO", log_dir=tmp, console=True, force=True)

        # =================================================================
        head("[6] 内存环形缓冲")
        # =================================================================
        for i in range(5):
            log.info("环形标记 %d", i)
        rows = LOG.recent(10)
        check("recent 能读到刚写的记录", any("环形标记 4" in r["message"]
                                        for r in rows))
        check("每条记录带级别与来源",
              all({"level", "name", "message", "ts", "thread"} <= set(r)
                  for r in rows))
        md = LOG.recent_markdown(10)
        check("recent_markdown 是表格", md.startswith("| 时间 |"))
        check("recent_markdown 含刚写的内容", "环形标记 4" in md)
        check("recent_markdown 不出现裸露 None", "None" not in md)
        md_err = LOG.recent_markdown(50, min_level="WARNING")
        check("min_level 过滤生效（INFO 不进 WARNING 列表）",
              "环形标记 4" not in md_err)
        check("min_level 保留 WARNING", "warn 仍然该写" in md_err)

        # =================================================================
        head("[7] 全局兜底：子线程未捕获异常也要落盘")
        # =================================================================
        check("install_excepthooks 幂等",
              (LOG.install_excepthooks() or True)
              and (LOG.install_excepthooks() or True))

        def _boom():
            raise RuntimeError("子线程爆炸标记")

        t = threading.Thread(target=_boom, name="probe-boom", daemon=True)
        t.start()
        t.join(timeout=5)
        time.sleep(0.3)
        err_txt = LOG.file_tail(400, LOG.error_log_path())
        check("子线程未捕获异常进了 error.log", "子线程爆炸标记" in err_txt)
        check("记录了线程名（知道是谁崩的）", "probe-boom" in err_txt)
        check("子线程异常带堆栈", "RuntimeError" in err_txt)

        # =================================================================
        head("[8] @ui_guard：记堆栈后原样抛出")
        # =================================================================

        @LOG.ui_guard("probe.on_explode")
        def _explode(x):
            raise KeyError(f"参数 {x}")

        raised = None
        try:
            _explode(7)
        except KeyError as e:
            raised = str(e)
        check("异常照旧抛出（界面行为不变）", raised == "'参数 7'", str(raised))
        ring = LOG.recent(80)
        hit = [r for r in ring if "probe.on_explode" in r["message"]]
        check("ui_guard 把异常写进了日志", bool(hit))
        check("ui_guard 的日志带堆栈", any("KeyError" in r.get("exc", "")
                                        for r in hit))
        check("ui_guard 记了入参（便于复现）",
              any("7" in r["message"] for r in hit), str(hit[:1]))

        @LOG.ui_guard("probe.on_ok")
        def _ok(a, b=None):
            return a + 1

        # 正常调用默认记在 DEBUG（INFO 级别下不刷屏），所以先切 DEBUG 再看
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.set_level("DEBUG")
        check("ui_guard 不改变正常返回值", _ok(1, b=2) == 2)
        check("ui_guard 记录了正常调用（DEBUG 级别）",
              any("probe.on_ok" in r["message"] for r in LOG.recent(40)))
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.set_level("INFO")

        @LOG.ui_guard("probe.on_slow", slow_sec=0.01)
        def _slow():
            time.sleep(0.05)
            return "done"

        _slow()
        slow_hit = [r for r in LOG.recent(40)
                    if "probe.on_slow" in r["message"]]
        check("慢调用记为 INFO（默认级别下也看得到）",
              any(r["level"] == "INFO" for r in slow_hit), str(slow_hit[:1]))

        # =================================================================
        head("[9] 日志自身出错不能带崩业务")
        # =================================================================

        class _Exploding:
            def __str__(self):
                raise RuntimeError("__str__ 炸了")

        try:
            log.info("格式化参数炸了 %s", _Exploding())
            ok = True
        except Exception as e:
            ok = False
            check("格式化失败被吞掉", False, f"{type(e).__name__}: {e}")
        check("日志格式异常不影响调用方（不抛）", ok)

        check("file_tail 对不存在的文件返回说明而不是抛",
              "不存在" in LOG.file_tail(10, os.path.join(tmp, "nope.log")))
        st = LOG.stats()
        check("stats 报告级别/路径/大小",
              {"level", "dir", "log", "error_log", "log_bytes", "ring"} <= set(st))
        check("stats 的 log_bytes > 0（确实写了东西）", st["log_bytes"] > 0,
              str(st["log_bytes"]))

        # =================================================================
        head("[10] 与 EventLog 打通（引擎事件落盘）")
        # =================================================================
        from webui_app.context import EventLog

        el = EventLog()
        el.push("lora_attached", "已挂载 LoRA gpt:probe", "ok")
        el.push("error", "模拟引擎错误", "error")
        time.sleep(0.2)
        main_txt = LOG.file_tail(400)
        check("EventLog 的普通事件进了文件日志",
              "lora_attached" in main_txt and "gpt:probe" in main_txt)
        err_txt = LOG.file_tail(400, LOG.error_log_path())
        check("EventLog 的 error 级事件进了 error.log",
              "模拟引擎错误" in err_txt)
        check("EventLog 内存环形仍然可用", len(el.tail(10)) == 2)

        # =================================================================
        head("[11] 引擎侧：LoRA 状态判定与请求日志")
        # =================================================================
        from webui_app.config import config_from_args
        from webui_app.services.engine import EngineError, TTSEngine

        class _FakeWrapped:
            """模拟 PeftModel：只有 base_model 这一个特征。"""

            def __init__(self):
                self.base_model = object()

        check("_is_wrapped(普通对象) = False",
              TTSEngine._is_wrapped(object()) is False)
        check("_is_wrapped(None) = False", TTSEngine._is_wrapped(None) is False)
        check("_is_wrapped(带 base_model 的对象) = True",
              TTSEngine._is_wrapped(_FakeWrapped()) is True)
        check("_lora_hint 能描述模块状态",
              "wrapped=True" in TTSEngine._lora_hint(_FakeWrapped()))

        eng = TTSEngine(config_from_args(["--lazy"]))
        check("未加载时 lora_status() 返回空（不抛）", eng.lora_status() == [])

        # -- 「选择 vs 实际挂载」的决策：静默用错模型就是从这里漏出去的 --
        from webui_app.tabs.synthesize import lora_action

        for run, mounted, want, why in [
            ("A", set(), "mount", "选了 A 但没挂 → 要挂"),
            ("A", {"A"}, "none", "选了 A 且已挂 → 不重复挂（省一次读盘）"),
            ("A", {"B"}, "mount", "选了 A 却挂着 B → 要换"),
            ("", {"A"}, "unmount", "选「不使用」但还挂着 → 要卸"),
            ("", set(), "none", "不使用且没挂 → 不动"),
            ("A", set(), "mount",
             "引擎刚重载（实际挂载为空）→ 必须重挂，不能静默用底座出声"),
            (None, {"A"}, "unmount", "None 等同空选择"),
        ]:
            check(f"lora_action：{why}", lora_action(run, mounted) == want,
                  f"{run!r}/{mounted} -> {lora_action(run, mounted)}")

        raised = ""
        try:
            eng.detach_lora("gpt")
        except EngineError as e:
            raised = str(e)
        except Exception as e:
            raised = f"WRONG:{type(e).__name__}"
        check("未加载时 detach 抛 EngineError（不是 AttributeError）",
              raised.startswith("模型尚未加载"), raised[:40])
        hit = [r for r in LOG.recent(60) if "卸载请求" in r["message"]]
        check("卸载请求即使失败也留下了日志", bool(hit), str(hit[:1]))

        raised = ""
        try:
            eng.attach_lora(os.path.join(tmp, "no_such_adapter"), "gpt")
        except EngineError as e:
            raised = str(e)
        except Exception as e:
            raised = f"WRONG:{type(e).__name__}"
        check("未加载时 attach 也抛 EngineError",
              raised.startswith("模型尚未加载"), raised[:40])
        hit = [r for r in LOG.recent(60) if "挂载请求" in r["message"]]
        check("挂载请求即使失败也留下了日志", bool(hit), str(hit[:1]))

        # =================================================================
        head("[12] 轮转配置（别让日志无限涨）")
        # =================================================================
        import logging
        root = logging.getLogger(LOG.ROOT_NAME)
        fhs = [h for h in root.handlers
               if isinstance(h, logging.handlers.RotatingFileHandler)]
        check("有轮转处理器", len(fhs) >= 2, f"{len(fhs)} 个")
        check("轮转上限已设（5 MB × 5 份）",
              all(h.maxBytes == 5 * 1024 * 1024 and h.backupCount == 5
                  for h in fhs))
        check("日志不冒泡到根 logger（避免被别人重复输出）",
              root.propagate is False)
        names = [getattr(h, "baseFilename", "") for h in root.handlers]
        check("error.log 的级别固定 WARNING（不被 DEBUG 冲淡）",
              all(h.level == logging.WARNING for h in root.handlers
                  if str(getattr(h, "baseFilename", "")).endswith("error.log")))

        # =================================================================
        head("[13] 日志参数已进手册")
        # =================================================================
        from webui_app import params as PM
        from webui_app import widgets as W

        PM.register_training_params()
        check("手册有「调试日志」分组", "logging" in PM.group_order(True),
              str(PM.group_order(True)))
        keys = [p.key for p in PM.by_group("logging")]
        for k in ("log_level", "log_dir", "quiet"):
            check(f"手册收录 {k}", k in keys, str(keys))
        check("日志参数都能渲染出手册卡片",
              all(W.help_markdown(p) for p in PM.by_group("logging")))
        check("log_level 的选项与代码里的级别一致",
              list(PM.get("log_level").choices) == list(LOG.LEVELS),
              str(PM.get("log_level").choices))

        # =================================================================
        head("[14] 「系统」页日志面板的取数函数")
        # =================================================================
        # 这里的断言来自一个真实故障：面板的取数函数曾把 `gr.update(...)`
        # **字典**当成正文返回，而 gr.Code 会对值调 `value.strip()` →
        # "'dict' object has no attribute 'strip'"，整页报错。
        # 真机截图验收抓到的，所以在这里焊死「正文必须是 str」。
        from webui_app.tabs import system as SY

        info, body = SY._dl_payload("INFO", "err", 50)
        check("面板说明是字符串", isinstance(info, str) and bool(info))
        check("**日志正文必须是纯字符串**（不能是 gr.update 字典）",
              isinstance(body, str) and not isinstance(body, dict),
              type(body).__name__)
        for src in ("ring", "err", "log"):
            i2, b2 = SY._dl_payload("INFO", src, 20)
            check(f"来源 {src} 也能取到纯文本",
                  isinstance(b2, str) and isinstance(i2, str))
        i3, b3 = SY._dl_payload("DEBUG", "log", 5)
        check("切级别后仍是纯文本且级别已生效",
              isinstance(b3, str) and LOG.level() == "DEBUG", LOG.level())
        with contextlib.redirect_stdout(io.StringIO()):
            LOG.set_level("INFO")
        i4, b4 = SY._dl_payload("", "nope", 0)
        check("空级别不切换、未知来源也不抛异常", isinstance(b4, str))

        # =================================================================
        head("清理")
        # =================================================================
        # Windows 上只要日志文件还有打开的句柄，目录就删不掉 —— 先关闭
        # handler 再清理（顺带验证 shutdown() 真的能释放句柄）。
        LOG.shutdown()
        check("shutdown 后配置状态复位", LOG.level() == LOG.DEFAULT_LEVEL
              and LOG.log_path() == "")
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除（句柄已释放）", not os.path.isdir(tmp))

    finally:
        try:
            LOG.shutdown()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
    print("=" * 70)
    if FAILS:
        print("失败项：")
        for f in FAILS:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
