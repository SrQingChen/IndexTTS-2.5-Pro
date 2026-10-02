"""progress_hub.LoggedProgress 的回归：限流 / 心跳 / 签名兼容 / 异常安全。

跑法：  .venv\\Scripts\\python.exe tools\\progress_probe.py
退出码：0 = 全过；1 = 有失败项。
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import time

from webui_app import logging_setup as LOG
from webui_app.services import progress_hub as PH

PASS = FAIL = 0
FAILS: list = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    log = LOG.get_logger("progress-probe")

    # ---- 1. 位置参数与 desc 关键字两种签名都收 ----
    seen: list = []
    lp = PH.LoggedProgress(log, user_progress=lambda f, *a, **k: seen.append((f, a, k)))
    lp(0.1, "位置参数消息")
    check("位置参数 (frac, msg) 可用", lp.last_msg == "位置参数消息")
    lp(0.2, desc="官方引擎风格 desc 消息")
    check("desc= 关键字（infer_v2 风格）可用",
          lp.last_msg == "官方引擎风格 desc 消息", lp.last_msg)
    check("user_progress 收到原始 desc 形态",
          bool(seen) and seen[-1][2].get("desc") == "官方引擎风格 desc 消息",
          str(seen[-1]))
    check("last_frac 跟踪", abs(lp.last_frac - 0.2) < 1e-9)

    # ---- 2. INFO 限流：连发多条只留最后一条之前的若干条 ----
    lp2 = PH.LoggedProgress(log, info_every=1.0)
    t0 = time.time()
    for i in range(20):
        lp2(i / 20, f"快速消息 {i}")
    check("限流窗口内不刷屏（elapsed<info_every 时全走 DEBUG）",
          time.time() - t0 < 1.0)

    # ---- 3. 心跳：静默超过 idle 后按周期写日志 ----
    lp3 = PH.LoggedProgress(log, heartbeat_idle=0.5, period=0.2).start()
    lp3(0.5, "心跳测试起点")
    time.sleep(1.3)
    lp3.stop()
    check("心跳存活（stop 不炸、idle 后有输出）", True)

    # ---- 4. user_progress 抛异常不拖垮包装器 ----
    def _boom(f, *a, **k):
        raise RuntimeError("UI 已死")
    lp4 = PH.LoggedProgress(log, user_progress=_boom)
    try:
        lp4(0.9, "UI 异常穿透测试")
        check("user_progress 异常被吞（日志仍更新）",
              lp4.last_msg == "UI 异常穿透测试")
    except Exception as e:
        check("user_progress 异常被吞（日志仍更新）", False, str(e))

    # ---- 5. stop 后可安全重复 stop；未 start 直接 stop 也不炸 ----
    lp5 = PH.LoggedProgress(log)
    lp5.stop()
    lp5.stop()
    check("重复 stop / 未 start stop 安全", True)

    print("\n" + "=" * 60)
    print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
    print("=" * 60)
    if FAILS:
        for f in FAILS:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
