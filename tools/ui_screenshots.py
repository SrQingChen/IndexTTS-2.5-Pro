"""为 README / 文档生成 WebUI 截图。

用系统自带的 Chrome/Edge 走 **CDP（Chrome DevTools Protocol）**，不引入
任何新依赖 —— `websockets` 已经随 gradio 装好了，而无头截图只需要
「开一个调试端口 → WebSocket 发两条命令」。

为什么不用 Playwright：为了 11 张文档截图去装一个 150 MB 的浏览器运行时
不划算，而且 Chrome 本来就装在机器上。

用法：
    # 1) 先启动界面（另一个窗口）
    .venv\\Scripts\\python.exe webui_pro.py --lazy

    # 2) 生成截图到 assets/ui/
    .venv\\Scripts\\python.exe tools\\ui_screenshots.py

    # 指定地址 / 输出目录 / 尺寸
    ... tools\\ui_screenshots.py --url http://127.0.0.1:7861 --out docs/img --width 1600

产物：
    assets/ui/00_overview.png      首屏（合成页 + 顶栏状态条）
    assets/ui/NN_<tab>.png         每个 Tab 一张
"""

from __future__ import annotations

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import argparse
import asyncio
import base64
import glob  # noqa: F401  (保留给后续按需筛选截图)
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

# 与 webui_app/app.py 的 TAB_SPECS 顺序一致；名字只用于文件名。
# 标签文本用于在页面上定位要点击的按钮（模糊匹配，前缀 emoji 可省）。
TABS = [
    ("syn", "01_synthesis", "合成"),
    ("lab", "02_audio_lab", "音频工作台"),
    ("batch", "03_batch", "批量"),
    ("preset", "04_presets", "预设"),
    ("data", "05_dataset", "数据集"),
    ("train", "06_training", "训练"),
    ("align", "07_alignment", "对齐"),
    ("deploy", "08_eval_deploy", "评测/部署"),
    ("model", "09_models", "模型"),
    ("sys", "10_system", "系统"),
    ("manual", "11_manual", "手册"),
]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    for p in CHROME_CANDIDATES:
        if os.path.isfile(p):
            return p
    env = os.environ.get("CHROME_PATH") or os.environ.get("EDGE_PATH")
    if env and os.path.isfile(env):
        return env
    raise FileNotFoundError(
        "找不到 Chrome/Edge。用 CHROME_PATH 环境变量指定浏览器可执行文件。")


def wait_for_port(port: int, timeout: float = 30.0) -> str:
    """等调试端口起来，返回 webSocketDebuggerUrl 的 host。"""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/version", timeout=1.0) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(0.4)
    raise TimeoutError(f"调试端口 {port} 在 {timeout}s 内没起来（{last}）")


def wait_for_page(port: int, want_url: str, timeout: float = 30.0) -> str:
    """从 /json/list 里找我们要的那个 page target 的 WebSocket 地址。

    不用 `/json/new`：新版 Chrome 要求它走 PUT，而且直接让浏览器带着 URL
    启动再找目标更省一次跳转（Gradio 首屏渲染本来就慢，少一次导航少一份不确定）。
    """
    t0 = time.time()
    last = []
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list", timeout=1.0) as r:
                targets = json.loads(r.read().decode("utf-8"))
            pages = [t for t in targets if t.get("type") == "page"
                     and t.get("webSocketDebuggerUrl")]
            last = [t.get("url") for t in pages]
            for t in pages:
                if str(t.get("url", "")).startswith(want_url[:22]):
                    return t["webSocketDebuggerUrl"]
            if pages:                       # 有页面但 URL 还没跟上，先用它
                return pages[0]["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise TimeoutError(f"没等到页面目标（看到的 URL：{last}）")


class CDP:
    """极简 CDP 客户端：够用就好（navigate / evaluate / screenshot）。"""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._id = 0
        self.ws = None

    async def __aenter__(self):
        import websockets
        # suppress_origin：Chrome 会拒绝带 Origin 头的 CDP 连接
        try:
            self.ws = await websockets.connect(self.ws_url,
                                               suppress_origin=True,
                                               max_size=64 * 1024 * 1024)
        except TypeError:
            self.ws = await websockets.connect(self.ws_url,
                                               max_size=64 * 1024 * 1024)
        return self

    async def __aexit__(self, *exc):
        if self.ws is not None:
            await self.ws.close()

    async def call(self, method: str, params: dict | None = None,
                   timeout: float = 60.0) -> dict:
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method,
                                       "params": params or {}}))
        t0 = time.time()
        while time.time() - t0 < timeout:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            if msg.get("id") != mid:
                continue                     # 事件通知，跳过
            if "error" in msg:
                raise RuntimeError(f"CDP {method} 失败：{msg['error']}")
            return msg.get("result") or {}
        raise TimeoutError(f"CDP {method} 超时")

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate",
                           {"expression": expr, "returnByValue": True,
                            "awaitPromise": True})
        res = r.get("result") or {}
        if r.get("exceptionDetails"):
            raise RuntimeError(f"页面 JS 异常：{r['exceptionDetails']}")
        return res.get("value")


# Gradio 5 的 Tab 栏在 DOM 里有**两份**同名按钮：
#   · 一组高度 1px 的隐藏副本（溢出菜单占位，无 role、无 aria-selected）
#   · 一组高度 32px 的真实 tab（role="tab" + aria-selected）
# 二者在文档顺序上前者在前，所以「按文本找 button」必然点到死元素 ——
# 必须同时卡 role="tab" 与真实高度，否则 11 张截图全是同一页而不报错。
CLICK_JS = """
(() => {
  const want = %s;
  const cands = [...document.querySelectorAll('button[role="tab"]')]
    .filter(e => e.getBoundingClientRect().height > 4);
  const norm = e => (e.textContent || '').replace(/\\s+/g, ' ').trim();
  const el = cands.find(e => norm(e).includes(want));
  if (!el) {
    return JSON.stringify({ok: false, seen: cands.map(norm)});
  }
  el.scrollIntoView({block: 'center'});
  el.click();
  return JSON.stringify({ok: true, text: norm(el)});
})()
"""

SEL_JS = """
(() => {
  const t = [...document.querySelectorAll('button[role="tab"][aria-selected="true"]')]
    .filter(e => e.getBoundingClientRect().height > 4)[0];
  return t ? (t.textContent || '').replace(/\\s+/g, ' ').trim() : '';
})()
"""

WAIT_JS = """
(() => {
  const want = %s;
  const t = [...document.querySelectorAll('button[role="tab"][aria-selected="true"]')]
    .filter(e => e.getBoundingClientRect().height > 4)[0];
  return t ? (t.textContent || '').replace(/\\s+/g, ' ').trim().includes(want) : false;
})()
"""


async def capture(url: str, out_dir: str, width: int, height: int,
                  keep_browser: bool = False) -> int:
    browser = find_browser()
    port = 9333
    profile = os.path.join(os.environ.get("TEMP", "."), "_ui_shot_profile")
    shutil.rmtree(profile, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    args = [
        browser, "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}", f"--window-size={width},{height}",
        "--hide-scrollbars", "--disable-gpu", "--no-first-run",
        "--no-default-browser-check", "--disable-extensions",
        "--force-device-scale-factor=1", url,
    ]
    print(f"  浏览器：{browser}")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        info = wait_for_port(port)
        print(f"  调试端口就绪：{info.get('Browser')}")

        ws_url = wait_for_page(port, url)

        got = 0
        async with CDP(ws_url) as cdp:
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Emulation.setDeviceMetricsOverride",
                           {"width": width, "height": height,
                            "deviceScaleFactor": 1, "mobile": False})
            await cdp.call("Page.navigate", {"url": url})
            await asyncio.sleep(6.0)         # 等 Gradio 首屏渲染

            for i, (tab_id, fname, label) in enumerate(TABS):
                if i > 0:
                    res = await cdp.js(CLICK_JS % json.dumps(label))
                    try:
                        r = json.loads(res) if res else {}
                    except Exception:
                        r = {}
                    if not r.get("ok"):
                        print(f"  [!] 点不到「{label}」，跳过（可见 tab："
                              f"{r.get('seen')}）")
                        continue
                    # 等它真的变成选中态，否则截到的还是上一页
                    ok = False
                    for _ in range(20):
                        await asyncio.sleep(0.25)
                        if await cdp.js(WAIT_JS % json.dumps(label)) is True:
                            ok = True
                            break
                    if not ok:
                        print(f"  [!] 「{label}」点了但没切过去，跳过")
                        continue
                    await asyncio.sleep(1.2)   # 等面板内容渲染完
                shot = await cdp.call("Page.captureScreenshot",
                                      {"format": "png",
                                       "captureBeyondViewport": True})
                path = os.path.join(out_dir, f"{fname}.png")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(shot["data"]))
                size_kb = os.path.getsize(path) / 1024
                active = await cdp.js(SEL_JS)
                print(f"  [{i+1}/{len(TABS)}] {label:10s} → {fname}.png "
                      f"（{size_kb:.0f} KB，当前选中：{active}）")
                got += 1

        # 顺带把首屏单独存一份 overview（README 顶部用）
        first = os.path.join(out_dir, f"{TABS[0][1]}.png")
        if os.path.isfile(first):
            shutil.copy2(first, os.path.join(out_dir, "00_overview.png"))
        print(f"  完成：{got} 张 → {out_dir}")
        return got
    finally:
        if not keep_browser:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 WebUI 截图")
    ap.add_argument("--url", default="http://127.0.0.1:7860",
                    help="界面地址（默认 http://127.0.0.1:7860）")
    ap.add_argument("--out", default=os.path.join("assets", "ui"),
                    help="输出目录（默认 assets/ui）")
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1050)
    ap.add_argument("--keep-browser", action="store_true",
                    help="截图后不关浏览器（调试用）")
    a = ap.parse_args()

    # 先确认界面活着，省得让用户对着超时发呆
    try:
        with urllib.request.urlopen(a.url, timeout=5) as r:
            code = r.status
    except Exception as e:
        print(f"✖ 打不开 {a.url}：{type(e).__name__}: {e}")
        print("  请先另开一个窗口启动界面：")
        print("      .venv\\Scripts\\python.exe webui_pro.py --lazy")
        return 2
    print(f"✓ 界面在线（HTTP {code}）：{a.url}")
    n = asyncio.run(capture(a.url, a.out, a.width, a.height,
                            a.keep_browser))
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
