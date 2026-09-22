"""音频工作台「智能切片候选」交互的真机端到端验收。

针对的回归场景（用户实测反馈）：

    长音频切分后列出所有候选 → 选中候选**直接试听**；
    试听 / 导出都**不得**替换主素材；只有显式「设为主素材」才替换。

流程（全自动：自己造素材、自己起服务器、自己开无头浏览器）：

    0. 用 numpy 合成 40 秒「音节串」音频（0.8s 音 + 0.25s 静音循环），
       先离线验证 find_segments 能出候选，不行就立刻报错退出
    1. 起服务器（--lazy）→ 浏览器进工作台
    2. 路径方式载入素材 → 断言主素材播放器有值
    3. 「扫描候选片段」→ 断言：候选下拉有选项，且自动选中 #0 时
       **预览播放器立即加载 _seg0**（选中即试听）
    4. 下拉切到 #1 → 断言：预览变为 _seg1；**主素材未被替换**；
       候选列表仍然在
    5. 点「📤 导出该片段」→ 断言：预览仍是 _seg1、主素材仍未替换
    6. 点「设为主素材」→ 断言：主素材**这时才**被替换为片段文件，
       候选列表被作废清空（显式操作的预期行为）

    .venv\\Scripts\\python.exe tools\\lab_ui_acceptance.py
"""

from __future__ import annotations

import _env                                            # noqa: F401  路径 + 控制台编码

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

from ui_screenshots import CDP, find_browser           # noqa: E402

PORT = 7867
URL = f"http://127.0.0.1:{PORT}"
STEM = f"zz_lab_e2e_{time.strftime('%H%M%S')}"
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def make_synthetic_wav(path: str, seconds: float = 40.0) -> None:
    """合成「音节串」：0.8 秒 220Hz 三和音 + 0.25 秒静音循环。"""
    import numpy as np
    import soundfile as sf
    sr = 22050
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # 用相位不连续的音节边界模拟辅音起声，让 voiced_ratio / 停顿检测都有信号
    gate = (np.arange(n) / sr) % 1.05 < 0.8
    wave = (0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.15 * np.sin(2 * np.pi * 440 * t)
            + 0.08 * np.sin(2 * np.pi * 660 * t))
    x = (wave * gate * 0.8 * 32767).astype(np.int16)
    sf.write(path, x, sr)


def wait_http(url: str, timeout: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def start_server() -> subprocess.Popen:
    py = os.path.join(".venv", "Scripts", "python.exe")
    proc = subprocess.Popen(
        [py, "webui_pro.py", "--lazy", "--port", str(PORT), "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_http(URL):
        proc.terminate()
        raise RuntimeError(f"服务器在 {URL} 没起来")
    return proc


# ---------- 页面 JS ----------
SET_TEXTBOX_JS = """
(() => {
  const lab = [...document.querySelectorAll('label')]
    .find(l => (l.textContent || '').includes('或直接填服务器上的文件路径'));
  if (!lab) return 'nolabel';
  let box = lab;
  for (let i = 0; i < 8; i++) {
    box = box.parentElement;
    if (box && box.querySelector('input,textarea')) break;
  }
  const input = box.querySelector('input,textarea');
  if (!input) return 'noinput';
  const proto = input.tagName === 'TEXTAREA'
    ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(input, %s);
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  return 'ok';
})()
"""

CLICK_JS = """
(() => {
  const want = %s;
  const btns = [...document.querySelectorAll('button')];
  const norm = e => (e.textContent || '').replace(/\\s+/g, ' ').trim();
  const el = btns.find(e => norm(e).includes(want));
  if (!el) return JSON.stringify({ok: false, seen: btns.map(norm).slice(0, 30)});
  el.click();
  return JSON.stringify({ok: true});
})()
"""

# 切 Tab 专用：Gradio 的 Tab 栏在 DOM 里有**两份**同名按钮（1px 隐藏副本 +
# 真实副本），必须卡 role="tab" 与真实高度，否则点到死元素上页面不会切换
# （与 tools/ui_screenshots.py 的 CLICK_JS 同一教训）。
TAB_CLICK_JS = """
(async () => {
  const want = %s;
  const cands = [...document.querySelectorAll('button[role="tab"]')]
    .filter(e => e.getBoundingClientRect().height > 4);
  const norm = e => (e.textContent || '').replace(/\\s+/g, ' ').trim();
  const el = cands.find(e => norm(e).includes(want));
  if (!el) return 'notab:' + cands.map(norm).join(',');
  el.scrollIntoView({block: 'center'});
  el.click();
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 300));
    const cur = [...document.querySelectorAll('button[role="tab"][aria-selected="true"]')]
      .filter(e => e.getBoundingClientRect().height > 4)[0];
    if (cur && norm(cur).includes(want)) return 'ok';
  }
  return 'noswitch';
})()
"""

# 读音频组件当前值。gr.Audio 的 <audio> 元素是懒创建的，读 src 不稳；
# 下载链接（a[download]）在值加载时就渲染，文件名即当前值 —— 用它当探针。
AUDIO_JS = """
(() => {
  const wrap = document.getElementById(%s);
  if (!wrap) return JSON.stringify({err: 'no-wrap'});
  const a = wrap.querySelector('a[download]');
  return JSON.stringify({
    name: a ? a.getAttribute('download') : null,
    href: a ? a.getAttribute('href') : null,
  });
})()
"""

# 下载按钮存在性：gradio 音频播放器的下载控件（a[download] 或 aria-label 含
# Download 的按钮；不同小版本实现有差异，两个都查）
DOWNLOAD_JS = """
(() => {
  const wrap = document.getElementById(%s);
  if (!wrap) return JSON.stringify({err: 'no-wrap'});
  const a = wrap.querySelector('a[download]');
  const btn = [...wrap.querySelectorAll('button')]
    .find(b => /download/i.test(b.getAttribute('aria-label') || '')
             || /download/i.test(b.getAttribute('title') || '')
             || /下载/.test(b.getAttribute('aria-label') || '')
             || /下载/.test(b.getAttribute('title') || ''));
  return JSON.stringify({
    has: !!(a || btn),
    a: a ? (a.getAttribute('download') || 'a') : null,
    btn: btn ? (btn.getAttribute('aria-label') || btn.getAttribute('title')) : null,
  });
})()
"""

SELECT_SEG_JS = """
(async () => {
  const wrap = document.getElementById('ix-lab-seg-pick');
  if (!wrap) return 'nowrap';
  const input = wrap.querySelector('input');
  if (!input) return 'noinput';
  input.focus();
  input.click();
  await new Promise(r => setTimeout(r, 600));
  const opts = [...document.querySelectorAll('li,[role="option"]')];
  // 选中项带 “✓ ” 前缀。用 includes('#N ')（后跟空格）匹配：
  // '#1  23.5s' 含 '#1 '，而 '#10 x' 的 '#1' 后是数字，不会误中。
  const want = %s + ' ';
  const target = opts.find(o => (o.textContent || '').includes(want));
  if (!target) return 'noopt:' + JSON.stringify(opts.map(o => o.textContent.trim()).slice(0, 10));
  target.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
  target.click();
  return 'ok';
})()
"""

SEG_OPTIONS_JS = """
(async () => {
  const wrap = document.getElementById('ix-lab-seg-pick');
  if (!wrap) return 'nowrap';
  const input = wrap.querySelector('input');
  input.focus();
  input.click();
  await new Promise(r => setTimeout(r, 600));
  const opts = [...document.querySelectorAll('li,[role="option"]')]
    .map(o => (o.textContent || '').trim());
  return JSON.stringify(opts);
})()
"""


async def js_json(cdp, js):
    raw = await cdp.js(js)
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {"raw": raw}


async def wait_audio(cdp, elem_id: str, contains: str, timeout: float = 25.0):
    """等某音频组件的当前文件名里出现指定子串（轮询直到超时）。"""
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout:
        last = await js_json(cdp, AUDIO_JS % json.dumps(elem_id))
        name = (last.get("name") or "") + " " + (last.get("href") or "")
        if contains in name:
            return True, last
        await asyncio.sleep(1.0)
    return False, last


async def with_page(coro):
    browser = find_browser()
    profile = os.path.join(os.environ.get("TEMP", "."), "_lab_e2e_profile")
    shutil.rmtree(profile, ignore_errors=True)
    proc = subprocess.Popen(
        [browser, "--headless=new", "--remote-debugging-port=9338",
         f"--user-data-dir={profile}", "--window-size=1680,1050",
         "--hide-scrollbars", "--disable-gpu", "--no-first-run", URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with urllib.request.urlopen("http://127.0.0.1:9338/json/list", timeout=10) as r:
            targets = json.loads(r.read().decode())
        ws_url = next((t["webSocketDebuggerUrl"] for t in targets
                       if t.get("type") == "page"), None)
        if not ws_url:
            raise RuntimeError("浏览器页面目标没找到")
        async with CDP(ws_url) as cdp:
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await asyncio.sleep(7.0)     # 等 Gradio 首屏渲染完
            # 工作台不是默认 Tab，先进它（等它真的变成选中态）
            r = await cdp.js(TAB_CLICK_JS % json.dumps("音频工作台"))
            if r != "ok":
                raise RuntimeError(f"切不到「音频工作台」Tab：{r}")
            await asyncio.sleep(2.0)
            return await coro(cdp)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


async def run_flow(cdp, wav_path: str):
    # ---------- 载入素材 ----------
    set_r = await cdp.js(SET_TEXTBOX_JS % json.dumps(wav_path))
    check("填入素材路径", set_r == "ok", str(set_r))
    await asyncio.sleep(0.5)
    r = json.loads(await cdp.js(CLICK_JS % json.dumps("用该路径")) or "{}")
    check("点「用该路径」", r.get("ok") is True)
    ok, st = await wait_audio(cdp, "ix-lab-src", STEM, timeout=20)
    check("主素材播放器已载入长音频", ok, str(st))
    dl = await js_json(cdp, DOWNLOAD_JS % json.dumps("ix-lab-src"))
    check("主素材（输入侧）播放器带下载按钮", dl.get("has") is True, str(dl))

    # ---------- 扫描候选 ----------
    r = json.loads(await cdp.js(CLICK_JS % json.dumps("扫描候选片段")) or "{}")
    check("点「扫描候选片段」", r.get("ok") is True)
    ok, st = await wait_audio(cdp, "ix-lab-seg-preview", "_seg0", timeout=30)
    check("扫描后自动选中 #0 且预览立即加载（选中即试听）", ok, str(st))
    dl2 = await js_json(cdp, DOWNLOAD_JS % json.dumps("ix-lab-seg-preview"))
    check("片段预览（输出侧）播放器带下载按钮", dl2.get("has") is True, str(dl2))
    opts = await js_json(cdp, SEG_OPTIONS_JS)
    import re as _re
    seg_opts = [t for t in (opts if isinstance(opts, list) else [])
                if _re.search(r"#\d", t)]
    check("候选下拉列出多个候选", len(seg_opts) >= 2,
          f"原始选项：{opts}" if len(seg_opts) < 2 else f"{len(seg_opts)} 个候选")
    src_before = str(await js_json(cdp, AUDIO_JS % json.dumps("ix-lab-src")))

    # ---------- 切到 #1 试听 ----------
    sel = await cdp.js(SELECT_SEG_JS % json.dumps("#1"))
    check("下拉选择候选 #1", sel == "ok", str(sel))
    ok, st = await wait_audio(cdp, "ix-lab-seg-preview", "_seg1", timeout=25)
    check("预览自动切换到 #1 片段", ok, str(st))
    src_after_pick = str(await js_json(cdp, AUDIO_JS % json.dumps("ix-lab-src")))
    check("试听 #1 后主素材未被替换（仍是长音频）",
          src_after_pick == src_before,
          f"{src_before} → {src_after_pick}")
    opts2 = await js_json(cdp, SEG_OPTIONS_JS)
    import re as _re
    n2 = len([t for t in (opts2 if isinstance(opts2, list) else [])
              if _re.search(r"#\d", t)])
    check("候选列表在试听后仍然有效", n2 >= 2, f"剩 {n2} 个")

    # ---------- 显式导出 ----------
    r = json.loads(await cdp.js(CLICK_JS % json.dumps("导出该片段")) or "{}")
    check("点「📤 导出该片段」", r.get("ok") is True)
    await asyncio.sleep(2.5)
    st = await js_json(cdp, AUDIO_JS % json.dumps("ix-lab-seg-preview"))
    check("导出后预览仍是 #1 片段", "_seg1" in (st.get("name") or ""), str(st))
    src_after_export = str(await js_json(cdp, AUDIO_JS % json.dumps("ix-lab-src")))
    check("导出不替换主素材", src_after_export == src_before,
          f"{src_before} → {src_after_export}")

    # ---------- 设为主素材（显式替换，允许发生） ----------
    r = json.loads(await cdp.js(CLICK_JS % json.dumps("设为主素材")) or "{}")
    check("点「设为主素材」", r.get("ok") is True)
    ok, st = await wait_audio(cdp, "ix-lab-src", "_seg1", timeout=25)
    check("显式操作后主素材才被替换为片段", ok, str(st))
    opts3 = await js_json(cdp, SEG_OPTIONS_JS)
    import re as _re
    n3 = len([t for t in (opts3 if isinstance(opts3, list) else [])
              if _re.search(r"#\d", t)])
    check("主素材更换后候选列表被作废清空", n3 == 0, f"剩 {n3} 个")


def main() -> int:
    from webui_app.services import audio_lab as AL

    wav_path = os.path.abspath(os.path.join(tempfile.gettempdir(), f"{STEM}.wav"))
    make_synthetic_wav(wav_path)

    # 先离线验证合成素材能扫出候选，省得白起一轮服务器
    segs = AL.find_segments(wav_path, target_sec=12.0, min_sec=6.0)
    if len(segs) < 2:
        print(f"✖ 合成素材只扫出 {len(segs)} 个候选，验收前提不成立，中止。")
        return 2

    server = None
    try:
        print(f"[1] 启动服务器 {URL}")
        server = start_server()
        asyncio.run(with_page(lambda cdp: run_flow(cdp, wav_path)))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()
        try:
            os.remove(wav_path)
        except OSError:
            pass
        # 清掉验收在工作目录里产生的片段文件
        import glob as _glob
        for p in _glob.glob(os.path.join("outputs", "lab", f"{STEM}_seg*")):
            try:
                os.remove(p)
            except OSError:
                pass

    print("\n" + "=" * 64)
    print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
    if FAIL:
        for f in FAIL:
            print(f"    FAIL {f}")
    print("=" * 64)
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
