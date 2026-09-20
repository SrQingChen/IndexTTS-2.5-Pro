"""参数记忆的真机端到端验收。

流程（全自动，自己起服务器、自己开浏览器）：

    A. 起服务器（--lazy 不加载模型）→ 无头浏览器进合成页
       1. 读初始值（应为注册表默认：temperature 0.8、vec 全 0）
       2. 点「⚡ 快速档」→ 8 个采样参数被程序化设置且落盘
       3. 点「随机一组」→ 8 维情感向量落盘
       4. 填名称点「💾 存为配置档」→ outputs/presets/ 出现、下拉出现
    B. 杀掉服务器进程（模拟重启）→ 重新起
       5. 读首屏值：temperature=0.7、num_beams=1、vec 有非零、记忆 chip 显示已恢复
       6. 下拉里能看到配置档
       7. 两步删除配置档（第一次布防、第二次真删）
       8. 点「清除记忆并恢复默认」→ 记忆文件消失、界面回默认
    C. 收尾：杀服务器，恢复验收前的真实记忆文件

    .venv\\Scripts\\python.exe tools\\mem_acceptance.py
"""

from __future__ import annotations

import _env                                            # noqa: F401  路径 + 控制台编码

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from ui_screenshots import CDP, find_browser           # noqa: E402  复用同目录的极简 CDP

PORT = 7866
URL = f"http://127.0.0.1:{PORT}"
PROFILE = f"e2e记忆验收_{time.strftime('%H%M%S')}"
PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


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


READ_JS = """
(() => {
  const slider = (id) => {
    const wrap = document.getElementById(id);
    if (!wrap) return null;
    const inputs = [...wrap.querySelectorAll('input')];
    for (const inp of inputs) {
      if (inp.value !== undefined && inp.value !== '') return inp.value;
    }
    return inputs.length ? inputs[0].value : null;
  };
  const chips = [...document.querySelectorAll('.ix-chip')]
    .map(c => c.textContent.replace(/\\s+/g, ' ').trim());
  return JSON.stringify({
    temperature: slider('ix-temperature'),
    num_beams: slider('ix-num_beams'),
    top_p: slider('ix-top_p'),
    vec: [0,1,2,3,4,5,6,7].map(i => {
      const v = slider('ix-emo_vec_' + i);
      return v === null ? null : parseFloat(v);
    }),
    dur: slider('ix-duration_factor'),
    chips,
  });
})()
"""

# 不做高度过滤：折叠面板 / 隐藏分组里的按钮也要能点（JS click 对
# display:none 的元素同样派发事件，Gradio 的监听器照常触发）。
# 「快速档 / 随机一组」这类文案全应用唯一，直接全文匹配即可。
CLICK_JS = """
(() => {
  const want = %s;
  const btns = [...document.querySelectorAll('button')];
  const norm = e => (e.textContent || '').replace(/\\s+/g, ' ').trim();
  const el = btns.find(e => norm(e).includes(want));
  if (!el) return JSON.stringify({ok: false, seen: btns.map(norm).slice(0, 40)});
  el.click();
  return JSON.stringify({ok: true});
})()
"""

# 在某 elem_id 的祖先范围内找按钮：防止点到别的 Tab 里同名按钮（如 🗑 删除）
CLICK_SCOPED_JS = """
(() => {
  const want = %s, scopeId = %s;
  let box = document.getElementById(scopeId);
  if (!box) return JSON.stringify({ok: false, why: 'scope not found'});
  for (let i = 0; i < 12 && box; i++) {
    const btn = [...box.querySelectorAll('button')]
      .find(e => (e.textContent || '').replace(/\\s+/g, ' ').trim().includes(want));
    if (btn) { btn.click(); return JSON.stringify({ok: true}); }
    box = box.parentElement;
  }
  return JSON.stringify({ok: false, why: 'button not found near scope'});
})()
"""

# 通用下拉选择：打开 elem_id 对应的下拉并点中包含指定文本的选项
# （Gradio 的选项列表渲染在全局 portal 里，不在组件内部）
SELECT_DROPDOWN_JS = """
(async () => {
  const wrap = document.getElementById(%s);
  if (!wrap) return 'nowrap';
  const input = wrap.querySelector('input');
  if (!input) return 'noinput';
  input.focus();
  input.click();
  await new Promise(r => setTimeout(r, 500));
  const opts = [...document.querySelectorAll('li,[role="option"]')];
  const target = opts.find(o => (o.textContent || '').includes(%s));
  if (!target) return 'noopt:' + opts.length;
  target.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
  target.click();
  return 'ok';
})()
"""

SET_NAME_JS = """
(() => {
  const lab = [...document.querySelectorAll('label')]
    .find(l => (l.textContent || '').includes('存为配置档'));
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

# 勾/取消「记住参数改动」复选框；want = 'true'/'false' 读当前勾选状态
TOGGLE_REMEMBER_JS = """
(() => {
  const lab = [...document.querySelectorAll('label')]
    .find(l => (l.textContent || '').includes('记住参数改动'));
  if (!lab) return 'nolabel';
  const input = lab.querySelector('input[type="checkbox"]');
  if (!input) return 'noinput';
  if (%s === 'read') return String(input.checked);
  input.click();
  return 'ok';
})()
"""

# 打开配置档下拉并选中指定项。定位用 elem_id（gradio 的 Dropdown 标签
# 不是 <label> 元素，靠文本找不稳）。
SELECT_PROFILE_JS = """
(async () => {
  const wrap = document.getElementById('ix-profile-dd');
  if (!wrap) return 'nowrap';
  const input = wrap.querySelector('input');
  if (!input) return 'noinput';
  input.focus();
  input.click();
  await new Promise(r => setTimeout(r, 600));
  const opts = [...document.querySelectorAll('li,[role="option"]')];
  const target = opts.find(o => (o.textContent || '').includes(%s));
  if (!target) return 'noopt:' + opts.length;
  target.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
  target.click();
  return 'ok';
})()
"""


async def with_page(coro):
    """起一个无头浏览器跑 coro(cdp)。"""
    browser = find_browser()
    profile = os.path.join(os.environ.get("TEMP", "."), "_mem_e2e_profile")
    shutil.rmtree(profile, ignore_errors=True)
    port = 9337
    proc = subprocess.Popen(
        [browser, "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--window-size=1680,1050",
         "--hide-scrollbars", "--disable-gpu", "--no-first-run", URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
            targets = json.loads(r.read().decode())
        ws_url = next((t["webSocketDebuggerUrl"] for t in targets
                       if t.get("type") == "page"), None)
        if not ws_url:
            raise RuntimeError("浏览器页面目标没找到")
        async with CDP(ws_url) as cdp:
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await asyncio.sleep(7.0)     # 等 Gradio 首屏渲染
            return await coro(cdp)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


async def phase_a(cdp) -> dict:
    """首次启动：默认值 → 改参数 → 存配置档。返回首读结果。"""
    raw = await cdp.js(READ_JS)
    first = json.loads(raw) if raw else {}
    print(f"    首读: temp={first.get('temperature')} beams={first.get('num_beams')} "
          f"vec={first.get('vec')}")

    # 先把情感模式切到「8 维向量」：向量分组 visible=False 时不在 DOM 里，
    # 「随机一组」按钮与向量滑杆都要等分组渲染出来才能操作
    sel = await cdp.js(SELECT_DROPDOWN_JS % (
        json.dumps("ix-emo_control_method"),
        json.dumps("使用 8 维情感向量")))
    check("情感模式切到「8 维向量」", sel == "ok", str(sel))
    await asyncio.sleep(1.2)

    for label in ("⚡ 快速档", "随机一组"):
        r = json.loads(await cdp.js(CLICK_JS % json.dumps(label)) or "{}")
        check(f"点「{label}」成功", r.get("ok") is True, str(r.get("seen"))[:200])
        await asyncio.sleep(1.5)

    r = json.loads(await cdp.js(READ_JS) or "{}")
    check("快速档已生效（temperature=0.7 / beams=1）",
          r.get("temperature") == "0.7" and r.get("num_beams") == "1",
          f"temp={r.get('temperature')} beams={r.get('num_beams')}")
    vec = [v for v in (r.get("vec") or []) if v is not None]
    check("随机情感向量已生效（8 维中至少一维非零）",
          any((v or 0) > 0 for v in vec), f"vec={vec}")

    set_r = await cdp.js(SET_NAME_JS % json.dumps(PROFILE))
    check("填写配置档名称", set_r == "ok", str(set_r))
    await asyncio.sleep(0.8)
    r2 = json.loads(await cdp.js(CLICK_JS % json.dumps("存为配置档")) or "{}")
    check("点「💾 存为配置档」", r2.get("ok") is True)
    await asyncio.sleep(1.5)
    return first


async def phase_b(cdp):
    """重启后：参数应恢复；然后选档、删档、清记忆。"""
    raw = await cdp.js(READ_JS)
    r = json.loads(raw) if raw else {}
    print(f"    重启后读: temp={r.get('temperature')} beams={r.get('num_beams')} "
          f"vec={r.get('vec')}")
    check("重启后 temperature 恢复为 0.7", r.get("temperature") == "0.7",
          str(r.get("temperature")))
    check("重启后 num_beams 恢复为 1", r.get("num_beams") == "1",
          str(r.get("num_beams")))
    check("重启后情感向量恢复",
          any((v or 0) > 0 for v in (r.get("vec") or [])))
    chips = " | ".join(r.get("chips") or [])
    check("记忆状态 chip 显示「已恢复」", "已恢复" in chips, chips[:160])

    # 先在下拉里选中配置档（删档的前置条件）
    sel = await cdp.js(SELECT_PROFILE_JS % json.dumps(PROFILE))
    check("配置档下拉可选中所存档位", sel == "ok", str(sel))
    await asyncio.sleep(1.0)

    # 两步删除：第一次布防、第二次真删（范围限定在配置档面板内）
    r1 = json.loads(await cdp.js(
        CLICK_SCOPED_JS % (json.dumps("删除"), json.dumps("ix-profile-dd"))) or "{}")
    await asyncio.sleep(0.8)
    r2 = json.loads(await cdp.js(
        CLICK_SCOPED_JS % (json.dumps("删除"), json.dumps("ix-profile-dd"))) or "{}")
    check("两步删除配置档的两次点击都成功",
          r1.get("ok") is True and r2.get("ok") is True,
          f"{r1} / {r2}")
    await asyncio.sleep(1.0)

    r3 = json.loads(await cdp.js(
        CLICK_JS % json.dumps("清除记忆并恢复默认")) or "{}")
    check("点「清除记忆并恢复默认」", r3.get("ok") is True)
    await asyncio.sleep(2.0)
    after = json.loads(await cdp.js(READ_JS) or "{}")
    check("清除后界面回到默认（temperature=0.8 / vec=0）",
          after.get("temperature") == "0.8"
          and not any((v or 0) for v in (after.get("vec") or [])),
          f"temp={after.get('temperature')} vec={after.get('vec')}")

    # 关闭「记住参数改动」：开关状态本身要能跨重启保持（阶段 C 验证）
    off = await cdp.js(TOGGLE_REMEMBER_JS % json.dumps("click"))
    check("取消勾选「记住参数改动」", off == "ok", str(off))
    await asyncio.sleep(1.2)


async def phase_c(cdp):
    """第三次重启后：记忆应保持关闭——不恢复参数、开关仍为未勾选。"""
    raw = await cdp.js(READ_JS)
    r = json.loads(raw) if raw else {}
    chips = " | ".join(r.get("chips") or [])
    check("关闭状态跨重启保持（chip 显示已关闭）", "已关闭" in chips, chips[:160])
    check("关闭状态下不恢复参数（temperature 为默认 0.8）",
          r.get("temperature") == "0.8", str(r.get("temperature")))
    state = await cdp.js(TOGGLE_REMEMBER_JS % json.dumps("read"))
    check("开关复选框保持未勾选", state == "false", str(state))


def main() -> int:
    from webui_app.services import synth_state as SS

    # 备份真实记忆（验收会覆写它）
    backups = {}
    for p in (SS.STATE_PATH,
              os.path.join(SS.STATE_DIR, SS.PROMPT_COPY),
              os.path.join(SS.STATE_DIR, SS.EMO_COPY)):
        if os.path.isfile(p):
            bak = p + ".acc_bak"
            shutil.copy2(p, bak)
            backups[p] = bak
    SS.forget()                          # 从干净状态开始

    server = None
    try:
        print(f"[A] 启动服务器 {URL}")
        server = start_server()
        asyncio.run(with_page(phase_a))

        # 磁盘侧核对
        check("记忆文件已落盘", os.path.isfile(SS.STATE_PATH))
        payload = SS.load_state()
        vals = payload.get("values", {})
        check("落盘值正确（temperature=0.7 / beams=1）",
              abs(vals.get("temperature", 0) - 0.7) < 1e-6
              and vals.get("num_beams") == 1,
              f"temp={vals.get('temperature')} beams={vals.get('num_beams')}")
        check("情感向量已落盘",
              any(vals.get(f"emo_vec_{i}", 0) for i in range(8)))
        from indextts.utils.presets import list_presets
        check("配置档出现在官方预设目录", PROFILE in list_presets())
        pdir = os.path.join("outputs", "presets", PROFILE)
        check("配置档含 preset.json", os.path.isfile(os.path.join(pdir, "preset.json")))

        print("[B] 杀掉服务器，模拟重启")
        server.terminate()
        server.wait(timeout=15)
        server = None
        time.sleep(2)

        server = start_server()
        asyncio.run(with_page(phase_b))

        check("清除记忆后状态回到默认",
              # 「清除」后的 2 秒抑制窗口内重置回声不落盘；即便窗口外
              # 有别的改动写入，值也应是默认值 —— 验收看值不看文件。
              (not os.path.isfile(SS.STATE_PATH)) or
              (SS.load_state().get("values", {}).get("temperature", 0.8) == 0.8
               and not any(SS.load_state().get("values", {})
                           .get(f"emo_vec_{i}", 0) for i in range(8))))
        from indextts.utils.presets import list_presets
        check("配置档已被两步删除", PROFILE not in list_presets())

        print("[C] 第三段重启：验证「关闭记忆」跨重启保持")
        server.terminate()
        server.wait(timeout=15)
        server = None
        time.sleep(2)
        server = start_server()
        asyncio.run(with_page(phase_c))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()
        # 恢复真实记忆；清掉验收残留（含被强杀的运行来不及清理的档位）
        SS.forget()
        for p, bak in backups.items():
            shutil.copy2(bak, p)
            os.remove(bak)
        import glob as _glob
        for pd in _glob.glob(os.path.join("outputs", "presets", "e2e记忆验收_*")):
            shutil.rmtree(pd, ignore_errors=True)
        shutil.rmtree(os.path.join("outputs", "presets", PROFILE),
                      ignore_errors=True)

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
