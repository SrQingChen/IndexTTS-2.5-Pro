"""UI 构建校验 —— 不启动服务、不加载模型，只装配 Blocks 并检查结构。

用途：改完 tabs/ 或 widgets/ 之后跑一遍，能在 10 秒内抓到
「Tab 渲染抛异常」「共享组件没注册」「预设写入控件数量对不上」
这类会让整个 UI 起不来的结构性错误，比每次真启动服务快得多。

    .venv\\Scripts\\python.exe tools\\build_check.py

真正要验证交互与推理，还是得启动 webui_pro.py。
"""

from __future__ import annotations

import os                                              # noqa: F401
import sys

import _env                                            # noqa: F401  路径 + 控制台编码
PROJECT_ROOT = _env.PROJECT_ROOT

from webui_app.config import config_from_args          # noqa: E402
from webui_app.context import AppContext                # noqa: E402

# 预设页要写入合成页的控件数量，必须与 tabs/presets.py 的 APPLY_COUNT 一致
EXPECTED_APPLY_TARGETS = 25


def main() -> int:
    AppContext.reset()
    cfg = config_from_args(["--lazy"])
    print(f"[cfg] version={cfg.version} backend={cfg.device.backend} "
          f"low_vram={cfg.device.low_vram} bf16={cfg.use_bf16}")

    from webui_app.app import TAB_SPECS, build_app      # noqa: E402

    demo, ctx = build_app(cfg, do_autoload=False)

    # 1. 所有 Tab 都必须渲染成功
    fails = ctx.shared.get("_render_failures", [])
    if fails:
        print("✖ 以下 Tab 渲染失败：")
        for f in fails:
            print(f"    {f}")
        return 1
    print(f"[tabs] {len(TAB_SPECS)} 个主 Tab 全部渲染成功")

    # 2. 顶栏状态条必须已注册（各 Tab 的回调都把它当 output）
    if "statusbar" not in ctx.shared:
        print("✖ ctx.shared['statusbar'] 未注册")
        return 1

    # 3. 合成页必须登记「预设可写入控件」，且数量与 APPLY_COUNT 一致
    targets = ctx.shared.get("synthesize_apply_targets")
    if not targets or len(targets) != EXPECTED_APPLY_TARGETS:
        n = len(targets) if targets else 0
        print(f"✖ synthesize_apply_targets 数量 {n}，"
              f"应为 {EXPECTED_APPLY_TARGETS}")
        return 1
    print(f"[shared] synthesize_apply_targets = {len(targets)} 个控件 OK")

    # 4. 生成 config —— Gradio 在这一步校验所有事件绑定的输入输出结构
    conf = demo.get_config_file()
    deps = conf.get("dependencies", [])
    print(f"[blocks] components={len(conf.get('components', []))} "
          f"dependencies={len(deps)}")

    # 5. 主 Tab 顺序（batch / manual 内部还有嵌套 Tabs，所以只查相对顺序）
    labels = [c["props"].get("label") for c in conf["components"]
              if c["type"] == "tabitem"]
    expected = [s[1] for s in TAB_SPECS]
    pos = -1
    for e in expected:
        try:
            pos = labels.index(e, pos + 1)
        except ValueError:
            print(f"✖ 主 Tab {e!r} 未找到或顺序错乱\n    实际 {labels}")
            return 1
    print(f"[tabs] 主 Tab 顺序正确（共 {len(labels)} 个 tabitem，含嵌套）")

    print("\n✅ 构建校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
