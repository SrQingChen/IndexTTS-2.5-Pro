from pathlib import Path

import pytest


def test_webui_python_source_parses():
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    source = webui_path.read_text(encoding="utf-8")

    assert compile(source, str(webui_path), "exec") is not None


def test_webui_all_tabs_render():
    """全部 Tab 真实渲染（E1-T 验收二轮教训：懒冒烟只数 blocks，
    build_app 会把渲染异常吞进 ctx.shared["_render_failures"]——
    曾有 Tab 内 NameError 而「构建成功」的盲区）。"""
    import os
    import sys

    gradio = pytest.importorskip("gradio")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ.setdefault("IX_WEBUI_LAZY", "1")
    os.chdir(root)

    from webui_app.app import build_app

    demo, ctx = build_app(do_autoload=False)
    failures = list(ctx.shared.get("_render_failures") or [])
    assert not failures, (
        f"有 Tab 渲染失败: {failures} —— 构建冒烟必须查 _render_failures，"
        "只数 blocks 是盲检")
