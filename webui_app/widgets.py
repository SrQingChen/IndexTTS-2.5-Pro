"""Param → Gradio 控件的自动工厂。

所有控件都从 params.REGISTRY 生成，界面与文档共用一份定义，
不会出现「界面写的默认值」和「文档写的默认值」不一致的情况。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gradio as gr

from webui_app import params as P
from webui_app import theme as T


def make_component(
    key: str,
    *,
    value: Any = "__default__",
    interactive: Optional[bool] = None,
    visible: bool = True,
    scale: Optional[int] = None,
    elem_classes: Optional[list] = None,
    show_key: bool = False,
    **overrides,
):
    """按注册表创建一个 Gradio 控件。

    Args:
        key: 参数键名（必须在 REGISTRY 中登记）
        value: 初始值，"__default__" 表示用注册表默认值
        interactive: 是否可交互，None 表示按 readonly 推导
        show_key: 标签里是否附带英文参数名
    """
    p = P.get(key)
    val = p.default if value == "__default__" else value
    inter = (not p.readonly) if interactive is None else interactive
    label = f"{p.label}  `{p.key}`" if show_key else p.label
    kw: Dict[str, Any] = dict(
        label=label, value=val, interactive=inter, visible=visible,
        elem_id=f"ix-{p.key}",
    )
    if scale is not None:
        kw["scale"] = scale
    if elem_classes:
        kw["elem_classes"] = elem_classes
    kw.update(overrides)

    kind = p.kind
    if kind == "slider":
        return gr.Slider(
            minimum=p.minimum, maximum=p.maximum, step=p.step,
            info=p.info or None, **kw,
        )
    if kind == "number":
        return gr.Number(precision=p.precision, info=p.info or None, **kw)
    if kind == "checkbox":
        kw.pop("value", None)
        return gr.Checkbox(value=bool(val), info=p.info or None, **kw)
    if kind == "dropdown":
        choices = list(p.choices or [])
        # 情感控制方式这类语义枚举，显示中文标签但传回索引
        if p.key == "emo_control_method":
            choices = EMO_MODE_LABELS
        return gr.Dropdown(
            choices=choices, allow_custom_value=False, info=p.info or None, **kw,
        )
    if kind == "textarea":
        kw.pop("value", None)
        return gr.Textbox(
            value=val, lines=p.lines, info=p.info or None,
            show_copy_button=True, **kw,
        )
    if kind == "text":
        kw.pop("value", None)
        return gr.Textbox(value=val or "", info=p.info or None, **kw)
    if kind == "audio":
        kw.pop("value", None)
        kw.pop("interactive", None)
        # 调用方显式给了值（参数记忆恢复上传音频）就带上；
        # 默认路径保持原行为（不带 value，控件为空）。
        if value != "__default__" and value:
            kw["value"] = value
        kw.setdefault("show_download_button", True)
        return gr.Audio(
            type="filepath", sources=["upload", "microphone"],
            elem_classes=["ix-audio-compact"] + (elem_classes or []), **kw,
        )
    raise ValueError(f"未知控件类型 {kind!r}（参数 {key}）")


EMO_MODE_LABELS = [
    "跟随音色参考音频（最稳，音色还原度最高）",
    "使用情感参考音频（音色与情感分别指定）",
    "使用 8 维情感向量（精确可复现）",
    "使用情感描述文本（实验功能，需 QwenEmotion）",
]

EMO_MODE_LABELS_SHORT = [
    "跟随音色音频", "情感参考音频", "8维情感向量", "情感描述文本",
]


def emo_mode_index(label_or_index) -> int:
    """把下拉框的值（标签或索引）统一转成 0~3 的索引。"""
    if isinstance(label_or_index, int):
        return label_or_index
    if label_or_index is None:
        return 0
    for i, lab in enumerate(EMO_MODE_LABELS):
        if lab == label_or_index:
            return i
    for i, lab in enumerate(EMO_MODE_LABELS_SHORT):
        if lab == label_or_index:
            return i
    try:
        return int(label_or_index)
    except (TypeError, ValueError):
        return 0


def help_markdown(p: P.Param) -> str:
    """把单个参数的完整文档渲染成 Markdown（参数手册 / 帮助抽屉共用）。"""
    tags = []
    if p.experimental:
        tags.append('<span class="ix-tag exp">实验功能</span>')
    if p.readonly:
        tags.append('<span class="ix-tag">只读</span>')
    if p.affects:
        tags.append(f'<span class="ix-tag">影响: {p.affects}</span>')
    if p.version != "all":
        tags.append(f'<span class="ix-tag">仅 v{p.version}</span>')
    tag_html = "".join(tags)

    rng = ""
    if p.kind == "slider" or p.kind == "number":
        bits = []
        if p.minimum is not None:
            bits.append(f"最小 {p.minimum}")
        if p.maximum is not None:
            bits.append(f"最大 {p.maximum}")
        if p.step is not None:
            bits.append(f"步长 {p.step}")
        if p.precision == 0:
            bits.append("仅整数")
        if bits:
            rng = f'<div class="ix-hint">取值范围：{" · ".join(bits)}'
            if p.unit:
                rng += f" · 单位 {p.unit}"
            rng += f" · 默认 <b>{p.default}</b></div>"

    parts = [
        f'<div class="ix-manual-card">',
        f'<h4 style="margin:6px 0 4px 0">'
        f'{p.label} &nbsp;<span class="ix-manual-key">{p.key}</span></h4>',
        tag_html,
        f'<p style="margin:7px 0 4px 0;font-size:13px">{p.summary}</p>',
        rng,
    ]
    if p.detail_md:
        parts += ["<details open><summary style='cursor:pointer;font-weight:600;"
                  "font-size:12.8px;margin:6px 0'>工作原理</summary>",
                  f"<div style='font-size:12.8px'>{p.detail_md}</div></details>"]
    if p.tuning_md:
        parts += ["<details><summary style='cursor:pointer;font-weight:600;"
                  "font-size:12.8px;margin:6px 0'>调优建议</summary>",
                  f"<div style='font-size:12.8px'>{p.tuning_md}</div></details>"]
    if p.pitfall_md:
        parts += ["<details><summary style='cursor:pointer;font-weight:600;"
                  "font-size:12.8px;margin:6px 0'>常见坑</summary>",
                  f"<div style='font-size:12.8px'>{p.pitfall_md}</div></details>"]
    parts.append("</div>")
    return "\n".join(parts)


def help_drawer(keys: list[str], title: str = "参数详解") -> gr.Accordion:
    """把一组参数的完整文档打包成一个可折叠抽屉。"""
    md = "\n\n".join(help_markdown(P.get(k)) for k in keys if k in P.REGISTRY)
    with gr.Accordion(f"📖 {title}", open=False) as acc:
        gr.HTML(md)
    return acc


def group_help(group: str, is_v25: bool = True, title: Optional[str] = None):
    """按分组生成帮助抽屉。"""
    keys = [p.key for p in P.by_group(group, is_v25)]
    return help_drawer(keys, title or P.GROUP_TITLES.get(group, group))
