"""Tab：产物清理。

把「用了一段时间之后磁盘上都堆了什么」回答清楚：
一键三连的评选产物、A/B 评测、训练运行（adapter + 保险库 checkpoint）、
LoRA 导出、数据集、音色库条目、工作台切片、单次合成、批量任务、缓存 ——
逐条列出体积与说明，勾选删除前必须过一次确认弹窗。

这一页**只看产物**：底模（checkpoints/）、项目代码、预设（outputs/presets/）、
日志（outputs/logs/）不在清理范围里，永远不出现 —— 详见 services/artifacts.py
的安全模型注释。删除是永久的（不进回收站），所以弹窗里把每项的路径与体积
都摆出来，让用户确认的每一眼都有信息依据。

确认弹窗复用 theme.py 里预留的 ix-modal 样式（覆盖层 + 居中卡片），
这是它第一次被真正用起来。
"""

from __future__ import annotations

import html
import time
from typing import Any, Dict, List

import gradio as gr

from webui_app import fsutil
from webui_app import logging_setup as LOG
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.services import artifacts as AR
from webui_app.services.monitor import human_size

# 单类最多逐条展示多少行 —— 防止一次合成几百条把页面撑爆，
# 超出的仍然可以用「清空此类」整类处理
MAX_ROWS_PER_KIND = 60


def _fmt_time(ts: float) -> str:
    if not ts:
        return "—"
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _esc(s: str) -> str:
    return html.escape(str(s), quote=False)


def _summary_html(items: List[Dict[str, Any]]) -> str:
    """头部统计条：总量 + 各分类 chip（数据来自 State，无需重建 dataclass）。"""
    if not items:
        return T.hint("当前没有任何产物 —— 干净得像刚装好一样。")
    total = sum(i["size"] for i in items)
    chips = []
    for kind, _icon, label, _desc in AR.CATEGORIES:
        group = [i for i in items if i["kind"] == kind]
        if not group:
            continue
        size = human_size(sum(i["size"] for i in group))
        chips.append(f'<span class="ix-chip">{_esc(label)} '
                     f'<b>{_esc(size)}</b> · {len(group)} 项</span>')
    return (
        '<div class="ix-statusbar">'
        f'<span class="ix-chip"><span class="ix-dot warn"></span>'
        f'产物总量 <b>{human_size(total)}</b> · {len(items)} 项</span>'
        + "".join(chips) + "</div>")


def _kind_header(kind: str, items: List[Dict[str, Any]]) -> str:
    icon, label, desc = AR.CATEGORY_INFO[kind]
    size = human_size(sum(i["size"] for i in items))
    prot = sum(1 for i in items if i.get("protected"))
    extra = f" · {prot} 项受保护" if prot else ""
    return (
        f'<div class="ix-section-title"><span class="ix-ico">{icon}</span>'
        f'{_esc(label)} · <span class="ix-mono">{size}</span> · {len(items)} 项{extra}</div>'
        f'<div class="ix-section-desc">{_esc(desc)}</div>')


def _item_row_html(it: Dict[str, Any]) -> str:
    meta = [human_size(it["size"]),
            f"{it['n_files']} 文件" if it["n_files"] > 1 else "1 文件",
            _fmt_time(it["mtime"])]
    if it.get("origin"):
        meta.append(_esc(it["origin"]))
    if it.get("note"):
        meta.append(_esc(it["note"]))
    return (
        '<div style="display:flex;align-items:baseline;gap:10px;'
        'padding:7px 0;border-bottom:1px solid var(--block-border-color)">'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-weight:600;overflow:hidden;text-overflow:ellipsis;'
        f'white-space:nowrap">{_esc(it["name"])}</div>'
        f'<div class="ix-mono" style="opacity:.7;overflow:hidden;'
        f'text-overflow:ellipsis;white-space:nowrap">{_esc(it["path"])}</div>'
        f'</div><div class="ix-mono" style="opacity:.85;white-space:nowrap;'
        f'font-size:11.5px">{" · ".join(meta)}</div></div>')


def _modal_detail_html(kind: str, items: List[Dict[str, Any]]) -> str:
    """确认弹窗里的清单。kind 为空串表示跨类混合删除。"""
    total = sum(i["size"] for i in items)
    rows = "".join(
        f'<tr><td>{_esc(i["name"])}</td>'
        f'<td class="ix-mono">{_esc(i["path"])}</td>'
        f'<td class="ix-mono" style="white-space:nowrap">{human_size(i["size"])}</td></tr>'
        for i in items[:MAX_ROWS_PER_KIND])
    more = (f'<tr><td colspan="3" style="opacity:.7">… 以及另外 '
            f'{len(items) - MAX_ROWS_PER_KIND} 项</td></tr>'
            if len(items) > MAX_ROWS_PER_KIND else "")
    title = f"「{AR.CATEGORY_INFO[kind][1]}」分类下的 {len(items)} 项产物" if kind \
        else f"{len(items)} 项产物"
    return (
        T.section("确认删除", "🗑",
                  f"即将永久删除 {title}，合计 <b>{human_size(total)}</b>。")
        + '<table class="ix-table"><tr><th>名称</th><th>路径</th><th>体积</th></tr>'
        + rows + more + "</table>"
        + T.warn("删除是<b>永久的</b>，不进回收站。数据集会连音频与已提取特征一起删；"
                 "训练运行会连 adapter 与保险库 checkpoint 一起删。底模与项目文件"
                 "不受影响。"))


def render(ctx: AppContext):
    cfg = ctx.cfg
    sb = ctx.component("statusbar")

    gr.HTML(T.section(
        "产物清理", "🧹",
        "盘点<b>除底模与项目文件之外</b>的全部产物：一键三连 / 自训练 / 工作台 /"
        "合成产生的一切中间文件与成品。每项都可单独删除，也可整类清空；"
        "删除前会弹出确认框列出明细。"))
    gr.HTML(T.hint(
        "受保护、永不出现在本页的：<code>checkpoints/</code>（底模）· 项目代码 · "
        "<code>outputs/presets/</code>（预设）· <code>outputs/logs/</code>（日志）· "
        "<code>training_runs/base_manifest.json</code>（底座快照凭证）。"
        "训练进行中的 run 会标记为不可删除。"))

    with gr.Column(elem_classes=["ix-section"]):
        with gr.Row():
            scan_btn = gr.Button("↻ 重新扫描", variant="primary", size="sm", scale=1)
            open_btn = gr.Button("📂 打开 outputs 目录", size="sm", scale=1)
            auto_cb = gr.Checkbox(True, label="进入本页时自动扫描", scale=2)
        summary_html = gr.HTML("<i>尚未扫描</i>")
        result_out = gr.HTML("")

    # 两次扫描之间页面组件树是静态的，条目行全部由 @gr.render 按 State 重绘
    items_state = gr.State({})
    pending_state = gr.State([])          # 等待确认的 key 列表
    pending_kind_state = gr.State("")     # 整类清空时的 kind（单删为 ""）

    # ------------------------------------------------------------------
    # 确认弹窗（复用 theme.py 预留的 ix-modal 覆盖层样式）
    # ------------------------------------------------------------------
    with gr.Group(visible=False, elem_classes=["ix-modal-overlay"]) as modal:
        with gr.Column(elem_classes=["ix-modal-box"]):
            modal_detail = gr.HTML("")
            with gr.Row():
                cancel_btn = gr.Button("取消", size="sm", scale=1)
                confirm_btn = gr.Button("🗑 确认永久删除", variant="stop", scale=1)

    # ------------------------------------------------------------------
    # 动态条目列表
    # ------------------------------------------------------------------
    @gr.render(inputs=[items_state])
    def draw_items(data: Dict[str, Any]):
        items: List[Dict[str, Any]] = (data or {}).get("items") or []
        if not items:
            gr.HTML(T.tip("没有扫描到任何产物。点上方「重新扫描」，或先去合成/训练一些东西。"))
            return

        by_kind: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            by_kind.setdefault(it["kind"], []).append(it)

        for kind, _icon, _label, _desc in AR.CATEGORIES:
            group = by_kind.get(kind) or []
            if not group:
                continue
            with gr.Column(elem_classes=["ix-section"]):
                with gr.Row():
                    gr.HTML(_kind_header(kind, group))
                    deletable = [i for i in group if not i.get("protected")]
                    if deletable:
                        wipe_btn = gr.Button(
                            f"清空此类（{len(deletable)}）",
                            size="sm", variant="secondary", scale=0, min_width=130)

                        def _wipe(k=kind, d=deletable):
                            return _open_modal(d, k)

                        wipe_btn.click(
                            _wipe, inputs=[], outputs=[pending_state, pending_kind_state, modal_detail, modal])

                hidden = max(0, len(group) - MAX_ROWS_PER_KIND)
                for it in group[:MAX_ROWS_PER_KIND]:
                    with gr.Row():
                        gr.HTML(_item_row_html(it))
                        if it.get("protected"):
                            gr.HTML('<span class="ix-chip"><span class="ix-dot err">'
                                    '</span>锁定</span>', scale=0, min_width=64)
                        else:
                            del_btn = gr.Button("🗑", size="sm", variant="stop",
                                                scale=0, min_width=48)

                            def _one(item=it):
                                return _open_modal([item], "")

                            del_btn.click(
                                _one, inputs=[],
                                outputs=[pending_state, pending_kind_state, modal_detail, modal])
                if hidden:
                    gr.HTML(T.hint(f"另有 {hidden} 项未逐条展示，"
                                   f"可用「清空此类」一次性处理。"))

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def _open_modal(sel: List[Dict[str, Any]], kind: str):
        """单删/整类清空共用的弹窗打开逻辑（渲染函数里闭包引用）。"""
        keys = [i["key"] for i in sel if not i.get("protected")]
        if not keys:
            return (gr.update(), gr.update(),
                    T.warn("没有可删除的项（可能全部处于保护状态）。"),
                    gr.update())
        return (keys, kind or "", _modal_detail_html(kind, sel),
                gr.update(visible=True))

    @LOG.ui_guard("cleanup.on_scan")
    def on_scan():
        r = AR.scan(cfg)
        return (r.to_dict(), _summary_html(r.to_dict()["items"]))

    scan_btn.click(on_scan, inputs=[], outputs=[items_state, summary_html])

    def on_open():
        if fsutil.open_in_explorer(cfg.output_dir):
            return T.hint(f"已打开 <code>{_esc(cfg.output_dir)}</code>")
        return T.warn(f"无法自动打开，请手动前往：<code>{_esc(cfg.output_dir)}</code>")

    open_btn.click(on_open, inputs=[], outputs=[result_out])

    def on_cancel():
        return ([], "", gr.update(), gr.update(visible=False))

    cancel_btn.click(on_cancel, inputs=[],
                     outputs=[pending_state, pending_kind_state, modal_detail, modal])

    @LOG.ui_guard("cleanup.on_confirm")
    def on_confirm(keys: List[str], _kind: str):
        """确认弹窗的「确认永久删除」：删除 → 重新扫描 → 关弹窗。"""
        if not keys:
            return ("", gr.update(), gr.update(), [], "",
                    gr.update(visible=False), gr.update())
        r = AR.delete_items(cfg, keys)
        scan = AR.scan(cfg)
        data = scan.to_dict()
        n = len(r["deleted"])
        msg = T.tip(f"✅ 已删除 <b>{n}</b> 项，释放 <b>{r['freed_text']}</b>。")
        if r["skipped"]:
            msg += T.hint(f"{len(r['skipped'])} 项已不存在（可能刚被别处删除），自动跳过。")
        if r["failed"]:
            msg += T.err("以下项删除失败：<br>"
                         + "<br>".join(_esc(f) for f in r["failed"]))
        ctx.log.push("artifacts_cleaned",
                     f"删除 {n} 项，释放 {r['freed_text']}", "ok")
        gr.Info(f"已删除 {n} 项，释放 {r['freed_text']}")
        return (msg, data, _summary_html(data["items"]), [], "",
                gr.update(visible=False), gr.update(value=ctx.status_html()))

    confirm_btn.click(
        on_confirm, inputs=[pending_state, pending_kind_state],
        outputs=[result_out, items_state, summary_html,
                 pending_state, pending_kind_state, modal, sb])

    def on_page_load():
        r = AR.scan(cfg)
        data = r.to_dict()
        return (data, _summary_html(data["items"]))

    return {
        "page_load": (on_page_load, [items_state, summary_html]),
        "components": {"summary_html": summary_html},
    }
