"""Gradio 主题与自定义 CSS。

设计取向：
    · 低饱和的靛蓝主色，深色/浅色双模式自适应
    · 卡片式分区（section card），弱化 Gradio 默认的方块堆叠感
    · 参数提示用统一的 hint / warn / tip 三种视觉语义
    · 顶栏状态条常驻，随时能看到引擎与显存状态
"""

from __future__ import annotations

import gradio as gr

ACCENT = "#5B6EE1"
ACCENT_SOFT = "#8B9BF0"

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EEF0FD", c100="#DDE1FB", c200="#BCC3F7",
        c300="#9BA5F3", c400="#7B88EF", c500=ACCENT,
        c600="#4A5BC7", c700="#3A489E", c800="#2A3575",
        c900="#1B224C", c950="#131834",
    ),
    secondary_hue=gr.themes.Color(
        c50="#F5F6F8", c100="#EAECF1", c200="#D5D9E3",
        c300="#B7BDCD", c400="#8E97AE", c500="#6B7590",
        c600="#545C74", c700="#414759", c800="#2E323F",
        c900="#1E2129", c950="#15171D",
    ),
    neutral_hue=gr.themes.Color(
        c50="#FAFAFB", c100="#F4F4F6", c200="#E6E7EB",
        c300="#D2D4DA", c400="#A9ACB6", c500="#7E828E",
        c600="#5F636D", c700="#474A52", c800="#31333A",
        c900="#212227", c950="#17181B",
    ),
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#F7F8FB",
    body_background_fill_dark="#131417",
    body_text_color="#1F2229",
    body_text_color_dark="#E4E6EB",
    body_text_size="14px",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#1B1D22",
    block_border_width="1px",
    block_border_color="#E6E7EB",
    block_border_color_dark="#2A2D34",
    block_label_text_color="#5F636D",
    block_label_text_color_dark="#A9ACB6",
    block_label_text_size="12px",
    block_label_text_weight="600",
    block_padding="14px",
    block_radius="12px",
    block_shadow="0 1px 2px rgba(16,24,40,.04)",
    block_shadow_dark="none",
    block_title_text_color="#1F2229",
    block_title_text_color_dark="#E4E6EB",
    block_title_text_size="14px",
    block_title_text_weight="600",
    button_large_padding="8px 18px",
    button_large_radius="10px",
    button_primary_background_fill=ACCENT,
    button_primary_background_fill_hover="#4A5BC7",
    button_primary_border_color=ACCENT,
    button_secondary_background_fill="#F4F4F6",
    button_secondary_background_fill_dark="#25272D",
    button_secondary_border_color="#E6E7EB",
    button_secondary_border_color_dark="#33363D",
    button_small_padding="5px 12px",
    button_small_radius="8px",
    checkbox_background_color_selected=ACCENT,
    checkbox_border_color_selected=ACCENT,
    input_background_fill="#FAFAFB",
    input_background_fill_dark="#212227",
    input_border_color="#E6E7EB",
    input_border_color_dark="#33363D",
    input_border_color_focus=ACCENT,
    input_border_color_focus_dark=ACCENT_SOFT,
    input_radius="10px",
    input_padding="8px 12px",
    panel_border_width="1px",
    table_border_color="#E6E7EB",
    table_border_color_dark="#2A2D34",
    table_even_background_fill="#FAFAFB",
    table_even_background_fill_dark="#1B1D22",
    table_odd_background_fill="#FFFFFF",
    table_odd_background_fill_dark="#17181B",
)
# 注：以上键名已逐个比对 Gradio 5.45 的 gr.themes.Base.set() 签名。
# 不支持的样式（如表格行内边距）改到下面的 CSS 里做，别往 set() 里塞 ——
# set() 对未知键名是直接抛 TypeError，会让整个 UI 起不来。


CSS = """
/* ---------- 全局 ---------- */
.gradio-container { max-width: 1560px !important; margin: 0 auto !important; }
footer { visibility: hidden; }

/* ---------- 顶栏 ---------- */
.ix-header {
    display: flex; align-items: center; gap: 18px;
    padding: 14px 22px; margin-bottom: 14px;
    border-radius: 16px;
    background: linear-gradient(115deg, #5B6EE1 0%, #7B5BD6 52%, #4A5BC7 100%);
    color: #fff; box-shadow: 0 6px 22px rgba(91,110,225,.24);
}
.ix-header .ix-logo {
    width: 42px; height: 42px; border-radius: 12px; flex: 0 0 auto;
    background: rgba(255,255,255,.18);
    display: flex; align-items: center; justify-content: center;
    font-size: 21px; font-weight: 700; letter-spacing: -.5px;
    backdrop-filter: blur(4px);
}
.ix-header .ix-title { font-size: 19px; font-weight: 700; letter-spacing: -.2px; line-height: 1.2; }
.ix-header .ix-sub { font-size: 12px; opacity: .82; margin-top: 3px; }
.ix-header .ix-spacer { flex: 1 1 auto; }
.ix-header .ix-badge {
    font-size: 11.5px; padding: 4px 11px; border-radius: 999px;
    background: rgba(255,255,255,.16); white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.ix-header .ix-badge b { font-weight: 700; }

/* ---------- 状态条 ---------- */
.ix-statusbar {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 9px 14px; margin-bottom: 16px;
    border-radius: 12px; font-size: 12.5px;
    background: var(--block-background-fill);
    border: 1px solid var(--block-border-color);
}
.ix-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 999px;
    background: var(--input-background-fill);
    border: 1px solid var(--input-border-color);
    font-variant-numeric: tabular-nums; white-space: nowrap;
}
.ix-dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto; background: #9AA0AC; }
.ix-dot.ok   { background: #2FBF71; box-shadow: 0 0 0 3px rgba(47,191,113,.16); }
.ix-dot.warn { background: #F2A93B; box-shadow: 0 0 0 3px rgba(242,169,59,.16); }
.ix-dot.err  { background: #E5484D; box-shadow: 0 0 0 3px rgba(229,72,77,.16); }
.ix-dot.idle { background: #9AA0AC; }
.ix-dot.load { background: #5B6EE1; box-shadow: 0 0 0 3px rgba(91,110,225,.18);
               animation: ix-pulse 1.2s ease-in-out infinite; }
@keyframes ix-pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ---------- 分区卡片 ---------- */
.ix-section {
    border: 1px solid var(--block-border-color) !important;
    border-radius: 14px !important;
    padding: 4px 16px 14px 16px !important;
    margin-bottom: 14px !important;
    background: var(--block-background-fill) !important;
}
.ix-section-title {
    display: flex; align-items: center; gap: 9px;
    font-size: 14.5px; font-weight: 700; letter-spacing: -.1px;
    margin: 12px 0 2px 0;
}
.ix-section-title .ix-ico {
    width: 24px; height: 24px; border-radius: 7px; flex: 0 0 auto;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px; background: rgba(91,110,225,.13); color: var(--primary-600);
}
.ix-section-desc {
    font-size: 12.5px; color: var(--body-text-color-subdued);
    margin: 0 0 10px 33px; line-height: 1.55;
}

/* ---------- 提示语义 ---------- */
.ix-hint {
    font-size: 12.3px; line-height: 1.62; padding: 9px 13px;
    border-radius: 10px; margin: 5px 0;
    background: var(--input-background-fill);
    border-left: 3px solid var(--primary-400);
    color: var(--body-text-color-subdued);
}
.ix-warn {
    font-size: 12.3px; line-height: 1.62; padding: 9px 13px;
    border-radius: 10px; margin: 5px 0;
    background: rgba(242,169,59,.10);
    border-left: 3px solid #F2A93B; color: var(--body-text-color);
}
.ix-tip {
    font-size: 12.3px; line-height: 1.62; padding: 9px 13px;
    border-radius: 10px; margin: 5px 0;
    background: rgba(47,191,113,.09);
    border-left: 3px solid #2FBF71; color: var(--body-text-color);
}
.ix-err {
    font-size: 12.3px; line-height: 1.62; padding: 9px 13px;
    border-radius: 10px; margin: 5px 0;
    background: rgba(229,72,77,.09);
    border-left: 3px solid #E5484D; color: var(--body-text-color);
}
.ix-mono { font-family: var(--font-mono); font-size: 12px; }

/* ---------- 参数手册 ---------- */
.ix-manual-card { border-bottom: 1px solid var(--block-border-color); padding: 4px 0 12px 0; }
.ix-manual-key {
    font-family: var(--font-mono); font-size: 11.5px; padding: 1px 7px;
    border-radius: 6px; background: rgba(91,110,225,.12); color: var(--primary-600);
}
.ix-tag {
    display:inline-block; font-size: 11px; padding: 1px 8px; border-radius: 999px;
    background: var(--input-background-fill); border: 1px solid var(--input-border-color);
    margin-right: 5px; color: var(--body-text-color-subdued);
}
.ix-tag.exp { background: rgba(242,169,59,.13); border-color: rgba(242,169,59,.4); }

/* ---------- 情感向量 ---------- */
.ix-emo-grid label { font-size: 12.5px !important; }
.ix-emo-meter {
    font-size: 12.5px; padding: 8px 13px; border-radius: 10px;
    background: var(--input-background-fill);
    border: 1px solid var(--input-border-color);
    font-variant-numeric: tabular-nums;
}

/* ---------- 紧凑音频控件 ---------- */
.ix-audio-compact .audio-container,
.ix-audio-compact .upload-container { min-height: 104px !important; }
.ix-audio-compact .empty { min-height: 74px !important; }

/* ---------- 表格 ---------- */
.ix-table table { font-size: 12.8px; }
.ix-table th { font-weight: 600 !important; }

/* 裸 HTML 表格（系统页的磁盘扫描等直接输出 <table class="ix-table">）。
   gr.themes 没有 table_row_padding 这个键，行内边距只能在这里给。 */
table.ix-table {
    width: 100%; border-collapse: collapse; font-size: 12.8px; margin: 6px 0 10px 0;
}
table.ix-table th, table.ix-table td {
    padding: 6px 10px; text-align: left; vertical-align: top;
    border-bottom: 1px solid var(--block-border-color);
}
table.ix-table th {
    font-weight: 600; background: var(--block-background-fill);
    color: var(--block-label-text-color);
}
table.ix-table tr:hover td { background: var(--block-background-fill); }

/* ---------- 进度条 ---------- */
.ix-progress-wrap {
    height: 7px; border-radius: 999px; overflow: hidden;
    background: var(--input-background-fill); border: 1px solid var(--input-border-color);
}
.ix-progress-bar { height: 100%; background: linear-gradient(90deg,#5B6EE1,#8B9BF0);
                   border-radius: 999px; transition: width .4s ease; }

/* ---------- Tab ----------
   内边距与字号刻意压紧：Gradio 5 会把放不下的 Tab 收进右侧「…」溢出菜单，
   7 个主 Tab 在 1366~1536px 宽的笔记本屏上很容易触发。 */
.tabs > .tab-nav { gap: 2px !important; padding: 4px 4px 0 4px !important;
                   border-bottom: 1px solid var(--block-border-color) !important; }
.tabs > .tab-nav > button {
    border-radius: 9px 9px 0 0 !important; font-size: 13px !important;
    font-weight: 600 !important; padding: 7px 11px !important;
    white-space: nowrap !important; letter-spacing: -.2px !important;
    border: 1px solid transparent !important; border-bottom: none !important;
}
.tabs > .tab-nav > button.selected {
    background: var(--block-background-fill) !important;
    border-color: var(--block-border-color) !important;
    color: var(--primary-600) !important;
    box-shadow: inset 0 2px 0 var(--primary-500) !important;
}

/* ---------- 模态框 ---------- */
.ix-modal-overlay {
    position: fixed !important; inset: 0 !important;
    width: 100vw !important; height: 100vh !important;
    background: rgba(15,17,21,.55) !important; backdrop-filter: blur(3px);
    z-index: 1000 !important; display: flex !important;
    justify-content: center !important; align-items: center !important;
    padding: 0 !important; margin: 0 !important;
}
.ix-modal-overlay > .column-wrap, .ix-modal-overlay > div { width:auto !important; height:auto !important; }
.ix-modal-box {
    background: var(--body-background-fill) !important;
    padding: 22px !important; border-radius: 16px !important;
    width: 92vw !important; max-width: 620px !important;
    max-height: 84vh !important; overflow-y: auto !important;
    box-shadow: 0 22px 60px rgba(0,0,0,.32) !important;
    border: 1px solid var(--block-border-color) !important;
}
"""


def header_html(title: str, subtitle: str, badges: list[tuple[str, str]] | None = None) -> str:
    """渲染顶栏。badges = [(文本, 值), ...]"""
    badges = badges or []
    chips = "".join(
        f'<span class="ix-badge">{t} <b>{v}</b></span>' for t, v in badges
    )
    return f"""
<div class="ix-header">
  <div class="ix-logo">Ix</div>
  <div>
    <div class="ix-title">{title}</div>
    <div class="ix-sub">{subtitle}</div>
  </div>
  <div class="ix-spacer"></div>
  {chips}
</div>"""


def section(title: str, icon: str = "", desc: str = "") -> str:
    """分区标题（配合 elem_classes=['ix-section'] 的 Column 使用）。"""
    d = f'<div class="ix-section-desc">{desc}</div>' if desc else ""
    i = f'<span class="ix-ico">{icon}</span>' if icon else ""
    return f'<div class="ix-section-title">{i}{title}</div>{d}'


def chip(label: str, value: str = "", state: str = "idle") -> str:
    return (f'<span class="ix-chip"><span class="ix-dot {state}"></span>'
            f'{label}{(" " + value) if value else ""}</span>')


def statusbar(chips: list[str]) -> str:
    return '<div class="ix-statusbar">' + "".join(chips) + "</div>"


def hint(text: str) -> str:
    return f'<div class="ix-hint">{text}</div>'


def warn(text: str) -> str:
    return f'<div class="ix-warn">⚠️ {text}</div>'


def tip(text: str) -> str:
    return f'<div class="ix-tip">💡 {text}</div>'


def err(text: str) -> str:
    return f'<div class="ix-err">✖ {text}</div>'


def progress_bar(ratio: float, label: str = "") -> str:
    pct = max(0.0, min(1.0, float(ratio))) * 100
    return (
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div class="ix-progress-wrap" style="flex:1">'
        f'<div class="ix-progress-bar" style="width:{pct:.1f}%"></div></div>'
        f'<span class="ix-mono" style="min-width:118px;text-align:right">'
        f'{label or f"{pct:.0f}%"}</span></div>'
    )
