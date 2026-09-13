"""Tab：模型资源管理。

把 tools/model_fetcher.py（镜像优先三级回退下载器）包成可视化界面：
资源审计 → 选择范围下载 → 后台线程执行 → 定时轮询进度 → 完成后联动引擎复检。

下载全程幂等：已就位且大小达标的文件自动跳过，大文件支持断点续传，
所以中断后重新点「开始下载」不会浪费已下载的流量。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from webui_app import fsutil
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.services import model_manager as MM
from webui_app.services.engine import EngineError

# 下载范围 → DownloadManager.start(only=...) 的取值
SCOPES: List[Tuple[str, Optional[str], str]] = [
    ("全部（推荐首次使用）", None,
     "主模型 + 辅助模型 + QwenEmotion + 示例音频，约 7.9 GB"),
    ("仅主模型", "main",
     "gpt.pth / codec.pth / s2mel.pth / config.yaml / tiktoken / feat1-2.pt，约 4.1 GB。"
     "推理必需"),
    ("仅辅助模型", "aux",
     "w2v-bert-2.0 / campplus / semantic_codec / bigvgan，约 2.8 GB。推理必需"),
    ("仅 QwenEmotion", "qwen_emo",
     "qwen0.6bemo4-merge，约 1.1 GB。只有情感模式 3（文本描述情感）需要"),
    ("仅示例音频", "examples",
     "examples/*.wav，约 3 MB。用于快速试听与音色库导入"),
]
SCOPE_LABELS = [s[0] for s in SCOPES]
SCOPE_ONLY = {s[0]: s[1] for s in SCOPES}
SCOPE_DESC = {s[0]: s[2] for s in SCOPES}


def _manager(ctx: AppContext) -> MM.DownloadManager:
    """每个进程一个 DownloadManager，挂到 ctx.shared 上。"""
    m = ctx.shared.get("_downloader")
    if m is None or m.model_dir != ctx.cfg.model_dir:
        m = MM.DownloadManager(ctx.cfg.model_dir)
        ctx.shared["_downloader"] = m
    return m


def render(ctx: AppContext):
    cfg = ctx.cfg
    eng = ctx.engine
    sb = ctx.component("statusbar")
    mgr = _manager(ctx)

    gr.HTML(T.section(
        "模型资源管理", "📦",
        f"模型目录 <code>{cfg.model_dir}</code>。下载走<b>镜像优先三级回退</b>"
        "（ModelScope → hf-mirror SDK → hf-mirror HTTP），全程幂等，中断后可续。"))

    with gr.Row(equal_height=False):
        # =================================================================
        # 左：下载控制
        # =================================================================
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("下载控制", "⬇️", ""))
                with gr.Row():
                    ver_dd = gr.Dropdown(
                        choices=["2.5", "2"], value=cfg.version,
                        label="IndexTTS 版本", scale=1,
                        info="2.5 = 当前项目；2 = 旧版（仅在你需要时下载）",
                    )
                    scope_dd = gr.Dropdown(
                        choices=SCOPE_LABELS, value=SCOPE_LABELS[0],
                        label="下载范围", scale=2, allow_custom_value=False,
                    )
                scope_desc = gr.HTML(T.hint(SCOPE_DESC[SCOPE_LABELS[0]]))
                with gr.Row():
                    dl_btn = gr.Button("🚀 开始下载", variant="primary", scale=2)
                    open_dir_btn = gr.Button("打开模型目录", scale=1, size="sm")
                dl_out = gr.HTML("")

                gr.HTML(T.section("实时进度", "📊", ""))
                progress_md = gr.Markdown(MM.progress_markdown(cfg.model_dir))
                with gr.Row():
                    refresh_btn = gr.Button("↻ 立即刷新", size="sm", scale=1)
                    auto_cb = gr.Checkbox(True, label="自动刷新（每 3 秒）", scale=2)

            with gr.Accordion("🌐 下载源与回退策略", open=False):
                gr.Markdown(MM.source_hint_markdown())

            with gr.Accordion("🧯 下载失败时的手动补救", open=False):
                gr.Markdown(_rescue_markdown(cfg.model_dir))

        # =================================================================
        # 右：资源审计
        # =================================================================
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("资源审计", "🔍",
                                  "逐个文件核对是否存在、大小是否达标。"
                                  "「推理就绪」为绿时才保证能加载模型。"))
                with gr.Row():
                    audit_btn = gr.Button("🔍 重新审计", variant="secondary", scale=2)
                    engine_chk_btn = gr.Button("引擎侧复检", scale=1, size="sm")
                audit_md = gr.Markdown("_点「重新审计」开始。_")
                engine_md = gr.HTML("")

    # =====================================================================
    # 回调
    # =====================================================================

    def on_scope_change(label):
        return T.hint(SCOPE_DESC.get(label, ""))

    scope_dd.change(on_scope_change, inputs=[scope_dd], outputs=[scope_desc])

    def on_start(version, scope):
        only = SCOPE_ONLY.get(scope)
        if version == "2" and only == "qwen_emo":
            return (T.err("QwenEmotion 是 IndexTTS-2.5 独有的组件，"
                          "2.0 版本没有这个模型。"),
                    gr.update(), gr.update(), gr.update())
        r = mgr.start(version=version, include_qwen_emo=True,
                      include_examples=True, only=only)
        if not r["started"]:
            return (T.warn(r["message"]), gr.update(),
                    MM.progress_markdown(cfg.model_dir), ctx.status_html())
        gr.Info("下载已在后台启动，可以切到别的 Tab 继续用")
        return (T.tip(f"✅ {r['message']}<br>范围：<b>{scope}</b> · 版本 <b>{version}</b>"),
                gr.update(interactive=False),
                MM.progress_markdown(cfg.model_dir),
                ctx.status_html())

    dl_btn.click(on_start, inputs=[ver_dd, scope_dd],
                 outputs=[dl_out, dl_btn, progress_md, sb])

    def on_open_dir():
        if fsutil.open_in_explorer(cfg.model_dir):
            return T.hint(f"已在文件管理器中打开：<code>{cfg.model_dir}</code>")
        return T.warn(f"无法自动打开，请手动复制：<code>{cfg.model_dir}</code>")

    open_dir_btn.click(on_open_dir, inputs=[], outputs=[dl_out])

    # ---------- 进度轮询 ----------
    # 空闲且内容无变化时返回空 update，避免定时器无谓地重绘 DOM。
    poll_cache: Dict[str, Any] = {"text": None, "busy": None}

    def on_poll(auto: bool):
        busy = mgr.busy
        prev_busy = poll_cache["busy"]
        text = MM.progress_markdown(cfg.model_dir)

        if auto and not busy and prev_busy is False and text == poll_cache["text"]:
            return gr.update(), gr.update(), gr.update(), gr.update()

        poll_cache["text"] = text
        poll_cache["busy"] = busy

        msg_up: Any = gr.update()
        audit_up: Any = gr.update()

        # 下载刚结束（busy → idle）：给出结论并自动复检一次
        if prev_busy and not busy:
            res = mgr.status().get("thread_result", {})
            msg = (T.tip(f"🎉 {res.get('message', '下载结束')}") if res.get("ok")
                   else T.err(res.get("message", "下载结束（可能出错，详见进度区）")))
            msg += T.hint("已自动重新审计；确认全绿后可回合成页加载引擎。")
            msg_up = gr.update(value=msg)
            try:
                audit_up = gr.update(value=MM.audit_markdown(mgr.audit(cfg.version)))
            except Exception:
                pass

        return (gr.update(value=text), gr.update(interactive=not busy),
                msg_up, audit_up)

    poll_outputs = [progress_md, dl_btn, dl_out, audit_md]
    timer = gr.Timer(value=3.0, active=True)
    timer.tick(on_poll, inputs=[auto_cb], outputs=poll_outputs)
    # 手动刷新走 auto=False 分支（无条件重算），所以不需要传 auto_cb
    refresh_btn.click(lambda: on_poll(False), inputs=[], outputs=poll_outputs)

    # ---------- 审计 ----------
    def on_audit(version):
        t0 = time.time()
        try:
            report = mgr.audit(version)
        except Exception as e:
            return T.err(f"审计失败：{type(e).__name__}: {e}"), gr.update()
        md = MM.audit_markdown(report)
        md += f"\n\n<sub>审计耗时 {time.time() - t0:.2f}s</sub>"
        return md, gr.update()

    audit_btn.click(on_audit, inputs=[ver_dd], outputs=[audit_md, engine_md])

    def on_engine_check():
        """从引擎的角度复检：直接问 IndexTTS 加载时需要哪些文件。"""
        if eng.loaded:
            return T.tip("引擎<b>已加载</b>，说明必需文件齐全。"
                         "（加载成功本身就是最强的审计）")
        try:
            missing = eng.audit_missing()
        except EngineError as e:
            return T.err(str(e))
        except Exception as e:
            return T.err(f"复检失败：{type(e).__name__}: {e}")
        if not missing:
            return T.tip("✅ 引擎侧复检通过：所有必需文件都在。")
        return T.err("引擎缺少以下文件：<br>"
                     + "<br>".join(f"<code>{m}</code>" for m in missing))

    engine_chk_btn.click(on_engine_check, inputs=[], outputs=[engine_md])

    def on_page_load():
        return (MM.progress_markdown(cfg.model_dir),
                MM.audit_markdown(mgr.audit(cfg.version)))

    return {
        "page_load": (on_page_load, [progress_md, audit_md]),
        "components": {"audit_md": audit_md, "progress_md": progress_md},
    }


def _rescue_markdown(model_dir: str) -> str:
    """手动补救指引：给出每个关键文件的官方仓库与镜像地址。"""
    from tools.model_fetcher import AUX_MODELS, MAIN_REPO

    lines = [
        "### 1. 优先重试",
        "",
        "本工具是幂等的，直接再点一次「开始下载」即可 —— 已完成的文件会跳过，",
        "大文件（>50 MB）通过 `.part` 临时文件断点续传。",
        "",
        "### 2. 命令行重试（可看到完整报错）",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe tools\\model_fetcher.py --version 2.5 --all",
        "```",
        "",
        "### 3. 手动下载后放到指定位置",
        "",
        f"模型目录：`{model_dir}`",
        "",
        "| 内容 | 官方仓库 | 镜像地址 |",
        "|---|---|---|",
    ]
    for v, repo in MAIN_REPO.items():
        lines.append(
            f"| IndexTTS-{v} 主模型 | `huggingface.co/{repo}` | "
            f"`hf-mirror.com/{repo}` · `modelscope.cn/models/{repo}` |")
    seen = set()
    for name, repo, remote, local, _req in AUX_MODELS:
        if repo in seen:
            continue
        seen.add(repo)
        lines.append(
            f"| {name} | `huggingface.co/{repo}` | "
            f"`hf-mirror.com/{repo}` |")
    lines += [
        "",
        "本地落盘路径（相对模型目录）：",
        "",
        "| 内容 | 本地路径 |",
        "|---|---|",
    ]
    for name, _repo, _remote, local, _req in AUX_MODELS:
        lines.append(f"| {name} | `{local}` |")
    lines += [
        "",
        "### 4. 常见失败原因",
        "",
        "| 现象 | 原因 | 处理 |",
        "|---|---|---|",
        "| 连接超时 | 直连了 `huggingface.co` | 本项目已默认 `HF_ENDPOINT=hf-mirror.com`，"
        "若你在外部环境运行请自行设置 |",
        "| 401 / 403 | 该仓库需要登录同意协议 | 用 `huggingface-cli login` 配置 token |",
        "| 磁盘写入失败 | 空间不足 | 需预留 ~10 GB（含 `.part` 临时文件） |",
        "| 大小不达标被判缺失 | 下载被截断 | 删掉该文件后重新下载 |",
        "| ModelScope 404 | 镜像仓库改名 | 工具会自动回退到 hf-mirror，无需干预 |",
    ]
    return "\n".join(lines)
