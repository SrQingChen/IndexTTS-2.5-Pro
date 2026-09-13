"""Tab：训练数据集（阶段 2）。

数据集 → 文本 → 划分 → 特征提取，是训练流水线的第一站。
界面操作直接映射 `training/dataset.py` 与 `training/features.py`：
    · 创建 / 导入音频（复制进数据集目录，训练几小时也不怕源文件被挪）
    · 补文本（特征提取的硬前提：没有文本就没有 text_tokens / mu）
    · train/val 划分（可重复：同 seed 同结果）
    · 特征提取（后台线程 + 进度 + 可中断 + 断点续提）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import fsutil
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.services.engine import EngineError
from webui_app.training import dataset as DS
from webui_app.training import features as FT
from webui_app.training.runner import get_runner

# 特征提取需要引擎里的 w2v-bert / codec / campplus —— 但**不用等它卸载**：
# 提取只借用前向，与推理可以共存（显存够时）。引擎没加载就先加载。
EXTRACT_NEEDS_ENGINE = True


def _datasets_dd_choices() -> List[str]:
    return DS.list_datasets()


def render(ctx: AppContext):
    eng = ctx.engine
    sb = ctx.component("statusbar")
    runner = get_runner()

    gr.HTML(T.section(
        "训练数据集", "🗂️",
        "创建数据集 → 导入音频 → 填文本 → 划分 → 特征提取。"
        "数据目录 <code>datasets/</code>，特征缓存与元信息都在里面，"
        "删数据集即全删。"))

    with gr.Row(equal_height=False):
        # ============================== 左：管理 ==============================
        with gr.Column(scale=1, min_width=380):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("创建与导入", "➕", ""))
                new_name = gr.Textbox(label="新数据集名", placeholder="my_voice")
                create_btn = gr.Button("创建", size="sm")
                create_out = gr.HTML("")

                ds_dd = gr.Dropdown(choices=_datasets_dd_choices(),
                                    label="当前数据集", allow_custom_value=False)
                with gr.Row():
                    refresh_btn = gr.Button("↻ 刷新列表", size="sm", scale=1)
                    del_btn = gr.Button("🗑 删除数据集", size="sm", scale=1)

                up_files = gr.File(label="导入音频（可多选，复制进数据集）",
                                   file_count="multiple",
                                   file_types=["audio"],
                                   type="filepath")
                lang_dd = gr.Dropdown(choices=["ZH", "EN", "JA", "ES"],
                                      value="ZH", label="语言")
                import_btn = gr.Button("📥 导入所选文件")
                import_out = gr.HTML("")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("补文本", "✍️",
                                  "特征提取的硬前提。一行一条 JSON："
                                  "<code>{\"id\": \"u001\", \"text\": \"…\"}</code>"))
                text_ta = gr.Textbox(label="批量文本（JSONL）", lines=6,
                                     placeholder='{"id": "u001", "text": "今天天气不错"}')
                text_btn = gr.Button("写入文本")
                text_out = gr.HTML("")

        # ============================== 右：状态与提取 ==============================
        with gr.Column(scale=1, min_width=380):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("数据集体检", "🩺", ""))
                stats_md = gr.Markdown("_选择数据集后点「重新体检」。_")
                stats_btn = gr.Button("🔍 重新体检")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("train / val 划分", "✂️", ""))
                val_ratio = gr.Slider(0.02, 0.4, value=0.1, step=0.01,
                                      label="验证集比例",
                                      info="小数据集建议 0.15~0.25，"
                                           "否则早停没有可看的 val")
                split_seed = gr.Number(42, label="随机种子", precision=0)
                split_btn = gr.Button("✂️ 重新划分（同种子结果一致）")
                split_out = gr.HTML("")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("特征提取", "⚙️",
                                  "离线缓存 text_tokens / codes / style / emo_vec /"
                                  " mel / mu —— 训练时完全不碰大模型。"
                                  "已提取的自动跳过（断点续提）。"))
                overwrite_cb = gr.Checkbox(False, label="覆盖已有特征")
                extract_btn = gr.Button("🚀 开始提取（后台）", variant="primary")
                stop_btn = gr.Button("⏹ 停止", size="sm")
                extract_out = gr.HTML("")
                progress_html = gr.HTML("")
                log_ta = gr.Textbox(label="日志", lines=8, max_lines=12,
                                    interactive=False)

    # =====================================================================
    # 回调
    # =====================================================================

    def _need_ds(name: str) -> Optional[str]:
        if not name or not DS.exists(name):
            return T.err(f"数据集 `{name or '(未选)'}` 不存在，先创建或刷新列表。")
        return None

    def on_create(name):
        name = (name or "").strip()
        if not name:
            return (T.err("名字不能为空。"), gr.update(),
                    gr.update(choices=_datasets_dd_choices()))
        try:
            DS.create(name, note="UI 创建")
        except Exception as e:
            return (T.err(f"创建失败：{e}"), gr.update(),
                    gr.update(choices=_datasets_dd_choices()))
        return (T.tip(f"✅ 数据集 <code>{name}</code> 已创建，可以导入音频了。"),
                gr.update(choices=_datasets_dd_choices(), value=name),
                gr.update(choices=_datasets_dd_choices()))

    create_btn.click(on_create, inputs=[new_name],
                     outputs=[create_out, ds_dd, ds_dd])
    refresh_btn.click(lambda: gr.update(choices=_datasets_dd_choices()),
                      inputs=[], outputs=[ds_dd])

    def on_delete(name):
        if not name or not DS.exists(name):
            return (T.err("先选一个存在的数据集。"),
                    gr.update(choices=_datasets_dd_choices()))
        if DS.delete(name):
            return (T.warn(f"已删除 <code>{name}</code>（音频与特征一并删除）。"),
                    gr.update(choices=_datasets_dd_choices(), value=None))
        return (T.err("删除失败。"), gr.update())

    del_btn.click(on_delete, inputs=[ds_dd], outputs=[create_out, ds_dd])

    def on_import(name, files, lang):
        err = _need_ds(name)
        if err:
            return err
        if not files:
            return T.warn("没有选择文件。")
        try:
            r = DS.import_audio(name, [f for f in files], copy=True, lang=lang)
        except Exception as e:
            return T.err(f"导入失败：{type(e).__name__}: {e}")
        msg = (f"✅ 导入 {r['added']} 条"
               + (f"，跳过 {len(r['skipped'])}" if r["skipped"] else "")
               + (f"，失败 {len(r['failed'])}" if r["failed"] else ""))
        out = T.tip(msg)
        if r["failed"]:
            out += T.err("<br>".join(r["failed"][:5]))
        out += T.hint("下一步：到「补文本」给每条样本写文本，然后体检。")
        return out

    import_btn.click(on_import, inputs=[ds_dd, up_files, lang_dd],
                     outputs=[import_out])

    def on_text(name, ta):
        err = _need_ds(name)
        if err:
            return err
        touched = {}
        bad = []
        for i, line in enumerate((ta or "").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                touched[str(d["id"])] = {"text": str(d.get("text", ""))}
            except Exception as e:
                bad.append(f"第 {i} 行：{e}")
        if not touched:
            return T.warn("没有可写入的行。每行一个 JSON："
                          '<code>{"id": "u001", "text": "…"}</code>')
        n = FT._apply_meta(name, touched)
        DS.refresh_all(name, require_features=False)
        out = T.tip(f"✅ 写入 {n} 条文本。")
        if bad:
            out += T.err("以下行解析失败：<br>" + "<br>".join(bad[:5]))
        return out

    text_btn.click(on_text, inputs=[ds_dd, text_ta], outputs=[text_out])

    def on_stats(name):
        err = _need_ds(name)
        if err:
            return err
        try:
            DS.refresh_all(name, require_features=False)
            return FT.stats_markdown(name)
        except Exception as e:
            return T.err(f"体检失败：{type(e).__name__}: {e}")

    stats_btn.click(on_stats, inputs=[ds_dd], outputs=[stats_md])

    def on_split(name, ratio, seed):
        err = _need_ds(name)
        if err:
            return err
        try:
            r = DS.make_split(name, val_ratio=float(ratio), seed=int(seed))
        except Exception as e:
            return T.err(f"划分失败：{e}")
        if not r.get("ok"):
            return T.err(r.get("error", "划分失败"))
        return T.tip(f"✅ 训练 {r['train']} 条 / 验证 {r['val']} 条"
                     f"（种子 {int(seed)}，同种子重跑结果一致）")

    split_btn.click(on_split, inputs=[ds_dd, val_ratio, split_seed],
                    outputs=[split_out])

    # ---------- 特征提取（runner 后台） ----------
    def on_extract(name, overwrite):
        err = _need_ds(name)
        if err:
            return err, gr.update(), gr.update()

        def fn(progress, should_stop):
            # 特征提取要借引擎里的 w2v-bert / codec —— 先确保加载
            if not eng.loaded:
                progress(0.01, "加载引擎（提取要借 w2v-bert / codec）…")
                try:
                    eng.load()
                except EngineError as e:
                    return {"ok": False, "error": f"引擎加载失败：{e}"}
            from webui_app.training import features as FTx
            return FTx.extract_dataset(
                name, overwrite=bool(overwrite), only="ready",
                progress=progress, should_stop=should_stop,
                tts=eng.tts, device=str(getattr(eng.tts, "device", "cuda")))

        r = runner.submit("extract", f"特征提取 · {name}", fn,
                          require_engine="none", engine=eng)
        if not r["ok"]:
            return T.err(r["message"]), gr.update(), gr.update()
        return (T.tip("🚀 特征提取已在后台运行，可切到「训练」页继续配置。"),
                gr.update(), gr.update())

    extract_btn.click(on_extract, inputs=[ds_dd, overwrite_cb],
                      outputs=[extract_out, progress_html, log_ta])

    stop_btn.click(lambda: runner.cancel(), inputs=[], outputs=[extract_out])

    # ---------- 轮询 ----------
    poll_cache: Dict[str, Any] = {"snap": None}

    def on_poll():
        snap = runner.snapshot()
        if snap == poll_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update()
        poll_cache["snap"] = snap
        bar = ""
        if snap["running"]:
            pct = int(snap["progress"] * 100)
            bar = (f'<div style="margin:4px 0"><b>{snap["label"]}</b> · '
                   f'{pct}% · {snap["message"]}'
                   + (" · <b>已请求停止</b>" if snap["stop_requested"] else "")
                   + f'<div style="background:var(--border-color-primary);'
                     f'border-radius:6px;height:8px;margin-top:4px">'
                     f'<div style="width:{pct}%;height:8px;border-radius:6px;'
                     f'background:var(--color-accent)"></div></div></div>')
            return gr.update(value=bar), gr.update(value=runner.log_text())
        # 刚结束
        if snap["ok"] is not None:
            icon = "🎉" if snap["ok"] else "🔴"
            bar = f'<div style="margin:4px 0">{icon} <b>{snap["label"]}</b> · ' \
                  f'{snap["message"]} · 用时 {snap["seconds"]}s</div>'
            return gr.update(value=bar), gr.update(value=runner.log_text())
        return gr.update(), gr.update()

    timer = gr.Timer(value=2.5, active=True)
    timer.tick(on_poll, inputs=[], outputs=[progress_html, log_ta])

    def on_page_load():
        return (gr.update(choices=_datasets_dd_choices()), "")

    return {
        "page_load": (on_page_load, [ds_dd, stats_md]),
        "components": {"ds_dd": ds_dd},
    }
