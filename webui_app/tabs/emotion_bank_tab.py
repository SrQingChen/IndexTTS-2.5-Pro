"""Tab：情感参考库。

导演模式逐句路由的数据源：把角色「最有戏」的几句台词按 角色×情绪 入库，
合成时每句自动挑对应情绪的参考音频（infer 的 emo_audio_prompt 路径）。

与音色库的分工（见 services/emotion_bank.py 的模块注释）：
    音色库回答「谁在说」；这里回答「怎么说」—— 同一个角色的
    怒/喜/哀/惧…各存一条示范，是语气起伏的主杠杆。

这一页只做增删改查与试听；路由发生在「🎙 合成」页的导演模式里。
"""

from __future__ import annotations

import os
from typing import List

import gradio as gr

from webui_app import logging_setup as LOG
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.services import emotion_bank as EB

# 下拉框用「角色 · 情绪(中文) · 名称」展示
def _entry_label(e: EB.EmoRefEntry) -> str:
    zh = EB.EMOTION_LABELS.get(e.emotion, e.emotion)
    return f"{e.character} · {zh} · {e.name}"


def _labels() -> List[str]:
    return [_entry_label(e) for e in EB.list_entries()]


def _route_hint(character: str) -> str:
    if not character:
        return T.hint("在「🎙 合成」页的导演模式里填角色名后，"
                      "逐句将按情绪在这里取参考。")
    rs = EB.route_summary(character)
    if not rs["n"]:
        return T.warn(f"角色 <code>{character}</code> 还没有任何情感参考 —— "
                      "导演模式逐句将全部回退「跟随音色参考」。")
    zh = ", ".join(EB.EMOTION_LABELS.get(k, k) for k in rs["have"])
    msg = (T.tip(f"角色 <code>{character}</code> 已归档 {rs['n']} 条 · "
                 f"已有情绪：{zh}"))
    if rs["missing"]:
        miss = ", ".join(EB.EMOTION_LABELS.get(k, k) for k in rs["missing"])
        msg += T.hint(f"还缺：{miss} —— 缺的情绪会回退到该角色的其他参考。")
    return msg


def render(ctx: AppContext):
    gr.HTML(T.section(
        "情感参考库", "🎭",
        "给每个角色按情绪归档「最有戏」的几句台词。导演模式（合成页）会按"
        "每句台词的情绪自动挑参考 —— 上游 issue #321 的社区结论：情感参考音频"
        "+ emo_alpha≈0.6~0.9 是表现力与音色相似的平衡点。素材可以从"
        "「🔬 音频工作台」的切片导出，也可以直接上传。"))

    with gr.Row(equal_height=False):
        # ---------------- 左：入库 ----------------
        with gr.Column(scale=1):
            gr.HTML("### ➕ 入库（角色 × 情绪）")
            in_character = gr.Textbox(
                label="角色名（路由键，建议与音色库条目同名）",
                placeholder="例如：卡提希娅")
            in_emotion = gr.Dropdown(
                choices=[(f"{EB.EMOTION_LABELS[k]} ({k})", k)
                         for k in EB.EMOTION_KEYS],
                value="happy", label="情绪",
                info="与官方 8 维情感向量一一对应")
            in_audio = gr.Audio(
                label="音频（建议 5~15s、情绪饱满、干净）",
                type="filepath", sources=["upload"])
            in_note = gr.Textbox(label="备注（可选）",
                                 placeholder="例如：主线第三章爆发戏")
            add_btn = gr.Button("➕ 入库", variant="primary")
            add_out = gr.HTML("")

            with gr.Accordion("🏷 自动打标入库（emotion2vec）", open=False):
                gr.HTML(T.hint(
                    "一次上传该角色的多段台词，emotion2vec 逐条判情绪后"
                    "自动入库（分不出的按「回退情绪」计）。模型跑在 CPU、"
                    "用完即卸，不与引擎抢显存 —— 首次会从 ModelScope 下载约 1 GB。"))
                auto_char_tb = gr.Textbox(
                    label="角色名", placeholder="例如：卡提希娅")
                auto_files = gr.File(
                    label="音频文件（可多选）", file_count="multiple",
                    file_types=["audio"])
                auto_fallback_dd = gr.Dropdown(
                    choices=[(f"{EB.EMOTION_LABELS[k]} ({k})", k)
                             for k in EB.EMOTION_KEYS],
                    value="calm", label="分不出时的回退情绪")
                auto_btn = gr.Button("🏷 全部自动打标入库", variant="primary",
                                     size="sm")
                auto_out = gr.HTML("")

            with gr.Accordion("🫁 呼吸库（角色本人的吸气采样）", open=False):
                gr.HTML(T.hint(
                    "从该角色的<b>数据集素材</b>里自动检测吸气段（语音前的"
                    "短气流段）入库，导演模式会在块边界按概率插入（贴下一句"
                    "开口，-24~-30dBFS）。呼吸是「真人感」最强的生理印记，"
                    "采样来自角色本人才有音色连续性。"))
                with gr.Row():
                    breath_char_tb = gr.Textbox(label="角色名", scale=1,
                                                placeholder="与导演模式一致")
                    from webui_app.training import dataset as _DS
                    breath_ds_dd = gr.Dropdown(choices=_DS.list_datasets(),
                                                label="数据集", scale=1)
                    breath_btn = gr.Button("🫁 挖吸气入库", size="sm",
                                           variant="primary")
                breath_out = gr.HTML("")
                breath_md = gr.Markdown("")

        # ---------------- 右：已有条目 ----------------
        with gr.Column(scale=2):
            gr.HTML("### 📇 已有条目")
            table_md = gr.Markdown(EB.table_markdown())
            with gr.Row():
                refresh_btn = gr.Button("↻ 刷新", size="sm")
            with gr.Accordion("▶ 试听 / 重新体检 / 删除", open=False):
                entry_dd = gr.Dropdown(
                    choices=_labels(), label="选择条目",
                    allow_custom_value=False)
                entry_audio = gr.Audio(label="试听", type="filepath",
                                       interactive=False)
                entry_info = gr.HTML("")
                with gr.Row():
                    reanalyze_btn = gr.Button("🔬 重新体检", size="sm")
                    del_btn = gr.Button("🗑 删除", variant="stop", size="sm")
                op_out = gr.HTML("")
            route_char_tb = gr.Textbox(
                label="路由预览：输入角色名", placeholder="例如：卡提希娅")
            route_hint = gr.HTML("")

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _refresh():
        labels = _labels()
        return (gr.update(value=EB.table_markdown()),
                gr.update(choices=labels,
                          value=labels[0] if labels else None))

    def on_add(character, emotion, audio_path, note):
        if not (character or "").strip():
            return T.err("角色名不能为空。")
        if not audio_path or not str(audio_path).strip():
            return T.err("请先上传音频。")
        try:
            e = EB.add(character.strip(), emotion, str(audio_path),
                       note=(note or "").strip())
        except Exception as ex:
            return T.err(f"入库失败：{type(ex).__name__}: {ex}")
        zh = EB.EMOTION_LABELS.get(e.emotion, e.emotion)
        msg = (f"✅ 已入库 <code>{e.name}</code>（{e.character} · {zh} · "
               f"体检 {e.score} 分/{e.grade}）")
        if e.duration < 5:
            msg += " · 提示：短于 5s 的情绪示范说服力有限，建议 5~15s"
        return T.tip(msg)

    add_btn.click(on_add,
                  inputs=[in_character, in_emotion, in_audio, in_note],
                  outputs=[add_out])

    @LOG.ui_guard("emobank.on_autotag", slow_sec=5.0)
    def on_autotag(character, files, fallback):
        char = (character or "").strip()
        if not char:
            return T.err("角色名不能为空。")
        paths = [f.name if hasattr(f, "name") else str(f)
                 for f in (files or [])]
        paths = [p for p in paths if p and os.path.isfile(p)]
        if not paths:
            return T.err("请先选择音频文件。")
        try:
            r = EB.autotag_add(char, paths, fallback_emotion=fallback or "calm")
        except Exception as ex:
            return T.err(f"自动打标失败：{type(ex).__name__}: {ex}")
        parts = [f"✅ 入库 {len(r['added'])} 条"]
        if r["added"]:
            parts.append("<code>" + "</code> <code>".join(r["added"][:12])
                         + ("…" if len(r["added"]) > 12 else "") + "</code>")
        if r["failed"]:
            parts.append(f"⚠️ 失败 {len(r['failed'])} 条："
                         + "；".join(f"{os.path.basename(x['path'])}: "
                                     f"{x['error'][:60]}"
                                     for x in r["failed"][:4]))
        return T.tip(" · ".join(parts))

    auto_btn.click(on_autotag,
                   inputs=[auto_char_tb, auto_files, auto_fallback_dd],
                   outputs=[auto_out])

    @LOG.ui_guard("emobank.on_breath", slow_sec=5.0)
    def on_breath(character, dataset):
        from webui_app.services import breath_bank as BB
        from webui_app.training import dataset as DS
        char = (character or "").strip()
        if not char:
            return T.err("角色名为空。"), gr.update()
        if not dataset or not DS.exists(dataset):
            return T.err("先选择数据集。"), gr.update()
        try:
            r = BB.build_from_dataset(char, dataset)
        except Exception as ex:
            return T.err(f"挖呼吸失败：{type(ex).__name__}: {ex}"), gr.update()
        if not r.get("ok"):
            return T.err(str(r.get("error", ""))), gr.update()
        msg = (f"🫁 扫描 {r['scanned']} 条素材，入库 {r['added']} 个吸气采样"
               + ("" if r["added"] else
                  " —— 没挖到（素材的吸气多被掐静音剪掉了？重跑一键三连"
                  "可关掉掐静音再试）"))
        return T.tip(msg), BB.table_markdown(char)

    breath_btn.click(on_breath,
                     inputs=[breath_char_tb, breath_ds_dd],
                     outputs=[breath_out, breath_md])

    def on_select(label):
        if not label:
            return gr.update(), ""
        for e in EB.list_entries():
            if _entry_label(e) == label:
                info = (f"<code>{e.name}</code> · {e.duration:.2f}s · "
                        f"{e.sample_rate} Hz · SNR {e.snr_db:.1f} dB · "
                        f"体检 {e.score}（{e.grade}）"
                        + (f"<br>备注：{e.note}" if e.note else ""))
                return e.audio_path, info
        return gr.update(), "条目已不存在，请刷新。"

    entry_dd.change(on_select, inputs=[entry_dd],
                    outputs=[entry_audio, entry_info])

    @LOG.ui_guard("emobank.on_reanalyze")
    def on_reanalyze(label):
        if not label:
            return T.err("先选择条目。")
        name = label.rsplit("·", 1)[-1].strip()
        e = EB.reanalyze(name)
        if e is None:
            return T.err(f"找不到条目：{name}")
        return T.tip(f"体检完成：{e.score} 分（{e.grade}）")

    reanalyze_btn.click(on_reanalyze, inputs=[entry_dd], outputs=[op_out])

    def on_delete(label):
        if not label:
            return T.err("先选择条目。")
        name = label.rsplit("·", 1)[-1].strip()
        ok = EB.remove(name)
        return (T.tip(f"已删除 <code>{name}</code>") if ok
                else T.err(f"删除失败：找不到 {name}"))

    del_btn.click(on_delete, inputs=[entry_dd], outputs=[op_out])

    refresh_btn.click(
        lambda: (*_refresh(), ""),
        outputs=[table_md, entry_dd, op_out])

    route_char_tb.change(lambda c: _route_hint((c or "").strip()),
                         inputs=[route_char_tb], outputs=[route_hint])

    def on_page_load():
        return (*_refresh(), _route_hint((route_char_tb.value or "").strip()))

    return {
        "page_load": (on_page_load, [table_md, entry_dd, route_hint]),
    }
