"""🎵 歌声训练（E1-T）Tab —— 从公开歌声数据到翻唱页可用的分步向导。

链路（docs/dev-plan-e1t-2026-10.md）：
    ① 引导复制（s2mel.pth → s2mel_singing.pth）
    ② 导入 GTSinger（公共歌声数据，一次导入全程复用）
    ③ singing 特征提取（F0/技巧标签，与 mel 帧对齐）
    ④ capability 训练（一次，全角色共享：教模型「怎么唱」）
    ⑤ 合并 → checkpoints/s2mel_singing.pth
    ⑥ 角色歌声身份 LoRA（每角色一次：教模型「谁在唱」）
    ⑦ 翻唱页使用（只练 CFM、不练 GPT；参考音频沿用现有）

后台执行：与一键三连同一套 TrainRunner（互斥/进度/停止/日志）。
服务逻辑全部在 webui_app/services/e1t_wizard.py，本文件只做 UI 编排。
"""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

from webui_app import logging_setup as LOG
from webui_app.context import AppContext
from webui_app.services import e1t_wizard as WZ
from webui_app.services import gts_fetch as GF
from webui_app.training.runner import get_runner
from webui_app.widgets import make_component

# 预设 label → key（与训练页 PRESETS 同一张表；这里不 import train_tab
# 是为了保持 tab 之间零耦合——改训练页预设不会静默改变这里的语义）
_PRESET_OF = {
    "🟢 保守（首跑推荐）": "conservative",
    "🟢 保守（推荐）": "conservative",
    "⚖️ 均衡": "balanced",
    "🎭 表现力": "expressive",
    "🔴 激进": "aggressive",
}

# 语言 label（⓪ 拉取的 CheckboxGroup）↔ 语言码
_LANG_OF = {GF.lang_label(c): c for c in GF.GTS_LANGS}

_LICENSE_MD = (
    "> ⚖️ **GTSinger 数据许可（务必读一次）**：检索显示 CC BY-NC-SA 4.0 ——"
    " **署名 + 非商用 + 相同方式共享**。个人研究/学习可直接使用；"
    " 免费公开基于它训练的权重需要署名 GTSinger 并按同协议共享；"
    "**任何商用形态（含商单配音、付费内容）需另行获得授权**。"
    " 本项目自身还受 bilibili 模型使用协议约束（衍生权重同约束），两层义务叠加。"
    "（信息如上，不构成法律意见；正式发布前请法务复核。）"
)

_REUSE_MD = (
    "**这些是一次性成本，之后全部复用**：\n"
    "- ②③ 的数据准备：所有角色共享；capability 换预设重训、后续加新角色都不用再导\n"
    "- ④⑤ 的歌声能力：全角色共享（「能力一次训，身份每次训」）\n"
    "- ⑥ 的角色素材：**现有 TTS 说话数据集直接用**（特征已提好，不必重提）；"
    "同一角色换歌曲也不用重训\n"
    "- 翻唱参考音频：沿用现在翻唱页用的那 15s 即可"
)


def render(ctx: AppContext) -> Dict[str, Any]:
    runner = get_runner()
    eng = ctx.engine

    gr.HTML("<div style='margin-top:2px'></div>")
    gr.Markdown(
        f"""### 🎵 歌声训练（E1-T）· 分步向导
把「音高/技巧 → 音色」的映射训进 CFM，让**只有说话参考的角色**也能唱出
音色不崩的高音。默认 TTS 全程不受影响（独立权重 `s2mel_singing.pth`，
原 `s2mel.pth` 字节级不动）。完整文档：`docs/dev-plan-e1t-2026-10.md`。

歌声数据默认自动拉取到 `datasets/_sources/GTSinger`（hf-mirror 镜像、断点续传，
不涉及 GitHub）；也可以把自备的 GTSinger 解压进同一目录跳过下载。

{_LICENSE_MD}""")

    # ---------------- 流程总览（七步状态） ----------------
    with gr.Row():
        status_md = gr.Markdown(WZ.status_markdown())
        refresh_btn = gr.Button("↻ 刷新状态", size="sm")

    gr.Markdown("---")

    # ---------------- 🚀 一键全链 ----------------
    with gr.Accordion("🚀 一键全链（推荐：配置好参数，一个按钮跑到底）",
                      open=True):
        gr.Markdown(
            "依次自动执行：**拉取数据（缺失时）→ ①引导 → ②导入 → ③特征 → "
            "④capability → ⑤合并 → ⑥身份（可选）→ ⑦接线翻唱页**。"
            "每段失败即停且成果保留——修好后**再点一次会跳过已完成的部分**。"
            "参数就用下方各步骤卡片里的同一批控件（在哪儿改都生效）。")
        with gr.Row():
            oneclick_identity_ck = gr.Checkbox(
                label="⑥ 顺带练角色身份 LoRA", value=False,
                info="勾选后用⑥卡片的角色数据集在歌声底座上练身份")
            oneclick_autofetch_ck = gr.Checkbox(
                label="⓪ 数据源为空时自动镜像拉取", value=True,
                info="语言用⓪卡片的选择；数据已就绪则跳过")
            oneclick_btn = gr.Button("🚀 一键全链（后台）",
                                     variant="primary")
        gr.Markdown(
            "完成后的「接线」= 写入翻唱页参数记忆（歌声模型开 + 身份 LoRA "
            "选好），下次进 🎤 AI翻唱 直接开始用。")

    # ---------------- ⓪ 镜像拉取 ----------------
    with gr.Accordion("⓪ 获取歌声数据（hf-mirror 镜像下载，断点续传）",
                      open=False):
        gr.Markdown(
            f"默认目录：`{GF.DEFAULT_DEST}`（项目内，已忽略 git 提交）。"
            "全量约 40GB 量级；**zh+en 子集约 15–20GB**，8GB 显存的训练规模"
            "建议从子集开始。GTSinger 是数万个小文件，下载需要耐心；"
            "中断后再点一次即续传。自备数据：解压到同一目录可跳过本步。")
        fetch_langs_cg = gr.CheckboxGroup(
            choices=[GF.lang_label(c) for c in GF.GTS_LANGS],
            value=[GF.lang_label("zh"), GF.lang_label("en")],
            label="语言子集")
        fetch_btn = gr.Button("⓪ 下载到默认目录（后台）",
                              variant="primary", size="sm")

    # ---------------- ① 引导复制 ----------------
    with gr.Accordion("① 引导复制（一次性，约 10 秒）", open=False):
        gr.Markdown(
            "首次使用前，把原底座复制一份作为歌声曲线的起点："
            "`checkpoints/s2mel.pth → checkpoints/s2mel_singing.pth`。"
            "**不会改动原文件**；第⑤步合并会把训练成果写进新文件。")
        boot_btn = gr.Button("① 生成歌声底座文件", variant="primary", size="sm")

    # ---------------- ② 导入 GTSinger ----------------
    with gr.Accordion("② 导入 GTSinger（公共歌声数据）", open=False):
        gr.Markdown(_REUSE_MD)
        with gr.Row():
            gts_src_tb = gr.Textbox(
                label="GTSinger 根目录",
                value=GF.DEFAULT_DEST,
                info="默认=⓪ 的镜像下载目录；自备数据改成你的路径",
                scale=2)
            gts_name_tb = gr.Textbox(label="目标数据集名", value="gtsinger_zh_en",
                                     scale=1)
        with gr.Row():
            gts_langs_tb = make_component("e1t_import_langs")
            gts_aug_tb = make_component("e1t_import_pitch_aug")
        with gr.Row():
            gts_max_n = make_component("e1t_import_max_items")
            gts_copy_ck = make_component("e1t_import_copy")
        import_btn = gr.Button("② 开始导入（后台）", variant="primary", size="sm")

    # ---------------- ③ singing 特征 ----------------
    with gr.Accordion("③ 提取歌声特征（F0 / 技巧标签）", open=False):
        gr.Markdown(
            "对②产出的数据集做 singing 模式特征提取：harvest F0 对齐到 mel 帧 + "
            "technique 标签展开。**普通说话数据集不需要这步**（角色素材走旧特征）。"
            "提取会临时加载模型（约 3GB 显存），期间引擎保持卸载。")
        with gr.Row():
            feat_ds_dd = gr.Dropdown(choices=WZ.singing_datasets(),
                                     label="歌声数据集（② 的产出）",
                                     interactive=True, scale=2)
            feat_refresh_btn = gr.Button("↻ 刷新列表", size="sm", scale=0)
        extract_btn = gr.Button("③ 提取歌声特征（后台）", variant="primary", size="sm")

    # ---------------- ④ capability 训练 ----------------
    with gr.Accordion("④ capability 训练（一次，全角色共享）", open=False):
        gr.Markdown(
            "在歌声底座上用公开歌声训练「怎么唱」：F0/技巧条件透传进 CFM + "
            "技巧均衡采样。**建议先小子集冒烟**（② 限 1500 条 + epochs 2–4）"
            "跑通全链，再放量。8GB 显存下全量 80h 是天级任务，量力而行。")
        with gr.Row():
            cap_ds_dd = gr.Dropdown(choices=WZ.singing_datasets(),
                                    label="歌声数据集", interactive=True, scale=2)
            cap_run_tb = gr.Textbox(label="run 名", value="singing_capability_v1",
                                    scale=1)
        with gr.Row():
            cap_preset_dd = make_component("e1t_cap_preset")
            cap_frames_sl = make_component("e1t_cap_frames")
            cap_epochs_sl = make_component("e1t_cap_epochs")
        cap_btn = gr.Button("④ 开始 capability 训练（后台，需引擎卸载）",
                            variant="primary", size="sm")

    # ---------------- ⑤ 合并 ----------------
    with gr.Accordion("⑤ 合并 capability → 歌声底座", open=False):
        gr.Markdown(
            "把④的 LoRA + TimbreConditioner 烘进 `checkpoints/s2mel_singing.pth`"
            "（CPU 合并；输出只允许这个歌声新名字——原底座有白名单守护，覆盖不进去）。"
            "合并是叠加式的：以后重训 capability 再合并即可，起点是当前成果。")
        with gr.Row():
            merge_run_dd = gr.Dropdown(choices=WZ.capability_runs(),
                                       label="capability run（④ 的产出）",
                                       interactive=True, scale=2)
            merge_btn = gr.Button("⑤ 合并（后台，CPU）",
                                  variant="primary", size="sm", scale=0)
        merge_md = gr.Markdown()

    # ---------------- ⑥ 角色身份 ----------------
    with gr.Accordion("⑥ 角色歌声身份 LoRA（每个角色一次）", open=False):
        gr.Markdown(
            "在歌声底座上、用**角色说话数据集**练身份锚点——**一键三连切好、"
            "降噪/响度锚定、ASR 对齐、特征已提好的那批音频就是它**，直接选"
            "（下拉按当时填的模型名称/角色名显示）。**只练 CFM、不练 GPT**"
            "（翻唱绕过 GPT）；条件不参与 → capability 的 conditioner 自动冻结，"
            "不会遗忘高音能力。唱得不像可以加素材重训。")
        with gr.Row():
            id_ds_dd = gr.Dropdown(choices=WZ.identity_dataset_choices(),
                                    label="角色说话数据集（一键三连产物会显示角色名）",
                                    interactive=True, scale=2)
            id_run_tb = gr.Textbox(label="run 名（选数据集后自动填）",
                                   value="", scale=1)
        with gr.Row():
            id_preset_dd = make_component("e1t_id_preset")
            id_epochs_sl = make_component("e1t_id_epochs")
        id_btn = gr.Button("⑥ 开始身份训练（后台，需引擎卸载）",
                           variant="primary", size="sm")

    # ---------------- ⑦ 使用指引 ----------------
    with gr.Accordion("⑦ 到翻唱页使用", open=False):
        gr.Markdown(
            """1. 🎤 **AI翻唱** 页 → 音色 LoRA 选 **⑥ 的身份 run**（不要选 capability——它已烘进底座）；
2. 展开「③c 歌声模型（E1-T 实验性）」→ 勾选 **歌声模型**；
3. 技巧条件先保持 `preserve`、音区映射 `auto`、BoN 先关——跑一版基线；
4. A/B 对比锚点：`python tools/e1t_baseline.py`（基线 `outputs/e1t_baseline.json` 已生成）；
5. 高音仍不稳时再开 **BoN 高音择优**（耗时 ×3 仅高音块）；戏腔/约德尔类快速换声区内容关掉「八度跳变修正」。

验收口径（G1a/G1b 详见母规划 §7）：高音子集 HNR、F0 50 音分内、八度错误——
**不要只看声纹相似度 SS**（「像但破」是失败态）。""")

    # ---------------- 运行面板（runner 轮询，一键三连同款） ----------------
    gr.Markdown("---")
    gr.Markdown("#### 后台任务（导入 / 提取 / 训练 / 合并 共用一条队列）")
    progress_html = gr.HTML("")
    log_ta = gr.Textbox(label="过程日志", lines=12, max_lines=24,
                        interactive=False)
    with gr.Row():
        stop_btn = gr.Button("⏹ 停止当前任务", variant="stop", size="sm")

    def _submit(kind: str, label: str, fn, require_engine: str = "none"):
        r = runner.submit(kind, label, fn,
                          require_engine=require_engine, engine=eng)
        if not r.get("ok"):
            raise gr.Error(r.get("message", "启动失败"))
        gr.Info(f"{label} 已在后台启动")
        return WZ.status_markdown()

    @LOG.ui_guard("e1t.refresh")
    def on_refresh():
        return (WZ.status_markdown(),
                gr.update(choices=WZ.singing_datasets()),
                gr.update(choices=WZ.singing_datasets()),
                gr.update(choices=WZ.capability_runs()),
                gr.update(choices=WZ.identity_dataset_choices()))

    refresh_btn.click(on_refresh, outputs=[status_md, feat_ds_dd, cap_ds_dd,
                                           merge_run_dd, id_ds_dd])
    feat_refresh_btn.click(
        lambda: gr.update(choices=WZ.singing_datasets()),
        outputs=[feat_ds_dd])

    @LOG.ui_guard("e1t.id_ds")
    def on_id_dataset(ds, current_run):
        """选了角色数据集 → 自动填默认 run 名（角色名 + 后缀）。

        只在 run 名为空、或仍是上一次自动填充值时覆盖——用户手改过就不动。
        """
        if not (ds or "").strip():
            return gr.update()
        new = WZ.identity_run_default(ds)
        cur = (current_run or "").strip()
        if not cur or cur.endswith("_singing_identity"):
            return gr.update(value=new)
        return gr.update()

    id_ds_dd.change(on_id_dataset, inputs=[id_ds_dd, id_run_tb],
                    outputs=[id_run_tb])

    @LOG.ui_guard("e1t.bootstrap")
    def on_bootstrap():
        return _submit("e1t_boot", "E1-T ① 引导复制", WZ.bootstrap_fn())

    @LOG.ui_guard("e1t.import")
    def on_import(src, name, langs, aug, max_items, copy_files):
        if not (src or "").strip():
            raise gr.Error("先填 GTSinger 根目录（需自行从 HuggingFace 下载）")
        if not (name or "").strip():
            raise gr.Error("先填目标数据集名")
        fn = WZ.import_fn(src, name, langs, aug, int(max_items or 0),
                          bool(copy_files))
        return _submit("e1t_import", f"E1-T ② 导入 → {name}", fn)

    @LOG.ui_guard("e1t.extract")
    def on_extract(ds):
        if not (ds or "").strip():
            raise gr.Error("先选择歌声数据集（没有就先做第②步）")
        # 自带模型实例 + 引擎保持卸载（与训练同档互斥）
        return _submit("e1t_extract", f"E1-T ③ 歌声特征 · {ds}",
                       WZ.extract_fn(ds), require_engine="unloaded")

    @LOG.ui_guard("e1t.capability")
    def on_capability(ds, run, preset_label, frames, epochs):
        if not (ds or "").strip():
            raise gr.Error("先选择歌声数据集")
        preset = _PRESET_OF.get(preset_label, "conservative")
        fn = WZ.capability_train_fn(ds, run, preset, int(frames),
                                    int(epochs or 0))
        return _submit("e1t_train", f"E1-T ④ capability · {ds}",
                       fn, require_engine="unloaded")

    @LOG.ui_guard("e1t.identity")
    def on_identity(ds, run, preset_label, epochs):
        if not (ds or "").strip():
            raise gr.Error("先选择角色说话数据集（现有的即可）")
        # 兜底：change 事件没触发时也保证有可区分的 run 名
        run = (run or "").strip() or WZ.identity_run_default(ds)
        preset = _PRESET_OF.get(preset_label, "conservative")
        fn = WZ.identity_train_fn(ds, run, preset, 0, int(epochs or 0))
        return _submit("e1t_train", f"E1-T ⑥ 身份 · {ds}",
                       fn, require_engine="unloaded")

    @LOG.ui_guard("e1t.merge")
    def on_merge(run):
        if not (run or "").strip():
            raise gr.Error("先选择 capability run（没有就先做第④步）")
        return _submit("e1t_merge", f"E1-T ⑤ 合并 · {run}", WZ.merge_fn(run))

    @LOG.ui_guard("e1t.fetch")
    def on_fetch(lang_labels):
        langs = [_LANG_OF.get(x, x) for x in (lang_labels or [])]
        if not langs:
            raise gr.Error("至少选一个语言；要全量请用 CLI："
                           "python tools/fetch_gtsinger.py --all")
        return _submit("e1t_fetch", f"E1-T ⓪ 拉取 · {','.join(langs)}",
                       WZ.fetch_fn(langs))

    @LOG.ui_guard("e1t.oneclick")
    def on_oneclick(src, name, lang_labels, aug, max_items,
                    cap_run, cap_preset_label, cap_frames, cap_epochs,
                    identity, id_ds, id_run, id_preset_label, id_epochs,
                    autofetch):
        if not (name or "").strip():
            raise gr.Error("先填②里的目标数据集名")
        if not (cap_run or "").strip():
            raise gr.Error("先填④里的 capability run 名")
        if identity and not (id_ds or "").strip():
            raise gr.Error("勾了「顺带练身份」：先在⑥选角色数据集")
        langs = [_LANG_OF.get(x, x) for x in (lang_labels or [])]

        def fn(progress, should_stop):
            # 特征提取与训练都要显存：引擎若在加载态先卸掉（若有翻唱/变声
            # 租约会在这里抛出——保护语义正确，原因会显示给用户）
            if getattr(eng, "loaded", False):
                eng.unload()
            id_run_eff = (id_run or "").strip()
            if not id_run_eff and identity and (id_ds or "").strip():
                id_run_eff = WZ.identity_run_default(id_ds)
            inner = WZ.pipeline_fn(
                src, name.strip(), langs, aug, int(max_items or 0),
                cap_run.strip(),
                _PRESET_OF.get(cap_preset_label, "conservative"),
                int(cap_frames or 1800), int(cap_epochs or 0),
                bool(identity), id_ds or "", id_run_eff,
                _PRESET_OF.get(id_preset_label, "conservative"),
                int(id_epochs or 0), bool(autofetch))
            return inner(progress, should_stop)

        return _submit("e1t_pipeline", f"E1-T 🚀 一键全链 · {name.strip()}",
                       fn, require_engine="unloaded")

    @LOG.ui_guard("e1t.stop")
    def on_stop():
        r = runner.cancel()
        return WZ.status_markdown()

    fetch_btn.click(on_fetch, inputs=[fetch_langs_cg], outputs=[status_md])
    oneclick_btn.click(
        on_oneclick,
        inputs=[gts_src_tb, gts_name_tb, fetch_langs_cg, gts_aug_tb,
                gts_max_n, cap_run_tb, cap_preset_dd, cap_frames_sl,
                cap_epochs_sl, oneclick_identity_ck, id_ds_dd, id_run_tb,
                id_preset_dd, id_epochs_sl, oneclick_autofetch_ck],
        outputs=[status_md])
    boot_btn.click(on_bootstrap, outputs=[status_md])
    import_btn.click(on_import,
                     inputs=[gts_src_tb, gts_name_tb, gts_langs_tb, gts_aug_tb,
                             gts_max_n, gts_copy_ck],
                     outputs=[status_md])
    extract_btn.click(on_extract, inputs=[feat_ds_dd], outputs=[status_md])
    cap_btn.click(on_capability,
                  inputs=[cap_ds_dd, cap_run_tb, cap_preset_dd, cap_frames_sl,
                          cap_epochs_sl],
                  outputs=[status_md])
    id_btn.click(on_identity,
                 inputs=[id_ds_dd, id_run_tb, id_preset_dd, id_epochs_sl],
                 outputs=[status_md])
    merge_btn.click(on_merge, inputs=[merge_run_dd],
                    outputs=[status_md])
    stop_btn.click(on_stop, outputs=[status_md])

    # ---------- 轮询：进度条 + 状态面板 + 合并报告 + 日志 ----------
    poll_cache: Dict[str, Any] = {"snap": None, "merge_md": None}

    def on_poll():
        snap = runner.snapshot()
        if snap == poll_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update(), gr.update(), gr.update()
        poll_cache["snap"] = snap

        done = snap["ok"] is not None and not snap["running"]
        bar = ""
        if snap["running"] or done:
            pct = int(snap["progress"] * 100)
            icon = "🎉" if (done and snap["ok"]) else ("🔴" if done else "▶️")
            bar = (f'<div style="margin:4px 0">{icon} <b>{snap["label"]}</b> · '
                   f'{pct}% · {snap["message"]}'
                   + (" · <b>已请求停止</b>" if snap["stop_requested"] else "")
                   + f' · {snap["seconds"]}s'
                   + (f'<br><span style="color:#c00">{snap["error"]}</span>'
                      if (done and not snap["ok"] and snap["error"]) else "")
                   + '<div style="background:var(--border-color-primary);'
                     'border-radius:6px;height:8px;margin-top:4px">'
                     f'<div style="width:{pct}%;height:8px;border-radius:6px;'
                     'background:var(--color-accent)"></div></div></div>')

        status = WZ.status_markdown() if done else gr.update()
        merge_out = gr.update()
        if done and snap["kind"] == "e1t_pipeline":
            res = snap.get("result") or {}
            if res.get("wire"):
                w = res["wire"]
                lora_note = (f"，LoRA={w['lora_run']}" if w.get("lora_run")
                             else "")
                merge_out = gr.update(value=(
                    "### 🎉 一键全链完成\n\n"
                    "- 阶段：" + " → ".join(res.get("stages") or []) + "\n"
                    f"- 合并产物：`{res.get('merged_to', 's2mel_singing.pth')}`\n"
                    "- ✅ 已接线翻唱页：歌声模型开" + lora_note
                    + "——到 🎤 AI翻唱 直接开始用。"))
            elif not snap["ok"] and snap.get("error"):
                stage = (snap.get("result") or {}).get("stage", "?")
                merge_out = gr.update(
                    value=(f"### 🔴 全链在第 **{stage}** 步停下\n\n"
                           f"{snap['error']}\n\n> 修好后**再点一次一键**——"
                           "已完成的部分会自动跳过。"))
        elif done and snap["kind"] == "e1t_merge":
            res = snap.get("result") or {}
            md = res.get("markdown")
            if md and poll_cache["merge_md"] != md:
                poll_cache["merge_md"] = md
                merge_out = gr.update(value=md)
        return (gr.update(value=bar), status, merge_out,
                gr.update(value=runner.log_text()))

    timer = gr.Timer(value=2.5, active=True)
    timer.tick(on_poll, inputs=[],
               outputs=[progress_html, status_md, merge_md, log_ta])

    return {
        "status_md": status_md,
        "progress_html": progress_html,
        "log_ta": log_ta,
    }
