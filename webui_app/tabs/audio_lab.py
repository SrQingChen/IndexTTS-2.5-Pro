"""Tab：参考音频工作台（L0）。

这是整套体系里**投入产出比最高**的一环。IndexTTS-2.5 是零样本 TTS，
音色几乎完全由参考音频决定（CAMPPlus 声纹 + w2v-BERT 情感特征 + ref_mel
声学模板三路注入），所以「把参考音频选对、处理好」往往比训练 LoRA 更有效、
更零风险。

流程：上传 → 体检打分 → 智能切片 → 增强处理 → 入库 → 在合成页直接调用。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import logging_setup as LOG
from webui_app import theme as T
from webui_app.config import OUTPUT_SAMPLE_RATE, REF_AUDIO_IDEAL
from webui_app.context import AppContext
from webui_app.services import audio_lab as AL
from webui_app.services import voice_bank

WORK_DIR = os.path.join("outputs", "lab")


def render(ctx: AppContext):
    cfg = ctx.cfg
    os.makedirs(WORK_DIR, exist_ok=True)
    sb = ctx.component("statusbar")

    gr.HTML(T.section(
        "参考音频工作台", "🔬",
        "官方 <code>_load_and_cut_audio(prompt, 15)</code> 是<b>取前 15 秒</b>而不是择优，"
        "所以参考音频<b>开头</b>的质量最关键。这里的每一项体检指标都对应推理链路里的"
        "一个真实约束，不是泛泛的「音质好不好」。"))

    with gr.Row(equal_height=False):
        # =================================================================
        # 左：输入与体检
        # =================================================================
        with gr.Column(scale=1, min_width=380):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("① 上传素材", "📤", ""))
                src_audio = gr.Audio(
                    label="原始音频", type="filepath",
                    sources=["upload", "microphone"],
                    elem_classes=["ix-audio-compact"],
                )
                src_path = gr.Textbox(
                    label="或直接填服务器上的文件路径",
                    placeholder="例如 examples/voice_01.wav 或 D:/record/take3.wav",
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔬 开始体检", variant="primary", scale=2)
                    use_path_btn = gr.Button("用该路径", scale=1, size="sm")
                report_md = gr.Markdown("_上传音频后点「开始体检」。_")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("② 智能切片", "✂️",
                                  "滑动窗口找出评分最高的连续片段。评分综合语音占比、"
                                  "信噪比、削波、响度，并优先让边界落在停顿处以免切断音节。"))
                with gr.Row():
                    seg_target = gr.Slider(
                        4.0, 15.0, 12.0, step=0.5, label="目标片段时长",
                        info="官方上限 15s；8~15s 是音色稳定性的甜区",
                    )
                    seg_min = gr.Slider(
                        2.0, 10.0, 6.0, step=0.5, label="最短可接受时长",
                        info="低于 3s 时 CAMPPlus 声纹统计不稳定",
                    )
                find_seg_btn = gr.Button("🔍 扫描候选片段")
                seg_table_md = gr.Markdown("_尚未扫描。_")
                seg_pick = gr.Dropdown(
                    choices=[], value=None, label="选择要导出的片段",
                    info="评分最高的是 #0，但内容/情绪是否合适请自己试听判断",
                )
                seg_audio = gr.Audio(label="片段预览", type="filepath")
                with gr.Row():
                    seg_export_btn = gr.Button("导出该片段", size="sm", scale=1)
                    seg_as_src_btn = gr.Button("设为主素材", size="sm", scale=1)

        # =================================================================
        # 右：增强与入库
        # =================================================================
        with gr.Column(scale=1, min_width=380):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("③ 增强处理", "🛠",
                                  "顺序刻意安排为：去DC → 裁静音 → 降噪 → 响度归一 → "
                                  "重采样 → 限长。先裁静音再降噪可省算力；"
                                  "先降噪再归一可避免把噪声一起放大。"))
                with gr.Row():
                    denoise_cb = gr.Checkbox(True, label="频谱降噪")
                    denoise_str = gr.Slider(
                        0.0, 1.0, 0.5, step=0.05, label="降噪强度",
                        info="0.3~0.6 较安全；过高会把语音细节一起削掉，听感发闷有水声",
                    )
                with gr.Row():
                    norm_cb = gr.Checkbox(True, label="响度归一")
                    norm_db = gr.Slider(
                        -35.0, -8.0, -20.0, step=0.5, label="目标 RMS (dBFS)",
                        info="-20 dBFS 附近最稳；太轻则声纹弱，太重则接近削波",
                    )
                with gr.Row():
                    trim_cb = gr.Checkbox(True, label="裁掉首尾静音")
                    trim_db = gr.Slider(
                        -60.0, -25.0, -45.0, step=1.0, label="静音阈值 (dBFS)",
                    )
                with gr.Row():
                    resample_cb = gr.Checkbox(True, label="重采样")
                    resample_sr = gr.Dropdown(
                        choices=[22050, 24000, 16000, 44100, 48000],
                        value=OUTPUT_SAMPLE_RATE, label="目标采样率",
                        info="22050 = 官方 ref_mel 的采样率，推荐",
                    )
                limit_sec = gr.Slider(
                    3.0, 15.0, 15.0, step=0.5, label="最大时长（秒）",
                    info="官方硬截断到 15s，主动裁到这个长度可避免意外丢弃",
                )
                with gr.Row():
                    enhance_btn = gr.Button("✨ 执行增强", variant="primary", scale=2)
                    compare_btn = gr.Button("对比试听", scale=1, size="sm")
                enh_audio = gr.Audio(label="处理后音频", type="filepath")
                orig_audio = gr.Audio(label="处理前音频（对比用）", type="filepath",
                                      visible=False)
                enhance_md = gr.Markdown("")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("④ 存入音色库", "📚",
                                  "入库时会连同体检报告一起保存，之后在合成页可以直接调用，"
                                  "并且能看到「这个音色当初为什么好/不好」。"))
                with gr.Row():
                    bank_name = gr.Textbox(label="音色名称", placeholder="例如 主播A_温暖",
                                           scale=2)
                    bank_lang = gr.Dropdown(
                        choices=["ZH", "EN", "JA", "AR", "ES"], value="ZH",
                        label="语言", scale=1,
                    )
                bank_tags = gr.Textbox(
                    label="标签（逗号分隔）",
                    placeholder="例如 女声, 温暖, 有声书",
                    info="标签只用于你自己检索，不影响推理",
                )
                bank_note = gr.Textbox(
                    label="备注", lines=2,
                    placeholder="记录这段素材的来源、适用场景、注意事项…",
                )
                with gr.Row():
                    save_src_btn = gr.Button("存入原始素材", scale=1)
                    save_enh_btn = gr.Button("存入增强结果", variant="primary", scale=1)
                save_out = gr.HTML("")

    # =====================================================================
    # 音色库管理
    # =====================================================================
    with gr.Accordion("📚 音色库管理", open=False):
        with gr.Row():
            bank_dd = gr.Dropdown(choices=[""] + voice_bank.names(), value="",
                                  label="已入库音色", scale=3,
                                  allow_custom_value=False)
            bank_reload = gr.Button("↻ 刷新", scale=1, size="sm")
            bank_reanalyze = gr.Button("重新体检", scale=1, size="sm")
            bank_delete = gr.Button("🗑 删除", scale=1, size="sm", variant="stop")
            bank_import_ex = gr.Button("导入官方示例音频", scale=1, size="sm")
        bank_table = gr.Markdown(voice_bank.table_markdown())
        bank_detail = gr.Markdown("_选择一个音色查看详情_")
        bank_preview = gr.Audio(label="试听", type="filepath", visible=False)

    gr.HTML(T.section("为什么这一步比训练更重要", "💡", ""))
    gr.HTML(
        T.tip("IndexTTS-2.5 的零样本说话人相似度已达 <b>77.10%</b>（CV3-Eval test-zh）。"
              "对「想克隆某人音色」这个需求，<b>很多时候最优解不是训练，而是把参考音频选对</b>。")
        + T.hint(
            "参考音频会兵分三路，每一路都对素材质量敏感：<br>"
            "① <b>CAMPPlus → 192维声纹</b>：需要足够时长做统计，"
            "有 BGM 时它会把音乐当成声纹的一部分<br>"
            "② <b>w2v-BERT 第17层 → 情感特征</b>：对噪声敏感，且决定默认情感基调<br>"
            "③ <b>ref_mel → CFM 的 prompt</b>：<b>声学音色细节的主要来源</b>，"
            "削波会让高频结构直接失真")
        + T.warn("官方示例 <code>examples/voice_01.wav</code> 实测只有 <b>2.44 秒</b>，"
                 "低于 3 秒最低推荐值。用它做参考音频时音色稳定性会打折 —— "
                 "这也是本页存在的理由。")
    )

    # =====================================================================
    # 回调
    # =====================================================================

    def _resolve_source(audio_val, path_val) -> Optional[str]:
        """优先用上传控件的值，其次用手填路径；返回绝对路径或 None。"""
        p = str(audio_val or path_val or "").strip()
        if not p:
            return None
        ap = p if os.path.isabs(p) else os.path.abspath(p)
        return ap if os.path.isfile(ap) else None

    def on_use_path(path_val):
        p = (path_val or "").strip()
        if not p:
            gr.Warning("请先填写路径")
            return gr.update()
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            gr.Error(f"文件不存在：{ap}")
            return gr.update()
        return gr.update(value=ap)

    use_path_btn.click(on_use_path, inputs=[src_path], outputs=[src_audio])

    @LOG.ui_guard("lab.on_analyze", slow_sec=2.0)
    def on_analyze(audio_val, path_val):
        p = _resolve_source(audio_val, path_val)
        if not p:
            return "_请先上传音频或填写有效路径。_", gr.update()
        rep = AL.analyze(p)
        if not rep.ok:
            LOG.get_logger("lab").warning("体检失败：%s · %s", p, rep.error)
            return f"**分析失败**：{rep.error}", gr.update()
        LOG.get_logger("lab").info(
            "体检：%s · %.2fs · 评分 %.1f（%s）· SNR %.1f dB",
            p, rep.duration, rep.score, rep.grade, rep.snr_db)
        return AL.report_markdown(rep), gr.update(value=rep.path)

    analyze_btn.click(on_analyze, inputs=[src_audio, src_path],
                      outputs=[report_md, seg_audio])

    # ---------- 切片 ----------
    seg_state = gr.State([])
    # 候选片段是**按哪个音频**扫出来的。必须记住它，否则「设为主素材」之后
    # src_audio 换成了短片段、而 seg_state 里还是长音频的秒数偏移，再点一次
    # 就会按越界范围去切 → 切出 0 个采样点（实测症状：44 字节的空 wav）。
    seg_source = gr.State("")

    @LOG.ui_guard("lab.on_find_segments", slow_sec=2.0)
    def on_find_segments(audio_val, path_val, target, minsec):
        p = _resolve_source(audio_val, path_val)
        if not p:
            return ("_请先上传音频。_", gr.update(choices=[], value=None), [], "")
        segs = AL.find_segments(p, target_sec=float(target), min_sec=float(minsec))
        choices = [
            f"#{i}  {s.start:.2f}s~{s.end:.2f}s  评分{s.score:.1f}  "
            f"SNR{s.snr_db:.0f}dB  语音{s.voiced_ratio*100:.0f}%"
            for i, s in enumerate(segs)
        ]
        LOG.get_logger("lab").info(
            "扫描候选片段：%s · 目标 %.1fs/最短 %.1fs → %d 个候选",
            p, float(target), float(minsec), len(segs))
        return (AL.segments_markdown(segs),
                gr.update(choices=choices, value=choices[0] if choices else None),
                segs, p)

    find_seg_btn.click(on_find_segments,
                       inputs=[src_audio, src_path, seg_target, seg_min],
                       outputs=[seg_table_md, seg_pick, seg_state, seg_source])

    # 换了主素材，之前扫的候选偏移就全部作废 —— 立刻清掉，避免用旧偏移去切
    def on_src_changed(audio_val):
        LOG.get_logger("lab").info("主素材已更换：%s（候选片段作废）", audio_val)
        return (gr.update(choices=[], value=None), [],
                "_主素材已更换，候选片段已作废，请重新「扫描候选片段」。_", "")

    src_audio.change(on_src_changed, inputs=[src_audio],
                     outputs=[seg_pick, seg_state, seg_table_md, seg_source])

    # 选中片段后只控制导出按钮的可用态；预览在导出后才给出
    seg_pick.change(lambda c: gr.update(interactive=bool(c)),
                    inputs=[seg_pick], outputs=[seg_export_btn])

    def _export_segment(choice, segs, scan_src, audio_val, path_val):
        """把选中的候选片段导出成一个 wav。返回 (输出的路径, 错误文案)。

        **切片一律针对「扫描时那个音频」**，而不是当前 src_audio：设为主素材之后
        src_audio 会变成短片段，此时再按长音频的秒数偏移去切就会越界
        （实测产出 44 字节的空 wav）。所以这里优先用扫描来源。
        """
        p = str(scan_src or "").strip() or _resolve_source(audio_val, path_val)
        if not p or not os.path.isfile(p):
            return None, "源音频不在了（可能被移动或删除），请重新选择。"
        if not choice or not segs:
            return None, "请先「扫描候选片段」并选择一个。"
        try:
            i = int(choice.split("#")[1].split()[0])
        except (ValueError, IndexError):
            return None, "无法解析所选片段，请重新扫描。"
        if i >= len(segs):
            return None, "这个片段不在当前候选列表里（可能刚换过素材），请重新扫描。"

        seg = segs[i]
        os.makedirs(WORK_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(WORK_DIR, f"{stem}_seg{i}_{int(time.time())}.wav")
        try:
            AL.extract_segment(p, seg, out)
        except Exception as e:
            LOG.get_logger("lab").error(
                "导出片段失败：%s #%d %.2f~%.2fs ← %s",
                p, i, seg.start, seg.end, e, exc_info=True)
            return None, f"导出失败：{e}"
        if not os.path.isfile(out) or os.path.getsize(out) < 1024:
            return None, (f"导出的文件异常（{os.path.getsize(out) if os.path.isfile(out) else 0} "
                          "字节），请重新扫描后再试。")
        LOG.get_logger("lab").info(
            "导出片段：%s #%d %.2f~%.2fs → %s（%.1f KB）",
            p, i, seg.start, seg.end, out, os.path.getsize(out) / 1024)
        return out, ""

    @LOG.ui_guard("lab.on_seg_export", slow_sec=2.0)
    def on_seg_export(choice, segs, scan_src, audio_val, path_val):
        out, err = _export_segment(choice, segs, scan_src, audio_val, path_val)
        if err:
            gr.Error(err)
            return gr.update(), gr.update()
        return (gr.update(value=out), gr.update(value=out))

    seg_export_btn.click(on_seg_export,
                         inputs=[seg_pick, seg_state, seg_source,
                                 src_audio, src_path],
                         outputs=[seg_audio, src_audio])

    @LOG.ui_guard("lab.on_seg_as_src", slow_sec=2.0)
    def on_seg_as_src(choice, segs, scan_src, audio_val, path_val):
        """导出片段 → 设为主素材 → 顺手体检它（这样能直接接着做增强与入库）。"""
        # 失败分支也必须返回**全部 6 个**输出，否则 Gradio 会在运行时抛
        # 「didn't return enough output values」（tools/ui_output_check.py 会拦）。
        def _fail(msg: str):
            gr.Error(msg)
            return (gr.update(), gr.update(), T.err(msg),
                    gr.update(), segs, scan_src)

        out, err = _export_segment(choice, segs, scan_src, audio_val, path_val)
        if err:
            return _fail(f"<b>设为主素材失败</b>：{err}")

        rep = AL.analyze(out)
        if not rep.ok or rep.duration <= 0:
            return _fail(f"<b>切出来的片段无法使用</b>："
                         f"{rep.error or '时长为 0'}")

        LOG.get_logger("lab").info(
            "设为主素材：%s（%.2fs · 评分 %.1f %s）",
            out, rep.duration, rep.score, rep.grade)
        head = T.tip(
            f"✅ 已把片段设为主素材：<code>{os.path.basename(out)}</code> · "
            f"{rep.duration:.2f}s · 评分 <b>{rep.score}</b>（{rep.grade}）。"
            "下面是它的体检报告，可直接接着做「③ 增强处理」或「④ 存入音色库」。")
        # 候选片段作废：主素材已经换成这个短片段，旧偏移不再适用
        body = head + "\n\n" + AL.report_markdown(rep)
        return (gr.update(value=out), gr.update(value=out), body,
                gr.update(choices=[], value=None), [], "")

    seg_as_src_btn.click(on_seg_as_src,
                         inputs=[seg_pick, seg_state, seg_source,
                                 src_audio, src_path],
                         outputs=[seg_audio, src_audio, report_md,
                                  seg_pick, seg_state, seg_source])

    # ---------- 增强 ----------
    @LOG.ui_guard("lab.on_enhance", slow_sec=3.0)
    def on_enhance(audio_val, path_val, denoise, dstr, norm, ndb,
                   trim, tdb, resample, rsr, limit):
        p = _resolve_source(audio_val, path_val)
        if not p:
            LOG.get_logger("lab").warning("增强：没有可处理的音频")
            return gr.update(), gr.update(), "_请先上传音频或选择一个切片。_"

        os.makedirs(WORK_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(WORK_DIR, f"{stem}_enh_{int(time.time())}.wav")
        res = AL.enhance(
            p, out,
            denoise=bool(denoise), denoise_strength=float(dstr),
            normalize=bool(norm), target_dbfs=float(ndb),
            trim_silence=bool(trim), silence_thresh_db=float(tdb),
            resample=bool(resample), target_sr=int(rsr),
            max_sec=float(limit),
        )
        if not res.ok:
            LOG.get_logger("lab").error("增强失败：%s · %s", p, res.error)
            return gr.update(), gr.update(), T.err(f"<b>处理失败</b>：{res.error}")
        LOG.get_logger("lab").info(
            "增强：%s → %s（%.2fs · 评分 %s → %s · %d 步）",
            p, res.path, res.duration,
            (res.before or {}).get("score"), (res.after or {}).get("score"),
            len(res.steps or []))

        b = res.before or {}
        a = res.after or {}
        gr.Info(f"增强完成，评分 {b.get('score')} → {a.get('score')}")
        return gr.update(value=res.path), gr.update(value=p), AL.enhance_markdown(res)

    enh_inputs = [src_audio, src_path, denoise_cb, denoise_str, norm_cb, norm_db,
                  trim_cb, trim_db, resample_cb, resample_sr, limit_sec]
    enh_outputs = [enh_audio, orig_audio, enhance_md]
    enhance_btn.click(on_enhance, inputs=enh_inputs, outputs=enh_outputs)
    # 「对比试听」与「执行增强」是同一件事，区别只在于前者不改变主素材；
    # 处理后的音频总是输出到右侧两个播放器供 A/B，所以共用一个回调即可。
    compare_btn.click(on_enhance, inputs=enh_inputs, outputs=enh_outputs)

    # ---------- 入库 ----------
    @LOG.ui_guard("lab.on_save", slow_sec=2.0)
    def on_save(name, lang, tags, note, use_enhanced, audio_val, path_val, enh_val):
        key = (name or "").strip()
        if not key:
            return T.err("请先填写音色名称。")
        src = enh_val if use_enhanced else _resolve_source(audio_val, path_val)
        if not src or not os.path.isfile(src):
            return T.err(
                "没有可入库的音频。"
                + ("请先执行增强处理。" if use_enhanced else "请先上传素材。"))
        try:
            e = voice_bank.add(
                key, src, note=(note or "").strip(),
                tags=[t.strip() for t in (tags or "").split(",") if t.strip()],
                lang=lang or "ZH",
            )
        except Exception as ex:
            LOG.get_logger("lab").error("入库失败：%s ← %s",
                                        key, src, exc_info=True)
            return T.err(f"入库失败：{type(ex).__name__}: {ex}")
        LOG.get_logger("lab").info(
            "入库：%s ← %s（%.2fs · 评分 %s %s · 源=%s）",
            key, src, e.duration, e.score, e.grade,
            "增强结果" if use_enhanced else "原始素材")

        badge = {"优秀": "🟢", "良好": "🟢", "可用": "🟡",
                 "勉强": "🟠", "不建议使用": "🔴"}.get(e.grade, "·")
        html = (f'<div class="ix-tip">✅ 已存入音色库：<b>{e.name}</b> '
                f'{badge} 评分 {e.score}/100（{e.grade}）</div>')
        if e.report and e.report.get("issues"):
            html += T.warn("该素材仍有遗留问题：" + "；".join(e.report["issues"][:3]))
        html += T.hint("现在可以到「语音合成」页的「从音色库载入」下拉框里直接选它。")
        gr.Info(f"音色「{e.name}」已入库")
        return html

    # 两个按钮共用一个逻辑，只用闭包固定 use_enhanced 的取值。
    # 不在 inputs 里临时新建 gr.State —— 那会在每次绑定时产生幽灵组件。
    bank_inputs = [bank_name, bank_lang, bank_tags, bank_note,
                   src_audio, src_path, enh_audio]

    def _make_saver(use_enhanced: bool):
        def handler(name, lang, tags, note, audio_val, path_val, enh_val):
            return on_save(name, lang, tags, note, use_enhanced,
                           audio_val, path_val, enh_val)
        return handler

    save_enh_btn.click(_make_saver(True), inputs=bank_inputs, outputs=[save_out])
    save_src_btn.click(_make_saver(False), inputs=bank_inputs, outputs=[save_out])

    # ---------- 音色库管理 ----------
    @LOG.ui_guard("lab.on_bank_reload")
    def on_bank_reload():
        entries = voice_bank.list_voices()
        return (gr.update(choices=[""] + [e.name for e in entries]),
                voice_bank.table_markdown(entries))

    bank_reload.click(on_bank_reload, inputs=[], outputs=[bank_dd, bank_table])

    def on_bank_select(name):
        e = voice_bank.get(name) if name else None
        if e is None:
            return "_选择一个音色查看详情_", gr.update(visible=False)
        return voice_bank.detail_markdown(name), gr.update(value=e.audio_path, visible=True)

    bank_dd.change(on_bank_select, inputs=[bank_dd], outputs=[bank_detail, bank_preview])

    @LOG.ui_guard("lab.on_bank_delete")
    def on_bank_delete(name):
        if not name:
            gr.Warning("请先选择要删除的音色")
            return gr.update(), gr.update(), gr.update()
        ok = voice_bank.remove(name)
        if ok:
            gr.Info(f"已删除音色「{name}」")
        else:
            gr.Warning("该音色不存在或已删除")
        entries = voice_bank.list_voices()
        return (gr.update(choices=[""] + [e.name for e in entries], value=""),
                voice_bank.table_markdown(entries),
                "_选择一个音色查看详情_")

    bank_delete.click(on_bank_delete, inputs=[bank_dd],
                      outputs=[bank_dd, bank_table, bank_detail])

    def on_bank_reanalyze(name):
        if not name:
            gr.Warning("请先选择音色")
            return gr.update(), gr.update()
        e = voice_bank.reanalyze(name)
        if e is None:
            gr.Warning("重新体检失败")
            return gr.update(), gr.update()
        gr.Info(f"「{name}」重新体检完成：{e.score}/100（{e.grade}）")
        return (voice_bank.table_markdown(voice_bank.list_voices()),
                voice_bank.detail_markdown(name))

    bank_reanalyze.click(on_bank_reanalyze, inputs=[bank_dd],
                         outputs=[bank_table, bank_detail])

    @LOG.ui_guard("lab.on_import_examples", slow_sec=5.0)
    def on_import_examples():
        """把 examples/ 下的官方示例音频批量入库，方便直接对比。"""
        ex_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "examples")
        if not os.path.isdir(ex_dir):
            return T.err(f"示例目录不存在：{ex_dir}"), gr.update(), gr.update()

        added, skipped = [], []
        for fn in sorted(os.listdir(ex_dir)):
            if not fn.lower().endswith(AL.AUDIO_EXTS):
                continue
            name = "示例_" + os.path.splitext(fn)[0]
            if voice_bank.get(name):
                skipped.append(fn)
                continue
            try:
                voice_bank.add(name, os.path.join(ex_dir, fn),
                               note="官方示例音频（自动导入）",
                               tags=["官方示例"], analyze_audio=True)
                added.append(fn)
            except Exception:
                skipped.append(fn)

        entries = voice_bank.list_voices()
        msg = T.tip(f"已导入 <b>{len(added)}</b> 个官方示例音频"
                    + (f"，跳过 {len(skipped)} 个（已存在或失败）" if skipped else "")
                    + "。")
        if added:
            msg += T.hint("导入后可以直接看到它们的体检评分 —— 你会发现官方示例里"
                          "有几条其实<b>低于推荐规格</b>（比如 voice_01.wav 只有 2.44 秒）。"
                          "这不是 bug，而是官方 demo 为了体积做的取舍。")
        return (msg,
                gr.update(choices=[""] + [e.name for e in entries]),
                voice_bank.table_markdown(entries))

    bank_import_ex.click(on_import_examples, inputs=[],
                         outputs=[save_out, bank_dd, bank_table])

    return {
        "page_load": (
            lambda: (voice_bank.table_markdown(),
                     gr.update(choices=[""] + voice_bank.names())),
            [bank_table, bank_dd],
        ),
        "components": {
            "src_audio": src_audio,
            "enh_audio": enh_audio,
            "bank_dd": bank_dd,
        },
    }
