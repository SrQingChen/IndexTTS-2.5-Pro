"""🎤 AI翻唱 Tab —— 源歌 → 角色音色（原生 IndexTTS2.5 VC 管线）。

链路（docs/dev-plan-ai-cover-vc-2026-10.md）：
    输入状态路由 → 分离/去混响/预处理 → svc_core 转换（引擎+CFM LoRA）
    → F0 旋律移植 → audio_lab 后处理 → 可选超分 → 可选自动合成 → 报告
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import logging_setup as LOG
from webui_app.context import AppContext
from webui_app.services import cover_pipeline as CP
from webui_app.services import progress_hub as PH
from webui_app.services import separation as SEP
from webui_app.services import svc_state, voice_bank
from webui_app.services.cover_pipeline import CoverRequest
from webui_app.widgets import make_component

LORA_NONE = "（不挂载）"
_STATE_TO_KEY = {"原歌曲（含伴奏）": "song",
                 "分离人声（可能有混响/残留）": "separated",
                 "干净人声（干声）": "clean"}
_KEY_TO_STATE = {v: k for k, v in _STATE_TO_KEY.items()}
_VARIANT_TO_KEY = {"A 量化重构（默认）": "A", "B 连续特征直通（实验）": "B"}
_F0_TO_KEY = {"auto 自动（推荐）": "auto", "force 全曲移植": "force", "off 关闭": "off"}
_F0_TO_LABEL = {v: k for k, v in _F0_TO_KEY.items()}


def _lora_cfm_choices() -> List[Any]:
    out: List[Any] = [(LORA_NONE, "")]
    try:
        from webui_app.training import runs as RN
        for r in RN.list_runs():
            if not getattr(r, "has_adapter", False):
                continue
            if str(getattr(r, "arch", "")) != "cfm":
                continue
            bv = (f"val {r.best_val:.4f}"
                  if getattr(r, "best_val", None) is not None else "val —")
            out.append((f"{r.name} · {bv}", r.name))
    except Exception:
        pass
    return out


def render(ctx: AppContext) -> Dict[str, Any]:
    cfg = ctx.cfg
    st = svc_state.load_cover()
    bank_names = voice_bank.names()
    _st_voice = st.get("ref_voice") if st.get("ref_voice") in bank_names else ""
    _lora_vals = [v for _l, v in _lora_cfm_choices()]
    _st_lora = st.get("lora_run") if st.get("lora_run") in _lora_vals else ""
    # 音色库选中项 → 参考音频初值（与合成页一致：选择即填入音频组件）
    _initial_ref_audio = None
    if _st_voice:
        _e = voice_bank.get(_st_voice)
        if _e is not None:
            _initial_ref_audio = _e.audio_path

    with gr.Column():
        gr.Markdown("## 🎤 AI 翻唱（原生管线：IndexTTS-2.5 + CFM LoRA）")
        gr.Markdown(
            "源歌人声 → 分离/去混响 → **内容/时长保持、音色替换**（引擎本体推理）"
            "→ F0 旋律移植 → 后处理 → 可选超分/BGM 回混。"
            "仅供个人研究/本地使用。")
        sb = ctx.component("statusbar")

        log_sink: deque = deque(maxlen=150)
        ctx.shared["cover_log"] = log_sink

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ① 输入")
                src_audio = gr.Audio(
                    label="源音频（歌曲/人声；原歌曲=自动分离，干净人声=直通）",
                    type="filepath",
                    sources=["upload"])
                state_dd = make_component("cover_input_state",
                                          value=_KEY_TO_STATE.get(st.get("input_state", "clean"),
                                                                  "干净人声（干声）"))
                sep_model_dd = gr.Dropdown(
                    label="人声分离模型", choices=list(SEP.VOCALS_MODELS.keys()),
                    value=st.get("separate_model") or list(SEP.VOCALS_MODELS.keys())[0],
                    info="BS-Roformer ViperX 社区默认；缺失时首次推理自动经镜像下载")
                with gr.Row():
                    sep_dl_btn = gr.Button("⬇ 预下载/补全分离模型（镜像）", size="sm")
                    sep_list_btn = gr.Button("🔄 模型状态", size="sm")
                dereverb_ck = make_component("cover_dereverb", value=bool(st.get("dereverb", True)))
                denoise_sl = make_component("cover_denoise", value=float(st.get("denoise", 0.0)))
                sep_status = gr.Markdown()

            with gr.Column(scale=1):
                gr.Markdown("### ② 音色目标")
                ref_voice_dd = gr.Dropdown(
                    label="音色库参考", choices=[""] + bank_names,
                    value=_st_voice,
                    info="选择后自动填入下方参考音频；也可直接上传覆盖")
                ref_reload_btn = gr.Button("🔄 刷新音色库", size="sm")
                ref_audio = gr.Audio(
                    label="参考音频（音色库选择或直接上传）", type="filepath",
                    sources=["upload"], value=_initial_ref_audio)
                ref_info = gr.Markdown()
                lora_run_dd = gr.Dropdown(
                    label="CFM LoRA（音色增强，可与参考音频叠加）",
                    choices=_lora_cfm_choices(),
                    value=_st_lora or "",
                    info="参考音频 + CFM LoRA 同时生效；GPT(语气) LoRA 在转换路径不生效")
                lora_scale_sl = gr.Slider(
                    minimum=0.0, maximum=1.5, step=0.05,
                    value=float(st.get("lora_scale", 1.0)), label="LoRA 强度")

        gr.Markdown("### ③ 转换核心参数")
        with gr.Row():
            variant_dd = make_component("cover_variant",
                                        value=("B 连续特征直通（实验）" if st.get("variant") == "B"
                                               else "A 量化重构（默认）"))
            steps_sl = make_component("cover_steps", value=int(st.get("steps", 25)))
            cfg_sl = make_component("cover_cfg", value=float(st.get("cfg", 0.7)))
        with gr.Row():
            ref_sec_sl = make_component("cover_ref_seconds", value=float(st.get("ref_seconds", 15.0)))
            chunk_sl = make_component("cover_chunk_sec", value=float(st.get("chunk_sec", 30.0)))
            f0_mode_dd = make_component("cover_f0_mode",
                                        value=_F0_TO_LABEL.get(st.get("f0_mode", "auto"),
                                                               "auto 自动（推荐）"))
        with gr.Row():
            tr_auto_ck = make_component("cover_transpose_auto", value=bool(st.get("transpose_auto", True)))
            tr_semis_sl = make_component("cover_transpose_semis", value=int(st.get("transpose_semis", 0)))

        gr.Markdown("### ④ 后处理 / 超分 / 混音")
        with gr.Row():
            deess_ck = make_component("cover_hf_deess", value=bool(st.get("hf_deess", True)))
            hflim_ck = make_component("cover_hf_limit", value=bool(st.get("hf_limit", True)))
            spec_ck = make_component("cover_spectral_match", value=bool(st.get("spectral_match", False)))
            breath_ck = make_component("cover_breath_soften", value=bool(st.get("breath_soften", False)))
        with gr.Row():
            hp_sl = make_component("cover_highpass", value=int(st.get("highpass", 70)))
            sr_ck = make_component("cover_sr_enable", value=bool(st.get("sr_enable", False)))
            sr_steps_sl = make_component("cover_sr_steps", value=int(st.get("sr_steps", 35)))
        with gr.Row():
            mix_ck = make_component("cover_auto_mix", value=bool(st.get("auto_mix", False)))
            vgain_sl = make_component("cover_vocal_gain", value=float(st.get("vocal_gain", 3.0)))
            ln_ck = make_component("cover_loudnorm", value=bool(st.get("loudnorm", False)))
        bgm_upload = gr.Audio(
            label="伴奏轨上传（输入状态≠原歌曲 且 勾选自动合成时使用）",
            type="filepath", sources=["upload"], visible=False)

        gr.Markdown("### ⑤ 运行")
        with gr.Row():
            start_btn = gr.Button("🚀 开始翻唱", variant="primary")
            stop_btn = gr.Button("⏹ 停止", variant="stop")
        log_box = gr.Textbox(label="过程日志", lines=10, max_lines=18, interactive=False)
        timer = gr.Timer(1.0)   # 常开（synthesize 日志窗同款），空闲时显示占位

        gr.Markdown("### ⑥ 结果")
        result_md = gr.Markdown()
        result_audio = gr.Audio(label="转换人声（成品试听）", type="filepath")
        mix_audio = gr.Audio(label="自动合成成品（勾选时）", type="filepath")
        result_files = gr.Files(label="产物文件（干声/BGM/成品/报告）")

        # ---------------- 回调 ----------------
        def _models_line() -> str:
            try:
                rows = SEP.installed_models()
                ready = sum(1 for r in rows if r["ready"])
                missing = [r["file"] for r in rows if not r["ready"]]
                line = f"分离/去混响模型：**{ready}/{len(rows)} 就绪**"
                if missing:
                    line += f"（缺失 {len(missing)} 个，首次推理自动镜像下载）"
                return line
            except Exception as exc:
                return f"模型状态读取失败: {exc}"

        def _sep_hint(state_label: str) -> str:
            if not SEP.available() and _STATE_TO_KEY.get(state_label) == "song":
                return ("⚠️ audio-separator 未安装；请改选『干净人声』或先装依赖：\n"
                        "`uv pip install \"audio-separator[gpu]\" "
                        "--default-index https://pypi.tuna.tsinghua.edu.cn/simple`")
            head = _models_line()
            if _STATE_TO_KEY.get(state_label) == "song":
                return (head + "\n\n链路：**人声分离 → 去混响 → 预处理 → VC → F0 → 后处理**"
                        "（分离约 1-3 分钟/首；权重缺失会自动下载，可用左侧按钮手动预下载）")
            if _STATE_TO_KEY.get(state_label) == "separated":
                return head + "\n\n链路：**去混响 → 预处理 → VC → F0 → 后处理**"
            return head + "\n\n链路：**预处理 → VC → F0 → 后处理**（直通，最快）"

        state_dd.change(_sep_hint, inputs=[state_dd], outputs=[sep_status])

        # ---------- 音色库联动（与合成页同款：选音色自动填入参考音频）----------
        def _on_voice_select(name):
            if not name:
                return gr.update(), ""
            e = voice_bank.get(name)
            if e is None:
                return gr.update(), f"音色 `{name}` 不存在"
            badge = {"优秀": "🟢", "良好": "🟢", "可用": "🟡",
                     "勉强": "🟠", "不建议使用": "🔴"}.get(e.grade, "·")
            html = (f"{badge} **{e.name}** · 评分 {e.score}/100（{e.grade}）"
                    f" · {e.duration:.2f}s · {e.sample_rate}Hz"
                    f" · SNR {e.snr_db:.1f}dB")
            return gr.update(value=e.audio_path), html

        ref_voice_dd.change(_on_voice_select, inputs=[ref_voice_dd],
                            outputs=[ref_audio, ref_info])

        def _reload_voices():
            return gr.update(choices=[""] + voice_bank.names())

        ref_reload_btn.click(_reload_voices, outputs=[ref_voice_dd])

        @LOG.ui_guard("cover.sep_status")
        def _sep_status_click(state_label):
            return _sep_hint(state_label)

        sep_list_btn.click(_sep_status_click, inputs=[state_dd], outputs=[sep_status])

        @LOG.ui_guard("cover.sep_prefetch")
        def _sep_prefetch(model_label, progress=gr.Progress()):
            name = SEP.VOCALS_MODELS.get(model_label, SEP.DEFAULT_VOCALS)
            log_sink.append(f"[模型] 检查 {name}（缺失则经镜像下载，支持续传）")
            r = SEP.ensure_model(
                name, lambda f, m: progress(min(0.98, 0.02 + 0.96 * f), desc=m))
            if not r.get("ok"):
                raise gr.Error(r.get("error", "下载失败"))
            log_sink.append(f"[模型] {name} 就绪")
            return _models_line()

        sep_dl_btn.click(_sep_prefetch, inputs=[sep_model_dd], outputs=[sep_status])

        mix_ck.change(lambda on: gr.update(visible=bool(on)),
                      inputs=[mix_ck], outputs=[bgm_upload])

        def _tick() -> str:
            lines = list(log_sink)
            return "\n".join(lines[-40:]) if lines else "（待运行）"

        timer.tick(_tick, outputs=[log_box])

        def _stop():
            ctx.shared["cover_stop"] = True
            return "停止请求已发出（当前块完成后中断）"

        stop_btn.click(_stop, outputs=[log_box])

        all_inputs = [src_audio, state_dd, sep_model_dd, dereverb_ck, denoise_sl,
                      ref_voice_dd, ref_audio, lora_run_dd, lora_scale_sl,
                      variant_dd, steps_sl, cfg_sl, ref_sec_sl, chunk_sl,
                      f0_mode_dd, tr_auto_ck, tr_semis_sl,
                      deess_ck, hflim_ck, spec_ck, breath_ck, hp_sl,
                      sr_ck, sr_steps_sl, mix_ck, bgm_upload, vgain_sl, ln_ck]
        assert len(all_inputs) == 28, f"cover all_inputs 接线数错误: {len(all_inputs)}"

        @LOG.ui_guard("cover.start")
        def _start(progress=gr.Progress(), *vals):
            (src, state_lbl, sep_model, dereverb, denoise,
             ref_bank, ref_audio_value, lora_run, lora_scale,
             variant_lbl, steps, cfg_rate, ref_seconds, chunk_sec,
             f0_lbl, tr_auto, tr_semis,
             deess, hflim, spec, breath, hp,
             sr_on, sr_steps, mix_on, bgm_up, vgain, loudnorm) = vals

            if not src:
                raise gr.Error("请先上传源音频")
            ref_path = ref_audio_value or ""
            if not ref_path:
                raise gr.Error("请选择音色库音色，或上传参考音频")

            lora_dir: Optional[str] = None
            if lora_run and lora_run != LORA_NONE:
                try:
                    from webui_app.training import merge as MG
                    lora_dir, _arch = MG.resolve_mount_dir(lora_run, "best")
                except Exception as exc:
                    raise gr.Error(f"LoRA 解析失败: {exc}")

            # 参数记忆
            svc_state.save_cover({
                "input_state": _STATE_TO_KEY.get(state_lbl, "clean"),
                "separate_model": sep_model, "dereverb": dereverb,
                "denoise": denoise, "variant": _VARIANT_TO_KEY.get(variant_lbl, "A"),
                "steps": steps, "cfg": cfg_rate, "ref_seconds": ref_seconds,
                "chunk_sec": chunk_sec, "f0_mode": _F0_TO_KEY.get(f0_lbl, "auto"),
                "transpose_auto": tr_auto, "transpose_semis": tr_semis,
                "hf_deess": deess, "hf_limit": hflim, "spectral_match": spec,
                "breath_soften": breath, "highpass": hp, "sr_enable": sr_on,
                "sr_steps": sr_steps, "auto_mix": mix_on, "vocal_gain": vgain,
                "loudnorm": loudnorm, "lora_run": lora_run if lora_run != LORA_NONE else "",
                "lora_scale": lora_scale, "ref_voice": ref_bank,
            })

            req = CoverRequest(
                src_path=src,
                input_state=_STATE_TO_KEY.get(state_lbl, "clean"),
                ref_path=ref_path,
                lora_cfm_dir=lora_dir,
                lora_cfm_scale=float(lora_scale),
                variant=_VARIANT_TO_KEY.get(variant_lbl, "A"),
                steps=int(steps), cfg_rate=float(cfg_rate),
                ref_seconds=float(ref_seconds), chunk_sec=float(chunk_sec),
                separate_model=sep_model,
                enable_dereverb=bool(dereverb),
                denoise_strength=float(denoise),
                f0_mode=_F0_TO_KEY.get(f0_lbl, "auto"),
                transpose_manual=None if tr_auto else int(tr_semis),
                hf_deess=bool(deess), hf_limit=bool(hflim),
                spectral_match=bool(spec), breath_soften=bool(breath),
                highpass_hz=int(hp),
                sr_enable=bool(sr_on), sr_steps=int(sr_steps),
                auto_mix=bool(mix_on),
                mix_bgm_path=bgm_up if mix_on else None,
                vocal_gain_db=float(vgain), loudness_normalize=bool(loudnorm),
            )

            log_sink.clear()
            ctx.shared["cover_stop"] = False

            def _should_stop():
                return bool(ctx.shared.get("cover_stop"))

            lp = PH.LoggedProgress(LOG.get_logger("cover"), user_progress=progress,
                                   heartbeat_idle=15, period=5, sink=log_sink)
            lp.start()
            try:
                report = CP.run_cover(ctx, req, progress=lp, should_stop=_should_stop,
                                      log=lambda m: log_sink.append(m))
            finally:
                lp.stop()

            if not report.get("ok"):
                LOG.get_logger("cover").warning("翻唱失败：%s",
                                                report.get("error", "未知错误"))
                return (f"❌ 失败：{report.get('error', '未知错误')}",
                        None, None, [])
            return _render_report(report)

        def _render_report(report: Dict[str, Any]) -> tuple:
            q = report.get("quality", {})
            vc = report.get("vc", {})
            f0 = report.get("f0", {})
            tm = report.get("timings", {})
            lines = ["## ✅ 翻唱完成", ""]
            lines.append(f"- 产物目录：`{report.get('out_dir')}`")
            if vc:
                extra = ""
                if vc.get("retried_smaller_chunk"):
                    extra = f"（显存不足已自动降到 {vc.get('chunk_sec_used')}s/块）"
                lines.append(f"- VC：{vc.get('chunks')} 块 · "
                             f"{vc.get('chunk_sec_used', '-')}s/块 · "
                             f"{vc.get('rtf')}×实时{extra}")
            if f0:
                lines.append(f"- F0：模式 {f0.get('mode_used', '-')}，轮廓相关 "
                             f"r={f0.get('f0_r', '-')}，八度错误 {f0.get('octave_err', '-')}，"
                             f"移调 {f0.get('transpose_semis', 0):+d}st")
            if "ss" in q:
                lines.append(f"- 音色相似度 SS = **{q['ss']}**（参考线 ≥0.75）")
            if "f0_r" in q:
                extra = (f"，50 音分内 {q['within_50c']:.0%}"
                         if q.get("within_50c") is not None else "")
                lines.append(f"- F0 轮廓相关 = **{q['f0_r']}**"
                             f"（移植前 {q.get('f0_r_pre', '-')}{extra}；参考线 ≥0.85）")
            an = q.get("analyze", {})
            if an:
                lines.append(f"- 体检：{an.get('grade', '-')} / SNR {an.get('snr_db', 0):.1f}dB"
                             f" / 削波 {an.get('clip_ratio', 0):.4f}")
                if q.get("analyze", {}).get("issues"):
                    lines.append(f"- ⚠️ {', '.join(an['issues'][:3])}")
            if tm:
                lines.append(f"- 阶段耗时：{tm}")
            files = [os.path.join(report["out_dir"], n) for n in report.get("files", {})]
            final = report.get("final_vocal")
            mix = (report.get("mix", {}) or {}).get("path")
            return ("\n".join(lines), final or None, mix or None,
                    [f for f in files if os.path.isfile(f)])

        start_btn.click(_start, inputs=all_inputs,
                        outputs=[result_md, result_audio, mix_audio, result_files])

    return {"page_load": (lambda: _sep_hint("干净人声（干声）"), [sep_status])}
