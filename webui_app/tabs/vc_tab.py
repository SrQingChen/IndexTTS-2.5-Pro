"""🎧 AI变声 Tab —— 实时麦克风变声 + 离线文件变声（同 VC 内核）。

实时模式输出走**本机声卡**（gradio 无法向浏览器流式送音频）：
监听选耳机、给别人听选 VB-Cable/VoiceMeeter 虚拟声卡。WebUI 是控制面板。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import logging_setup as LOG
from webui_app.context import AppContext
from webui_app.services import progress_hub as PH
from webui_app.services import svc_core, svc_state, voice_bank, vc_realtime as VR
from webui_app.widgets import make_component

LORA_NONE = "（不挂载）"
_VC_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "vc"))


def _lora_cfm_choices() -> List[Any]:
    out: List[Any] = [(LORA_NONE, "")]
    try:
        from webui_app.training import runs as RN
        for r in RN.list_runs():
            if not getattr(r, "has_adapter", False):
                continue
            if str(getattr(r, "arch", "")) != "cfm":
                continue
            out.append((r.name, r.name))
    except Exception:
        pass
    return out


def render(ctx: AppContext) -> Dict[str, Any]:
    st = svc_state.load_vc()
    bank_names = voice_bank.names()
    _st_voice = st.get("ref_voice") if st.get("ref_voice") in bank_names else ""
    _lora_vals = [v for _l, v in _lora_cfm_choices()]
    _st_lora = st.get("lora_run") if st.get("lora_run") in _lora_vals else ""
    _initial_ref_audio = None
    if _st_voice:
        _e = voice_bank.get(_st_voice)
        if _e is not None:
            _initial_ref_audio = _e.audio_path
    session: Dict[str, Optional[VR.RealtimeVC]] = {"rt": None}

    with gr.Column():
        gr.Markdown("## 🎧 AI 变声（原生 VC 内核，准实时/离线）")
        gr.Markdown(
            "实时模式：麦克风 → 分块转换（左上下文 + CFM + 交叉淡化）→ 本机声卡输出。"
            "输出设备选**耳机**=自己监听，选 **VB-Cable/虚拟声卡**=给 Discord/OBS 采集。"
            "仅供个人研究/本地使用。")

        with gr.Tabs():
            # ================= 实时模式 =================
            with gr.TabItem("🎙 实时变声"):
                gr.Markdown("### 音色目标")
                with gr.Row():
                    ref_bank_dd = gr.Dropdown(
                        label="音色库参考", choices=[""] + bank_names,
                        value=_st_voice,
                        info="选择后自动填入下方参考音频；也可直接上传覆盖")
                    ref_reload_btn = gr.Button("🔄 刷新音色库", size="sm")
                ref_upload = gr.Audio(
                    label="参考音频（音色库选择或直接上传）", type="filepath",
                    sources=["upload"], value=_initial_ref_audio)
                ref_info = gr.Markdown()
                with gr.Row():
                    lora_run_dd = gr.Dropdown(
                        label="CFM LoRA（音色增强，可与参考音频叠加）",
                        choices=_lora_cfm_choices(), value=_st_lora or "")
                    lora_scale_sl = gr.Slider(0.0, 1.5, step=0.05,
                                              value=float(st.get("lora_scale", 1.0)),
                                              label="LoRA 强度")
                gr.Markdown("### 设备与档位")
                _ins, _outs = VR.list_devices()
                try:
                    import sounddevice as _sd
                    _dflt_in, _dflt_out = _sd.default.device
                except Exception:
                    _dflt_in = _dflt_out = None
                with gr.Row():
                    in_dev_dd = gr.Dropdown(
                        label="输入设备（麦克风）",
                        choices=[n for n, _ in _ins],
                        value=next((n for n, i in _ins if i == _dflt_in),
                                   _ins[0][0] if _ins else None))
                    out_dev_dd = gr.Dropdown(
                        label="输出设备（监听/虚拟声卡）",
                        choices=[n for n, _ in _outs],
                        value=next((n for n, i in _outs if i == _dflt_out),
                                   _outs[0][0] if _outs else None))
                with gr.Row():
                    preset_dd = make_component(
                        "vc_preset",
                        value=(st.get("preset") if st.get("preset") in VR.PRESETS
                               else "均衡（推荐）"))
                    agc_ck = make_component("vc_agc", value=bool(st.get("agc", True)))
                    bypass_ck = make_component("vc_silence_bypass",
                                               value=bool(st.get("silence_bypass", True)))
                    rec_ck = make_component("vc_record", value=bool(st.get("record", False)))
                with gr.Row():
                    rt_start_btn = gr.Button("▶ 开始", variant="primary")
                    rt_stop_btn = gr.Button("⏹ 停止", variant="stop")
                    latency_btn = gr.Button("📏 回环延迟实测")
                rt_status_md = gr.Markdown("（未启动）")
                rec_out = gr.Audio(label="录制回放（停止后出现）", type="filepath")
                rt_timer = gr.Timer(1.0)   # 常开，轻量轮询状态行

                def _dev_index(name: str, kind: str) -> Optional[int]:
                    ins, outs = VR.list_devices()
                    lst = ins if kind == "in" else outs
                    for n, i in lst:
                        if n == name:
                            return i
                    return None

                def _fmt_status() -> str:
                    s = session["rt"]
                    if s is None:
                        return "（未启动）"
                    stt = s.status()
                    bar_in = "▉" * max(0, min(20, int((stt.level_in_db + 60) / 3)))
                    bar_out = "▉" * max(0, min(20, int((stt.level_out_db + 60) / 3)))
                    lines = [
                        f"{'🟢 运行中' if stt.running else '⚪ 已停止'}"
                        f" · 块 {stt.blocks} · 单块推理 {stt.last_infer_ms:.0f}ms"
                        f" · 欠载 {stt.underruns} · 静音占比 {stt.silent_ratio:.0%}",
                        f"入电平 `{bar_in}` {stt.level_in_db:.0f}dB",
                        f"出电平 `{bar_out}` {stt.level_out_db:.0f}dB",
                        f"算法延迟 ≈{stt.algo_latency_ms:.0f}ms（2×块+右上下文）"
                        + (f" · 回环实测 {stt.measured_latency_ms:.0f}ms"
                           if stt.measured_latency_ms > 0 else " · 回环实测未做"),
                    ]
                    if stt.error:
                        lines.append(f"⚠️ {stt.error}")
                    return "\n\n".join(lines)

                rt_timer.tick(_fmt_status, outputs=[rt_status_md])

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

                def _reload_voices():
                    return gr.update(choices=[""] + voice_bank.names())

                ref_bank_dd.change(_on_voice_select, inputs=[ref_bank_dd],
                                   outputs=[ref_upload, ref_info])
                ref_reload_btn.click(_reload_voices, outputs=[ref_bank_dd])

                @LOG.ui_guard("vc.rt_start")
                def _rt_start(in_dev, out_dev, ref_bank, ref_audio_value, lora_run, lora_scale,
                              preset, agc, bypass, rec):
                    if not VR.sd_available():
                        raise gr.Error("sounddevice 未安装：pip install sounddevice"
                                       "（或改用『文件变声』模式）")
                    if session["rt"] is not None:
                        raise gr.Error("会话已在运行，请先停止")
                    ref_path = ref_audio_value or ""
                    if not ref_path:
                        raise gr.Error("请选择音色库音色，或上传参考音频")
                    svc_state.save_vc({"preset": preset, "agc": agc,
                                       "silence_bypass": bypass, "record": rec,
                                       "ref_voice": ref_bank,
                                       "lora_run": lora_run if lora_run != LORA_NONE else "",
                                       "lora_scale": lora_scale,
                                       "in_device": _dev_index(in_dev, "in"),
                                       "out_device": _dev_index(out_dev, "out")})
                    engine = ctx.engine
                    if not engine.loaded:
                        try:
                            engine.load()
                        except Exception as exc:
                            raise gr.Error(f"引擎加载失败: {exc}")
                    # LoRA 挂载（会话期间保持）
                    if lora_run and lora_run != LORA_NONE:
                        try:
                            from webui_app.training import merge as MG
                            d, _a = MG.resolve_mount_dir(lora_run, "best")
                            engine.attach_lora(d, target="cfm")
                            if abs(float(lora_scale) - 1.0) > 1e-6:
                                MG.set_scale(engine, float(lora_scale), target="cfm")
                        except Exception as exc:
                            raise gr.Error(f"LoRA 挂载失败: {exc}")
                    rt = VR.RealtimeVC(engine, ref_path)
                    s = rt.start(_dev_index(in_dev, "in"), _dev_index(out_dev, "out"),
                                 preset, agc=agc, silence_bypass=bypass, record=rec)
                    if not s.running:
                        try:
                            engine.detach_lora(target="cfm")
                        except Exception:
                            pass
                        raise gr.Error(f"启动失败: {s.error}")
                    session["rt"] = rt
                    return _fmt_status()

                rt_start_btn.click(
                    _rt_start,
                    inputs=[in_dev_dd, out_dev_dd, ref_bank_dd, ref_upload,
                            lora_run_dd, lora_scale_sl, preset_dd, agc_ck,
                            bypass_ck, rec_ck],
                    outputs=[rt_status_md])

                def _rt_stop():
                    rt = session["rt"]
                    if rt is None:
                        return "（未启动）", None
                    path = rt.stop()
                    session["rt"] = None
                    try:
                        ctx.engine.detach_lora(target="cfm")
                    except Exception:
                        pass
                    extra = f"\n\n💾 录制已保存" if path else ""
                    return _fmt_status() + extra, path

                rt_stop_btn.click(_rt_stop, outputs=[rt_status_md, rec_out])

                @LOG.ui_guard("vc.measure")
                def _measure(in_dev, out_dev):
                    if session["rt"] is not None:
                        raise gr.Error("请先停止实时会话再测延迟")
                    r = VR.measure_roundtrip(_dev_index(in_dev, "in"),
                                             _dev_index(out_dev, "out"))
                    if not r.get("ok"):
                        raise gr.Error(r.get("error", "测量失败"))
                    return (f"📏 回环（往返）延迟 ≈ **{r['ms']}ms**（单程约一半；"
                            f"置信度 {r.get('confidence', '-')}）。要求：扬声器→麦克风"
                            f"有声音回路，或输出选 VB-Cable 且其为本页输入。")

                latency_btn.click(_measure, inputs=[in_dev_dd, out_dev_dd],
                                  outputs=[rt_status_md])

            # ================= 文件模式 =================
            with gr.TabItem("📁 文件变声（离线）"):
                gr.Markdown("输入一般已是较干净人声 → 直接转换（同翻唱核心，"
                            "语音参数组：F0 默认关、10 步、cfg 0）。")
                with gr.Row():
                    f_src = gr.Audio(label="输入音频（干净人声）", type="filepath",
                                     sources=["upload"])
                    f_ref_bank = gr.Dropdown(
                        label="音色库参考", choices=[""] + bank_names,
                        value=_st_voice,
                        info="选择后自动填入下方参考音频")
                    f_ref_reload = gr.Button("🔄 刷新音色库", size="sm")
                f_ref_up = gr.Audio(
                    label="参考音频（音色库选择或直接上传）", type="filepath",
                    sources=["upload"], value=_initial_ref_audio)
                f_ref_info = gr.Markdown()

                f_ref_bank.change(_on_voice_select, inputs=[f_ref_bank],
                                  outputs=[f_ref_up, f_ref_info])
                f_ref_reload.click(_reload_voices, outputs=[f_ref_bank])
                with gr.Row():
                    f_lora = gr.Dropdown(label="CFM LoRA", choices=_lora_cfm_choices(),
                                         value=_st_lora or "")
                    f_steps = gr.Slider(5, 25, step=1, value=10, label="CFM 步数",
                                        info="离线语音 10 步接近 25 步听感")
                    f_cfg = gr.Slider(0.0, 1.0, step=0.05, value=0.0, label="CFG",
                                      info="0 提速一倍，差异细微")
                    f_f0 = gr.Dropdown(label="语调（F0）", choices=["跟随（推荐）", "关闭"],
                                       value="跟随（推荐）",
                                       info="跟随=移植源语调趋势（smooth）")
                f_start = gr.Button("🚀 转换", variant="primary")
                f_out = gr.Audio(label="转换结果", type="filepath")
                f_md = gr.Markdown()

                @LOG.ui_guard("vc.file_convert")
                def _file_convert(src, ref_bank, ref_audio_value, lora_run, steps, cfg_rate, f0_lbl):
                    if not src:
                        raise gr.Error("请上传输入音频")
                    ref_path = ref_audio_value or ""
                    if not ref_path:
                        raise gr.Error("请选择音色库音色，或上传参考音频")
                    engine = ctx.engine
                    mounted = False
                    try:
                        engine.load()
                    except Exception as exc:
                        raise gr.Error(f"引擎加载失败: {exc}")
                    engine.begin_svc("文件变声")
                    try:
                        if lora_run and lora_run != LORA_NONE:
                            from webui_app.training import merge as MG
                            d, _a = MG.resolve_mount_dir(lora_run, "best")
                            engine.attach_lora(d, target="cfm")
                            mounted = True
                        ref = svc_core.build_reference(engine, ref_path, 8.0)
                        res = svc_core.convert_file(engine, src, ref, steps=int(steps),
                                                    cfg_rate=float(cfg_rate), chunk_sec=60.0)
                        if not res.ok:
                            raise gr.Error(f"转换失败: {res.reason}")
                        vocal = res.wav
                        notes = [f"VC {res.chunks} 块 / {res.timing.get('rtf')}×实时"]
                        if f0_lbl.startswith("跟随"):
                            from webui_app.services import cover_pipeline as CP
                            from webui_app.services import svc_f0
                            y, sr = CP.AL.load_audio(src)
                            y = CP.resample(y, sr, 22050)
                            src_f0 = svc_f0.harvest_f0(y, 22050)
                            ref_y, ref_sr = CP.AL.load_audio(ref_path)
                            ref_f0 = svc_f0.harvest_f0(
                                CP.resample(ref_y, ref_sr, 22050), 22050)
                            semis = svc_f0.octave_shift_semis(src_f0, ref_f0)
                            mapped = svc_f0.smooth_ratio_curve(
                                svc_f0.shift_f0(src_f0, semis), 41, 21)
                            r2, met = svc_f0.transplant_f0(vocal, 22050, mapped,
                                                            out_f0=None, mode="force")
                            if r2.ok:
                                vocal = r2.y
                                notes.append(f"语调移植 {semis:+d}st（smooth）")
                            else:
                                notes.append(f"语调移植回退: {r2.reason}")
                        os.makedirs(_VC_ROOT, exist_ok=True)
                        out_path = os.path.join(_VC_ROOT, time.strftime(
                            "vcfile-%Y%m%d-%H%M%S.wav"))
                        from webui_app.services import audio_lab as AL
                        AL.save_audio(out_path, vocal, 22050)
                        notes.append(f"`{out_path}`")
                        return out_path, "✅ " + " · ".join(notes)
                    finally:
                        if mounted:
                            try:
                                engine.detach_lora(target="cfm")
                            except Exception:
                                pass
                        engine.end_svc("文件变声")

                f_start.click(_file_convert,
                              inputs=[f_src, f_ref_bank, f_ref_up, f_lora, f_steps,
                                      f_cfg, f_f0],
                              outputs=[f_out, f_md])

        if not VR.sd_available():
            gr.Markdown("ℹ️ 实时模式需要 `pip install sounddevice`；未安装时『文件变声』仍可用。")

    return {}
