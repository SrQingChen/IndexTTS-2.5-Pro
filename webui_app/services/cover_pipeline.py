"""cover_pipeline — AI 翻唱链路编排（预处理→分离路由→VC→F0→后处理→超分→混音→报告）。

架构（dev-plan §1.1）：
    输入状态路由（原歌曲→分离 / 分离人声→去混响 / 干净人声→直通）
    → 翻唱专用预处理（去DC/边缘静音/轻降噪/RMS保护，不动段内停顿）
    → svc_core.convert_file（引擎 + CFM LoRA）
    → svc_f0.transplant_f0（旋律移植）
    → audio_lab 后处理（HF治理/频谱对齐/气声软化，翻唱标定）
    → 可选 AudioSR 超分（默认关）
    → 可选自动合成（分离产出 BGM + 转换人声混音）
    → 质量报告（campplus SS / F0 相关 / 八度错误 / AL.analyze 体检）

显存编排（D7）：全链 try/finally 恢复进入前引擎状态；分离与超分段
临时卸载引擎（audio_sr 模式）；LoRA 结束恢复进入前挂载。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from webui_app.services import audio_lab as AL
from webui_app.services import svc_core, svc_f0

ENGINE_SR = svc_core.ENGINE_SR
COVER_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "cover"))

INPUT_STATES = {
    "song": "原歌曲（含伴奏）",
    "separated": "分离人声（可能有混响/残留）",
    "clean": "干净人声（干声）",
}


@dataclass
class CoverRequest:
    src_path: str
    input_state: str = "clean"          # song / separated / clean
    ref_path: str = ""
    # LoRA（cfm=音色；gpt 在 VC 路径不生效——旁路 GPT）
    lora_cfm_dir: Optional[str] = None
    lora_cfm_scale: float = 1.0
    # VC 转换
    variant: str = "A"                   # A=量化重构(默认) / B=连续直通
    steps: int = 25
    cfg_rate: float = 0.7
    ref_seconds: float = 15.0
    chunk_sec: float = 30.0
    # 分离/去混响
    separate_model: str = "标准（BS-Roformer ViperX，SDR 12.98）"
    enable_dereverb: bool = True
    dereverb_model: str = "标准（UVR-DeEcho-DeReverb）"
    # 预处理
    denoise_strength: float = 0.0        # 0=关
    # F0
    f0_mode: str = "auto"                # off / auto / force
    transpose_manual: Optional[int] = None
    # 音域适配（B 方案；默认 off。非纯增益：可能改调或引入二次移调伪影）
    key_mode: str = "off"                # off / auto / manual
    key_semitones: int = 0               # manual: -12..0
    key_output: str = "lower_key"        # lower_key / original_key
    key_ceiling_hz: float = 600.0
    # 后处理
    hf_deess: bool = True
    hf_limit: bool = True
    spectral_match: bool = False         # 频谱画像对齐（实验）
    breath_soften: bool = False
    highpass_hz: int = 70
    # 超分
    sr_enable: bool = False
    sr_steps: int = 35
    # 混音
    auto_mix: bool = False
    mix_bgm_path: Optional[str] = None   # 状态≠song 时的伴奏上传
    vocal_gain_db: float = 3.0
    loudness_normalize: bool = False


# ---------------------------------------------------------------------------
# 预处理（翻唱专用：不动段内停顿——演唱节奏红线）
# ---------------------------------------------------------------------------

def preprocess_vocal(y: np.ndarray, sr: int,
                     denoise_strength: float = 0.0,
                     trim_edges: bool = True) -> Tuple[np.ndarray, int]:
    """去DC → 边缘静音保守裁剪（-45dB，仅首尾）→ 可选轻降噪 → 增益保护。

    返回 (y, head_cut_samples)：head_cut 是**头部被裁样本数**——自动合成
    （BGM 回混）必须把它补回人声头部，否则有前奏的歌会整体提前进拍
    （对拍红线，交付验收#4）。段内停顿（气口）一律不动。
    """
    y = np.asarray(y, dtype=np.float32)
    dc = float(np.mean(y))
    if abs(dc) > 0.001:
        y = y - dc

    head_cut = 0
    if trim_edges:
        def _edge_len(y_in: np.ndarray, head: bool) -> int:
            thresh = 10 ** (-45.0 / 20.0)
            if head:
                idx = int(np.argmax(np.abs(y_in) > thresh))
                return max(0, idx - int(0.05 * sr))
            idx = len(y_in) - int(np.argmax(np.abs(y_in[::-1]) > thresh))
            return max(0, len(y_in) - min(len(y_in), idx + int(0.05 * sr)))

        head_cut = _edge_len(y, True)
        y = y[head_cut:]
        y = y[:len(y) - _edge_len(y, False)]

    if denoise_strength > 0.01:
        try:
            import noisereduce as nr
            y = nr.reduce_noise(y=y, sr=sr,
                                prop_decrease=float(np.clip(denoise_strength, 0.05, 1.0)))
            y = np.asarray(y, dtype=np.float32)
        except Exception:
            pass

    # 增益保护：峰值 >0.891(-1dB) 压回；整体 RMS < -35dB 提到 -24dB（弱信号抬升）
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.891:
        y = y * (0.891 / peak)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    if rms < 10 ** (-35.0 / 20.0) and rms > 1e-6:
        gain = min(10 ** (-24.0 / 20.0) / rms, 8.0)
        y = np.clip(y * gain, -0.98, 0.98)
    return y, head_cut


def highpass(y: np.ndarray, sr: int, hz: float = 70.0) -> np.ndarray:
    if hz <= 0:
        return y
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, hz, btype="highpass", fs=sr, output="sos")
        return np.asarray(sosfiltfilt(sos, y), dtype=np.float32)
    except Exception:
        return y


def shift_vocal_key(y: np.ndarray, sr: int, semis: int):
    """源人声预移调（B 方案）：优先 WORLD（保共振峰），失败回退相位声码器。"""
    if semis == 0:
        return y, "none"
    f0 = svc_f0.harvest_f0(y, sr)
    res = svc_f0.world_replace_f0(y, sr, svc_f0.shift_f0(f0, semis))
    if res.ok:
        return res.y, "world"
    import librosa
    return (np.asarray(librosa.effects.pitch_shift(y, sr=sr, n_steps=semis),
                       dtype=np.float32), "librosa")


def shift_audio_librosa(y: np.ndarray, sr: int, semis: int) -> np.ndarray:
    """伴奏/后移调：相位声码器（时长保持）。大移调会有伪影，报告里会提示。"""
    if semis == 0:
        return y
    import librosa
    return np.asarray(librosa.effects.pitch_shift(y, sr=sr, n_steps=semis),
                      dtype=np.float32)


def resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    import librosa
    return np.asarray(librosa.resample(np.asarray(y, dtype=np.float32),
                                       orig_sr=sr, target_sr=target), dtype=np.float32)


# ---------------------------------------------------------------------------
# 混音
# ---------------------------------------------------------------------------

def mixdown(vocal: np.ndarray, vocal_sr: int, bgm: np.ndarray, bgm_sr: int,
            vocal_gain_db: float = 3.0, normalize_rms: bool = False,
            head_pad_samples: int = 0) -> Dict:
    """人声贴回伴奏：统一采样率 → 头部偏移补回（对拍红线）→ 时长对齐
    （pad/trim，不做 time-stretch）→ 人声增益 → 混合 → 峰值保护（-1dBTP）。

    head_pad_samples：预处理裁掉的前奏静音样本数（vocal 采样率域），
    自动合成时补回，保证与同源伴奏时间轴一致。
    """
    out_sr = max(vocal_sr, bgm_sr, 44100)
    v = resample(vocal, vocal_sr, out_sr)
    b = resample(bgm, bgm_sr, out_sr)
    if head_pad_samples > 0:
        pad_out = int(round(head_pad_samples * out_sr / vocal_sr))
        v = np.pad(v, (pad_out, 0))
    n = max(len(v), len(b))
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    v = v * (10 ** (float(vocal_gain_db) / 20.0))
    mix = v + b
    peak = float(np.max(np.abs(mix)))
    if peak > 0.891:
        mix = mix * (0.891 / peak)
    if normalize_rms:
        rms = float(np.sqrt(np.mean(mix ** 2)) + 1e-9)
        target = 10 ** (-14.0 / 20.0)
        if rms > 1e-5:
            mix = np.clip(mix * min(target / rms, 2.0), -0.98, 0.98)
            peak = float(np.max(np.abs(mix)))
            if peak > 0.98:
                mix = mix * (0.98 / peak)
    return {"wav": mix.astype(np.float32), "sr": out_sr,
            "seconds": round(n / out_sr, 2)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

class _Stop(Exception):
    pass


class _Fail(Exception):
    """可预期的失败：走统一出口（finally 恢复引擎 + 落 report.json）。"""


def _check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise _Stop()


def run_cover(ctx, req: CoverRequest, progress: Optional[Callable] = None,
              should_stop: Optional[Callable[[], bool]] = None,
              log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """执行翻唱全链。返回报告 dict（同时落盘 report.json）。

    ctx: AppContext（取 engine）。
    """
    def _p(frac, msg):
        if progress is not None:
            try:
                progress(frac, msg)
            except Exception:
                pass

    def _log(msg):
        if log is not None:
            log(msg)

    if not os.path.isfile(req.src_path):
        return {"ok": False, "error": f"源音频不存在: {req.src_path}"}
    if not req.ref_path or not os.path.isfile(req.ref_path):
        return {"ok": False, "error": "缺少音色参考音频"}

    os.makedirs(COVER_ROOT, exist_ok=True)
    out_dir = os.path.join(COVER_ROOT, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    report: Dict[str, Any] = {"ok": False, "out_dir": out_dir, "request": asdict(req)}
    timings: Dict[str, float] = {}

    engine = ctx.engine
    was_loaded = bool(getattr(engine.stats, "loaded", False))
    mounted_by_us = False
    # 引擎租约：与实时变声/其他长会话互斥（并发共享引擎会显存叠加）。
    # 内部卸载（分离/SR 段）用 bypass_lease=自己的 token，只绕自己的租约。
    svc_lease = engine.begin_svc("AI翻唱")
    gpt_offloaded = False
    try:
        # ---------- 1) 输入路由：分离 / 去混响 ----------
        t0 = time.perf_counter()
        src_y, src_sr = AL.load_audio(req.src_path)      # 原采样率
        report["src"] = {"sr": src_sr, "seconds": round(len(src_y) / src_sr, 2),
                         "state": req.input_state}
        vocal_path = os.path.join(out_dir, "01_source.wav")
        bgm_path: Optional[str] = None

        if req.input_state == "song":
            from webui_app.services import separation as SEP
            if not SEP.available():
                raise _Fail("audio-separator 未安装：pip install "
                            "\"audio-separator[gpu]\"，或在输入状态选择"
                            "『干净人声』绕过分离")
            _log("[分离] 卸载引擎 → 人声分离（不改时长，保对拍）")
            if was_loaded or bool(getattr(engine.stats, "loaded", False)):
                engine.unload(bypass_lease=svc_lease)
            _p(0.02, "人声分离中（Roformer，首跑需下载权重）")
            sep_res = SEP.separate_vocals(req.src_path, out_dir, req.separate_model,
                                          progress=_p, should_stop=should_stop)
            SEP.release_all()
            if sep_res.get("cancelled"):
                raise _Stop()
            _check_stop(should_stop)
            if not sep_res.get("ok"):
                raise _Fail(f"分离失败: {sep_res.get('error')}")
            vocal_path = sep_res["vocals"]
            bgm_path = sep_res.get("instrumental")
            _log(f"[分离] 完成 {sep_res['seconds']}s → {os.path.basename(vocal_path)}")
        else:
            # separated / clean：源文件即人声（clean 曾因漏赋值走虚构路径必炸——验收 P0）
            vocal_path = req.src_path

        if req.input_state == "separated" and req.enable_dereverb:
            from webui_app.services import separation as SEP
            if SEP.available():
                if bool(getattr(engine.stats, "loaded", False)):
                    engine.unload(bypass_lease=svc_lease)
                _p(0.10, "去混响中（DeEcho-DeReverb）")
                dr = SEP.dereverb(vocal_path, out_dir, req.dereverb_model,
                                   progress=_p, should_stop=should_stop)
                SEP.release_all()
                if dr.get("cancelled"):
                    raise _Stop()
                _check_stop(should_stop)
                if dr.get("ok"):
                    vocal_path = dr["dry"]
                    _log("[去混响] 完成")
                else:
                    _log(f"[去混响] 跳过: {dr.get('error')}")
        timings["separation"] = round(time.perf_counter() - t0, 1)

        # ---------- 2) 预处理（22k 工作域） ----------
        t0 = time.perf_counter()
        _p(0.18, "预处理（去DC/边缘静音/增益保护）")
        y, sr = AL.load_audio(vocal_path)
        y = resample(y, sr, ENGINE_SR)
        # 自动合成时头部偏移必须补回（对拍）；不做回混则裁头无害
        y, head_cut = preprocess_vocal(y, ENGINE_SR, req.denoise_strength,
                                       trim_edges=True)
        report["head_cut_22k"] = int(head_cut)
        # 音域适配（B）：先分析源 F0，必要时把源人声整体降调
        key_info = {"mode": req.key_mode, "output": req.key_output}
        key_semis = 0
        src_f0_key = None
        if req.key_mode != "off":
            src_f0_key = svc_f0.harvest_f0(y, ENGINE_SR)
            vv = src_f0_key[src_f0_key > 0]
            key_info["p99_before"] = (round(float(np.percentile(vv, 99)), 1)
                                      if len(vv) else 0.0)
            key_semis = svc_f0.plan_key_shift(src_f0_key, req.key_mode,
                                              req.key_semitones,
                                              req.key_ceiling_hz)
            key_info["semitones"] = key_semis
            if key_semis != 0:
                _p(0.20, f"音域适配：源人声降 {-key_semis} 半音")
                y, keng = shift_vocal_key(y, ENGINE_SR, key_semis)
                key_info["engine"] = keng
                if keng == "librosa":
                    key_info["warn"] = "WORLD 预降调失败，已回退相位声码器"
                _log(f"[音域] 预降 {key_semis}st（{keng}）")
            else:
                _log("[音域] 源音域在舒适上限内，不降调")
        report["key"] = key_info
        dry_path = os.path.join(out_dir, "02_dry_vocal_22k.wav")
        AL.save_audio(dry_path, y, ENGINE_SR)
        timings["preprocess"] = round(time.perf_counter() - t0, 1)
        _check_stop(should_stop)

        # ---------- 3) 引擎 + LoRA + 参考 ----------
        t0 = time.perf_counter()
        _p(0.22, "加载推理引擎")
        engine.load(bypass_lease=svc_lease)
        # VC 绕过 GPT：低显存卡把它移到 CPU，让出 ~1.5GB 给分块推理
        gpt_offloaded = engine.offload_gpt_for_svc()
        # LoRA 挂载（替换语义：已挂先卸——attach_lora 内建）。结束时卸掉
        # 我们挂的；进入前若有别的挂载，报告中提示需重挂。
        pre_mounted = any(it.get("target") == "cfm" and it.get("wrapped")
                          for it in engine.lora_status()) \
            if isinstance(engine.lora_status(), list) else False
        mounted_by_us = False
        if req.lora_cfm_dir:
            try:
                tag = engine.attach_lora(req.lora_cfm_dir, target="cfm")
                mounted_by_us = True
                if abs(req.lora_cfm_scale - 1.0) > 1e-6:
                    from webui_app.training import merge as MG
                    MG.set_scale(engine, req.lora_cfm_scale, target="cfm")
                _log(f"[LoRA] CFM 音色挂载: {tag}")
            except Exception as exc:
                _log(f"[LoRA] 挂载失败（继续用纯参考音色）: {exc}")

        ref = svc_core.build_reference(engine, req.ref_path, req.ref_seconds)
        _p(0.25, "参考特征构建完成")

        # ---------- 4) VC 转换 ----------
        conv = svc_core.convert_file(engine, dry_path, ref,
                                     variant=req.variant, steps=req.steps,
                                     cfg_rate=req.cfg_rate, chunk_sec=req.chunk_sec,
                                     progress=lambda f, m: _p(0.25 + 0.45 * f, m),
                                     should_stop=should_stop)
        if not conv.ok:
            if conv.reason == "用户停止":
                raise _Stop()
            raise _Fail(f"VC 转换失败: {conv.reason}")
        vc_path = os.path.join(out_dir, "03_converted_22k.wav")
        AL.save_audio(vc_path, conv.wav, ENGINE_SR)
        report["vc"] = {"chunks": conv.chunks,
                        "chunk_sec_used": getattr(conv, "chunk_sec_used", None),
                        "retried_smaller_chunk": getattr(conv, "retried", False),
                        **conv.timing}
        timings["vc_convert"] = round(conv.timing.get("elapsed", 0.0), 1)
        _log(f"[VC] {conv.chunks} 块 / {conv.timing.get('rtf', 0)}×实时")
        _check_stop(should_stop)

        # LoRA 卸载统一走 finally（VC 后任一阶段异常也不残留挂载）

        # ---------- 5) F0 移植 ----------
        t0 = time.perf_counter()
        vocal: np.ndarray = conv.wav
        f0_report: Dict[str, Any] = {"mode": "off"}
        if req.f0_mode != "off":
            _p(0.72, "F0 提取（源）")
            src_f0 = (src_f0_key if (src_f0_key is not None and key_semis == 0)
                      else svc_f0.harvest_f0(y, ENGINE_SR))
            _p(0.75, "F0 提取（参考音区）")
            ref_y, ref_sr = AL.load_audio(req.ref_path)
            ref_f0 = svc_f0.harvest_f0(resample(ref_y, ref_sr, ENGINE_SR), ENGINE_SR)
            semis = svc_f0.octave_shift_semis(src_f0, ref_f0, req.transpose_manual)
            mapped = svc_f0.shift_f0(src_f0, semis)
            mapped = svc_f0.smooth_ratio_curve(mapped)
            _p(0.78, f"F0 移植（{req.f0_mode} 模式，移调 {semis:+d} 半音）")
            res, metrics = svc_f0.transplant_f0(vocal, ENGINE_SR, mapped,
                                                out_f0=None, mode=req.f0_mode)
            f0_report = {**metrics, "transpose_semis": semis,
                         "key_shift_semis": key_semis,
                         "transplant_ok": res.ok, "transplant_reason": res.reason}
            if res.ok:
                vocal = res.y
            _log(f"[F0] r={metrics.get('f0_r', '-')} → {metrics.get('mode_used', '-')}，"
                 f"移调 {semis:+d}st"
                 + (f"（50 音分内 {metrics.get('within_50c', 0):.0%}）"
                    if metrics.get("within_50c") is not None else ""))
            _p(0.82, f"F0 完成（{metrics.get('mode_used', '-')}，"
                     f"移调 {semis:+d} 半音），进入后处理")
        f0_path = os.path.join(out_dir, "04_f0_vocal.wav")
        AL.save_audio(f0_path, vocal, ENGINE_SR)
        report["f0"] = f0_report
        timings["f0"] = round(time.perf_counter() - t0, 1)
        _check_stop(should_stop)

        # ---------- 6) 后处理链（audio_lab，翻唱标定） ----------
        t0 = time.perf_counter()
        applied: List[str] = []
        if req.highpass_hz > 0:
            vocal = highpass(vocal, ENGINE_SR, req.highpass_hz)
            applied.append(f"高通{req.highpass_hz}Hz")
        if req.hf_deess:
            vocal = AL.suppress_wet_hf(vocal, ENGINE_SR, max_cut_db=10.0)[0]
            applied.append("间隙de-esser")
        if req.hf_limit:
            vocal = AL.limit_voiced_hf(vocal, ENGINE_SR, ceiling_db=12.0)[0]
            applied.append("浊音HF限幅")
        if req.spectral_match:
            try:
                target = AL.band_profile(req.ref_path)
                vocal = AL.match_band_profile(vocal, ENGINE_SR, target, max_db=4.0)
                applied.append("频谱画像对齐")
            except Exception as exc:
                _log(f"[后处理] 频谱对齐跳过: {exc}")
        if req.breath_soften:
            vocal = AL.soften_breath_hiss(vocal, ENGINE_SR)[0]
            applied.append("气声软化")
        peak = float(np.max(np.abs(vocal))) if len(vocal) else 0.0
        if peak > 0.98:
            vocal = vocal * (0.98 / peak)
            applied.append("峰值保护")
        report["post"] = {"applied": applied}
        timings["post"] = round(time.perf_counter() - t0, 1)
        # 音域适配（B）：仅人声升回原调（会引入二次移调伪影）
        if key_semis != 0 and req.key_output == "original_key":
            _p(0.845, f"音域适配：人声升回 {-key_semis} 半音（有移调伪影风险）")
            vocal = shift_audio_librosa(vocal, ENGINE_SR, -key_semis)
            key_info["post_shift"] = "librosa"
            if -key_semis > 4:
                key_info["warn"] = ("升回幅度较大（>4 半音），建议改用"
                                    "『整首降调』输出")
        _p(0.84, "后处理完成" + ("（" + "、".join(applied) + "）" if applied else ""))

        # ---------- 7) 超分（可选，默认关；17.8× 实时耗时警示） ----------
        vocal_sr = ENGINE_SR
        sr_path: Optional[str] = None
        if req.sr_enable:
            t0 = time.perf_counter()
            from webui_app.services import audio_sr as SR
            if not SR.available():
                _log("[超分] audiosr 未安装，跳过")
            else:
                dur_s = len(vocal) / ENGINE_SR
                est_min = dur_s * 17.8 / 60.0
                _p(0.85, "超分前卸载引擎（AudioSR 显存互斥）")
                _log(f"[超分] 卸载引擎 → AudioSR（约 17.8× 实时；"
                     f"{dur_s:.0f}s 人声预计 {est_min:.0f} 分钟）")
                if bool(getattr(engine.stats, "loaded", False)):
                    engine.unload(bypass_lease=svc_lease)
                base = os.path.join(out_dir, "05_vocal_final_22k.wav")
                AL.save_audio(base, vocal, ENGINE_SR)
                # 这一步可能长达几十分钟：进度必须停在“超分中”，
                # 否则心跳会一直显示上一条 F0 文案，看起来像卡死。
                _p(0.86, f"超分中（AudioSR → 48k，预计约 {est_min:.0f} 分钟）")
                r = SR.enhance_file(base, ddim_steps=int(req.sr_steps))
                SR.release_all()
                if r.get("ok"):
                    sr_path = r["path"]
                    vocal, _ = AL.load_audio(sr_path)
                    vocal_sr = r.get("sr", 48000)
                    report["sr"] = {"ok": True, "seconds": r.get("seconds"),
                                    "sr": vocal_sr, "steps": req.sr_steps}
                    _p(0.92, f"超分完成（{r.get('seconds', 0):.0f}s）")
                else:
                    report["sr"] = {"ok": False, "error": r.get("error")}
                    _p(0.92, "超分失败（回退 22k 人声）")
                timings["superres"] = round(time.perf_counter() - t0, 1)

        final_path = sr_path or os.path.join(out_dir, "05_vocal_final_22k.wav")
        if not req.sr_enable or sr_path is None:
            AL.save_audio(final_path, vocal, ENGINE_SR)

        # ---------- 8) 自动合成（可选：分离"废弃物"BGM 回混） ----------
        mix_info: Optional[Dict] = None
        if req.auto_mix:
            t0 = time.perf_counter()
            bgm_src = bgm_path if bgm_path else req.mix_bgm_path
            if bgm_src and os.path.isfile(bgm_src):
                bgm_y, bgm_sr = AL.load_audio(bgm_src)
                if key_semis != 0 and req.key_output == "lower_key":
                    bgm_y = shift_audio_librosa(bgm_y, bgm_sr, key_semis)
                    key_info["bgm_shift"] = "librosa"
                _p(0.93, "自动合成（人声+伴奏混音）")
                # 预处理裁掉的前奏静音补回头部（对拍红线：人声与伴奏同源同时轴）
                head_pad = int(round(head_cut * vocal_sr / ENGINE_SR))
                m = mixdown(vocal, vocal_sr, bgm_y, bgm_sr,
                            req.vocal_gain_db, req.loudness_normalize,
                            head_pad_samples=head_pad)
                mix_path = os.path.join(out_dir, "06_mix.wav")
                AL.save_audio(mix_path, m["wav"], m["sr"])
                mix_info = {"ok": True, "path": mix_path, "sr": m["sr"],
                            "seconds": m["seconds"],
                            "vocal_gain_db": req.vocal_gain_db,
                            "head_pad_ms": round(head_pad / vocal_sr * 1000, 1)}
                _log(f"[混音] {m['sr']}Hz / {m['seconds']}s")
            else:
                mix_info = {"ok": False, "error": "无伴奏轨（原歌曲分离未产出或未上传伴奏）"}
            report["mix"] = mix_info
            timings["mix"] = round(time.perf_counter() - t0, 1)

        # ---------- 9) 质量报告 ----------
        t0 = time.perf_counter()
        _p(0.96, "质量评估（campplus SS / F0 / 体检）")
        report["quality"] = _quality_report(final_path, req.ref_path,
                                            f0_report, vocal, vocal_sr)
        timings["quality"] = round(time.perf_counter() - t0, 1)

        # 产物清单
        files = {}
        for name in sorted(os.listdir(out_dir)):
            p = os.path.join(out_dir, name)
            if os.path.isfile(p):
                files[name] = round(os.path.getsize(p) / 1024, 1)
        report["files"] = files
        report["ok"] = True
        report["final_vocal"] = final_path
        report["timings"] = timings
        _p(1.0, "完成")

    except _Stop:
        report["error"] = "用户停止"
        report["stopped"] = True
        report["timings"] = timings
    except _Fail as exc:
        report["error"] = str(exc)
        report["timings"] = timings
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["timings"] = timings
        import traceback
        report["traceback"] = traceback.format_exc()
    finally:
        if gpt_offloaded:
            try:
                engine.restore_gpt_after_svc()
            except Exception:
                pass
        # LoRA 卸载（我们挂的才卸；异常路径也不残留——验收建议#6）
        if mounted_by_us:
            try:
                engine.detach_lora(target="cfm")
            except Exception:
                pass
            if pre_mounted:
                _log("[LoRA] 提示：转换前 cfm 通道已有其他挂载，"
                     "如需继续使用请到合成页重新挂载")
        # 引擎状态恢复（D7：进入前什么样，离开时什么样）
        try:
            if was_loaded:
                if not bool(getattr(engine.stats, "loaded", False)):
                    engine.load(bypass_lease=svc_lease)
            else:
                if bool(getattr(engine.stats, "loaded", False)):
                    engine.unload(bypass_lease=svc_lease)
        except Exception:
            pass
        finally:
            engine.end_svc(svc_lease)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return report


def _quality_report(vocal_path: str, ref_path: str, f0_report: Dict,
                    vocal: np.ndarray, sr: int) -> Dict:
    """campplus SS（对参考）+ F0 指标（沿用移植阶段）+ AL.analyze 体检。"""
    q: Dict[str, Any] = {}
    try:
        from webui_app.training.reward import RewardScorer, RewardOptions
        scorer = RewardScorer(RewardOptions(device="cpu"))
        a = scorer.embed(vocal_path)
        b = scorer.embed(ref_path)
        q["ss"] = round(RewardScorer.cosine(a, b), 4)
        scorer.unload()
    except Exception as exc:
        q["ss_error"] = str(exc)
    try:
        rep = AL.analyze(vocal_path)
        d = rep.to_dict()
        q["analyze"] = {k: d.get(k) for k in
                        ("score", "grade", "snr_db", "clip_ratio", "dc_offset",
                         "spectral_centroid_hz", "silence_ratio")}
        issues = [x for x in d.get("issues", [])
                  if ("15 秒" not in x and "15s" not in x)]
        q["analyze"]["issues"] = issues[:5]
    except Exception as exc:
        q["analyze_error"] = str(exc)
    if "f0_r" in f0_report:
        q["f0_r"] = f0_report["f0_r"]          # 移植后探针（20s 代表段）
        q["f0_r_pre"] = f0_report.get("f0_r_pre")
        q["octave_err"] = f0_report.get("octave_err")
        q["within_50c"] = f0_report.get("within_50c")
        q["f0_mode_used"] = f0_report.get("mode_used")
    q["pass_line"] = {
        "ss": "≥0.75（对角色参考）",
        "f0_r": "≥0.85（auto 模式处理后）",
        "octave_err": "<5%",
        "note": "参考线见 dev-plan §3.3；歌声素材上下浮动属正常",
    }
    return q
