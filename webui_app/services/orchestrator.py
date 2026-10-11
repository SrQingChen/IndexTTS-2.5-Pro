"""句级编排器（Orchestrator）：把导演台本逐句合成为一条音频。

为什么需要它（对照官方 infer 的两大机械感来源）：
    1. 官方把长文本切段后，每两段之间插入**完全相同**的 interval_silence
       （默认 200ms）—— 真人句间停顿是 150ms~1s+ 的浮动分布。
    2. 官方整段文本共用一份情感条件 —— 无法逐句切换情绪参考。

本模块在 webui 层逐句调用 engine.infer()（单句=单段），自己控制：
    · 每句的情感参考（情感库路由）与 emo_alpha（按表演强度缩放）
    · 每句的随机种子（整篇可复现）
    · 句间静音 = 台本的 pause_after_ms（标点+情绪感知，已含抖动）

不动 indextts/ 上游一行代码；引擎的参考缓存（spk/emo）逐句复用，
换情感参考只重提 w2v-BERT 特征（一次前向，代价小）。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from webui_app import logging_setup as LOG
from webui_app import PROCESS_STARTED_AT
from webui_app.services import emotion_bank as EB
from webui_app.services import audio_lab as AL
from webui_app.services import inference as INF
from webui_app.services.director import (
    DirectorScript, comma_pause_band as _comma_band,
    min_punct_pause as _min_punct_pause)
from webui_app.services.engine import EngineError, TTSEngine

SR = 22050  # 官方输出采样率（int16 单声道）


def code_info() -> Dict[str, Any]:
    """合成时代码与进程的版本信息（旁车落盘，归因用）。

    2026-10-03 教训：连续两轮把「帧量化盲区」误判成「旧进程在跑」——
    事后归因靠行为反推太脆弱。现在旁车直接记录四件事：
        rev / commit_ts  ：磁盘上代码的版本与提交时刻
        dirty            ：工作树是否有未提交修改（同版本不同行为的隐患）
        process_started_at / process_older_than_code：
            进程启动时刻，以及「进程比代码旧」的自动判定——True 即合成
            跑在代码更新之前启动的进程上（改代码没重启 WebUI 的实锤）。
    git 不可用时各字段尽力而为，不影响主流程。
    """
    import subprocess as _sp
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    def _git(*args: str) -> str:
        try:
            return _sp.run(["git", *args], cwd=root, capture_output=True,
                           text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    rev = _git("rev-parse", "--short", "HEAD")
    try:
        commit_ts = float(_git("log", "-1", "--format=%ct") or 0) or None
    except ValueError:
        commit_ts = None
    dirty = bool(_git("status", "--porcelain"))
    older = (commit_ts is not None
             and PROCESS_STARTED_AT < commit_ts)
    return {
        "rev": rev,
        "commit_ts": commit_ts,
        "dirty": dirty,
        "process_started_at": PROCESS_STARTED_AT,
        "process_older_than_code": older,
    }


def _tail_truncation_penalty(path: str) -> float:
    """候选末字被截的惩罚（BoN 专用）。

    AR 模型偶发 EOS 提前，末字衰减被硬切（2026-10-03 爱弥斯实锤：块尾
    在 -29dB 处直接数字零，听感「收音戛然而止」）。SenseVoice 转写会
    **容错补全**缺尾音的句子，reward 的 WER 项扣不到分 —— 坏候选照样
    可能被选中。用波形尾部检查兜底。

    判据按真机校准（2026-10-03）：自然收尾的末 50ms RMS ≤ -41dB（衰减
    完成）；被截断的末 50ms ≈ -30dB（还有语音能量就断了）。取分界
    **-35dB**——比旧判据（末 100ms > -25dB）严格得多：旧线放走了这次
    实锤的截断候选（末 100ms -27dB 却未触发）。只影响候选间相对排序，
    保守罚分。模块级便于探针直测。
    """
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = int(min(0.05, len(y) / sr) * sr)
        if n < int(0.02 * sr):
            return 0.0
        tail = y[-n:]
        rms = 20.0 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12)
        return 0.30 if rms > -35.0 else 0.0
    except Exception:
        return 0.0


def _register_jump_st(path: str, prev_path: str) -> float:
    """候选首音区与上一块尾音区的差（半音，绝对值）。异常/测不出→inf。

    实测（30 块长文本）：块尾→下一块首的 F0 中位差 ±2~8 半音，与块内
    词组过渡同量级——块是独立渲染的，起始音区随机；以前块间 30ms 无缝
    交叉淡化把跳变掩盖了，句间停顿拉长后每一处边界都成了干净 A/B 对比
    （用户听感「分块前后语感突变」）。
    """
    try:
        import soundfile as sf
        import pyworld as pw

        def _med(p: str, head: bool) -> float:
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            n = int(min(0.30, len(y) / sr * 0.5) * sr)
            if n < int(0.1 * sr):
                return float("nan")
            seg = (y[:n] if head else y[-n:]).astype(np.float64)
            f0, _t = pw.harvest(seg, sr, f0_floor=70.0, f0_ceil=600.0)
            v = f0[f0 > 0]
            if len(v) < 15:
                return float("nan")
            return float(np.median(v))

        a = _med(prev_path, head=False)
        b = _med(path, head=True)
        if np.isnan(a) or np.isnan(b):
            return float("inf")
        return abs(12.0 * np.log2(b / a))
    except Exception:
        return float("inf")


def _centroid_jump(path: str, prev_path: str) -> float:
    """候选首 300ms 与上一块尾 300ms 的谱质心差（Hz/400 归一）。

    音色连续性的决胜项之一（2026-10-04）：谱质心块间 6 倍摆动是「一句
    尖锐嘶哑、一句很低沉」的实体；与 _register_jump_st 联合构成并列组
    决胜键（音区 + 音色）。异常 → 0（不影响排序）。
    """
    try:
        import soundfile as sf

        def _cen(p: str, head: bool) -> float:
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            n = int(min(0.30, len(y) / sr * 0.5) * sr)
            if n < int(0.1 * sr):
                return float("nan")
            seg = (y[:n] if head else y[-n:]) * np.hanning(n)
            S = np.abs(np.fft.rfft(seg)) ** 2
            ff = np.fft.rfftfreq(n, 1.0 / sr)
            return float(np.sum(ff * S) / (np.sum(S) + 1e-12))

        a = _cen(prev_path, head=False)
        b = _cen(path, head=True)
        if np.isnan(a) or np.isnan(b):
            return 0.0
        return abs(b - a) / 400.0
    except Exception:
        return 0.0


def _register_jump_penalty(path: str, prev_path: str,
                           free_st: float = 3.0,
                           max_pen: float = 0.30) -> float:
    """_register_jump_st 的惩罚形态（BoN 软罚，线性 >free_st 起罚，封顶）。

    软罚在「某候选奖励领先过多」时买不动（实测 0.93 vs 0.86 差距下
    0.30 罚不动第一名）——真正的决胜在 chosen 选择处的并列组择优
    （奖励差 ≤0.10 为并列，组内挑音区衔接最好的），本惩罚负责把
    「奖励本来就接近」的候选排对顺序。模块级便于探针直测。
    """
    st = _register_jump_st(path, prev_path)
    if not np.isfinite(st):
        return 0.0
    return float(min(max_pen, max(0.0, st - free_st) * 0.06))


def _read_mono_int16(path: str) -> np.ndarray:
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="int16", always_2d=False)
    except Exception as e:
        raise EngineError(f"读取句音频失败 {os.path.basename(path)}：{e}")
    if sr != SR:
        # 官方输出恒为 22050；走到这里说明上游行为变了，宁可报错不出怪声
        raise EngineError(f"句音频采样率 {sr} ≠ {SR}，请检查上游 infer 输出")
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    return data


def _line_emo_kwargs(
    req: INF.GenRequest,
    entry: Optional[EB.EmoRefEntry],
    intensity: float,
    extrapolate: bool,
) -> Dict[str, Any]:
    """一句的情感 kwargs：命中参考→模式1语义；未命中→模式0语义（跟随音色）。

    命中时的 alpha = 全局 emo_alpha × (0.5 + intensity)：
    intensity 1.0（爆发句）→ 1.5×全局；0.3（平静句）→ 0.8×全局。
    外推开关关闭时夹回官方语义的 [0, 1]。
    """
    if entry is None:
        return {"emo_audio_prompt": None, "emo_alpha": 1.0,
                "emo_vector": None, "use_emo_text": False, "emo_text": None,
                "use_random": bool(req.use_random)}
    alpha = float(req.emo_alpha or 0.65) * (0.5 + max(0.0, min(1.0, intensity)))
    if not extrapolate:
        alpha = max(0.0, min(1.0, alpha))
    else:
        alpha = max(0.0, min(1.5, alpha))
    return {"emo_audio_prompt": entry.audio_path, "emo_alpha": alpha,
            "emo_vector": None, "use_emo_text": False, "emo_text": None,
            "use_random": bool(req.use_random)}


def perform(
    engine: TTSEngine,
    req: INF.GenRequest,
    script: DirectorScript,
    route: bool = True,
    character: str = "",
    extrapolate: bool = False,
    progress=None,
    output_path: Optional[str] = None,
    bon_n: int = 0,
    bon_keep: bool = False,
    lora_run: str = "",
    continuity: bool = True,
    breath: bool = False,
    pause_cap_ms: int = 300,
    stress_enable: bool = True,
    stress_gain_db: float = 2.0,
    f0_restore: bool = True,
    f0_expand_max: float = 1.4,
    flatten_ref: bool = True,
    flatten_cap_ms: int = 120,
    pause_scale: float = 1.0,
) -> Dict[str, Any]:
    """表演块编排（v2）：把台本行合并成「表演块」再逐块合成拼接。

    v1 的逐行合成 + 行间垫静音被实测证伪：每行末尾 AR 模型本来就会做
    末延长+收束，再垫人工静音 = 双重句末动作，听感「超级散」。v2 依据
    语音学事实（句内节奏=10~30% 时长伸缩，停顿只在强边界）与官方设计
    （预训练用多句 ≤25s 段、块内自由、块间垫音）重做：

        1. **块合并**：情绪相同的相邻行合并，直到 ≤120 文本 token
           （对齐官方 split_text_by_tokens 预算）；情绪突变/预算到顶断块，
           逗号永不切。块内是**一次合成调用**——微时序、末延长、收束
           全部由 25Hz 语义 token 自发完成，零人工静音。
        2. **块间停顿按等级**：取块末行的 pause_after_ms（标点+情绪
           感知），经 pause_scale 缩放；直连（gap=0）时做 30ms 交叉淡化。
        3. **情感路由/BoN 升为块级**：块=情感一致的表演单元。

    bon_n > 0 时启用**块级择优（Best-of-N）**：每块合成 N 个候选
    （不同种子），用 reward 打分选最好的一条（文献背书：arXiv 2608.31035，
    BoN-8 ≈ GRPO 训练收益）。打分器整体跑在 **CPU**，与 GPU 引擎零竞争。

    Raises:
        EngineError: 输入不合法 / 某一块合成失败（含块号与临时目录位置）
    """
    if not script or not script.lines:
        raise EngineError("导演台本为空，无法编排。")
    # 入参快照（追踪件，2026-10-03）：函数入口的 locals() 恰好就是全部
    # 位置/关键字参数，取标量子集落盘旁车——事后归因时「当时到底传了什
    # 么」不再依赖 UI 状态文件的时序猜测。新增参数自动纳入。
    perform_args = {k: v for k, v in locals().copy().items()
                    if isinstance(v, (bool, int, float, str))}
    # 参考链路审计：基础参考与每块滚动参考的构建/摊平统计
    ref_log: List[Dict[str, Any]] = []
    empty = [i + 1 for i, ln in enumerate(script.lines) if not (ln.text or "").strip()]
    if empty:
        raise EngineError(
            f"台本里第 {empty[:5]} 句台词为空 —— 请检查导演层输出"
            "（旁车 script.json 里有完整台本）。")
    bon_n = int(max(0, min(8, bon_n or 0)))
    cfg = engine.cfg
    log = LOG.get_logger("orchestrator")

    # 响度指纹：挂载的 GPT run 训练时随 run.json 交付的数据集响度分布。
    # 逐块目标响度从它采样（爆发响/平静轻），没有指纹则保持旧行为。
    _loudness_fp: Dict[str, Any] = {}
    _f0_fp: Dict[str, Any] = {}
    if lora_run:
        try:
            from webui_app.training import runs as _RN
            _rj = _RN.read_run(lora_run) or {}
            _loudness_fp = dict(_rj.get("loudness_fingerprint") or {})
            _f0_fp = dict(_rj.get("f0_fingerprint") or {})
        except Exception:
            _loudness_fp = {}
    if _loudness_fp:
        log.info("响度指纹命中（%s）：median=%s std=%s",
                 lora_run, _loudness_fp.get("median"), _loudness_fp.get("std"))
    import random as _random
    _rng = _random.Random(INF.resolve_seed(req.seed))   # 逐块响度采样可复现

    # 基础 kwargs（采样参数/分句/语速等）走既有装配，情感部分逐句覆盖
    out_base = output_path or INF.output_filename(cfg)
    base = INF.build_infer_kwargs(engine, req, out_base)

    base_seed = INF.resolve_seed(req.seed)
    run_dir = os.path.join(cfg.cache_dir, "perform", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    n = len(script.lines)
    routed, fallback = 0, 0
    line_infos: List[Dict[str, Any]] = []
    wavs: List[np.ndarray] = []
    t0 = time.perf_counter()

    # BoN 打分器：懒建（bon_n=0 时完全不实例化），整体跑 CPU —— 与
    # GPU 上的推理引擎零显存竞争。权重：读对 0.40 / 音色像 0.30 /
    # 情绪贴合 0.25（路由命中才有）/ 停顿人味 0.05（带区 5~20%——
    # 2026-09-29 校准：句内静音是稀缺品，不再奖励 8~25% 的旧带区）。
    _scorer = None
    if bon_n > 0:
        from webui_app.training import reward as RW
        _scorer = RW.RewardScorer(RW.RewardOptions(
            asr_engine="sensevoice", whisper_size="small", device="cpu",
            wer_weight=0.40, sim_weight=0.30,
            emo_weight=0.25, pause_weight=0.05))
        log.info("BoN 择优开启：N=%d · 打分器 CPU（sensevoice+campplus"
                 "+emotion2vec+停顿启发）", bon_n)

    def _count_tokens(text: str) -> int:
        fn = getattr(engine, "count_tokens", None)
        if callable(fn):
            try:
                k = int(fn(text))
                if k > 0:
                    return k
            except Exception:
                pass
        return max(1, int(len(text) / 1.6) + 1)   # 中文 ≈1.6 字/token 兜底

    def _merge_blocks() -> List[Dict[str, Any]]:
        """台本行 → 表演块：情绪相同的相邻行合并，预算 = min(120 token,
        **40 字符**)。

        40 字符上限是关键（2026-09-29 用户猜想证实）：8GB 卡上引擎的
        低显存模式对 >40 字的文本会**自行按标点再分块**（连逗号都切，
        块间垫 interval_silence）——那会绕过编排器在块内埋下隐藏的碎片
        化。把块压到 ≤40 字，低显存路径永远不触发，块内始终是一次合成。
        与官方 split_text_by_tokens 同 token 预算；情绪突变 / 预算到顶 /
        **省略号破折号结尾**断块（戏剧性停顿拍必须是块边界才能拿到台本
        长停顿）。块间接缝由滚动参考续合成（_rolling_prompt）弥合。
        """
        blocks: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None

        def _close():
            nonlocal cur
            if cur is not None:
                blocks.append(cur)
                cur = None

        def _feed(ln) -> None:
            """一行进块（含 40 字/token 双预算与戏剧拍断块）。"""
            nonlocal cur
            cost = _count_tokens(ln.text)
            fits = (cur is not None and ln.emotion == cur["emotion"]
                    and cur["tokens"] + cost <= 120
                    and len(cur["text"]) + len(ln.text) + 1 <= 40)
            if fits:
                cur["lines"].append(ln)
                cur["tokens"] += cost
                cur["intensity"] = max(cur["intensity"], ln.intensity)
                cur["stress_words"] = list(dict.fromkeys(
                    cur.get("stress_words", []) + list(ln.stress)))
                joiner = "" if cur["text"][-1:] in "。！？…！？.!?\"" else "。"
                cur["text"] = cur["text"] + joiner + ln.text
            else:
                _close()
                cur = {"emotion": ln.emotion, "lines": [ln], "tokens": cost,
                       "intensity": ln.intensity, "text": ln.text,
                       "stress_words": list(ln.stress)}
            # 戏剧性停顿拍：……/—— 结尾绝不与后文同块
            if cur is not None and cur["text"].endswith(("……", "…", "——", "—")):
                _close()

        from webui_app.services.director import ScriptLine as _SL
        import re as _re

        for ln in script.lines:
            # 长行守卫：>40 字的单行（LLM 台本常见）先按强标点拆成子行，
            # 否则它会成为一个 >40 字的块、触发引擎低显存的内部再分块
            if len(ln.text) > 40:
                parts = [p for p in _re.split(
                    r"(?<=[。！？!?…；;])", ln.text) if p]
                # 强标点不够用时回退逗号级（子块由滚动参考弥合，仍好于
                # 触发引擎低显存的内部再分块——那里没有续接上下文）
                finer = []
                for p in parts:
                    if len(p) <= 40:
                        finer.append(p)
                        continue
                    acc = ""
                    for q in [x for x in _re.split(r"(?<=[，,、])", p) if x]:
                        if acc and len(acc) + len(q) > 40:
                            finer.append(acc)
                            acc = q
                        else:
                            acc += q
                    if acc:
                        finer.append(acc)
                parts = finer
                chunks, cur_p = [], ""
                for p in parts:
                    if cur_p and len(cur_p) + len(p) > 40:
                        chunks.append(cur_p)
                        cur_p = p
                    else:
                        cur_p += p
                if cur_p:
                    chunks.append(cur_p)
                for ci, ch in enumerate(chunks):
                    _feed(_SL(text=ch, emotion=ln.emotion,
                              intensity=ln.intensity,
                              pause_after_ms=(ln.pause_after_ms
                                              if ci == len(chunks) - 1 else 0),
                              note=ln.note))
                continue
            _feed(ln)
        if cur is not None:
            blocks.append(cur)
        return blocks

    blocks = _merge_blocks()
    n = len(blocks)
    log.info("表演块合并：%d 行 → %d 块（≤40 字/块，情绪一致，低显存路径不触发）",
             len(script.lines), n)

    # ---- 频谱画像（2026-09-29 不饱满/缺频段根治）----
    # 实测：音色库参考 300-1kHz 占比 47.5% → 输出 33.7%（克隆链忠实遗传
    # 参考的频谱包络，参考被降噪链搞闷了）。这里把参考的 6 频段包络向
    # **角色本人训练素材的均值画像**（run.json 的 spectral_profile）靠拢
    # （±5dB 平滑曲线、相位不变），块 1 与滚动参考共用这份匹配后的参考。
    _spectral_fp: Dict[str, Any] = {}
    if lora_run:
        try:
            from webui_app.training import runs as _RN
            _rj2 = _RN.read_run(lora_run) or {}
            _spectral_fp = dict(_rj2.get("spectral_profile") or {})
        except Exception:
            _spectral_fp = {}

    _base_ref_path = str(base.get("spk_audio_prompt") or "")

    # ---- 参考韵律摊平（2026-10-02 散感/断层根因修复）----
    # A/B 实锤（同模型同 seed 同文本仅换参考）：戏剧化参考（游戏配音，
    # 每词组收束）→ 输出 67 停顿/分钟、1.1s 大收束；摊平参考 → 32/分钟。
    # **引擎的词组化韵律直接克隆自参考**——输出里「每 2-4 字一组、组间
    # 像不同合成」的调型断裂，源头在这里，任何输出侧静音手术都剪不掉。
    # 把参考里 >flatten_cap_ms 的停顿压平（时长封顶，不动音色/频谱），
    # 只在首次进入克隆链前做一次；滚动参考继承输出的改善，无需再处理。
    def _flatten_ref_prosody(path: str) -> str:
        if not flatten_ref or not path or not os.path.isfile(path):
            return path
        try:
            y, sr = AL.load_audio(path)
            if sr != SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                sr = SR
            y2 = AL.cap_interior_pauses(y, SR, int(flatten_cap_ms),
                                        thresh_db=-30.0)
            ref_log.append({
                "stage": "flatten", "file": os.path.basename(path),
                "cap_ms": int(flatten_cap_ms),
                "before_s": round(len(y) / SR, 3),
                "after_s": round(len(y2) / SR, 3),
                "changed": not (len(y2) == len(y) and np.array_equal(y2, y)),
            })
            if len(y2) == len(y) and np.array_equal(y2, y):
                return path
            p = os.path.join(run_dir, "prompt_base_flat.wav")
            AL.save_audio(p, y2, sr)
            log.info("参考韵律已摊平（>%dms 停顿封顶，时长 %.2f→%.2fs）",
                     int(flatten_cap_ms), len(y) / sr, len(y2) / sr)
            return p
        except Exception as e:
            log.warning("参考韵律摊平失败（用原参考）：%s", e)
            return path

    def _matched_base_ref() -> str:
        """块 1 的（可选频谱匹配 + 韵律摊平后的）参考；失败时原样返回。"""
        if not (_spectral_fp and _base_ref_path):
            return _flatten_ref_prosody(_base_ref_path)
        try:
            y, sr = AL.load_audio(_base_ref_path)
            if sr != SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                sr = SR
            y2 = AL.match_band_profile(y, sr, _spectral_fp, max_db=5.0)
            p = os.path.join(run_dir, "prompt_base_matched.wav")
            AL.save_audio(p, y2, sr)
            log.info("参考已做频谱匹配（画像质心 %s → 目标 %s Hz）",
                     AL.band_profile(y, sr).get("centroid"),
                     _spectral_fp.get("centroid"))
            return _flatten_ref_prosody(p)
        except Exception as e:
            log.warning("参考频谱匹配失败（用原参考）：%s", e)
            return _flatten_ref_prosody(_base_ref_path)

    _matched_ref = _matched_base_ref()

    # ---- 滚动参考续合成（continuity，2026-09-29 用户猜想的正式实现） ----
    # 第 N 块的音色参考 = [参考填充段 + 上一块成品(≤11s)]，定长 14.0s、
    # **结尾是上一块**（最新语境在末尾：GPT 的 w2v-BERT 条件是时序分布的、
    # CFM 的 ref_mel 是前缀续写——两者都拿到「刚说完那句」的语速/语调/
    # 收束，跨块语气接得上；MoonCast/VoiceStar 式前缀续接的零改动实现）。
    # 不足 14s 时**用参考音频补满前面（绝不补零）**——静音占大头的参考会
    # 稀释 CAMPPlus/w2v 条件（实测闷/房间感的来源之一）。每块独立临时
    # 文件：引擎参考缓存按路径命中，同路径换内容会被旧缓存骗过。
    def _rolling_prompt(prev_wav: str, base_ref: str, bi: int) -> str:
        """块 N 参考 = [参考连续段 | 60ms 交叉淡化 | 上一块尾部]，定长 14.0s。

        2026-09-30 重构（v1 的平铺拼接被听感证伪）：不再「3s 参考尾 + 平铺
        重复参考」——重复内容与无淡化的平铺接缝会污染 w2v/CAMPPlus 条件
        （用户听到的「空间音效/忽高忽低」嫌疑之一）。现在：
          · 参考只取**一条连续段**（尾部对齐，不重复）；
          · 上一块占尾部（最新语境在末尾，CFM 前缀续写吃到它）；
          · 两段先做**电平匹配**（±6dB 限幅），再 60ms 等功率交叉淡化；
          · 定长 14.0s（cuDNN 形状恒定）；素材实在太短才前补零（罕见）。
        """
        try:
            ref_y, ref_sr = AL.load_audio(base_ref)
            prev_y, _ = AL.load_audio(prev_wav)
            if ref_sr != SR:
                import librosa
                ref_y = librosa.resample(ref_y, orig_sr=ref_sr, target_sr=SR)
            fixed = int(14.0 * SR)
            xf = int(0.060 * SR)

            # 2026-10-04 尾巴占比反转：11s→5s。11s 尾巴占 14s 参考的
            # 79%，音色/音区漂移链式放大（谱质心 6 倍摆动实锤）；5s 尾巴
            # （35%）足以承载「最新语境接续」，9s 光谱对齐过的稳定基础
            # 参考把每块条件锚回角色均值。
            prev_tail = prev_y[-int(5.0 * SR):]
            # 参考韵律摊平对滚动参考同样生效（2026-10-03 连续性修复）：
            # prev_tail 是上一块**未经输出治理的原始波形**，带着模型自发
            # 的词组化停顿（实测 44~53 个/分钟）——不摊平的话，第 2 块起
            # 的参考把散感韵律重新教回模型，阶段28只治了块 1（实测用户
            # 成品停顿密度 147/分钟 vs 单次直出 98）。这里用与基础参考同一
            # 把刀（>flatten_cap_ms 停顿封顶，中段截，语音/音色不动）。
            _pt_before = len(prev_tail)
            if flatten_ref:
                prev_tail = AL.cap_interior_pauses(
                    prev_tail, SR, int(flatten_cap_ms), thresh_db=-30.0)
            # 频谱画像对齐（2026-10-04 音色漂移根治件）：滚动参考里 11s
            # 的 prev_tail 主导 w2v/CAMPPlus 条件，而它从未做频谱匹配——
            # 明亮块生出明亮参考（更亮）、暗块反之，音色链式漂移（实测
            # 30 块谱质心 775~4690Hz 摆 6 倍 = 用户听感「一句尖锐嘶哑、
            # 一句很低沉」；最初导演模式每块共用同一参考所以无此问题）。
            # 与块 1 基础参考同一把刀：向角色 spectral_profile 对齐
            # （±5dB 平滑增益、相位不变），把每块的音色条件拉回角色均值。
            _spec_matched = False
            if _spectral_fp:
                try:
                    prev_tail = AL.match_band_profile(
                        prev_tail, SR, _spectral_fp, max_db=5.0)
                    _spec_matched = True
                except Exception:
                    pass
            ref_log.append({
                "stage": "rolling", "block": bi, "prev": os.path.basename(prev_wav),
                "prev_tail_before_s": round(_pt_before / SR, 3),
                "prev_tail_after_s": round(len(prev_tail) / SR, 3),
                "flatten_on": bool(flatten_ref),
                "spectral_matched": _spec_matched,
            })
            ref_part = ref_y[max(0, len(ref_y) - (fixed - len(prev_tail))):]
            # 参考不够填 → 拉长上一块的占用；两者合计仍不足 14s 就**短着用**
            # ——绝不补零（静音参考稀释条件是实测过的病）。代价是这种罕见
            # 场景多一次 cuDNN 形状调优；正常素材（参考≥6s）恒为 14.0s。
            if len(ref_part) + len(prev_tail) < fixed:
                _need = fixed - len(ref_part)
                prev_tail = (prev_y[-_need:] if _need <= len(prev_y)
                             else prev_y)
                if flatten_ref:
                    prev_tail = AL.cap_interior_pauses(
                        prev_tail, SR, int(flatten_cap_ms), thresh_db=-30.0)

            # 段间电平匹配（按各自语音 RMS，±6dB 限幅）—— 电平差会让
            # 交叉淡化处出现台阶，条件特征读到「音量突变」
            def _srms(x):
                if not len(x):
                    return 1e-4
                fr = int(SR * 0.025)
                n = len(x) // fr
                r = np.sqrt(np.mean(
                    x[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
                act = np.percentile(r, 60)
                return max(float(act), 1e-4)
            gain = float(np.clip(_srms(ref_part) / _srms(prev_tail),
                                 10 ** (-6 / 20), 10 ** (6 / 20)))
            prev_tail = prev_tail * gain

            f = min(xf, len(ref_part) // 2, len(prev_tail) // 2)
            a, b = ref_part.copy(), prev_tail.copy()
            if f > 0:
                # 等功率（余弦）交叉淡化：不相干素材拼接听感更平滑
                a[-f:] *= np.cos(np.linspace(0, np.pi / 2, f)) ** 1
                b[:f] *= np.sin(np.linspace(0, np.pi / 2, f)) ** 1
            mixed = np.concatenate([a, b])
            if len(mixed) > fixed:
                mixed = mixed[-fixed:]
            # 首尾 5ms 微淡化（防文件边界 click）
            e = int(0.005 * SR)
            if len(mixed) > 2 * e:
                mixed = mixed.copy()
                mixed[:e] *= np.linspace(0, 1, e, dtype=np.float32)
                mixed[-e:] *= np.linspace(1, 0, e, dtype=np.float32)
            p = os.path.join(run_dir, f"prompt_blk{bi:04d}.wav")
            AL.save_audio(p, mixed, SR)
            return p
        except Exception as e:
            log.warning("滚动参考构建失败（块 %d 回退原参考）：%s", bi, e)
            return base_ref

    def _score_one(path: str, text: str, emo_ref: str,
                   prev_path: str = "") -> float:
        r = _scorer.score(path, text, base["spk_audio_prompt"],
                          emo_ref_path=(emo_ref or None))
        if not r.get("ok") or r.get("reward") is None:
            return -1.0
        pen = _tail_truncation_penalty(path)
        if prev_path:
            # 衔接连续性择优：同情绪块间才罚（情绪转换是有意的风格变化）
            pen += _register_jump_penalty(path, prev_path)
            # 音色亮度连续性（2026-10-11 发闷周期治理）：候选头部质心偏离
            # 上一块尾部越远罚越重（双向对称，只选连续不选暗/亮方向）——
            # 抑制「几句发闷几句恢复」的块间音色漂移
            pen += min(0.10, _centroid_jump(path, prev_path) * 0.05)
        return float(r["reward"]) - pen

    # 逐块目标响度：先采完整序列再做**限步平滑**（首块=中位，相邻 ≤2.5dB）
    # —— 独立采样在宽分布角色上会 ±9dB 跳变，听感「忽高忽低」/混响抽吸
    block_targets: List[Optional[float]] = [None] * n
    if _loudness_fp:
        _raw = [AL.sample_target_loudness(_loudness_fp, b["intensity"], _rng)
                for b in blocks]
        block_targets = AL.smooth_loudness_sequence(
            _raw, max_step_db=2.5,
            anchor_db=float(_loudness_fp.get("median", -20.0)))

    prev_chosen: Optional[str] = None
    try:
        for bi, blk in enumerate(blocks):
            if progress:
                try:
                    progress(0.05 + 0.85 * bi / n,
                             desc=f"导演编排 块 {bi + 1}/{n} · {blk['emotion']}")
                except Exception:
                    pass

            entry = EB.pick(character, blk["emotion"]) if route else None
            if route:
                if entry is not None:
                    routed += 1
                else:
                    fallback += 1

            emo_kw = _line_emo_kwargs(req, entry, blk["intensity"], extrapolate)
            block_seed = (base_seed + 977 * (bi + 1)) % (2 ** 31)
            emo_ref_path = entry.audio_path if entry is not None else ""
            blk_text = blk["text"]

            # 参考：块 1 用（可选频谱匹配后的）参考；第 2+ 块滚动续接
            used_rolling = False
            if continuity and bi > 0 and prev_chosen:
                blk_ref = _rolling_prompt(prev_chosen, _matched_ref, bi)
                used_rolling = (blk_ref != _matched_ref)
            else:
                blk_ref = _matched_ref

            # ---- 合成：单候选，或 BoN 的 N 个候选（块级） ----
            cand_paths: List[str] = []
            cand_seeds: List[int] = []
            n_try = max(1, bon_n)
            for k in range(n_try):
                cand_seed = (block_seed + 131 * k) % (2 ** 31)
                INF.apply_seed(cand_seed)
                cpath = os.path.join(run_dir, f"blk_{bi:04d}_c{k}.wav")
                kw = dict(base)
                kw["text"] = blk_text
                kw["output_path"] = cpath
                kw["spk_audio_prompt"] = blk_ref
                # 块 ≤40 字 ⇒ 引擎低显存路径（>40 字触发）永不命中；
                # 万一有超长单句漏网，内部垫 0 也好过叠一层人工静音
                kw["interval_silence"] = 0
                kw.update(emo_kw)
                if progress and bon_n > 0:
                    try:
                        progress(0.05 + 0.85 * (bi + 0.5 * (k + 1) / n_try) / n,
                                 desc=f"导演编排 块 {bi + 1}/{n} · 候选 {k + 1}"
                                      f"/{n_try} · {blk['emotion']}")
                    except Exception:
                        pass
                try:
                    ret = engine.infer(progress=None, **kw)
                except Exception as e:
                    raise EngineError(
                        f"块 {bi + 1}/{n} 合成失败（情绪 {blk['emotion']}，"
                        f"候选 {k + 1}/{n_try}）："
                        f"{type(e).__name__}: {e}。临时音频保留在 {run_dir}")
                actual = ret if isinstance(ret, str) else cpath
                if not os.path.isfile(actual):
                    raise EngineError(
                        f"块 {bi + 1}/{n} 没有产出音频（文本可能为空或被"
                        f"提前终止）：{blk_text[:40]}…")
                cand_paths.append(actual)
                cand_seeds.append(cand_seed)

            # ---- 择优（块级） ----
            rewards: List[float] = []
            chosen_k = 0
            bon_info: Dict[str, Any] = {}
            if bon_n > 1:
                # 衔接连续性：同情绪的相邻块间，候选首音区偏离上一选中
                # 块尾音区会被罚——「分块前后语感突变」的择优治理
                # （2026-10-04，实测边界音区差 ±2~8 半音与块内词组过渡
                # 同级，句间停顿拉长后暴露）。
                _prev_for_cont = (prev_chosen if (
                    bi > 0 and prev_chosen
                    and blocks[bi]["emotion"] == blocks[bi - 1]["emotion"])
                    else "")
                for k, cp in enumerate(cand_paths):
                    if progress:
                        try:
                            progress(0.05 + 0.85 * (bi + 0.75) / n,
                                     desc=f"打分 块 {bi + 1}/{n} · 候选 {k + 1}"
                                          f"/{n_try}")
                        except Exception:
                            pass
                    rewards.append(round(_score_one(cp, blk_text,
                                                    emo_ref_path,
                                                    prev_path=_prev_for_cont),
                                         4))
                ok_idx = [k for k, r in enumerate(rewards) if r >= 0]
                chosen_k = 0
                _tied_n = 1
                if ok_idx:
                    best_r = max(rewards[k] for k in ok_idx)
                    # 并列组择优（2026-10-04 衔接连续性决胜局）：奖励差
                    # ≤0.10 的候选视为并列（准确率优先不妥协），组内挑
                    # 与上一块尾音区衔接最好的——软罚买不动领先者时由
                    # 这里兜底（实测 0.93 vs 0.86 的差距下 0.30 罚不动）。
                    tied = [k for k in ok_idx if rewards[k] >= best_r - 0.10]
                    _tied_n = len(tied)
                    if len(tied) > 1 and _prev_for_cont:
                        # 联合连续度决胜：音区（半音）+ 音色（质心/400）
                        chosen_k = min(
                            tied, key=lambda k: _register_jump_st(
                                cand_paths[k], _prev_for_cont)
                            + _centroid_jump(cand_paths[k], _prev_for_cont))
                    else:
                        chosen_k = max(ok_idx, key=lambda k: rewards[k])
                bon_info = {"n": n_try, "chosen": chosen_k,
                            "rewards": rewards,
                            "tied_n": _tied_n,
                            "continuity": bool(_prev_for_cont)}
                # bon_keep：把最优/最差落成持久文件 —— 对齐页的
                # 「从 BoN 台本导入」用它们构造 DPO 偏好对（C2 桥接）。
                if bon_keep and len(ok_idx) >= 2:
                    worst_k = min(ok_idx, key=lambda k: rewards[k])
                    bon_dir = os.path.join(
                        cfg.output_dir, "bon",
                        os.path.splitext(os.path.basename(out_base))[0])
                    try:
                        os.makedirs(bon_dir, exist_ok=True)
                        bp = os.path.join(bon_dir, f"blk_{bi:04d}_best.wav")
                        wp = os.path.join(bon_dir, f"blk_{bi:04d}_worst.wav")
                        shutil.copy2(cand_paths[chosen_k], bp)
                        shutil.copy2(cand_paths[worst_k], wp)
                        bon_info.update({
                            "best_path": bp, "worst_path": wp,
                            "best_k": chosen_k, "worst_k": worst_k,
                            "reward_best": rewards[chosen_k],
                            "reward_worst": rewards[worst_k],
                        })
                    except OSError as e:
                        log.warning("BoN 候选持久化失败（块 %d）：%s",
                                    bi + 1, e)
                # 落选候选即删（run_dir 反正要清，删掉省磁盘峰值）
                for k, cp in enumerate(cand_paths):
                    if k != chosen_k:
                        try:
                            os.remove(cp)
                        except OSError:
                            pass

            wav = _read_mono_int16(cand_paths[chosen_k])

            # ---- 间隙「痰状」高频爆冲抑制（2026-10-10 锁暝取证根治件）----
            # 实体：6~11kHz 平坦、快调制、电平近语音的噪声簇（音节间隙），
            # 源素材（游戏VO MP3）自带，底座克隆即复现（四配置差分实锤）。
            # 只动间隙里的高频带（≤12dB 限幅），浊音/擦音零接触；处理结果
            # 回写候选文件——滚动参考链拿到干净音频，切断
            # 「爆冲→参考→下一块再克隆」的反馈回路。
            try:
                wav, _n_wet = AL.suppress_wet_hf(wav, SR)
                wav, _n_tear = AL.limit_voiced_hf(wav, SR)
                if _n_wet or _n_tear:
                    sf.write(cand_paths[chosen_k], wav, SR,
                             subtype="PCM_16")
            except Exception as _e:
                _n_wet = _n_tear = 0
                log.warning("湿高频抑制失败（跳过）：%s", _e)

            # ---- 块内停顿「数量+时长」双治理（2026-10-02 散感彻底根治）----
            # 实测：音色库参考 28 停顿/分钟语音 + AR 先验放大 → 引擎输出
            # 44~53 个/分钟，「两三个字一停」。时长封顶治了长治不了多；
            # 按用户产品规范（短句内 0 明显停顿、长句最多 1 处）改配额制：
            # ≤12 字保留 0 个、13~27 字保留 1 个最长、28~40 字保留 2 个，
            # 保留的压到 cap（默认 220ms），其余压到 60ms 词间隙水平。
            # 块边界的台本停顿在拼接层，不受影响。
            # 第五轮（2026-10-02 逗号保底位）：参考摊平后模型在逗号处的
            # 停顿只剩 75~150ms，短块配额还会把它压到 35ms——「逗号几乎
            # 没有停顿」。标点感知模式：块文本里每个逗号类标点按位置匹配
            # 已实现的静音，保证落在导演逗号带（×停顿系数，与块边界同一
            # 真源）；无标点路径行为不变。
            #
            # 层级锚定（2026-10-03）：配额拍的上限不再只看封顶滑条，而是
            # 锚到**缩放后的逗号带上限**（cap_eff = min(滑条, 逗号带 hi)）
            # ——块内皆句内，句内停顿不过逗号级。此前 pause_scale=0.4 时
            # 边界逗号被压到 84ms 而块内配额拍仍可达 300ms，「逗号/句号
            # 比句内节奏停顿还短」的倒挂由此而来；句号/省略号等更强的
            # 停顿永远由块边界（台本带）承载，层级在任何停顿系数下成立。
            # 默认（scale=1.0, 封顶220）下 220→210ms，听感无差。
            if pause_cap_ms and int(pause_cap_ms) > 0:
                _pf, _pc = _comma_band(float(pause_scale or 1.0))
                _cap_eff = min(int(pause_cap_ms), int(_pc))
                gov_audit: List[Dict[str, Any]] = []
                _raw_samples = int(wav.shape[0])
                wav = AL.normalize_intra_pauses(
                    wav.astype(np.float32) / 32768.0, SR,
                    text_chars=len(blk_text),
                    cap_ms=_cap_eff,
                    text=blk_text,
                    punct_floor_ms=_pf, punct_cap_ms=_pc,
                    audit_out=gov_audit)
                wav = np.clip(wav * 32767.0, -32767,
                              32767).astype(np.int16)
                _gov_entry = {"raw_samples": _raw_samples,
                              "out_samples": int(wav.shape[0]),
                              "cap_ms_effective": _cap_eff,
                              "punct_band": [_pf, _pc],
                              "wet_hf_clusters": _n_wet,
                              "voiced_tear_clusters": _n_tear,
                              "audit": gov_audit}
            else:
                _gov_entry = {"enabled": False,
                              "reason": "pause_cap_ms<=0",
                              "wet_hf_clusters": _n_wet}

            # ---- F0 音域恢复（WORLD 重合成，向角色指纹靠拢并扩张）----
            f0_applied = False
            f0_note = ""
            if f0_restore and _f0_fp:
                y = wav.astype(np.float32) / 32768.0
                rr = AL.restore_f0_range(y, SR, _f0_fp,
                                         intensity=blk["intensity"],
                                         max_expand=float(f0_expand_max))
                if rr.get("ok") and len(rr.get("y", [])) == len(wav):
                    wav = np.clip(rr["y"] * 32767.0, -32767,
                                  32767).astype(np.int16)
                    f0_applied = True
                f0_note = str(rr.get("error") or
                              (f"k={rr.get('k')}" if rr.get("ok") else ""))

            # ---- 块输出频谱对齐（2026-10-11 发闷周期根治件）----
            # 实测（达妮娅新 9 块）：块间谱质心 963~1597 摆 1.66 倍，发闷块
            # 的 2-5kHz 共鸣峰区塌陷（响度正常）——CFM 扩散采样+BoN 选择的
            # 联合方差。把每块输出频谱拉向**演员自己的平均画像**
            #（spectral_profile，±4dB 温和钳制）：偏离平均=偏离演员音色，
            # 拉回即恢复。已在目标附近的块增益≈0 零接触。
            if _spectral_fp:
                try:
                    _yf = wav.astype(np.float32) / 32768.0
                    _ym = AL.match_band_profile(_yf, SR, _spectral_fp,
                                                max_db=4.0)
                    wav = np.clip(_ym * 32767.0, -32767,
                                  32767).astype(np.int16)
                except Exception as _e:
                    log.warning("块输出频谱对齐失败（跳过）：%s", _e)

            # ---- 逐块响度（响度指纹迁移）：从角色自己的响度分布采样目标 ----
            # 爆发块从响端、平静块从轻端 —— "这句该响那句该轻"来自角色本人
            # 的习惯（run.json 的 loudness_fingerprint），不是随机抖动。
            # 无指纹（旧 run / 纯底座）时返回 -20，行为与从前一致。
            block_lufs = block_targets[bi] if _loudness_fp else None
            if block_lufs is not None:
                y = wav.astype(np.float32) / 32768.0
                y = AL.apply_block_loudness(y, block_lufs)
                wav = np.clip(y * 32767.0, -32767, 32767).astype(np.int16)

            # ---- 词级重音（LLM/规则标注 + 字符时间映射 + 局部增益）----
            # 短块（<10 字）跳过（2026-10-03 阶梯感修复）：整块只有一个
            # 韵律词组时，重音增益不是「聚焦」而是电平台阶——真机 4 字块
            # 「因为这样……」上给末 2 字 +2dB，用户听到明显的两次合成感。
            stressed: List[str] = []
            if (stress_enable and stress_gain_db > 0
                    and blk.get("stress_words") and len(blk_text) >= 10):
                try:
                    _y = wav.astype(np.float32) / 32768.0
                    _fr = int(SR * 0.025)
                    _nn = len(_y) // _fr
                    _sil = []
                    if _nn >= 2:
                        _rms = np.sqrt(np.mean(
                            _y[:_nn * _fr].reshape(_nn, _fr) ** 2,
                            axis=1) + 1e-12)
                        _db = 20 * np.log10(_rms + 1e-10)
                        _q = _db < -40.0
                        _i2 = 0
                        while _i2 < _nn:
                            if _q[_i2]:
                                _j2 = _i2
                                while _j2 < _nn and _q[_j2]:
                                    _j2 += 1
                                if (_j2 - _i2) * 0.025 >= 0.12:
                                    _sil.append((_i2 * 0.025,
                                                 (_j2 - _i2) * 0.025))
                                _i2 = _j2
                            else:
                                _i2 += 1
                    _spans = AL.char_time_spans(blk_text, len(_y) / SR, _sil)
                    _cand = []
                    for w in blk["stress_words"]:
                        idx = blk_text.find(w)
                        if 0 <= idx and idx + len(w) <= len(_spans):
                            _cand.append((_spans[idx][0],
                                          _spans[idx + len(w) - 1][1], w))
                    _cand.sort()
                    _last_end = -1.0
                    for a, b, w in _cand:
                        if a >= _last_end:
                            stressed.append(w)
                            _last_end = b
                    if _cand:
                        wav = AL.apply_stress_gains(
                            wav, SR, [(a, b) for a, b, _w in _cand],
                            gain_db=float(stress_gain_db))
                except Exception as e:
                    log.warning("重音定位失败（跳过）：%s", e)

            # ---- 韵尾延长（2026-10-03 收音急促根治件，第三处定稿位）----
            # 爱弥斯类 LoRA 把标点前收音渲染成 ≤30ms 陡坡（8 候选全如此，
            # 选择侧无解）——对真急收点前的尾段做 WORLD 1.5x 拉伸（音高/
            # 谱不变），把「戛然而止」拖成可感知的末音节延长。
            # **必须在响度归一与重音之后**：静音判定线（-30dB）是绝对电平，
            # 引擎原始块比成品响 ~7dB，早做会让停顿检出整体缩水、零触发
            # （真机实锤两轮）；放在这里 = 判据电平与听众听到的一致。
            # 保守触发：主体线（非静音帧 p75，电平不变）坡 <80ms、停顿
            # ≥100ms、浊音守卫 + 谱自检，每块 ≤4 处；拼接层随后的边界
            # 记账（tail_s 重测）自动吸收新尾长。
            if isinstance(_gov_entry, dict):
                wav, _releases = AL.lengthen_abrupt_releases(wav, SR)
                _gov_entry["releases"] = _releases
                _gov_entry["out_samples"] = int(wav.shape[0])
                if _releases:
                    log.info("块 %d 韵尾延长 %d 处：%s", bi + 1,
                             len(_releases), _releases)

            wavs.append(wav)
            # 块后停顿 = 块末行的 pause_after_ms（导演层已按标点/情绪给量）；
            # 块内行的停顿标记作废 —— 那部分节奏交还模型
            info = {
                "idx": bi + 1, "text": blk_text, "emotion": blk["emotion"],
                "intensity": blk["intensity"],
                "pause_after_ms": blk["lines"][-1].pause_after_ms,
                "seed": cand_seeds[chosen_k],
                "emo_ref": entry.name if entry else "",
                "emo_alpha": (emo_kw["emo_alpha"] if entry is not None else None),
                "target_lufs": block_lufs,
                "rolling_ref": used_rolling,
                "stress": stressed,
                "f0_restored": f0_applied,
                "f0_note": f0_note,
                "ref_flattened": bool(flatten_ref),
                "samples": int(wav.shape[0]),
                "governance": _gov_entry,
                "lines": [l.text for l in blk["lines"]],
            }
            if bon_n > 1:
                info["bon"] = bon_info
            line_infos.append(info)
            prev_chosen = cand_paths[chosen_k]

        # ---- 拼接：块间停顿**精确化**（2026-10-02 散感根治）----
        # 旧逻辑 gap 纯叠加在块尾/块头静音之上：模型的块尾收束（0.2~0.3s）
        # + 块头起振（0.2~0.3s）+ 台本 gap 三重叠加，实测每处 0.4~0.9s ——
        # 长句被导演拆成多块后「每一小句都拖长停顿、整体很散」。
        # 现在：量出两侧实际静音，超出台本 gap 就从外侧修剪、不足就垫零 ——
        # 块间总停顿精确等于 pause_after_ms（下限 30ms 防粘连），台本给的
        # 层级（词间<逗号≤句间/戏剧拍）原样落地。
        # 吸气（2026-09）：有停顿的块边界按概率插入**角色本人**的吸气采样
        # （breath_bank），贴下一句开口放置。默认关（breath=False）。
        from webui_app.services import breath_bank as BB
        fade = int(SR * 0.03)
        inhales_used = 0
        # 拼接层审计（追踪件）：每个块边界实际量到的两侧静音与修剪/垫足
        # 决策——「边界停顿为什么是这个长度」从旁车直接读，不用反推。
        joints: List[Dict[str, Any]] = []
        final = wavs[0] if wavs else np.zeros(1, np.int16)
        for i in range(1, n):
            gap_s = max(0.0, float(line_infos[i - 1]["pause_after_ms"])) / 1000.0
            # 标点下限（2026-10-03 句间反转修复）：LLM 的 0 语义是「想合
            # 并成一块」，40 字预算合并不了时 0 会落到 30ms 交叉淡化，
            # 被块内逗号保底(150~290ms)反超——「句间比句内还短」。行尾
            # 标点才是停顿等级的真源：逗号结尾→至少逗号带下限；无标点
            # （强行句中切）→保持 30ms 无缝。
            _punct_floor_ms = 0
            if gap_s < 0.05:
                _punct_floor_ms = _min_punct_pause(
                    line_infos[i - 1]["text"], float(pause_scale or 1.0))
                gap_s = max(gap_s, _punct_floor_ms / 1000.0)
            gap_s = max(gap_s, 0.03)             # 台本 0 也至少留 30ms
            nxt = wavs[i]

            # 量两侧实际静音（块尾收束 / 块头起振）
            _fin_f = final.astype(np.float32) / 32768.0
            _nxt_f = nxt.astype(np.float32) / 32768.0
            tail_s = AL.edge_silence_len(_fin_f, SR, from_end=True)
            head_s = AL.edge_silence_len(_nxt_f, SR, from_end=False)

            over = tail_s + head_s - gap_s
            if over > 0:
                # 超出：从两侧**外侧**修剪（各承担一半，不超过各自静音量）
                cut_tail = min(over / 2.0, max(0.0, tail_s - 0.02))
                cut_head = min(over - cut_tail, max(0.0, head_s - 0.02))
                if cut_tail > 0:
                    final = AL.trim_edge_silence(final, SR, cut_tail,
                                                 from_end=True).astype(np.int16)
                if cut_head > 0:
                    nxt = AL.trim_edge_silence(nxt, SR, cut_head,
                                               from_end=False).astype(np.int16)
                gap = 0                          # 已贴紧，走交叉淡化
            else:
                gap = int(SR * (gap_s - tail_s - head_s) * 1000.0 / 1000.0)

            joints.append({
                "after_block": i,
                "script_gap_ms": int(round(gap_s * 1000)),
                "punct_floor_ms": _punct_floor_ms,
                "tail_s": round(tail_s, 3), "head_s": round(head_s, 3),
                "over_s": round(over, 3),
                "trim_tail_ms": int(round(min(over / 2.0, max(0.0, tail_s - 0.02)) * 1000)) if over > 0 else 0,
                "trim_head_ms": int(round(min(over - min(over / 2.0, max(0.0, tail_s - 0.02)), max(0.0, head_s - 0.02)) * 1000)) if over > 0 else 0,
                "bed_ms": int(round(gap / SR * 1000)) if gap > 0 else 0,
                "mode": "bed" if gap > 0 else "crossfade",
            })

            if gap > 0:
                gap_n = gap
                bed = np.zeros(gap_n, dtype=np.int16)
                if breath and character:
                    try:
                        inh = BB.maybe_inhale(character, gap_n * 1000 // SR,
                                              float(line_infos[i].get(
                                                  "intensity") or 0.5), _rng)
                    except Exception:
                        inh = None
                    if inh is not None and len(inh) < gap_n:
                        off = max(0, gap_n - 30 - len(inh))
                        bed[off: off + len(inh)] = inh
                        inhales_used += 1
                final = np.concatenate([final, bed, nxt])
            else:
                f = min(fade, len(final) // 2, len(nxt) // 2)
                if f > 0:
                    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
                    mixed = (final[-f:].astype(np.float32) * (1.0 - ramp)
                             + nxt[:f].astype(np.float32) * ramp)
                    final[-f:] = mixed.astype(np.int16)
                    final = np.concatenate([final, nxt[f:]])
                else:
                    final = np.concatenate([final, nxt])

        # ---- 起止整形：首尾 80ms 余白 + 12ms 淡化（治"被咬掉"感） ----
        # 余白给下游混音留呼吸口；淡入让第一个字的起振不被切。
        # 末尾硬截断兜底（2026-10-03）：AR EOS 提前时成品在语音能量中直接
        # 数字零（爱弥斯实锤 -29dB 硬切）。BoN 惩罚收紧后仍可能全员被截
        # （素材本身急收尾），这里把悬崖变成 100ms 快速收束——自然收尾
        # （尾部静音 ≥30ms）原样不动。
        _tail_fixed = AL.repair_truncated_tail(final, SR)
        tail_repaired = not np.array_equal(_tail_fixed, final)
        if tail_repaired:
            log.info("成品末尾疑似截断，已做 100ms 收束淡化")
        final = _tail_fixed
        final = AL.shape_edges(final, SR, pad_ms=80.0, fade_ms=12.0)

        try:
            import soundfile as sf
            sf.write(out_base, final, SR, subtype="PCM_16")
        except Exception as e:
            raise EngineError(f"写拼接结果失败：{e}")
    except Exception:
        # 失败保留 run_dir 便于排查（成功则清掉）
        log.warning("编排失败，句音频保留在 %s", run_dir, exc_info=True)
        raise
    finally:
        if _scorer is not None:
            try:
                _scorer.unload()      # 连带 funasr_hub 一起卸
            except Exception:
                pass

    try:
        shutil.rmtree(run_dir, ignore_errors=True)
    except Exception:
        pass

    seconds = time.perf_counter() - t0
    dur = INF.audio_duration(out_base)

    # 台本落盘为旁车文件（观测/复现用；清理页会随 outputs/ 一起盘点）
    # 2026-10-03 追踪件全套：code（代码版本+进程年龄自动判定）、
    # perform_args（本次编排全部入参）、refs（参考链路与摊平统计）、
    # joints（拼接层逐边界审计）、逐块 governance（每段静音的处置原因码
    # 与逗号匹配详情）。事后归因从旁车直接读——本轮「旧进程」误判两次的
    # 教训：行为反推不可靠，让产物自己作证。
    sidecar = os.path.splitext(out_base)[0] + ".script.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "backend": script.backend, "ok": script.ok,
                "error": script.error, "character": character,
                "routed": routed, "fallback": fallback,
                "base_seed": base_seed, "lines_in": len(script.lines),
                "blocks": n,
                "code": code_info(),
                "perform_args": perform_args,
                "refs": ref_log,
                "joints": joints,
                "tail_repair": tail_repaired,
                "lines": line_infos,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        log.warning("台本旁车文件写失败：%s", sidecar, exc_info=True)

    # kwargs 摘要（合成页结果表用）：把整篇单次的情感字段替换为编排统计
    kw_summary = {k: v for k, v in base.items()
                  if k not in ("text", "output_path", "emo_audio_prompt",
                               "emo_alpha", "emo_vector")}
    log.info("编排完成：%d 行→%d 块 · 路由 %d / 回退 %d · %.1fs · %s",
             len(script.lines), n, routed, fallback, seconds,
             os.path.basename(out_base))

    return {
        "path": out_base,
        "seconds": seconds,
        "audio_duration": dur,
        "rtf": (seconds / dur) if dur else None,
        "seed": base_seed,
        "kwargs": kw_summary,
        "director": {
            "backend": script.backend, "n": n, "routed": routed,
            "fallback": fallback, "bon_n": bon_n,
            "lines_in": len(script.lines),
            # 台本缓存命中（cached_api_direct 的 note 标记）——UI 显示
            # 「缓存命中/LLM 重新生成」，让省掉的 LLM 调用肉眼可见。
            "cache_hit": (script.note == "cache-hit"),
            "bon_kept": sum(1 for x in line_infos
                            if (x.get("bon") or {}).get("best_path")),
            "avg_pause_ms": (sum(x["pause_after_ms"] for x in line_infos[:-1])
                             / max(1, n - 1)) if n > 1 else 0,
            "inhales": inhales_used,
            "sidecar": sidecar,
        },
    }
