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

from webui_app import logging_setup as LOG
from webui_app.services import emotion_bank as EB
from webui_app.services import audio_lab as AL
from webui_app.services import inference as INF
from webui_app.services.director import DirectorScript
from webui_app.services.engine import EngineError, TTSEngine

SR = 22050  # 官方输出采样率（int16 单声道）


def _read_mono_int16(path: str) -> np.ndarray:
    """读一个官方输出的 wav → (T,) int16。失败抛 EngineError。"""
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
    if lora_run:
        try:
            from webui_app.training import runs as _RN
            _rj = _RN.read_run(lora_run) or {}
            _loudness_fp = dict(_rj.get("loudness_fingerprint") or {})
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
                joiner = "" if cur["text"][-1:] in "。！？…！？.!?\"" else "。"
                cur["text"] = cur["text"] + joiner + ln.text
            else:
                _close()
                cur = {"emotion": ln.emotion, "lines": [ln], "tokens": cost,
                       "intensity": ln.intensity, "text": ln.text}
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

    def _matched_base_ref() -> str:
        """块 1 的（可选频谱匹配后的）参考；无画像/失败时原样返回。"""
        if not (_spectral_fp and _base_ref_path):
            return _base_ref_path
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
            return p
        except Exception as e:
            log.warning("参考频谱匹配失败（用原参考）：%s", e)
            return _base_ref_path

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

            prev_tail = prev_y[-int(11.0 * SR):]
            ref_part = ref_y[max(0, len(ref_y) - (fixed - len(prev_tail))):]
            # 参考不够填 → 拉长上一块的占用；两者合计仍不足 14s 就**短着用**
            # ——绝不补零（静音参考稀释条件是实测过的病）。代价是这种罕见
            # 场景多一次 cuDNN 形状调优；正常素材（参考≥6s）恒为 14.0s。
            if len(ref_part) + len(prev_tail) < fixed:
                prev_tail = prev_y[-(fixed - len(ref_part)):]                     if fixed - len(ref_part) <= len(prev_y) else prev_y

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

    def _score_one(path: str, text: str, emo_ref: str) -> float:
        r = _scorer.score(path, text, base["spk_audio_prompt"],
                          emo_ref_path=(emo_ref or None))
        if not r.get("ok") or r.get("reward") is None:
            return -1.0
        return float(r["reward"])

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
                for k, cp in enumerate(cand_paths):
                    if progress:
                        try:
                            progress(0.05 + 0.85 * (bi + 0.75) / n,
                                     desc=f"打分 块 {bi + 1}/{n} · 候选 {k + 1}"
                                          f"/{n_try}")
                        except Exception:
                            pass
                    rewards.append(round(_score_one(cp, blk_text,
                                                    emo_ref_path), 4))
                ok_idx = [k for k, r in enumerate(rewards) if r >= 0]
                chosen_k = (max(ok_idx, key=lambda k: rewards[k])
                            if ok_idx else 0)
                bon_info = {"n": n_try, "chosen": chosen_k,
                            "rewards": rewards}
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

            # ---- 块内停顿封顶（输出侧对称刀，2026-09-30）----
            # 模型在逗号处仍会生成 500~660ms 停顿（训练封顶管素材、
            # 底模先验管不着）——对块内静音做同样的中段压缩。块边界的
            # 台本停顿在拼接层，不受影响。
            if pause_cap_ms and int(pause_cap_ms) > 0:
                wav = AL.cap_interior_pauses(wav, SR, int(pause_cap_ms),
                                         thresh_db=-40.0)

            # ---- 逐块响度（响度指纹迁移）：从角色自己的响度分布采样目标 ----
            # 爆发块从响端、平静块从轻端 —— "这句该响那句该轻"来自角色本人
            # 的习惯（run.json 的 loudness_fingerprint），不是随机抖动。
            # 无指纹（旧 run / 纯底座）时返回 -20，行为与从前一致。
            block_lufs = block_targets[bi] if _loudness_fp else None
            if block_lufs is not None:
                y = wav.astype(np.float32) / 32768.0
                y = AL.apply_block_loudness(y, block_lufs)
                wav = np.clip(y * 32767.0, -32767, 32767).astype(np.int16)

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
                "samples": int(wav.shape[0]),
                "lines": [l.text for l in blk["lines"]],
            }
            if bon_n > 1:
                info["bon"] = bon_info
            line_infos.append(info)
            prev_chosen = cand_paths[chosen_k]

        # ---- 拼接：块间停顿（按边界等级）+ 直连时 30ms 交叉淡化 ----
        # 块间垫**数字零**：实际使用会在停顿段垫 BGM，零底最干净；人声的
        # 自然衰减在块内由模型完成（见 _merge_blocks 的收束说明），零只
        # 出现在「已经说完」之后，不存在戛然而止。
        # 吸气（2026-09）：有停顿的块边界按概率插入**角色本人**的吸气采样
        # （breath_bank），贴下一句开口放置（真人就是「吸完立刻说」）。
        # 默认关（breath=False）：启发式挖取的采样质量未经耳检，贸然常开
        # 会往听感里掺噪声（用户实测「嘈杂不干净」的嫌疑之一）。
        from webui_app.services import breath_bank as BB
        fade = int(SR * 0.03)
        inhales_used = 0
        final = wavs[0] if wavs else np.zeros(1, np.int16)
        for i in range(1, n):
            gap = int(line_infos[i - 1]["pause_after_ms"])
            nxt = wavs[i]
            if gap > 0:
                gap_n = int(SR * gap / 1000.0)
                bed = np.zeros(gap_n, dtype=np.int16)
                if breath and character:
                    try:
                        inh = BB.maybe_inhale(character, gap,
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
    sidecar = os.path.splitext(out_base)[0] + ".script.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({
                "backend": script.backend, "ok": script.ok,
                "error": script.error, "character": character,
                "routed": routed, "fallback": fallback,
                "base_seed": base_seed, "lines_in": len(script.lines),
                "blocks": n, "lines": line_infos,
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
            "bon_kept": sum(1 for x in line_infos
                            if (x.get("bon") or {}).get("best_path")),
            "avg_pause_ms": (sum(x["pause_after_ms"] for x in line_infos[:-1])
                             / max(1, n - 1)) if n > 1 else 0,
            "inhales": inhales_used,
            "sidecar": sidecar,
        },
    }
