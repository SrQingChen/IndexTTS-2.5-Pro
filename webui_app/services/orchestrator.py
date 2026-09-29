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
) -> Dict[str, Any]:
    """逐句合成并拼接。返回结构对齐 INF.generate()，额外带导演信息。

    bon_n > 0 时启用**逐句择优（Best-of-N）**：每句合成 N 个候选
    （不同种子），用 reward 打分选最好的一条 —— 「以时间换质量」的
    落点（文献背书：arXiv 2608.31035，BoN-8 ≈ GRPO 训练收益）。
    打分器整体跑在 **CPU**（SenseVoice 转写 + campplus 声纹 +
    emotion2vec 情绪 + 停顿启发），与 GPU 上的推理引擎零显存竞争；
    代价是每句多花 N×(合成+CPU打分) 的时间，适合离线精修。

    Raises:
        EngineError: 输入不合法 / 某一句合成失败（含句号与临时目录位置）
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
    # GPU 上的推理引擎零显存竞争。权重设计：读得对 0.35 / 音色像 0.25 /
    # 情绪贴合 0.30（路由命中才有）/ 停顿人味 0.10。
    _scorer = None
    if bon_n > 0:
        from webui_app.training import reward as RW
        _scorer = RW.RewardScorer(RW.RewardOptions(
            asr_engine="sensevoice", whisper_size="small", device="cpu",
            wer_weight=0.35, sim_weight=0.25,
            emo_weight=0.30, pause_weight=0.10))
        log.info("BoN 择优开启：N=%d · 打分器 CPU（sensevoice+campplus"
                 "+emotion2vec+停顿启发）", bon_n)

    def _score_one(path: str, text: str, emo_ref: str) -> float:
        r = _scorer.score(path, text, base["spk_audio_prompt"],
                          emo_ref_path=(emo_ref or None))
        if not r.get("ok") or r.get("reward") is None:
            return -1.0
        return float(r["reward"])

    try:
        for i, ln in enumerate(script.lines):
            if progress:
                try:
                    progress(0.05 + 0.85 * i / n,
                             desc=f"导演编排 {i + 1}/{n} · {ln.emotion}")
                except Exception:
                    pass

            entry = EB.pick(character, ln.emotion) if route else None
            if route:
                if entry is not None:
                    routed += 1
                else:
                    fallback += 1

            emo_kw = _line_emo_kwargs(req, entry, ln.intensity, extrapolate)
            line_seed = (base_seed + 977 * (i + 1)) % (2 ** 31)
            emo_ref_path = entry.audio_path if entry is not None else ""

            # ---- 合成：单候选，或 BoN 的 N 个候选 ----
            cand_paths: List[str] = []
            cand_seeds: List[int] = []
            n_try = max(1, bon_n)
            for k in range(n_try):
                cand_seed = (line_seed + 131 * k) % (2 ** 31)
                INF.apply_seed(cand_seed)
                cpath = os.path.join(run_dir, f"line_{i:04d}_c{k}.wav")
                kw = dict(base)
                kw["text"] = ln.text
                kw["output_path"] = cpath
                # 句内若再被切分（超长句），沿用用户的全局句内静音
                kw.update(emo_kw)
                if progress and bon_n > 0:
                    try:
                        progress(0.05 + 0.85 * (i + 0.5 * (k + 1) / n_try) / n,
                                 desc=f"导演编排 {i + 1}/{n} · 候选 {k + 1}"
                                      f"/{n_try} · {ln.emotion}")
                    except Exception:
                        pass
                try:
                    ret = engine.infer(progress=None, **kw)
                except Exception as e:
                    raise EngineError(
                        f"第 {i + 1}/{n} 句合成失败（情绪 {ln.emotion}，"
                        f"候选 {k + 1}/{n_try}）："
                        f"{type(e).__name__}: {e}。临时句音频保留在 {run_dir}")
                actual = ret if isinstance(ret, str) else cpath
                if not os.path.isfile(actual):
                    raise EngineError(
                        f"第 {i + 1}/{n} 句没有产出音频（文本可能为空或被"
                        f"提前终止）：{ln.text[:40]}…")
                cand_paths.append(actual)
                cand_seeds.append(cand_seed)

            # ---- 择优 ----
            rewards: List[float] = []
            chosen_k = 0
            bon_info: Dict[str, Any] = {}
            if bon_n > 1:
                for k, cp in enumerate(cand_paths):
                    if progress:
                        try:
                            progress(0.05 + 0.85 * (i + 0.75) / n,
                                     desc=f"打分 {i + 1}/{n} · 候选 {k + 1}"
                                          f"/{n_try}")
                        except Exception:
                            pass
                    rewards.append(round(_score_one(cp, ln.text, emo_ref_path), 4))
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
                        bp = os.path.join(bon_dir, f"line_{i:04d}_best.wav")
                        wp = os.path.join(bon_dir, f"line_{i:04d}_worst.wav")
                        shutil.copy2(cand_paths[chosen_k], bp)
                        shutil.copy2(cand_paths[worst_k], wp)
                        bon_info.update({
                            "best_path": bp, "worst_path": wp,
                            "best_k": chosen_k, "worst_k": worst_k,
                            "reward_best": rewards[chosen_k],
                            "reward_worst": rewards[worst_k],
                        })
                    except OSError as e:
                        log.warning("BoN 候选持久化失败（第 %d 句）：%s",
                                    i + 1, e)
                # 落选候选即删（run_dir 反正要清，删掉省磁盘峰值）
                for k, cp in enumerate(cand_paths):
                    if k != chosen_k:
                        try:
                            os.remove(cp)
                        except OSError:
                            pass

            wav = _read_mono_int16(cand_paths[chosen_k])
            wavs.append(wav)
            info = {
                "idx": i + 1, "text": ln.text, "emotion": ln.emotion,
                "intensity": ln.intensity, "pause_after_ms": ln.pause_after_ms,
                "seed": cand_seeds[chosen_k],
                "emo_ref": entry.name if entry else "",
                "emo_alpha": (emo_kw["emo_alpha"] if entry is not None else None),
                "samples": int(wav.shape[0]),
            }
            if bon_n > 1:
                info["bon"] = bon_info
            line_infos.append(info)

        # ---- 拼接：句间插入台本停顿（最后一句不再加） ----
        parts: List[np.ndarray] = []
        for i, wav in enumerate(wavs):
            parts.append(wav)
            if i < n - 1:
                gap = int(SR * script.lines[i].pause_after_ms / 1000.0)
                parts.append(np.zeros(gap, dtype=np.int16))
        final = np.concatenate(parts, axis=0) if parts else np.zeros(1, np.int16)

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
                "base_seed": base_seed, "lines": line_infos,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        log.warning("台本旁车文件写失败：%s", sidecar, exc_info=True)

    # kwargs 摘要（合成页结果表用）：把整篇单次的情感字段替换为编排统计
    kw_summary = {k: v for k, v in base.items()
                  if k not in ("text", "output_path", "emo_audio_prompt",
                               "emo_alpha", "emo_vector")}
    log.info("编排完成：%d 句 · 路由 %d / 回退 %d · %.1fs · %s",
             n, routed, fallback, seconds, os.path.basename(out_base))

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
            "bon_kept": sum(1 for x in line_infos
                            if (x.get("bon") or {}).get("best_path")),
            "avg_pause_ms": (sum(l.pause_after_ms for l in script.lines[:-1])
                             / max(1, n - 1)) if n > 1 else 0,
            "sidecar": sidecar,
        },
    }
