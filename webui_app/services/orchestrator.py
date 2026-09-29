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
        """台本行 → 表演块：情绪相同的相邻行合并，≤120 文本 token。

        与官方 split_text_by_tokens 同预算；情绪突变 / 预算到顶断块。
        **省略号/破折号结尾强制断块**（2026-09-29）：那是戏剧性停顿拍，
        必须成为块边界才能拿到台本的长停顿——留在块内会被模型读成
        普通逗号级停顿（实测「……」比句内停顿还短）。
        块内文本一次合成调用 —— 这是「块内零人工静音、节奏交还模型」
        的机制保证。
        """
        blocks: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None

        def _close():
            nonlocal cur
            if cur is not None:
                blocks.append(cur)
                cur = None

        for ln in script.lines:
            cost = _count_tokens(ln.text)
            if (cur is not None and ln.emotion == cur["emotion"]
                    and cur["tokens"] + cost <= 120):
                cur["lines"].append(ln)
                cur["tokens"] += cost
                cur["intensity"] = max(cur["intensity"], ln.intensity)
                joiner = "" if cur["text"][-1:] in "。！？…！？.!?\"" else "。"
                cur["text"] = cur["text"] + joiner + ln.text
            else:
                _close()
                cur = {"emotion": ln.emotion, "lines": [ln], "tokens": cost,
                       "intensity": ln.intensity, "text": ln.text}
            # 戏剧性停顿拍：……/—— 结尾的行绝不与后文同块
            if cur is not None and cur["text"].endswith(("……", "…", "——", "—")):
                _close()
        if cur is not None:
            blocks.append(cur)
        return blocks

    blocks = _merge_blocks()
    n = len(blocks)
    log.info("表演块合并：%d 行 → %d 块（≤120 token/块，情绪一致）",
             len(script.lines), n)

    def _score_one(path: str, text: str, emo_ref: str) -> float:
        r = _scorer.score(path, text, base["spk_audio_prompt"],
                          emo_ref_path=(emo_ref or None))
        if not r.get("ok") or r.get("reward") is None:
            return -1.0
        return float(r["reward"])

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
                # 块内零人工静音的保证：块 ≤120 token 时官方不会再切，
                # 即便计数误差触发了内部分段，也把垫音压到最小
                kw["interval_silence"] = min(int(base.get("interval_silence", 200)
                                                 or 200), 120)
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
                "samples": int(wav.shape[0]),
                "lines": [l.text for l in blk["lines"]],
            }
            if bon_n > 1:
                info["bon"] = bon_info
            line_infos.append(info)

        # ---- 拼接：块间停顿（按边界等级）+ 直连时 30ms 交叉淡化 ----
        # 块间垫**数字零**：实际使用会在停顿段垫 BGM，零底最干净；人声的
        # 自然衰减在块内由模型完成（见 _merge_blocks 的收束说明），零只
        # 出现在「已经说完」之后，不存在戛然而止。
        fade = int(SR * 0.03)
        final = wavs[0] if wavs else np.zeros(1, np.int16)
        for i in range(1, n):
            gap = int(line_infos[i - 1]["pause_after_ms"])
            nxt = wavs[i]
            if gap > 0:
                final = np.concatenate(
                    [final, np.zeros(int(SR * gap / 1000.0), dtype=np.int16), nxt])
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
            "sidecar": sidecar,
        },
    }
