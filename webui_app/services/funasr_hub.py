"""FunASR 模型仓库：SenseVoice（转写）与 emotion2vec（情绪）的统一生命周期。

为什么集中在一个模块：
    · 两个模型共用 funasr 这一个依赖，加载/设备选择/卸载逻辑只写一遍；
    · 8GB 显存是硬约束 —— 这里的策略与 QwenEmotionSafe 一致：**要么全
      GPU、要么全 CPU**，显存不够时明确告知并回退 CPU，绝不静默 offload
      （那种做法慢 20~30 倍且不报错）；
    · 调用方（一键三连的识别阶段 / 打分器 / 情感库打标）都是「用完就
      release」的短生命周期，仓库不常驻任何模型。

模型（首次使用自动从 ModelScope 下载，缓存在用户目录）：
    · iic/SenseVoiceSmall + fsmn-vad + ct-punc —— 中文转写质量优于
      whisper-medium（专名/标点/口语填充词），且**一条流水线附带情绪与
      音频事件标签**（<|HAPPY|> / <|laugh|>…），转写+情绪打标一次完成；
    · iic/emotion2vec_plus_large —— 句级情绪嵌入/分类（IndexTTS2 论文
      的 ES 指标即基于此族模型），供 reward 的情绪项与 BoN 择优用。
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from webui_app import logging_setup as LOG
from webui_app.config import PROJECT_ROOT

__all__ = [
    "FunASRError", "sensevoice_available", "transcribe_batch", "transcribe",
    "emotion_available", "emo_embed", "emo_classify", "release_all",
    "loaded_summary", "apply_glossary",
]

# 显存预估（GB，fp32 实测量级）。宁可高估触发 CPU 回退，不赌静默降速。
_SV_VRAM_GB = 1.6      # SenseVoiceSmall + vad + punc
_E2V_VRAM_GB = 1.2     # emotion2vec_plus_large

# SenseVoice 富转写标签 → 本项目 8 情绪键（EMO_VECTOR_KEYS）
# 没有对应原型的（无 melancholic/surprised）留空，不硬凑。
_SV_EMO_MAP = {
    "HAPPY": "happy", "ANGRY": "angry", "SAD": "sad",
    "FEARFUL": "afraid", "DISGUSTED": "disgusted", "NEUTRAL": "calm",
    "EMO_UNKNOWN": "",
}
# 音频事件标签（副语言线索，入库 meta 的 events 字段）
_SV_EVENT_RE = re.compile(r"<\|(laugh|applause|music|noise|speech|breath)\|>")

_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {"sv": None, "e2v": None,
                          "sv_device": "", "e2v_device": "",
                          "sv_loaded_at": 0.0, "e2v_loaded_at": 0.0,
                          "sv_infer": 0, "e2v_infer": 0}


class FunASRError(RuntimeError):
    """funasr 不可用 / 模型加载失败。调用方应能回退 whisper。"""


# ---------------------------------------------------------------------------
# 依赖与设备
# ---------------------------------------------------------------------------

def sensevoice_available() -> bool:
    try:
        import funasr  # noqa: F401
        return True
    except Exception:
        return False


def _pick_device(need_gb: float) -> str:
    """显存够就上 CUDA，不够回退 CPU（明确告知，不静默 offload）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free, _total = torch.cuda.mem_get_info(0)
        if free / (1024 ** 3) >= need_gb:
            return "cuda:0"
        LOG.get_logger("funasr_hub").warning(
            "空闲显存 %.2f GB < 需要 %.2f GB，FunASR 模型回退 CPU"
            "（慢但零显存风险）", free / (1024 ** 3), need_gb)
        return "cpu"
    except Exception:
        return "cpu"


def _free_vram_before(tag: str):
    try:
        from webui_app.training import guard as GD
        return GD.free_vram(tag, LOG.get_logger("funasr_hub"))
    except Exception:
        return {}


def release_all() -> Dict[str, Any]:
    """卸掉全部 FunASR 模型，归还显存。返回卸载摘要。"""
    import gc
    out: Dict[str, Any] = {}
    with _LOCK:
        for key in ("sv", "e2v"):
            if _STATE[key] is not None:
                out[key] = {"device": _STATE[key + "_device"],
                            "infers": _STATE[key + "_infer"],
                            "alive_s": round(time.time()
                                             - _STATE[key + "_loaded_at"], 1)}
                _STATE[key] = None
                _STATE[key + "_device"] = ""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    return out


def loaded_summary() -> Dict[str, Any]:
    with _LOCK:
        return {
            "sensevoice": bool(_STATE["sv"]),
            "sensevoice_device": _STATE["sv_device"],
            "emotion2vec": bool(_STATE["e2v"]),
            "emotion2vec_device": _STATE["e2v_device"],
        }


# ---------------------------------------------------------------------------
# SenseVoice：转写（富输出：文本 + 情绪 + 事件）
# ---------------------------------------------------------------------------

def _sv_model(disable_punc: bool = False):
    if _STATE["sv"] is not None:
        return _STATE["sv"]
    if not sensevoice_available():
        raise FunASRError(
            "funasr 未安装（uv pip install funasr）。装好前转写走 whisper。")
    from funasr import AutoModel

    log = LOG.get_logger("funasr_hub")
    _free_vram_before("加载 SenseVoice 前")
    device = _pick_device(_SV_VRAM_GB)
    t0 = time.perf_counter()
    log.info("加载 SenseVoiceSmall（device=%s，首次会从 ModelScope 下载）",
             device)
    kwargs: Dict[str, Any] = dict(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        disable_update=True,
    )
    if not disable_punc:
        # ct-punc：CT-Transformer 标点模型，保证中文标点丰富（，。？！）。
        # 标点=停顿=韵律，这是 A2「文本修复」的核心诉求之一。
        kwargs["punc_model"] = "ct-punc"
    try:
        _STATE["sv"] = AutoModel(**kwargs)
    except TypeError:
        kwargs.pop("disable_update", None)
        _STATE["sv"] = AutoModel(**kwargs)
    except Exception as e:
        raise FunASRError(f"SenseVoice 加载失败：{type(e).__name__}: {e}") from e
    _STATE["sv_device"] = device
    _STATE["sv_loaded_at"] = time.time()
    log.info("SenseVoiceSmall 就绪 · %.1fs · device=%s",
             time.perf_counter() - t0, device)
    return _STATE["sv"]


def _parse_rich(raw: str) -> Dict[str, Any]:
    """剥离 <|...|> 标签：文本 / 语言 / 情绪 / 事件 各归各。

    2026-09-29 关键修复：ct-punc 会输出**重复/连缀标点**（「。。」「，，」
    「。，」——真机实测 99% 的转写含此类污染）。标点是模型学习「何时停」
    的监督信号，双标点直接教出「每隔几个字停一下」的碎裂模型 —— 在解析
    层就地清洗，绝不放行。
    """
    text = raw or ""
    events = sorted(set(m.group(1) for m in _SV_EVENT_RE.finditer(text)))
    text = _SV_EVENT_RE.sub("", text)
    emo = ""
    for tag, key in _SV_EMO_MAP.items():
        if f"<|{tag}|>" in text:
            emo = key
            text = text.replace(f"<|{tag}|>", "")
            break
    text = re.sub(r"<\|[^|]*\|>", "", text)     # 其余 <|zh|> 之类
    text = sanitize_punct(re.sub(r"\s+", " ", text))
    return {"text": text, "emotion": emo, "events": events}


# 与数据集侧同一套规则（webui_app/training/dataset.py:sanitize_text_punct
# 是权威实现）；这里延迟导入避免环，失败则退化为本地最小清洗。
def sanitize_punct(text: str) -> str:
    try:
        from webui_app.training.dataset import sanitize_text_punct
        return sanitize_text_punct(text)
    except Exception:
        return re.sub(r"(?<=([，。！？；、,.!?;]))\1+", "", text or "")


def transcribe_batch(paths: List[str], lang: str = "zh",
                     use_itn: bool = True, merge_vad: bool = True,
                     disable_punc: bool = False) -> List[Dict[str, Any]]:
    """批量转写。返回每条 {text, emotion, events, raw}。

    失败抛 FunASRError（调用方决定回退 whisper 还是中止）。
    """
    m = _sv_model(disable_punc=disable_punc)
    with _LOCK:
        res = m.generate(
            input=list(paths), cache={},
            language=("zh" if (lang or "zh").lower().startswith("zh") else "auto"),
            use_itn=use_itn, batch_size_s=60,
            merge_vad=merge_vad, merge_length_s=10000,
        )
        _STATE["sv_infer"] += len(paths)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(res or []):
        raw = ""
        if isinstance(item, dict):
            raw = str(item.get("text") or "")
        out.append({**_parse_rich(raw), "raw": raw[:500],
                    "path": paths[i] if i < len(paths) else ""})
    return out


def transcribe(path: str, lang: str = "zh", **kw) -> Dict[str, Any]:
    return transcribe_batch([path], lang=lang, **kw)[0]


# ---------------------------------------------------------------------------
# emotion2vec：句级情绪嵌入 / 分类
# ---------------------------------------------------------------------------

def _e2v_model():
    if _STATE["e2v"] is not None:
        return _STATE["e2v"]
    if not sensevoice_available():
        raise FunASRError("funasr 未安装，emotion2vec 不可用")
    from funasr import AutoModel

    log = LOG.get_logger("funasr_hub")
    _free_vram_before("加载 emotion2vec 前")
    device = _pick_device(_E2V_VRAM_GB)
    t0 = time.perf_counter()
    log.info("加载 emotion2vec_plus_large（device=%s，首次会下载）", device)
    try:
        _STATE["e2v"] = AutoModel(model="iic/emotion2vec_plus_large",
                                  device=device, disable_update=True)
    except TypeError:
        _STATE["e2v"] = AutoModel(model="iic/emotion2vec_plus_large",
                                  device=device)
    except Exception as e:
        raise FunASRError(
            f"emotion2vec 加载失败：{type(e).__name__}: {e}") from e
    _STATE["e2v_device"] = device
    _STATE["e2v_loaded_at"] = time.time()
    log.info("emotion2vec_plus_large 就绪 · %.1fs · device=%s",
             time.perf_counter() - t0, device)
    return _STATE["e2v"]


def emotion_available() -> bool:
    return sensevoice_available()


# emotion2vec_plus 系列的类别表（双语标签）→ 本项目 8 情绪键
# 实测（funasr 1.4.16）：labels 是**全类别表**、scores 是对应分布、
# feats 是 1024 维句级嵌入 —— 嵌入键名不叫 embedding/embeddings。
_E2V_LABEL_MAP = {
    "生气/angry": "angry", "厌恶/disgusted": "disgusted",
    "恐惧/fearful": "afraid", "开心/happy": "happy",
    "中立/neutral": "calm", "难过/sad": "sad",
    "吃惊/surprised": "surprised", "其他/other": "", "<unk>": "",
    # 纯英文/纯中文兜底
    "angry": "angry", "disgusted": "disgusted", "fearful": "afraid",
    "happy": "happy", "neutral": "calm", "sad": "sad",
    "surprised": "surprised",
}


def e2v_label_to_key(label: str) -> str:
    """emotion2vec 的类别标签 → 8 键之一（无对应返回 ""）。"""
    s = str(label or "").strip()
    if s in _E2V_LABEL_MAP:
        return _E2V_LABEL_MAP[s]
    # 前缀兜底：标签可能带模型侧微调
    for k, v in _E2V_LABEL_MAP.items():
        if s.startswith(k):
            return v
    return ""


def emo_classify(paths: List[str]) -> List[Dict[str, Any]]:
    """句级情绪分析。返回 [{labels, scores, top, top_key, embedding}]。

    实测返回结构（funasr 1.4.16，emotion2vec_plus_large）：
        labels = 全类别表（9 类，含「<unk>」「其他/other」）
        scores = 与 labels 对齐的分布
        feats  = 1024 维句级嵌入
    不同版本做过防御式兼容（embedding/embeddings 旧键名也认）。
    """
    import numpy as np

    m = _e2v_model()
    with _LOCK:
        res = m.generate(list(paths), granularity="utterance",
                         extract_embedding=True)
        _STATE["e2v_infer"] += len(paths)
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(res or []):
        d: Dict[str, Any] = {"labels": [], "scores": None, "top": "",
                             "top_key": "", "embedding": None,
                             "path": paths[i] if i < len(paths) else ""}
        if isinstance(item, dict):
            labs = item.get("labels")
            d["labels"] = [str(x) for x in labs] if isinstance(
                labs, (list, tuple)) else []
            scores = item.get("scores", item.get("probs"))
            if scores is not None:
                try:
                    arr = np.asarray(scores, dtype=float).reshape(-1)
                    d["scores"] = arr.tolist()
                    if len(arr) == len(d["labels"]):
                        top_i = int(np.argmax(arr))
                        d["top"] = d["labels"][top_i]
                        d["top_key"] = e2v_label_to_key(d["top"])
                except Exception:
                    pass
            emb = item.get("feats")
            if emb is None:
                emb = item.get("embedding")
            if emb is None:
                emb = item.get("embeddings")
            if emb is not None:
                try:
                    arr2 = emb.detach().cpu().numpy() if hasattr(emb, "detach") \
                        else np.asarray(emb)
                    d["embedding"] = arr2.reshape(-1)
                except Exception:
                    pass
        out.append(d)
    return out


def emo_embed(path: str):
    """单条 → (D,) 情绪嵌入（np.ndarray）。取不到嵌入时抛 FunASRError。"""
    r = emo_classify([path])[0]
    if r["embedding"] is None:
        raise FunASRError("emotion2vec 未返回嵌入（funasr 版本差异）")
    return r["embedding"]


def emo_cosine(a_path: str, b_path: str) -> float:
    """两条音频的情绪嵌入余弦 —— reward 情绪项 / BoN 择优的核心量。"""
    import numpy as np
    a, b = emo_embed(a_path), emo_embed(b_path)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# 热词后纠（专名纠错）：拼音等价替换
# ---------------------------------------------------------------------------

def apply_glossary(text: str, names: List[str]) -> Tuple[str, List[str]]:
    """把转写里与「标准名」拼音相同/相近的片段替换回标准名。

    背景：ASR 对游戏专名（卡提希娅/弗洛德利斯）常写出同音错字
    （哈提西亚）。SenseVoice 没有热词接口，这里用拼音比对做后纠。

    匹配单位是**滑窗**而不是整段汉字：错字嵌在长句中间
    （“哈提西亚对弗洛德利斯说话”），整段比对永远匹配不上。
    每个汉字恰好对应一个音节，所以「字数 == 名字字数」的窗口
    拼出来正好与名字拼音同长 —— 精确相等或编辑距离 ≤1 才替换，
    宁缺毋滥。
    返回 (纠正后文本, [被纠正的标准名])。
    """
    names = [n.strip() for n in (names or []) if n and n.strip()]
    if not names or not text:
        return text, []

    from pypinyin import lazy_pinyin

    def _py(s: str) -> str:
        return "".join(lazy_pinyin(s))

    han = re.compile(r"[\u4e00-\u9fff]+")
    changed: List[str] = []
    fixed_notes: List[str] = []

    def _sub(m: re.Match) -> str:
        seg = m.group(0)
        chars = list(seg)
        pys = lazy_pinyin(seg)                # 每字一音节，与 chars 对齐
        for name in names:
            n_len = len(name)
            if len(chars) < max(2, n_len - 1):
                continue
            tp = _py(name)
            for w in (n_len, n_len + 1, n_len - 1):
                if w < 2 or w > len(chars):
                    continue
                for i in range(0, len(chars) - w + 1):
                    piece = "".join(pys[i:i + w])
                    cand = "".join(chars[i:i + w])
                    if cand == name:
                        continue              # 已经是对的
                    if piece == tp or (len(tp) >= 6 and _edit1(piece, tp)):
                        out = (seg[:i] + name + seg[i + w:])
                        if name not in changed:
                            changed.append(name)
                        fixed_notes.append(f"{cand}→{name}")
                        return out
        return seg

    out = han.sub(_sub, text)
    return out, changed


def _edit1(a: str, b: str) -> bool:
    """汉明式近似：等长允许 ≤1 处不同；差 1 长允许单插/删。"""
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if abs(len(a) - len(b)) == 1:
        if len(a) > len(b):
            a, b = b, a
        i = 0
        while i < len(a) and a[i] == b[i]:
            i += 1
        return a[i:] == b[i + 1:]
    return False
