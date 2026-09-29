"""呼吸声库（Breath Bank）：从角色素材挖「本人的吸气采样」，编排器在
块边界按停顿层级插入 —— 「真人感」高感知维度里投入产出比最高的一件。

证据锚点（2026-09 调研）：给合成句插入**真实吸气声**显著提升听者回忆
成绩，频谱等效的非呼吸噪声无效（Whalen et al. 1995, JASA）——起作用的
是「这个人在呼吸」的识别。因此采样必须来自**角色本人**（音色连续），
且只插**块边界（句子开始前）**的吸气；呼气/叹气是表演决策，入副语言
库按剧本用，不自动插。

检测：MVP 用能量+频谱斜率启发式（吸气 = 语音前 150~600ms 的低幅段，
能量显著低于后续语音且高频占比偏高）。不依赖外部模型，随时可跑；
后续可换 Respiro-en（arXiv:2402.00288）逐帧检测提精度 —— 接口不变。
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from webui_app import logging_setup as LOG
from webui_app.config import PROJECT_ROOT
from webui_app.services import audio_lab as AL

BANK_DIR = os.path.join(PROJECT_ROOT, "breath_bank")
INDEX_FILE = os.path.join(BANK_DIR, "index.json")
AUDIO_SUBDIR = "audio"

SR = 22050


def safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "untitled"


# ---------------------------------------------------------------------------
# 吸气段检测（启发式 MVP）
# ---------------------------------------------------------------------------

def _frames(y: np.ndarray, sr: int, frame_ms: float = 20.0):
    n = int(len(y) / (sr * frame_ms / 1000.0))
    if n < 2:
        return None
    fr = int(sr * frame_ms / 1000.0)
    seg = y[: n * fr].reshape(n, fr)
    rms = np.sqrt(np.mean(seg.astype(np.float64) ** 2, axis=1) + 1e-12)
    # 高频占比：每帧一阶差分能量 / 总能量（吸气气流噪声的粗糙代理）
    diff = np.sqrt(np.mean(np.diff(seg, axis=1).astype(np.float64) ** 2,
                           axis=1) + 1e-12)
    return rms, diff / (rms + 1e-12)


def detect_inhales(path: str, max_take: int = 6) -> List[Dict[str, Any]]:
    """一条素材 → 吸气候选段列表 [{start, end, score}]。

    判据：某段帧 (a) RMS 低于全片语音中位 RMS 的 25%，(b) 粗糙度高于
    中位（气流噪声），(c) 时长 150~600ms，(d) 后面 300ms 内有语音
    （吸气是「为说话抢气」，孤立的静音不算）。
    """
    try:
        y, sr = AL.load_audio(path)
    except Exception:
        return []
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    if len(y) < SR * 0.5:
        return []
    res = _frames(y, sr)
    if res is None:
        return []
    rms, rough = res
    speech_med = float(np.median(rms[rms > np.percentile(rms, 60)]))
    rough_med = float(np.median(rough))
    if speech_med <= 0:
        return []
    quiet = (rms < speech_med * 0.25) & (rough > rough_med)
    frame_ms = 20.0
    out: List[Dict[str, Any]] = []
    i, n = 0, len(rms)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            dur = (j - i) * frame_ms / 1000.0
            # 吸气后 300ms 内要有语音
            look = rms[j: j + int(300 / frame_ms)]
            if 0.15 <= dur <= 0.6 and len(look) and \
                    float(np.max(look)) > speech_med * 0.5:
                depth = float(1.0 - np.mean(rms[i:j]) / speech_med)
                air = float(np.mean(rough[i:j]) / max(rough_med, 1e-9))
                out.append({"start": round(i * frame_ms / 1000.0, 3),
                            "end": round(j * frame_ms / 1000.0, 3),
                            "score": round(min(1.0, 0.6 * depth + 0.4 * air), 3)})
            i = j
        else:
            i += 1
    out.sort(key=lambda d: -d["score"])
    return out[:max_take]


# ---------------------------------------------------------------------------
# 库
# ---------------------------------------------------------------------------

@dataclass
class BreathEntry:
    name: str
    character: str
    audio: str                    # 相对 BANK_DIR
    duration: float = 0.0
    score: float = 0.0
    source: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def audio_path(self) -> str:
        return os.path.join(BANK_DIR, self.audio) if self.audio else ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load() -> List[Dict[str, Any]]:
    if not os.path.isfile(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save(items: List[Dict[str, Any]]):
    os.makedirs(BANK_DIR, exist_ok=True)
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_FILE)


def list_entries(character: str = "") -> List[BreathEntry]:
    out = []
    for d in _load():
        e = BreathEntry(**{k: v for k, v in d.items()
                           if k in BreathEntry.__dataclass_fields__})
        if e.audio and not os.path.isfile(e.audio_path):
            continue
        if character and e.character != character:
            continue
        out.append(e)
    out.sort(key=lambda x: -x.score)
    return out


def build_from_dataset(character: str, dataset: str,
                       per_clip: int = 2, min_score: float = 0.35,
                       progress=None) -> Dict[str, Any]:
    """从数据集素材挖吸气采样入库（角色本人的呼吸，音色连续）。

    音频复制进库并**响度归一到 -26dBFS**（插入时相对台词 -24~-30dBFS，
    这里先统一到一个中性电平，插入端再调）。返回统计。
    """
    from webui_app.training import dataset as DS

    character = (character or "").strip()
    if not character:
        return {"ok": False, "error": "角色名为空"}
    added, scanned = 0, 0
    os.makedirs(os.path.join(BANK_DIR, AUDIO_SUBDIR), exist_ok=True)
    items = DS.load_meta(dataset)
    for i, u in enumerate(items):
        ap = u.audio_abs(DS.dir_of(dataset))
        if not (ap and os.path.isfile(ap)):
            continue
        scanned += 1
        if progress:
            try:
                progress(i / max(1, len(items)),
                         f"挖呼吸 {i + 1}/{len(items)} · {u.id}")
            except Exception:
                pass
        for k, seg in enumerate(detect_inhales(ap, max_take=per_clip)):
            if seg["score"] < min_score:
                continue
            try:
                y, sr = AL.load_audio(ap)
                a = int(seg["start"] * sr)
                b = int(seg["end"] * sr)
                piece = y[max(0, a - int(0.05 * sr)): b + int(0.05 * sr)]
                piece = _fade_edges(piece, sr, 15)
                piece = _to_level(piece, -26.0)
                name = f"{safe_name(character)}_{i:03d}_{k}"
                rel = os.path.join(AUDIO_SUBDIR, name + ".wav").replace("\\", "/")
                dst = os.path.join(BANK_DIR, rel)
                AL.save_audio(dst, piece, SR)
                items_idx = _load()
                items_idx = [d for d in items_idx if d.get("name") != name]
                items_idx.append(BreathEntry(
                    name=name, character=character, audio=rel,
                    duration=round(len(piece) / SR, 3),
                    score=seg["score"], source=u.id).to_dict())
                _save(items_idx)
                added += 1
            except Exception as e:
                LOG.get_logger("breath_bank").warning(
                    "呼吸采样 %s_%d 入库失败：%s", u.id, k, e)
    return {"ok": True, "scanned": scanned, "added": added}


def remove(name: str) -> bool:
    key = safe_name(name)
    items = _load()
    kept = [d for d in items if d.get("name") != key]
    if len(kept) == len(items):
        return False
    for d in items:
        if d.get("name") == key and d.get("audio"):
            try:
                p = os.path.join(BANK_DIR, d["audio"])
                os.path.isfile(p) and os.remove(p)
            except OSError:
                pass
    _save(kept)
    return True


# ---------------------------------------------------------------------------
# 插入（编排器调用）
# ---------------------------------------------------------------------------

def maybe_inhale(character: str, gap_ms: int, intensity: float,
                 rng: random.Random) -> Optional[np.ndarray]:
    """决定并在块边界返回一段吸气波形（int16 @SR），不插返回 None。

    规则（对齐真人统计）：
        · 只在「有停顿的块边界」考虑（gap≥200ms —— 刚抢完气才需要再吸）；
        · 短句边界不插（人不会每句都大换气）：插入概率 ~50%，情绪强度
          高（气口急）概率略升；
        · 位置抖动由调用方拼接时的余白承担，这里返回整段。
    """
    if gap_ms < 200:
        return None
    if rng.random() > (0.45 + 0.15 * max(0.0, min(1.0, intensity))):
        return None
    entries = list_entries(character)
    if not entries:
        return None
    e = entries[rng.randrange(min(4, len(entries)))]
    try:
        y, sr = AL.load_audio(e.audio_path)
        if sr != SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        # 目标电平：相对台词 -24~-30dBFS 随机。load_audio 返回 ±1 域
        # float32，电平化后一次转 int16。
        target = rng.uniform(-30.0, -24.0)
        y = _to_level(np.asarray(y, dtype=np.float32), target)
        return np.clip(y * 32767.0, -32767, 32767).astype(np.int16)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _fade_edges(y: np.ndarray, sr: int, ms: float) -> np.ndarray:
    f = max(1, int(sr * ms / 1000.0))
    out = y.copy()
    if len(out) > 2 * f:
        out[:f] *= np.linspace(0.0, 1.0, f, dtype=y.dtype)
        out[-f:] *= np.linspace(1.0, 0.0, f, dtype=y.dtype)
    return out


def _to_level(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    cur = AL.rms_dbfs(y)
    if cur <= -100:
        return y
    out = y * 10 ** ((target_dbfs - cur) / 20.0)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 10 ** (-1.0 / 20.0):
        out = out * (10 ** (-1.0 / 20.0) / peak)
    return out.astype(np.float32)


def table_markdown(character: str = "") -> str:
    es = list_entries(character)
    if not es:
        return ("_呼吸库为空。_ 用「🎭 情感参考库」页的「从数据集挖呼吸」"
                "或编排器自动提示，把角色本人的吸气采样入库（导演模式"
                "会在块边界自动插入）。")
    lines = ["| 采样 | 角色 | 时长 | 质量分 | 来源 |", "|---|---|---|---|---|"]
    for e in es:
        lines.append(f"| `{e.name}` | {e.character} | {e.duration:.2f}s | "
                     f"{e.score:.2f} | {e.source or '-'} |")
    return "\n".join(lines)
