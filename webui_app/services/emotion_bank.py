"""情感参考库（Emotion Bank）。

与音色库（voice_bank）的分工：
    · 音色库条目 = 「谁在说」的参考音频（CAMPPlus 声纹 / ref_mel 模板）
    · 情感参考库条目 = 「怎么说的」的参考音频 —— 按角色×情绪打标，
      供导演模式逐句路由（infer 的 emo_audio_prompt 路径）。

动机（2026-09 表现力调研结论）：默认模式下情感完全来自音色参考音频本身
（infer_v2_5 里 emo_alpha 被强制 1.0），一条平稳参考 = 整篇平淡。
上游 issue #321 的社区结论是「情感参考音频 + emo_alpha≈0.6」是
表现力/音色相似度的平衡点 —— 本库就是那个「情感参考音频」的数据源。

条目结构沿用 voice_bank 的约定（index.json + audio/ 子目录 + 体检报告），
方便复用清理页 / 工作台的既有习惯。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from webui_app.config import PROJECT_ROOT, EMO_VECTOR_KEYS, EMO_VECTOR_LABELS
from webui_app.services import audio_lab as AL

BANK_DIR = os.path.join(PROJECT_ROOT, "emotion_bank")
INDEX_FILE = os.path.join(BANK_DIR, "index.json")
AUDIO_SUBDIR = "audio"

# 情绪键与官方 8 维向量一一对应（顺序同 EMO_VECTOR_KEYS）
EMOTION_KEYS = list(EMO_VECTOR_KEYS)            # happy/angry/...
EMOTION_LABELS = dict(zip(EMO_VECTOR_KEYS, EMO_VECTOR_LABELS))   # happy→喜


def safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "untitled"


def norm_emotion(v: str) -> str:
    """把任意输入归一化成 8 个官方情绪键之一；认不出的返回 ""。"""
    s = (v or "").strip().lower()
    if not s:
        return ""
    if s in EMOTION_KEYS:
        return s
    # 中文标签反查（喜/怒/哀/惧/厌恶/低落/惊喜/平静）
    for k, zh in EMOTION_LABELS.items():
        if s == zh or s == zh.lower():
            return k
    return ""


@dataclass
class EmoRefEntry:
    """一条情感参考：某角色的某种情绪的示范音频。"""

    name: str                          # 全局唯一：角色_情绪_序号
    character: str = ""                # 角色名（路由键，建议与音色库条目同名）
    emotion: str = "calm"              # 8 键之一
    audio: str = ""                    # 相对 BANK_DIR 的路径
    note: str = ""
    lang: str = "ZH"
    created_at: float = field(default_factory=time.time)
    source: str = ""
    score: float = 0.0
    grade: str = "-"
    duration: float = 0.0
    sample_rate: int = 0
    snr_db: float = 0.0
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def audio_path(self) -> str:
        return os.path.join(BANK_DIR, self.audio) if self.audio else ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 索引读写（与 voice_bank 相同的原子写法）
# ---------------------------------------------------------------------------

def _ensure_dir():
    os.makedirs(BANK_DIR, exist_ok=True)
    os.makedirs(os.path.join(BANK_DIR, AUDIO_SUBDIR), exist_ok=True)


def _load_index() -> List[Dict[str, Any]]:
    if not os.path.isfile(INDEX_FILE):
        return []
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_index(items: List[Dict[str, Any]]):
    _ensure_dir()
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_FILE)


def list_entries() -> List[EmoRefEntry]:
    """全部条目（角色→情绪排序），顺手剔除音频已丢失的死条目。"""
    out: List[EmoRefEntry] = []
    stale = False
    for d in _load_index():
        e = EmoRefEntry(**{k: v for k, v in d.items()
                           if k in EmoRefEntry.__dataclass_fields__})
        if e.audio and not os.path.isfile(e.audio_path):
            stale = True
            continue
        out.append(e)
    if stale:
        _save_index([e.to_dict() for e in out])
    out.sort(key=lambda x: (x.character.lower(), x.emotion, -x.score))
    return out


def characters() -> List[str]:
    seen, out = set(), []
    for e in list_entries():
        if e.character and e.character not in seen:
            seen.add(e.character)
            out.append(e.character)
    return out


def get(name: str) -> Optional[EmoRefEntry]:
    for e in list_entries():
        if e.name == name:
            return e
    return None


# ---------------------------------------------------------------------------
# 路由（导演模式逐句取参考的核心）
# ---------------------------------------------------------------------------

def pick(character: str, emotion: str) -> Optional[EmoRefEntry]:
    """按（角色, 情绪）取最佳情感参考。

    回退链（两条安全边界）：
        1. **只在同角色内回退** —— 跨角色情感参考会把别人的音色气息带
           进来，宁可退回模式 0 也不用错人；
        2. **平静句不做情绪借用** —— calm 没有精确命中时直接返回 None
           （跟随音色参考）。给平静台词挂一条愤怒参考是有害的：同角色
           借用只服务「想演但没归档该情绪」的句子。
    """
    emo = norm_emotion(emotion)
    entries = [e for e in list_entries()
               if e.character and e.character == (character or "").strip()]
    if not entries:
        return None
    exact = [e for e in entries if e.emotion == emo] if emo else []
    if exact:
        return max(exact, key=lambda e: (e.score, e.duration))
    if emo == "calm":
        return None
    return max(entries, key=lambda e: (e.score, e.duration))


def route_summary(character: str) -> Dict[str, Any]:
    """给 UI 的路由预览：该角色已有哪些情绪、缺哪些。"""
    entries = [e for e in list_entries()
               if e.character == (character or "").strip()]
    have = sorted({e.emotion for e in entries})
    return {
        "character": character,
        "n": len(entries),
        "have": have,
        "missing": [k for k in EMOTION_KEYS if k not in have],
    }


# ---------------------------------------------------------------------------
# 增删改
# ---------------------------------------------------------------------------

def add(
    character: str,
    emotion: str,
    audio_path: str,
    note: str = "",
    lang: str = "ZH",
    analyze_audio: bool = True,
) -> EmoRefEntry:
    """把一段音频按 角色×情绪 入库（复制进库目录并体检）。"""
    _ensure_dir()
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    char = safe_name(character)
    emo = norm_emotion(emotion)
    if not char or char == "untitled":
        raise ValueError("角色名不能为空")
    if not emo:
        raise ValueError(
            f"情绪必须是 8 键之一（{'/'.join(EMOTION_KEYS)}），收到：{emotion!r}")

    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    # 同角色同情绪允许多条（路由自动取评分最高），文件名加序号防覆盖。
    # 入库即 wav（2026-09-29）：非 wav 转码（引擎 torchaudio 读 mp3 不可靠），
    # 转码失败按原扩展名保留（绝不把 mp3 内容拷进 .wav 名）
    seq = 1
    while True:
        name = f"{char}_{emo}_{seq:02d}" if seq > 1 else f"{char}_{emo}"
        rel = os.path.join(AUDIO_SUBDIR, name + ".wav").replace("\\", "/")
        dst = os.path.join(BANK_DIR, rel)
        if not os.path.isfile(dst) or os.path.abspath(dst) == os.path.abspath(audio_path):
            break
        seq += 1
    if os.path.abspath(dst) != os.path.abspath(audio_path):
        if ext == ".wav":
            shutil.copy2(audio_path, dst)
        else:
            try:
                AL.ensure_wav(audio_path, dst)
            except Exception:
                rel = os.path.join(AUDIO_SUBDIR,
                                   name + ext).replace("\\", "/")
                dst = os.path.join(BANK_DIR, rel)
                shutil.copy2(audio_path, dst)

    entry = EmoRefEntry(
        name=name, character=character.strip(), emotion=emo, audio=rel,
        note=note or "", lang=lang or "ZH",
        source=os.path.basename(audio_path),
    )

    if analyze_audio:
        rep = AL.analyze(dst)
        if not rep.ok:
            # 体检不过不入库（同 voice_bank 的理由：坏素材进库只会在
            # 路由命中时让合成失败，比当场报错难查得多）
            try:
                if os.path.abspath(dst) != os.path.abspath(audio_path):
                    os.remove(dst)
            except OSError:
                pass
            raise ValueError(f"这个音频无法入库：{rep.error or '体检未通过'}")
        entry.report = rep.to_dict()
        entry.score = rep.score
        entry.grade = rep.grade
        entry.duration = rep.duration
        entry.sample_rate = rep.sample_rate
        entry.snr_db = rep.snr_db

    items = _load_index()
    items = [d for d in items if d.get("name") != name]
    items.append(entry.to_dict())
    _save_index(items)
    return entry


def remove(name: str) -> bool:
    key = safe_name(name)
    items = _load_index()
    kept = [d for d in items if d.get("name") != key]
    if len(kept) == len(items):
        return False
    for d in items:
        if d.get("name") == key and d.get("audio"):
            p = os.path.join(BANK_DIR, d["audio"])
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
    _save_index(kept)
    return True


def update(name: str, **fields) -> Optional[EmoRefEntry]:
    """更新 note / emotion / character 等元数据。"""
    items = _load_index()
    key = safe_name(name)
    changed = None
    for d in items:
        if d.get("name") == key:
            for k, v in fields.items():
                if k == "emotion":
                    v = norm_emotion(v) or d.get("emotion")
                if k in EmoRefEntry.__dataclass_fields__ and k not in ("name", "audio"):
                    d[k] = v
            changed = d
    if changed:
        _save_index(items)
        return EmoRefEntry(**{k: v for k, v in changed.items()
                              if k in EmoRefEntry.__dataclass_fields__})
    return None


def reanalyze(name: str) -> Optional[EmoRefEntry]:
    e = get(name)
    if e is None:
        return None
    rep = AL.analyze(e.audio_path)
    return update(
        name, report=rep.to_dict(), score=rep.score, grade=rep.grade,
        duration=rep.duration, sample_rate=rep.sample_rate, snr_db=rep.snr_db,
    )


# ---------------------------------------------------------------------------
# 自动打标（emotion2vec）
# ---------------------------------------------------------------------------

def suggest_emotions(paths: List[str]) -> List[Dict[str, Any]]:
    """一批音频 → 每条的建议情绪（emotion2vec 句级分类的 top 标签）。

    返回 [{path, key(8键之一或""), top(原始标签), ok, error}]。
    模型用完**不在这里卸**（调用方批量入库完统一 release_all ——
    逐条卸载会把同一模型反复加载，8GB 卡上纯属自虐）。
    """
    from webui_app.services import funasr_hub as FH

    out: List[Dict[str, Any]] = []
    try:
        rs = FH.emo_classify(list(paths))
    except Exception as e:
        return [{"path": p, "key": "", "top": "", "ok": False,
                 "error": f"{type(e).__name__}: {e}"} for p in paths]
    for r in rs:
        out.append({
            "path": r.get("path", ""), "key": r.get("top_key", ""),
            "top": r.get("top", ""), "ok": True, "error": "",
        })
    return out


def autotag_add(character: str, paths: List[str],
                fallback_emotion: str = "calm") -> Dict[str, Any]:
    """批量自动打标入库：emotion2vec 定情绪 → 体检入库。

    分不出情绪的（top_key 为空，如「其他/other」）按 fallback_emotion 入库。
    返回 {added: [names], failed: [{path, error}]}，并在最后释放模型。
    """
    from webui_app.services import funasr_hub as FH

    character = (character or "").strip()
    added: List[str] = []
    failed: List[Dict[str, str]] = []
    if not character:
        return {"added": added, "failed": [{"path": p, "error": "角色名为空"}
                                           for p in paths]}
    if paths:
        sugg = suggest_emotions(paths)
        for s in sugg:
            emo = s.get("key") or fallback_emotion
            try:
                e = add(character, emo, s["path"],
                        note=(f"自动打标：{s.get('top') or 'fallback'}"
                              if s.get("ok") else f"打分失败回退：{s.get('error')}"))
                added.append(f"{e.name}({emo})")
            except Exception as ex:
                failed.append({"path": s["path"],
                               "error": f"{type(ex).__name__}: {ex}"})
    try:
        FH.release_all()
    except Exception:
        pass
    return {"added": added, "failed": failed}


# ---------------------------------------------------------------------------
# 展示
# ---------------------------------------------------------------------------

def table_markdown(entries: Optional[List[EmoRefEntry]] = None) -> str:
    entries = entries if entries is not None else list_entries()
    if not entries:
        return ("_情感参考库为空。_\n\n"
                "到「🎭 情感参考库」页，把角色最有戏的几句台词按情绪入库"
                "（怒/喜/哀/惧…各留一条），合成页的「导演模式」就能逐句"
                "自动路由到对应情绪的参考音频。")
    lines = [
        "| 条目 | 角色 | 情绪 | 评分 | 时长 | 信噪比 | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        badge = {"优秀": "🟢", "良好": "🟢", "可用": "🟡",
                 "勉强": "🟠", "不建议使用": "🔴"}.get(e.grade, "·")
        zh = EMOTION_LABELS.get(e.emotion, e.emotion)
        lines.append(
            f"| `{e.name}` | {e.character} | {zh} ({e.emotion}) | "
            f"{badge} **{e.score}** | {e.duration:.2f}s | "
            f"{e.snr_db:.1f} dB | {(e.note or '-')[:40]} |"
        )
    return "\n".join(lines)
