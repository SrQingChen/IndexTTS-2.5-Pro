"""音色库（Voice Bank）。

与官方「预设」的区别：
    · 预设 = 参数快照（含情感/采样参数），存在 outputs/presets/
    · 音色库条目 = **处理好的参考音频 + 体检报告 + 标签**，存在 voice_bank/

音色库解决的是「我有一段调好的参考音频，下次还想用，并且要记得它为什么好」。
每个条目自带 audio_lab 的评分报告，这样你能一眼看出哪个音色素材质量最高。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from webui_app.config import PROJECT_ROOT
from webui_app.services import audio_lab as AL

BANK_DIR = os.path.join(PROJECT_ROOT, "voice_bank")
INDEX_FILE = os.path.join(BANK_DIR, "index.json")
AUDIO_SUBDIR = "audio"


def safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "untitled"


@dataclass
class VoiceEntry:
    """一个音色库条目。"""

    name: str
    audio: str = ""                    # 相对 BANK_DIR 的音频路径
    note: str = ""
    tags: List[str] = field(default_factory=list)
    lang: str = "ZH"
    created_at: float = field(default_factory=time.time)
    source: str = ""                   # 原始上传文件名
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
# 索引读写
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


def list_voices() -> List[VoiceEntry]:
    """按名称排序返回全部条目，并顺手剔除音频文件已丢失的。"""
    out = []
    stale = False
    for d in _load_index():
        e = VoiceEntry(**{k: v for k, v in d.items() if k in VoiceEntry.__dataclass_fields__})
        if e.audio and not os.path.isfile(e.audio_path):
            stale = True
            continue
        out.append(e)
    if stale:
        _save_index([e.to_dict() for e in out])
    out.sort(key=lambda x: x.name.lower())
    return out


def names() -> List[str]:
    return [e.name for e in list_voices()]


def get(name: str) -> Optional[VoiceEntry]:
    for e in list_voices():
        if e.name == name:
            return e
    return None


# ---------------------------------------------------------------------------
# 增删改
# ---------------------------------------------------------------------------

def add(
    name: str,
    audio_path: str,
    note: str = "",
    tags: Optional[List[str]] = None,
    lang: str = "ZH",
    analyze_audio: bool = True,
) -> VoiceEntry:
    """把一个音频文件加入音色库（会复制进库目录并做体检）。"""
    _ensure_dir()
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    key = safe_name(name)
    if not key:
        raise ValueError("音色名称不能为空")

    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    rel = os.path.join(AUDIO_SUBDIR, key + ext).replace("\\", "/")
    dst = os.path.join(BANK_DIR, rel)
    if os.path.abspath(dst) != os.path.abspath(audio_path):
        shutil.copy2(audio_path, dst)

    entry = VoiceEntry(
        name=key, audio=rel, note=note or "",
        tags=[t.strip() for t in (tags or []) if t and t.strip()],
        lang=lang or "ZH",
        source=os.path.basename(audio_path),
    )

    if analyze_audio:
        rep = AL.analyze(dst)
        if not rep.ok:
            # 体检都过不了的音频（空文件、损坏文件）不该进库：留着只会在合成页
            # 被选中然后推理失败，比当场报错难查得多。顺手把刚复制的文件删掉。
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
    items = [d for d in items if d.get("name") != key]
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


def rename(old: str, new: str) -> VoiceEntry:
    e = get(old)
    if e is None:
        raise KeyError(f"音色不存在：{old}")
    newkey = safe_name(new)
    if not newkey:
        raise ValueError("新名称不能为空")
    items = _load_index()
    for d in items:
        if d.get("name") == e.name:
            d["name"] = newkey
    # 音频文件名不必改，索引里存的是相对路径
    _save_index(items)
    return get(newkey) or e


def update(name: str, **fields) -> Optional[VoiceEntry]:
    """更新 note / tags / lang 等元数据。"""
    items = _load_index()
    key = safe_name(name)
    changed = None
    for d in items:
        if d.get("name") == key:
            for k, v in fields.items():
                if k in VoiceEntry.__dataclass_fields__ and k not in ("name", "audio"):
                    d[k] = v
            changed = d
    if changed:
        _save_index(items)
        return VoiceEntry(**{k: v for k, v in changed.items()
                             if k in VoiceEntry.__dataclass_fields__})
    return None


def reanalyze(name: str) -> Optional[VoiceEntry]:
    e = get(name)
    if e is None:
        return None
    rep = AL.analyze(e.audio_path)
    return update(
        name, report=rep.to_dict(), score=rep.score, grade=rep.grade,
        duration=rep.duration, sample_rate=rep.sample_rate, snr_db=rep.snr_db,
    )


# ---------------------------------------------------------------------------
# 展示
# ---------------------------------------------------------------------------

def table_markdown(entries: Optional[List[VoiceEntry]] = None) -> str:
    entries = entries if entries is not None else list_voices()
    if not entries:
        return ("_音色库为空。_\n\n"
                "到「参考音频工作台」上传一段音频 → 体检 → 增强 → "
                "点「存入音色库」，就能在这里管理。")
    lines = [
        "| 名称 | 评分 | 时长 | 采样率 | 信噪比 | 语言 | 标签 | 备注 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        badge = {"优秀": "🟢", "良好": "🟢", "可用": "🟡",
                 "勉强": "🟠", "不建议使用": "🔴"}.get(e.grade, "·")
        lines.append(
            f"| `{e.name}` | {badge} **{e.score}** | {e.duration:.2f}s | "
            f"{e.sample_rate} Hz | {e.snr_db:.1f} dB | {e.lang} | "
            f"{', '.join(e.tags) or '-'} | {(e.note or '-')[:40]} |"
        )
    return "\n".join(lines)


def detail_markdown(name: str) -> str:
    e = get(name)
    if e is None:
        return "_请选择一个音色_"
    parts = [
        f"### 音色 `{e.name}`",
        "",
        f"- **评分**：{e.score} / 100（{e.grade}）",
        f"- **音频**：`{e.audio}`",
        f"- **来源**：`{e.source or '-'}`",
        f"- **时长 / 采样率**：{e.duration:.2f}s / {e.sample_rate} Hz",
        f"- **信噪比**：{e.snr_db:.1f} dB",
        f"- **语言**：{e.lang}",
        f"- **标签**：{', '.join(e.tags) or '-'}",
        f"- **备注**：{e.note or '-'}",
        f"- **入库时间**：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e.created_at))}",
    ]
    if e.report:
        issues = e.report.get("issues") or []
        if issues:
            parts += ["", "**体检发现的问题**", ""]
            parts += [f"- {i}" for i in issues]
        sugg = e.report.get("suggestions") or []
        if sugg:
            parts += ["", "**改进建议**", ""]
            parts += [f"- {s}" for s in sugg]
        if not issues:
            parts += ["", "🟢 体检未发现问题。"]
    return "\n".join(parts)
