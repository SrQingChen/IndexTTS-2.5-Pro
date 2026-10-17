"""gts_import — GTSinger → 本项目数据集导入（服务层，UI/CLI 共用）。

从 tools/gtsinger_import.py 的脚本逻辑演化而来：核心导入能力函数化
（progress/should_stop 回调），CLI 壳与「🎵 歌声训练」向导页共用同一份。

**可复用性**（向导页的核心引导信息，两处保持一致）：
- 导入是**一次性**成本：capability 重训、换预设试验、补练角色都直接
  复用同一个数据集，不必重复导入；
- 移调增广副本与原始条目同入 meta，singing 特征提取一视同仁。

许可边界（如实提示，不构成法律意见）：GTSinger 检索显示 CC BY-NC-SA 4.0——
署名 + 非商用 + 相同方式共享。个人研究可用；免费公开衍生权重需署名并按
同协议共享；商用需另行获得授权。
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Dict, List, Optional

from webui_app.training import dataset as DS
from webui_app.training.features import TECHNIQUE_IDS

# GTSinger 技巧标签 → 本项目 tech_id（词表 features.TECHNIQUE_NAMES）
GTS_TECH_MAP = {
    "mixed": TECHNIQUE_IDS["mixed"],
    "falsetto": TECHNIQUE_IDS["falsetto"],
    "breathy": TECHNIQUE_IDS["breathy"],
    "pharyngeal": TECHNIQUE_IDS["pharyngeal"],
    "vibrato": TECHNIQUE_IDS["vibrato"],
    "glissando": TECHNIQUE_IDS["glissando"],
}


def parse_pitch_aug(spec: str) -> List[int]:
    """'-12,+12' → [-12, 12]（空/非法→[]）。范围与母规划 SPA-SVC 区间一致。"""
    if not spec:
        return []
    out: List[int] = []
    for tok in spec.replace("，", ",").split(","):
        tok = tok.replace("−", "-").strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if -18 <= v <= 18 and v != 0:
            out.append(v)
    return out


def find_gtsinger_root(src: str) -> bool:
    """宽松判定 GTSinger 根目录：json/ 与 audio/ 结构（按语言分的两种布局都认）。"""
    if not os.path.isdir(src):
        return False
    for root, dirs, _files in os.walk(src):
        rel = os.path.relpath(root, src).replace("\\", "/")
        if rel.split("/")[0] in ("json",) or rel.endswith("/json"):
            return True
        if "audio" in dirs and "json" in dirs:
            return True
    return False


def iter_phrases(src: str, langs, singers):
    """遍历 (audio, singer, song, phrase, lang, tech_dir, stem, json_guess)。"""
    langs = set(langs or [])
    singers = set(singers or [])
    for root, _dirs, files in os.walk(src):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            ap = os.path.join(root, fn)
            rel = os.path.relpath(ap, src).replace("\\", "/")
            parts = rel.split("/")
            if "audio" not in parts:
                continue
            i = parts.index("audio")
            singer = parts[i + 1] if len(parts) > i + 1 else "unknown"
            tech_dir = parts[i + 2] if len(parts) > i + 2 else ""
            stem = os.path.splitext(parts[-1])[0]
            song_id, _, phrase_id = stem.partition("__")
            lang = parts[0] if parts[0] in ("zh", "en", "ja", "ko") else ""
            if langs and lang not in langs:
                continue
            if singers and singer not in singers:
                continue
            yield dict(audio=ap, singer=singer, song_id=song_id,
                       phrase=phrase_id or stem, lang=lang,
                       tech_dir=tech_dir, stem=stem,
                       json_guess=os.path.join(
                           root.replace(os.sep, "/").replace(
                               "/audio/", "/json/"), stem + ".json"))


def load_transcription(json_path: str):
    """读 GTSinger phrase JSON → (text, technique_segments)。字段宽松匹配。"""
    if not json_path or not os.path.isfile(json_path):
        return "", []
    try:
        d = json.load(open(json_path, encoding="utf-8"))
    except Exception:
        return "", []
    text = str(d.get("text") or d.get("lyric") or "").strip()
    segs = []
    t0 = 0.0
    for n in d.get("notes") or []:
        for p in (n.get("phns") or []):
            dur = float(p.get("dur") or 0.0)
            tech = str(p.get("technique") or "").strip().lower()
            tid = GTS_TECH_MAP.get(tech)
            if tid is not None and dur > 0:
                segs.append([round(t0, 3), round(t0 + dur, 3), tid])
            t0 += dur
    return text, segs


def run_import(src: str, dataset: str,
               langs: Optional[List[str]] = None,
               singers: Optional[List[str]] = None,
               copy: bool = False,
               pitch_aug: str = "",
               holdout_singers: str = "",
               holdout_technique: str = "",
               max_items: int = 0,
               progress: Optional[Callable[[float, str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> Dict:
    """执行导入。返回摘要 dict（UI 与 CLI 各自渲染）。

    语义：幂等（uid 去重、已存在的音频/technique.json 跳过重写），
    重复跑同一 src 只补新条目——「一次导入，全程复用」的操作基础。
    """
    if not find_gtsinger_root(src):
        return {"ok": False,
                "error": f"{src} 不像 GTSinger 根目录（找不到 json/ + audio/ 结构）。\n"
                "请从 https://huggingface.co/datasets/GTSinger/GTSinger 下载后重试。"}
    aug_semis = parse_pitch_aug(pitch_aug)

    ds_dir = DS.dir_of(dataset)
    audio_dir = os.path.join(ds_dir, DS.AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)

    tech_map: Dict[str, List] = {}
    tp = os.path.join(ds_dir, "technique.json")
    if os.path.isfile(tp):
        try:
            tech_map = json.load(open(tp, encoding="utf-8"))
        except Exception:
            tech_map = {}

    have_ids = set()
    if os.path.isfile(os.path.join(ds_dir, DS.META_FILE)):
        have_ids = {u.id for u in DS.load_meta(dataset)}

    new_items = []
    aug_todo = []
    n = 0
    stopped = False
    for it in iter_phrases(src, langs, singers):
        if max_items and n >= max_items:
            break
        if should_stop is not None and should_stop():
            stopped = True
            break
        uid = f"gts_{it['singer']}_{it['stem']}"[:80]
        if uid in have_ids:
            continue
        ext = os.path.splitext(it["audio"])[1] or ".wav"
        dst = os.path.join(audio_dir, uid + ext)
        if not os.path.isfile(dst):
            try:
                if copy:
                    shutil.copy2(it["audio"], dst)
                else:
                    os.link(it["audio"], dst)
            except OSError:
                shutil.copy2(it["audio"], dst)
        text, segs = load_transcription(it.get("json_guess"))
        if segs:
            tech_map[uid] = segs
        new_items.append(DS.Utterance(
            id=uid, audio=os.path.join(DS.AUDIO_SUBDIR, os.path.basename(dst)),
            source=os.path.abspath(it["audio"]),
            text=text, lang=(it["lang"] or "zh").upper(),
            status="ready" if text else "no_text"))
        aug_todo.append((uid, dst))
        n += 1
        if progress and n % 100 == 0:
            progress(0.0, f"导入 {n} 条…")

    if new_items:
        with DS._META_LOCK:
            items = list(DS.load_meta(dataset))
            have = {u.id for u in items}
            items.extend([u for u in new_items if u.id not in have])
            DS.save_meta(dataset, items)

    aug_entries = []
    if aug_semis and not stopped:
        import librosa
        import soundfile as sf
        for k, (uid, wav) in enumerate(aug_todo):
            if should_stop is not None and should_stop():
                stopped = True
                break
            base_u = next((u for u in new_items if u.id == uid), None)
            for s in aug_semis:
                auid = f"{uid}_ps{'+%d' % s if s > 0 else s}"
                adst = os.path.join(audio_dir, auid + ".wav")
                if os.path.isfile(adst):
                    continue
                y, sr = librosa.load(wav, sr=None, mono=True)
                y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=s)
                sf.write(adst, y2, sr)
                if uid in tech_map:
                    tech_map[auid] = tech_map[uid]
                if base_u is not None:
                    aug_entries.append(DS.Utterance(
                        id=auid,
                        audio=os.path.join(DS.AUDIO_SUBDIR, auid + ".wav"),
                        source=os.path.abspath(adst),
                        text=base_u.text, lang=base_u.lang,
                        status=base_u.status))
            if progress and (k + 1) % 25 == 0:
                progress(0.3 + 0.5 * (k + 1) / max(1, len(aug_todo)),
                         f"音高增广 {k + 1}/{len(aug_todo)}")
        if aug_entries:
            with DS._META_LOCK:
                items = list(DS.load_meta(dataset))
                have = {u.id for u in items}
                items.extend([u for u in aug_entries if u.id not in have])
                DS.save_meta(dataset, items)

    if tech_map:
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tech_map, f, ensure_ascii=False)

    note = [
        f"# {dataset} split 建议（E1-T 母规划 Step 1.5）", "",
        "- 按歌手/歌曲划分 train/val/test，禁止按帧随机划分。",
        f"- unseen singer 留出：{holdout_singers or '（未指定）'}",
        f"- unseen technique 留出：{holdout_technique or '（未指定）'}",
        "- 每歌手最高 10% F0 片段留作 high-pitch test。",
        "- 特征提取请用 singing 模式（带 F0/technique 字段）。",
    ]
    with open(os.path.join(ds_dir, "split_note.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(note))

    return {"ok": not stopped, "stopped": stopped, "dataset": dataset,
            "imported": len(new_items), "augmented": len(aug_entries),
            "pitch_aug": aug_semis, "tech_labels": len(tech_map),
            "out_dir": ds_dir}
