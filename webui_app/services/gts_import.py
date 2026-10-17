"""gts_import — GTSinger → 本项目数据集导入（服务层，UI/CLI 共用）。

**真实仓库布局（2026-10 实测探明，与最初猜测的 audio/+json/ 不同）**：
    Chinese/ZH-Alto-1/<Technique>/<Song>/<Group>/<NNNN>.{wav,json,...}
    - json 与 wav 同目录同名，是**逐词列表**：
      [{"word","start_time","end_time","ph","ph_start","ph_end",
        "mix"/"falsetto"/"breathy"/"pharyngeal"/"glissando"/"vibrato"
        （逐音素 "0"/"1" 标志）, "note", "tech", ...}]
      文本 = 拼接 word；技巧段 = 逐词看六标志（无顶层 text/notes）。
    - Group：`<Tech>_Group` 带技巧 / `Control_Group` 同句无技巧（同一首歌
      会在多个技巧目录下重复出现——**uid 不含技巧目录**，靠 meta 去重）/
      `Paired_Speech_Group` 配对语音（capability 不导入）。
    - 顶层是语言全名（Chinese/…），语言码映射见 gts_fetch.LANG_DIR。

**可复用性**（向导页的核心引导信息，两处保持一致）：
- 导入是**一次性**成本：capability 重训、换预设试验、补练角色都直接
  复用同一个数据集，不必重复导入（uid 级幂等）；
- 移调增广副本与原始条目同入 meta，singing 特征提取一视同仁。

许可边界（如实提示，不构成法律意见）：GTSinger 检索显示 CC BY-NC-SA 4.0——
署名 + 非商用 + 相同方式共享。个人研究可用；免费公开衍生权重需署名并按
同协议共享；商用需另行获得授权。
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Dict, List, Optional, Tuple

from webui_app.services.gts_fetch import DIR_LANG, LANG_DIR
from webui_app.training import dataset as DS
from webui_app.training.features import TECHNIQUE_IDS

# 六技巧的 json 标志名 → 本项目 tech_id（判定优先级同序）
_FLAG_TO_ID = [
    ("mix", TECHNIQUE_IDS["mixed"]),
    ("falsetto", TECHNIQUE_IDS["falsetto"]),
    ("breathy", TECHNIQUE_IDS["breathy"]),
    ("pharyngeal", TECHNIQUE_IDS["pharyngeal"]),
    ("glissando", TECHNIQUE_IDS["glissando"]),
    ("vibrato", TECHNIQUE_IDS["vibrato"]),
]


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
    """判定 GTSinger 根目录：任一 wav 带同名 json 兄弟文件即认（实测布局），
    兼容旧的 audio/+json/ 布局。"""
    if not os.path.isdir(src):
        return False
    if os.path.isdir(os.path.join(src, "audio")) and \
            os.path.isdir(os.path.join(src, "json")):
        return True
    n_checked = 0
    for root, _dirs, files in os.walk(src):
        for f in files:
            if f.lower().endswith(".wav"):
                if os.path.isfile(os.path.join(root, os.path.splitext(f)[0]
                                               + ".json")):
                    return True
                n_checked += 1
                if n_checked >= 50:      # 50 个 wav 都没有兄弟 json，不是它
                    return False
    return False


def _lang_of_top(seg: str) -> str:
    s = seg or ""
    if s in DIR_LANG:
        return DIR_LANG[s]
    low = s.lower()
    return low if low in LANG_DIR else ""


def iter_phrases(src: str, langs=None, singers=None,
                 include_speech: bool = False):
    """遍历 (audio, singer, song, phrase, lang, tech_dir, group, json_path)。

    真实布局：`<Lang>/<Singer>/<Technique>/<Song>/<Group>/<NNNN>.wav`，
    Group 段以 `_Group` 结尾（从后往前找）。Paired_Speech_Group 默认跳过
    （配对语音不是歌声，capability 不用；要导语音做对照再开）。
    兼容旧布局：路径含 `audio/` 段、json 在平行的 `json/` 树里。
    """
    langs = set(langs or [])
    singers = set(singers or [])
    for root, _dirs, files in os.walk(src):
        wavs = sorted(f for f in files if f.lower().endswith(".wav"))
        if not wavs:
            continue
        parts = os.path.relpath(root, src).replace("\\", "/").split("/")
        # ---- 旧布局：audio/<singer>/<technique>/<song>__<phrase>.wav ----
        if "audio" in parts:
            i = parts.index("audio")
            singer = parts[i + 1] if len(parts) > i + 1 else "unknown"
            tech_dir = parts[i + 2] if len(parts) > i + 2 else ""
            for f in wavs:
                ap = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                lang = _lang_of_top(parts[0])
                if langs and lang not in langs:
                    continue
                if singers and singer not in singers:
                    continue
                yield dict(audio=ap, singer=singer,
                           song=stem.partition("__")[0], phrase=stem,
                           lang=lang, tech_dir=tech_dir, group="",
                           json_guess=os.path.join(
                               root.replace(os.sep, "/").replace(
                                   "/audio/", "/json/"), stem + ".json"))
            continue
        # ---- 实测布局：<Lang>/<Singer>/<Tech>/<Song>/<Group>/NNNN.wav ----
        group_idx = next((i for i in range(len(parts) - 1, -1, -1)
                          if parts[i].endswith("_Group")), -1)
        if group_idx < 1:
            continue
        group = parts[group_idx]
        if group == "Paired_Speech_Group" and not include_speech:
            continue
        lang = _lang_of_top(parts[0])
        if langs and lang not in langs:
            continue
        singer = parts[1] if len(parts) > 1 else "unknown"
        if singers and singer not in singers:
            continue
        song = parts[group_idx - 1]
        tech_dir = parts[2] if len(parts) > 2 else ""
        for f in wavs:
            stem = os.path.splitext(f)[0]
            ap = os.path.join(root, f)
            jp = os.path.join(root, stem + ".json")
            yield dict(audio=ap, singer=singer, song=song, phrase=stem,
                       lang=lang, tech_dir=tech_dir, group=group,
                       json_guess=jp)


def load_transcription(json_path: str) -> Tuple[str, List[List]]:
    """读 phrase json → (text, technique_segments)。

    实测 schema（逐词列表）：
      text = 拼接 word（全 CJK 无空格，含拉丁词用空格连接）；
      技巧段 = 逐词查六标志（任一音素为 "1" 即算），相邻同技巧合并为
      [start_time, end_time, tech_id]；无技巧词不产生段（= control）。
    兼容旧猜测 schema（dict 带 notes/phns/dur/technique）。
    """
    if not json_path or not os.path.isfile(json_path):
        return "", []
    try:
        d = json.load(open(json_path, encoding="utf-8"))
    except Exception:
        return "", []
    if isinstance(d, dict):
        # 旧猜测 schema（保留兼容）：notes/phns 里的 technique 名直接映射
        name_to_id = {flag: tid for flag, tid in _FLAG_TO_ID}
        text = str(d.get("text") or d.get("lyric") or "").strip()
        segs = []
        t0 = 0.0
        for n in d.get("notes") or []:
            for p in (n.get("phns") or []):
                dur = float(p.get("dur") or 0.0)
                tid = name_to_id.get(
                    str(p.get("technique") or "").strip().lower())
                if tid is not None and dur > 0:
                    segs.append([round(t0, 3), round(t0 + dur, 3), tid])
                t0 += dur
        return text, segs
    if not isinstance(d, list) or not d:
        return "", []
    words = [w for w in d if isinstance(w, dict)]
    words.sort(key=lambda w: float(w.get("start_time") or 0.0))
    if not words:
        return "", []
    # 文本：CJK 紧排、拉丁词空格连接（一个粗但够用的判别）
    text = ""
    prev_latin = False
    for w in words:
        word = str(w.get("word") or "")
        if not word:
            continue
        is_latin = all(ord(c) < 0x2E80 for c in word)
        if text and (is_latin or prev_latin):
            text += " "
        text += word
        prev_latin = is_latin

    def _word_tech(w) -> Optional[int]:
        for flag, tid in _FLAG_TO_ID:
            marks = w.get(flag) or []
            marks = [str(m) for m in (marks if isinstance(marks, list)
                                      else [marks])]
            if any(m in ("1", "1.0", "true") for m in marks):
                return tid
        return None

    segs = []
    for w in words:
        tid = _word_tech(w)
        if tid is None:
            continue
        s = float(w.get("start_time") or 0.0)
        e = float(w.get("end_time") or s)
        if e <= s:
            continue
        if segs and segs[-1][2] == tid and abs(segs[-1][1] - s) < 0.02:
            segs[-1][1] = round(e, 3)      # 相邻同技巧合并
        else:
            segs.append([round(s, 3), round(e, 3), tid])
    return text, segs


def _uid_of(it: dict) -> str:
    """uid 不含技巧目录：Control_Group 在多个技巧目录下重复出现时，
    同一首歌同一条 phrase 自然去重（have_ids 跳过）。"""
    if it.get("group"):
        stem = f"{it['song']}_{it['group']}_{it['phrase']}"
    else:
        stem = it["phrase"]
    return f"gts_{it['singer']}_{stem}"[:80]


def run_import(src: str, dataset: str,
               langs: Optional[List[str]] = None,
               singers: Optional[List[str]] = None,
               copy: bool = False,
               pitch_aug: str = "",
               holdout_singers: str = "",
               holdout_technique: str = "",
               max_items: int = 0,
               include_speech: bool = False,
               progress: Optional[Callable[[float, str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> Dict:
    """执行导入。返回摘要 dict（UI 与 CLI 各自渲染）。

    语义：幂等（uid 去重、已存在的音频/technique.json 跳过重写），
    重复跑同一 src 只补新条目——「一次导入，全程复用」的操作基础。
    """
    if not find_gtsinger_root(src):
        return {"ok": False,
                "error": f"{src} 不像 GTSinger 根目录（找不到 wav+同名json "
                "结构）。请从 https://huggingface.co/datasets/GTSinger/GTSinger "
                "下载（或用向导⓪镜像拉取）后重试。"}
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
    for it in iter_phrases(src, langs, singers, include_speech):
        if max_items and n >= max_items:
            break
        if should_stop is not None and should_stop():
            stopped = True
            break
        uid = _uid_of(it)
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
        f"- unseen singer 留出：{holdout_singers or '（未指定；中文仅 2 位歌手，建议留 ZH-Tenor-1 整人）'}",
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
