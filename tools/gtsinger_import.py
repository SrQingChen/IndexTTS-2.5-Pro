"""gtsinger_import — GTSinger → 本项目训练数据集导入工具（E1-T Step 1 入口）。

只做导入/整理，**不下载**数据（GTSinger 需自行从 HuggingFace 获取：
https://huggingface.co/datasets/GTSinger/GTSinger ，检索显示 CC BY-NC-SA 4.0，
商用需另获授权——详见 docs/dev-plan-e1t-2026-10.md §11）。

用法示例：
    python tools/gtsinger_import.py --src D:/data/GTSinger --dataset gtsinger_zh_en \
        --langs zh,en --singers all --pitch-aug -12,+12

产出（datasets/<name>/）：
    audio/  （或软链/复制，见 --copy）＋ meta.jsonl ＋ technique.json
    （technique.json: {uid: [[start_sec, end_sec, tech_id], ...]}，
     tech_id 词表见 webui_app/training/features.py TECHNIQUE_NAMES）

split 建议（写入 datasets/<name>/split_note.md，不自动划分——
E1-T 母规划要求 unseen singer / unseen technique / top10% F0 三种留出，
必须按歌手/歌曲切，不能按帧随机切）：
    --holdout-singers "Singer1,Singer2"   留出整歌手做 unseen singer 测试
    --holdout-technique falsetto          留出一种技巧做 unseen technique

音高增广（--pitch-aug，SPA-SVC 思想的导入期实现）：对每条样本生成
移调副本（librosa 相位声码器，保时长），F0/tech 特征随音频重提取——
**必须重提取而不是改标签**：只改 F0 标签不改 mel 会教模型忽略条件。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.training import dataset as DS  # noqa: E402
from webui_app.training.features import TECHNIQUE_IDS  # noqa: E402

# GTSinger 技巧标签 → 本项目 tech_id（词表 features.TECHNIQUE_NAMES）
GTS_TECH_MAP = {
    "mixed": TECHNIQUE_IDS["mixed"],
    "falsetto": TECHNIQUE_IDS["falsetto"],
    "breathy": TECHNIQUE_IDS["breathy"],
    "pharyngeal": TECHNIQUE_IDS["pharyngeal"],
    "vibrato": TECHNIQUE_IDS["vibrato"],
    "glissando": TECHNIQUE_IDS["glissando"],
}


def parse_pitch_aug(spec: str):
    """'−12,+12' → [-12, 12]（空/非法→[]）"""
    if not spec:
        return []
    out = []
    for tok in spec.replace("，", ",").split(","):
        tok = tok.replace("−", "-").strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            print(f"[跳过] 非法移调值 {tok!r}")
            continue
        if -18 <= v <= 18 and v != 0:
            out.append(v)
        else:
            print(f"[跳过] 移调值超范围（-18..18，非0）：{v}")
    return out


def find_gtsinger_root(src: str) -> bool:
    """宽松判定 GTSinger 根目录：要有 json/ 与 audio/（或按语言分的子目录）。"""
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
    """遍历 (audio_path, singer, song_id, phrase_id, lang)。

    GTSinger 目录组织存在版本差异（根下按语言分 / 直接 json+audio），
    这里两种都兼容；JSON 元数据匹配不到时 yield 空 text/tech，
    由调用方按「无标签全 control」处理（与母规划一致）。
    """
    langs = set(langs or [])
    singers = set(singers or [])
    for root, _dirs, files in os.walk(src):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            ap = os.path.join(root, fn)
            rel = os.path.relpath(ap, src).replace("\\", "/")
            parts = rel.split("/")
            # .../audio/<singer>/<technique>/<song>__<phrase>.wav
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
    """读 GTSinger 的 phrase JSON → (text, technique_segments)。

    schema（按 GTSinger 公开描述，字段名做宽松匹配；匹配不上返回空）：
    { "text": ..., "notes": [{"word","phns":[{"phn","dur","technique"}]}] }
    durations 是相对 phrase 起点的秒数（累计）。
    """
    if not json_path or not os.path.isfile(json_path):
        return "", []
    try:
        d = json.load(open(json_path, encoding="utf-8"))
    except Exception:
        return "", []
    text = str(d.get("text") or d.get("lyric") or "").strip()
    segs = []
    t0 = 0.0
    notes = d.get("notes") or []
    for n in notes:
        for p in (n.get("phns") or []):
            dur = float(p.get("dur") or 0.0)
            tech = str(p.get("technique") or "").strip().lower()
            tid = GTS_TECH_MAP.get(tech)
            if tid is not None and dur > 0:
                segs.append([round(t0, 3), round(t0 + dur, 3), tid])
            t0 += dur
    return text, segs


def main():
    ap = argparse.ArgumentParser(description="GTSinger → 本项目数据集导入")
    ap.add_argument("--src", required=True, help="GTSinger 根目录（自行下载）")
    ap.add_argument("--dataset", required=True, help="目标数据集名")
    ap.add_argument("--langs", default="", help="逗号分隔语言过滤（zh,en,…空=全部）")
    ap.add_argument("--singers", default="", help="逗号分隔歌手过滤（空=全部）")
    ap.add_argument("--copy", action="store_true",
                    help="复制音频（默认硬链接，失败退回复制）")
    ap.add_argument("--pitch-aug", default="",
                    help="导入期音高增广，如 '-12,+12'（生成移调副本并重提特征）")
    ap.add_argument("--holdout-singers", default="",
                    help="建议留出歌手（写进 split_note.md，不进 meta）")
    ap.add_argument("--holdout-technique", default="",
                    help="建议留出技巧（写进 split_note.md）")
    ap.add_argument("--max-items", type=int, default=0, help="调试上限（0=不限）")
    args = ap.parse_args()

    if not find_gtsinger_root(args.src):
        print(f"[错误] {args.src} 不像 GTSinger 根目录（找不到 json/ + audio/ 结构）。\n"
              "请从 https://huggingface.co/datasets/GTSinger/GTSinger 下载后重试。")
        return 2

    langs = [x for x in args.langs.replace("，", ",").split(",") if x]
    singers = [x for x in args.singers.replace("，", ",").split(",") if x]
    aug_semis = parse_pitch_aug(args.pitch_aug)

    ds_dir = DS.dir_of(args.dataset)
    audio_dir = os.path.join(ds_dir, DS.AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)

    tech_map: dict = {}
    n = 0
    aug_todo = []
    new_items = []
    have_ids = set()
    if os.path.isfile(os.path.join(ds_dir, DS.META_FILE)):
        have_ids = {u.id for u in DS.load_meta(args.dataset)}
    for it in iter_phrases(args.src, langs, singers):
        if args.max_items and n >= args.max_items:
            break
        uid = f"gts_{it['singer']}_{it['stem']}"[:80]
        if uid in have_ids:
            continue
        ext = os.path.splitext(it["audio"])[1] or ".wav"
        dst = os.path.join(audio_dir, uid + ext)
        if not os.path.isfile(dst):
            try:
                if args.copy:
                    shutil.copy2(it["audio"], dst)
                else:
                    os.link(it["audio"], dst)
            except OSError:
                shutil.copy2(it["audio"], dst)
        text, segs = load_transcription(it.get("json_guess"))
        if segs:
            tech_map[uid] = segs
        u = DS.Utterance(id=uid, audio=os.path.join(DS.AUDIO_SUBDIR,
                                                    os.path.basename(dst)),
                         source=os.path.abspath(it["audio"]),
                         text=text, lang=(it["lang"] or "zh").upper(),
                         status="ready" if text else "no_text")
        new_items.append(u)
        aug_todo.append((uid, dst))
        n += 1
        if n % 200 == 0:
            print(f"[进度] 已导入 {n} 条")

    if new_items:
        # 批量入册（锁外拼接，锁内整体重写一次——逐条 update 是 O(n²)）
        with DS._META_LOCK:
            items = list(DS.load_meta(args.dataset))
            have = {u.id for u in items}
            items.extend([u for u in new_items if u.id not in have])
            DS.save_meta(args.dataset, items)

    if tech_map:
        with open(os.path.join(ds_dir, "technique.json"), "w",
                  encoding="utf-8") as f:
            json.dump(tech_map, f, ensure_ascii=False)

    # 音高增广副本（音频域移调；特征需 singing 模式重提取）
    # P0（评审实锤）：副本必须同步入册 meta.jsonl——特征提取/体检只遍历
    # meta，漏了入册的副本是孤儿文件，「增广已生成」的报告是假的。
    if aug_semis:
        import librosa
        import soundfile as sf
        aug_entries = []
        for k, (uid, wav) in enumerate(aug_todo):
            base_u = next((u for u in new_items if u.id == uid), None)
            for s in aug_semis:
                auid = f"{uid}_ps{'+%d' % s if s > 0 else s}"
                adst = os.path.join(audio_dir, auid + ".wav")
                if os.path.isfile(adst):
                    continue
                y, sr = librosa.load(wav, sr=None, mono=True)
                y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=s)
                sf.write(adst, y2, sr)
                # 移调副本继承 technique 标签（时间轴不变，保时长移调）
                if uid in tech_map:
                    tech_map[auid] = tech_map[uid]
                if base_u is not None:
                    aug_entries.append(DS.Utterance(
                        id=auid,
                        audio=os.path.join(DS.AUDIO_SUBDIR, auid + ".wav"),
                        source=os.path.abspath(adst),
                        text=base_u.text, lang=base_u.lang,
                        status=base_u.status))
            if (k + 1) % 50 == 0:
                print(f"[增广] {k + 1}/{len(aug_todo)}")
        if aug_entries:
            with DS._META_LOCK:
                items = list(DS.load_meta(args.dataset))
                have = {u.id for u in items}
                items.extend([u for u in aug_entries if u.id not in have])
                DS.save_meta(args.dataset, items)
            print(f"[增广] {len(aug_entries)} 个移调副本已入册 meta.jsonl")
        if tech_map:
            with open(os.path.join(ds_dir, "technique.json"), "w",
                      encoding="utf-8") as f:
                json.dump(tech_map, f, ensure_ascii=False)

    note = [
        f"# {args.dataset} split 建议（E1-T 母规划 Step 1.5）", "",
        "- 按歌手/歌曲划分 train/val/test，禁止按帧随机划分。",
        f"- unseen singer 留出：{args.holdout_singers or '（未指定）'}",
        f"- unseen technique 留出：{args.holdout_technique or '（未指定）'}",
        "- 每歌手最高 10% F0 片段留作 high-pitch test。",
        "- 特征提取请用 singing 模式（带 F0/technique 字段）。",
    ]
    with open(os.path.join(ds_dir, "split_note.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(note))
    print(f"[完成] 导入 {n} 条 → {ds_dir}（technique 标签 {len(tech_map)} 条；"
          f"音高增调 {aug_semis}）\n下一步：数据集页体检 → singing 模式提取特征。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
