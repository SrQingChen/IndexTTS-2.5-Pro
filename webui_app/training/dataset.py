"""训练数据集管理。

一个数据集 = 一批「音频 + 文本」配对，是 GPT/CFM LoRA 训练的原料。

目录结构（datasets/<name>/）：
    dataset.json    数据集元信息
    meta.jsonl      每条样本一行（音频路径、文本、语言、时长、体检指标、状态）
    audio/          导入的音频（默认复制进来，避免外部文件被移动后数据集失效）
    features/       离线预提取的特征 <id>.pt（见 features.py）
    split.json      train/val 划分

设计取舍：
    · 用 JSONL 而不是 sqlite —— 便于人工检查、diff、手工修补个别条目
    · 音频默认**复制**而非引用 —— 训练可能跑几小时，期间源文件被移动就全废了
    · 体检指标入库时算一次并缓存 —— 列表页要显示几百条，不能每次重算

时长上限来自架构硬约束（已实测核实）：
    语义 token 为 25Hz，GPT 的 max_mel_tokens=1815
    → 单条音频理论上限 1815/25 ≈ 72.6s，超过就装不进位置编码
    但 8GB 显存下 CFM 的注意力开销随 mel 帧数平方增长，
    → 实际上限设 20s，超过建议先用「参考音频工作台」切片
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT

DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

# meta.jsonl 的读-改-写互斥：UI 线程（写文本/体检重算）与 runner 线程
# （特征提取结束批量回写）都会整文件 load→modify→save，不加锁时
# 后写者会拿旧快照覆盖先写者 —— 用户刚补的文本被静默抹掉。
_META_LOCK = threading.RLock()

AUDIO_SUBDIR = "audio"
FEATURE_SUBDIR = "features"
META_FILE = "meta.jsonl"
INFO_FILE = "dataset.json"
SPLIT_FILE = "split.json"

# ---- 时长约束（秒）--------------------------------------------------------
HARD_MAX_SEC = 72.0        # 架构硬上限：1815 语义token ÷ 25Hz
MAX_TRAIN_SEC = 20.0       # 8GB 显存下的实际上限，超过建议切分
IDEAL_MIN_SEC = 3.0
IDEAL_MAX_SEC = 15.0
MIN_SEC = 1.0              # 低于这个长度学不到有效韵律
MIN_SNR_DB = 12.0

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")

# 状态 → (图标, 是否可用于训练)
STATUS_INFO: Dict[str, tuple] = {
    "ready":            ("✅", True),
    "no_text":          ("📝", False),
    "missing_audio":    ("❌", False),
    "too_short":        ("⏱", False),
    "too_long":         ("✂️", False),
    "over_hard_limit":  ("🚫", False),
    "low_snr":          ("🔇", False),
    "no_features":      ("🧮", False),
}


@dataclass
class Utterance:
    """一条训练样本。字段名会直接序列化进 meta.jsonl，不要随意改。"""

    id: str
    audio: str = ""                 # 相对数据集目录的路径
    source: str = ""                # 导入时的源文件绝对路径（去重用）
    text: str = ""                  # 训练用文本（人工校对后）
    asr_text: str = ""              # whisper 原始转写，仅作参考
    asr_model: str = ""
    lang: str = "ZH"

    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    snr_db: float = 0.0
    score: float = 0.0

    status: str = "no_text"
    problems: List[str] = field(default_factory=list)
    note: str = ""

    has_features: bool = False
    features_at: float = 0.0
    added_at: float = field(default_factory=time.time)

    # ---- 特征提取后回写的长度信息 ----
    # 存在 meta 里而不是每次去读 .pt：训练器要按长度排序与过滤（减少 padding、
    # 剔除超长样本），而这两个动作不该触发磁盘 IO。几个 int 而已，
    # 相比一条 700 KB 的特征文件，读 meta 便宜三个量级。
    n_codes: int = 0              # 语义 token 帧数（25Hz）
    n_text_tokens: int = 0        # 文本 token 数（含语言前缀）
    mel_len: int = 0              # mel 帧数（86.13Hz）
    lang_token: int = 1           # 语言编号，喂给 lang_embedding

    # ---- 派生 ----
    @property
    def usable(self) -> bool:
        return self.status == "ready"

    def audio_abs(self, ds_dir: str) -> str:
        p = self.audio
        return p if os.path.isabs(p) else os.path.join(ds_dir, p)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Utterance":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# 路径与元信息
# ---------------------------------------------------------------------------

def root() -> str:
    os.makedirs(DATASETS_ROOT, exist_ok=True)
    return DATASETS_ROOT


def dir_of(name: str) -> str:
    """数据集名 → 目录。

    必须跟 `create()` 用**同一套**名称规则：create 会先过 safe_dataset_name，
    而其它函数都走 dir_of。两边不一致的话，`create("_x")` 会建出
    `datasets/x`，而 `import_audio("_x")` 却往 `datasets/_x` 里写 ——
    两个目录同时存在，数据静默分裂。

    已存在的目录优先按原名走，避免早期建错的历史目录被「改名」后找不到。
    """
    raw = os.path.join(root(), name or "")
    if os.path.isdir(raw):
        return raw
    return os.path.join(root(), safe_dataset_name(name))


def safe_dataset_name(name: str) -> str:
    """数据集名会当目录名用，清洗规则与官方 safe_preset_name 保持一致。"""
    import re
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "untitled"


def list_datasets() -> List[str]:
    r = root()
    out = []
    for n in sorted(os.listdir(r)):
        if os.path.isfile(os.path.join(r, n, INFO_FILE)):
            out.append(n)
    return out


def exists(name: str) -> bool:
    return os.path.isfile(os.path.join(dir_of(name), INFO_FILE))


def create(name: str, note: str = "", lang_default: str = "ZH") -> str:
    name = safe_dataset_name(name)
    d = dir_of(name)
    os.makedirs(os.path.join(d, AUDIO_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(d, FEATURE_SUBDIR), exist_ok=True)
    info_path = os.path.join(d, INFO_FILE)
    if not os.path.isfile(info_path):
        _write_json(info_path, {
            "name": name, "note": note, "lang_default": lang_default,
            "created_at": time.time(),
        })
    if not os.path.isfile(os.path.join(d, META_FILE)):
        open(os.path.join(d, META_FILE), "w", encoding="utf-8").close()
    return name


def info(name: str) -> Dict[str, Any]:
    return _read_json(os.path.join(dir_of(name), INFO_FILE), {})


def set_info(name: str, **fields) -> Dict[str, Any]:
    d = info(name)
    d.update(fields)
    _write_json(os.path.join(dir_of(name), INFO_FILE), d)
    return d


def delete(name: str) -> bool:
    d = dir_of(name)
    if not os.path.isdir(d):
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not os.path.isdir(d)


def _read_json(path: str, default):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# meta.jsonl 读写
# ---------------------------------------------------------------------------

def load_meta(name: str) -> List[Utterance]:
    path = os.path.join(dir_of(name), META_FILE)
    if not os.path.isfile(path):
        return []
    out: List[Utterance] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(Utterance.from_dict(json.loads(s)))
            except Exception:
                continue        # 单行损坏不该让整个数据集打不开
    return out


def save_meta(name: str, items: Sequence[Utterance]) -> None:
    path = os.path.join(dir_of(name), META_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def get(name: str, uid: str) -> Optional[Utterance]:
    for u in load_meta(name):
        if u.id == uid:
            return u
    return None


def update(name: str, uid: str, **fields) -> Optional[Utterance]:
    with _META_LOCK:
        items = load_meta(name)
        target = None
        for u in items:
            if u.id == uid:
                for k, v in fields.items():
                    if hasattr(u, k):
                        setattr(u, k, v)
                target = u
                break
        if target is not None:
            save_meta(name, items)
        return target


def apply_fields(name: str, touched: Dict[str, Dict[str, Any]]) -> int:
    """一次性把多个样本的字段回写 meta.jsonl（读-改-写，加锁）。

    供特征提取结束时的批量回写与 UI 的批量写文本共用。
    不用逐条 update()：那个每次都 load+save 整个文件，循环里调是 O(n²)。
    """
    if not touched:
        return 0
    with _META_LOCK:
        items = load_meta(name)
        n = 0
        for u in items:
            fields = touched.get(u.id)
            if not fields:
                continue
            for k, v in fields.items():
                if hasattr(u, k):
                    setattr(u, k, v)
            n += 1
        if n:
            save_meta(name, items)
    return n


def remove_utterances(name: str, uids: Sequence[str]) -> int:
    """删除条目，同时删掉对应的音频与特征文件。"""
    uids = set(uids)
    d = dir_of(name)
    items = load_meta(name)
    kept = [u for u in items if u.id not in uids]
    for u in items:
        if u.id in uids:
            for p in (u.audio_abs(d),
                      os.path.join(d, FEATURE_SUBDIR, f"{u.id}.pt")):
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                except OSError:
                    pass
    save_meta(name, kept)
    return len(items) - len(kept)


def _next_id(items: Sequence[Utterance]) -> Callable[[], str]:
    mx = 0
    for u in items:
        try:
            mx = max(mx, int(str(u.id).split("_")[-1]))
        except (ValueError, IndexError):
            pass
    counter = [mx]

    def gen() -> str:
        counter[0] += 1
        return f"utt_{counter[0]:05d}"
    return gen


# ---------------------------------------------------------------------------
# 体检与状态判定
# ---------------------------------------------------------------------------

def evaluate(u: Utterance, ds_dir: str, require_features: bool = True) -> Utterance:
    """重算一条样本的指标与状态。不写盘。"""
    from webui_app.services import audio_lab as AL

    problems: List[str] = []
    ap = u.audio_abs(ds_dir)
    if not u.audio or not os.path.isfile(ap):
        u.status = "missing_audio"
        u.problems = ["音频文件不存在"]
        return u

    try:
        r = AL.analyze(ap)
        u.duration = round(r.duration, 3)
        u.sample_rate = int(r.sample_rate)
        u.channels = int(r.channels)
        u.snr_db = round(r.snr_db, 1)
        u.score = round(r.score, 1)
    except Exception as e:
        u.status = "missing_audio"
        u.problems = [f"音频无法解析：{type(e).__name__}: {e}"]
        return u

    if not (u.text or "").strip():
        problems.append("没有文本（需 ASR 转写或手工填写）")
    if u.duration < MIN_SEC:
        problems.append(f"时长 {u.duration:.2f}s 低于下限 {MIN_SEC:.0f}s")
    if u.duration > HARD_MAX_SEC:
        problems.append(
            f"时长 {u.duration:.1f}s 超过架构硬上限 {HARD_MAX_SEC:.0f}s"
            f"（1815 语义token ÷ 25Hz），必须切分")
    elif u.duration > MAX_TRAIN_SEC:
        problems.append(
            f"时长 {u.duration:.1f}s 超过推荐上限 {MAX_TRAIN_SEC:.0f}s，"
            "8GB 显存下建议切分")
    if u.snr_db < MIN_SNR_DB:
        problems.append(f"信噪比 {u.snr_db:.1f}dB 偏低（建议 ≥{MIN_SNR_DB:.0f}dB）")
    if require_features and not u.has_features:
        problems.append("尚未预提取特征")

    u.problems = problems
    if not problems:
        u.status = "ready"
    elif not (u.text or "").strip():
        u.status = "no_text"
    elif u.duration > HARD_MAX_SEC:
        u.status = "over_hard_limit"
    elif u.duration > MAX_TRAIN_SEC:
        u.status = "too_long"
    elif u.duration < MIN_SEC:
        u.status = "too_short"
    elif u.snr_db < MIN_SNR_DB:
        u.status = "low_snr"
    elif require_features and not u.has_features:
        u.status = "no_features"
    else:
        u.status = "no_text"
    return u


def refresh_all(name: str, require_features: bool = True,
                progress: Optional[Callable[[float, str], None]] = None
                ) -> Dict[str, int]:
    """重算全部样本的状态。导入后/提取特征后调用。"""
    d = dir_of(name)
    with _META_LOCK:
        items = load_meta(name)
        n = len(items)
        for i, u in enumerate(items):
            evaluate(u, d, require_features)
            if progress and n:
                progress((i + 1) / n, f"体检 {i+1}/{n}")
        save_meta(name, items)
    return stats(name)


# ---------------------------------------------------------------------------
# 导入音频
# ---------------------------------------------------------------------------

def import_audio(name: str, paths: Sequence[str], copy: bool = True,
                 lang: str = "ZH",
                 progress: Optional[Callable[[float, str], None]] = None
                 ) -> Dict[str, Any]:
    """把一批音频文件导入数据集。

    copy=True 时复制进 datasets/<name>/audio/，False 时只记绝对路径引用。
    默认复制 —— 训练可能跑几小时，源文件被移动/删除会让数据集静默失效。
    """
    d = dir_of(name)
    audio_dir = os.path.join(d, AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)

    items = load_meta(name)
    # 按源文件去重：同一批文件导两遍不该让样本静默翻倍。
    # 新 meta 用 source 字段（导入时的源绝对路径）；旧数据里 copy=False
    # 导入的 audio 本身就是绝对路径，也算进去。
    seen_sources = {os.path.normcase(os.path.abspath(u.source))
                    for u in items if getattr(u, "source", "")}
    seen_sources |= {os.path.normcase(os.path.abspath(u.audio))
                     for u in items if os.path.isabs(u.audio or "")}
    gen = _next_id(items)

    added, skipped, failed = [], [], []
    staged: List[tuple] = []          # (uid, rel, src_abs)：拷完再统一入册
    paths = [p for p in paths if p]
    for i, src in enumerate(paths):
        if progress and paths:
            progress(i / len(paths), f"导入 {i+1}/{len(paths)}")
        src = str(src).strip()
        if not src:
            continue
        if not os.path.isfile(src):
            failed.append(f"{src}: 文件不存在")
            continue
        if not src.lower().endswith(AUDIO_EXTS):
            skipped.append(f"{os.path.basename(src)}: 不是支持的音频格式")
            continue
        src_abs = os.path.abspath(src)
        key = os.path.normcase(src_abs)
        if key in seen_sources:
            skipped.append(f"{os.path.basename(src)}: 已导入过（同一源文件）")
            continue

        if copy:
            base = os.path.basename(src)
            stem, ext = os.path.splitext(base)
            # 统一重命名为 `<uid>.<ext>`：既避开同名覆盖，也让 meta 里的
            # audio 字段与 id 一一对应（排查问题时看文件名就知道是哪条）。
            uid = gen()
            dst_name = f"{uid}{ext.lower()}"
            dst = os.path.join(audio_dir, dst_name)
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                failed.append(f"{base}: 复制失败 {e}")
                continue
            rel = os.path.join(AUDIO_SUBDIR, dst_name)
        else:
            uid = gen()
            rel = src_abs

        seen_sources.add(key)
        staged.append((uid, rel, src_abs))

    # ---- 入册（读-改-写，加锁）。拷贝是慢操作放在锁外；锁内重读 meta，
    # 中途别处写入的字段（比如用户同时补文本）不会丢。
    if staged:
        with _META_LOCK:
            items = load_meta(name)
            have = {u.id for u in items}
            for uid, rel, src_abs in staged:
                if uid in have:               # 有并发导入时 uid 撞号，重发一个
                    uid = _next_id(items)()
                u = Utterance(id=uid, audio=rel, lang=lang, source=src_abs)
                evaluate(u, d, require_features=False)
                items.append(u)
                added.append(uid)
            save_meta(name, items)
    if progress:
        progress(1.0, "导入完成")
    return {"added": len(added), "skipped": skipped, "failed": failed,
            "ids": added}


# ---------------------------------------------------------------------------
# 长音频切分
# ---------------------------------------------------------------------------

def split_long(name: str, uid: str, target_sec: float = 12.0,
               min_sec: float = 4.0, max_pieces: int = 12,
               text: str = "") -> Dict[str, Any]:
    """把一条过长的样本切成多条。

    复用「参考音频工作台」的 find_segments —— 它按语音占比/信噪比/削波/响度
    打分，并优先让切分点落在停顿处，比等长硬切好得多。

    切分产物本身无法继承文本（各片说的不是同一句话），所以 `text` 默认留空：
    由调用方补（「一键三连」的做法是先切片再**逐片转写**，见
    training/oneclick.py 的对齐说明），或用户在数据集页手填。

    并发：切片是慢 IO（每个片段都要解码+重采样+写文件），所以**文件写出放在
    锁外**，只在最后入册时加锁重读 meta —— 否则用户在 UI 里补文本会被
    这里最后那次 save 用旧快照覆盖掉（import_audio 用的是同一套 staged 模式）。
    """
    from webui_app.services import audio_lab as AL

    d = dir_of(name)
    with _META_LOCK:
        items = load_meta(name)
    src_u = next((u for u in items if u.id == uid), None)
    if src_u is None:
        return {"ok": False, "error": f"样本 {uid} 不存在"}
    ap = src_u.audio_abs(d)
    if not os.path.isfile(ap):
        return {"ok": False, "error": "源音频文件不存在"}

    segs = AL.find_segments(ap, target_sec=target_sec, min_sec=min_sec,
                            max_candidates=max_pieces * 2, hop_sec=1.0)
    if not segs:
        return {"ok": False, "error": "找不到合适的切分点（音频可能太短或全是静音）"}

    # 去掉重叠的片段，按时间顺序排
    segs = sorted(segs, key=lambda s: s.start)
    picked = []
    last_end = -1.0
    for s in segs:
        if s.start >= last_end - 0.05:
            picked.append(s)
            last_end = s.end
        if len(picked) >= max_pieces:
            break

    # ---- 锁外：切片文件写盘（慢）----
    gen = _next_id(items)
    staged: List[Tuple[str, str, int, Any]] = []     # (uid, rel, 片号, seg)
    fails: List[str] = []
    for k, s in enumerate(picked):
        new_id = gen()
        out_rel = os.path.join(AUDIO_SUBDIR, f"{new_id}.wav")
        try:
            AL.extract_segment(ap, s, os.path.join(d, out_rel))
        except Exception as e:
            fails.append(f"片段 {k + 1}: {type(e).__name__}: {e}")
            continue
        staged.append((new_id, out_rel, k, s))

    # ---- 锁内：重读 meta 再入册，避免覆盖并发写入 ----
    created: List[str] = []
    if staged:
        with _META_LOCK:
            items = load_meta(name)
            have = {u.id for u in items}
            for new_id, out_rel, k, s in staged:
                if new_id in have:                  # 并发导入撞号，重发一个
                    old_abs = os.path.join(d, out_rel)
                    new_id = _next_id(items)()
                    out_rel = os.path.join(AUDIO_SUBDIR, f"{new_id}.wav")
                    try:
                        os.replace(old_abs, os.path.join(d, out_rel))
                    except OSError:
                        pass
                nu = Utterance(
                    id=new_id, audio=out_rel, lang=src_u.lang, text=text,
                    note=f"由 {uid} 切分而来（片段 {k + 1}/{len(picked)}，"
                         f"{s.start:.2f}~{s.end:.2f}s，评分 {s.score:.0f}）",
                )
                evaluate(nu, d, require_features=False)
                items.append(nu)
                created.append(new_id)
            save_meta(name, items)

    out: Dict[str, Any] = {"ok": True, "created": len(created), "ids": created,
                           "segments": [(round(s.start, 2), round(s.end, 2),
                                         round(s.score, 1))
                                        for s in picked]}
    if fails:
        out["failed"] = fails
    return out


# ---------------------------------------------------------------------------
# 训练/验证划分
# ---------------------------------------------------------------------------

def make_split(name: str, val_ratio: float = 0.05, seed: int = 42,
               min_val: int = 1) -> Dict[str, Any]:
    """只在 status == ready 的样本里划分。"""
    import random
    items = load_meta(name)
    ready = [u.id for u in items if u.status == "ready"]
    if not ready:
        return {"ok": False, "error": "没有状态为 ready 的样本，无法划分"}
    rng = random.Random(seed)
    ids = list(ready)
    rng.shuffle(ids)
    n_val = max(min_val, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
    n_val = min(n_val, len(ids) - 1)      # 至少留一条给训练
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])
    data = {"train": train, "val": val, "seed": seed,
            "val_ratio": val_ratio, "created_at": time.time(),
            "total_ready": len(ready)}
    _write_json(os.path.join(dir_of(name), SPLIT_FILE), data)
    return {"ok": True, "train": len(train), "val": len(val),
            "total_ready": len(ready)}


def load_split(name: str) -> Dict[str, Any]:
    return _read_json(os.path.join(dir_of(name), SPLIT_FILE), {})


# ---------------------------------------------------------------------------
# 统计与渲染
# ---------------------------------------------------------------------------

def stats(name: str) -> Dict[str, Any]:
    items = load_meta(name)
    by_status: Dict[str, int] = {}
    total_sec = 0.0
    ready_sec = 0.0
    with_feat = 0
    for u in items:
        by_status[u.status] = by_status.get(u.status, 0) + 1
        total_sec += u.duration
        if u.status == "ready":
            ready_sec += u.duration
        if u.has_features:
            with_feat += 1
    sp = load_split(name)
    return {
        "name": name,
        "total": len(items),
        "ready": by_status.get("ready", 0),
        "by_status": by_status,
        "total_minutes": round(total_sec / 60.0, 2),
        "ready_minutes": round(ready_sec / 60.0, 2),
        "with_features": with_feat,
        "train": len(sp.get("train", [])),
        "val": len(sp.get("val", [])),
        "has_split": bool(sp),
    }


def stats_markdown(name: str) -> str:
    s = stats(name)
    icon = lambda k: STATUS_INFO.get(k, ("·",))[0]      # noqa: E731
    L = [
        f"### 数据集 `{name}`",
        "",
        "| 项 | 值 |", "|---|---|",
        f"| 样本总数 | **{s['total']}** |",
        f"| 可训练（ready） | **{s['ready']}** |",
        f"| 总时长 | {s['total_minutes']} 分钟 |",
        f"| 可训练时长 | **{s['ready_minutes']} 分钟** |",
        f"| 已预提取特征 | {s['with_features']} / {s['total']} |",
        f"| train / val | {s['train']} / {s['val']}"
        f"{'（未划分）' if not s['has_split'] else ''} |",
        "",
    ]
    if s["by_status"]:
        L += ["**状态分布**", "", "| 状态 | 数量 | 可训练 |", "|---|---|---|"]
        for k, v in sorted(s["by_status"].items(), key=lambda x: -x[1]):
            usable = "✅" if STATUS_INFO.get(k, ("", False))[1] else "—"
            L.append(f"| {icon(k)} `{k}` | {v} | {usable} |")
        L.append("")

    n_ready = s["ready"]
    if n_ready == 0:
        L.append("> 🔴 **还没有可训练样本**。请先补齐文本（ASR 转写）并预提取特征。")
    else:
        mins = s["ready_minutes"]
        if mins < 1:
            L.append(f"> 🟠 可训练素材仅 **{mins} 分钟**。LoRA 能跑，"
                     "但音色/韵律的改变会很有限，容易过拟合。")
        elif mins < 5:
            L.append(f"> 🟡 可训练素材 **{mins} 分钟**。适合做**音色微调**"
                     "（r=8~16，少量 epoch，注意过拟合）。")
        elif mins < 30:
            L.append(f"> 🟢 可训练素材 **{mins} 分钟**。这是 LoRA 微调的甜区，"
                     "足以学到稳定的音色与说话习惯。")
        else:
            L.append(f"> 🟢 可训练素材 **{mins} 分钟**，相当充足。"
                     "可以考虑更大的 rank 或更多 epoch。")
    return "\n".join(L)


def table_markdown(name: str, limit: int = 400,
                   only: Optional[str] = None) -> str:
    items = load_meta(name)
    if only:
        items = [u for u in items if u.status == only]
    shown = items[:limit]
    if not shown:
        return "_（空）_"
    L = ["| ID | 状态 | 时长 | SNR | 评分 | 语言 | 特征 | 文本 |",
         "|---|---|---|---|---|---|---|---|"]
    for u in shown:
        ic = STATUS_INFO.get(u.status, ("·",))[0]
        txt = (u.text or u.asr_text or "").replace("|", "\\|").replace("\n", " ")
        if len(txt) > 46:
            txt = txt[:46] + "…"
        if not txt:
            txt = "_（无文本）_"
        L.append(f"| `{u.id}` | {ic} | {u.duration:.2f}s | {u.snr_db:.0f} | "
                 f"{u.score:.0f} | {u.lang} | {'✅' if u.has_features else '—'} "
                 f"| {txt} |")
    if len(items) > limit:
        L.append(f"\n_仅显示前 {limit} 条，共 {len(items)} 条。_")
    return "\n".join(L)


def detail_markdown(name: str, uid: str) -> str:
    d = dir_of(name)
    u = get(name, uid)
    if u is None:
        return f"_样本 `{uid}` 不存在_"
    L = [
        f"### 样本 `{u.id}`",
        "",
        "| 项 | 值 |", "|---|---|",
        f"| 状态 | {STATUS_INFO.get(u.status, ('·',))[0]} `{u.status}` |",
        f"| 音频 | `{u.audio}` |",
        f"| 时长 | {u.duration:.3f} s |",
        f"| 采样率 | {u.sample_rate} Hz |",
        f"| 声道 | {u.channels} |",
        f"| 信噪比 | {u.snr_db:.1f} dB |",
        f"| 体检评分 | {u.score:.1f} |",
        f"| 语言 | {u.lang} |",
        f"| 已预提取特征 | {'✅ ' + time.strftime('%m-%d %H:%M', time.localtime(u.features_at)) if u.has_features else '—'} |",
        f"| ASR 模型 | {u.asr_model or '—'} |",
        "",
        "**训练文本**", "", f"> {u.text or '_（空）_'}", "",
    ]
    if u.asr_text and u.asr_text != u.text:
        L += ["**ASR 原始转写**（未校对）", "", f"> {u.asr_text}", ""]
    if u.problems:
        L += ["**存在的问题**", ""] + [f"- {p}" for p in u.problems] + [""]
    if u.note:
        L += ["**备注**", "", u.note, ""]
    ap = u.audio_abs(d)
    if not os.path.isfile(ap):
        L.append(f"> ❌ 音频文件丢失：`{ap}`")
    return "\n".join(L)


def audio_path(name: str, uid: str) -> Optional[str]:
    u = get(name, uid)
    if u is None:
        return None
    p = u.audio_abs(dir_of(name))
    return p if os.path.isfile(p) else None


def ids(name: str, only_ready: bool = True) -> List[str]:
    return [u.id for u in load_meta(name)
            if (u.status == "ready" or not only_ready)]
