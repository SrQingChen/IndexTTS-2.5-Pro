"""services/artifacts.py —— 产物盘点与安全清理。

「清理」页的后端。回答两个问题：

    1. 这台机器上有哪些**产物**？（一键三连 / 自训练 / 工作台 / 合成产生的
       中间文件、数据集、LoRA、音色库条目……）
    2. 哪些可以安全地删掉？

安全模型（这一页的删除是**永久删除**，不进回收站，所以宁可保守）：

    · 白名单根目录：只有 artifacts.py 自己登记的根（outputs 子目录、
      datasets/、training_runs/、voice_bank/audio/）里的**子项**可以被删，
      根目录本身永远不删 —— 引擎与各 Tab 都假设这些目录存在。
    · 永不出现在盘点里的：checkpoints/（底模）、项目代码、outputs/presets/
      （用户预设）、outputs/logs/（正被本进程写入，Windows 下删不掉且会丢
      排查现场）、training_runs/base_manifest.json（底座只读快照的凭证）。
    · 训练进行中的 run 标记 protected，按钮置灰不可删。
    · 删除前按 key 重新核对磁盘（扫描结果可能已过期），路径做 realpath
      包含性校验，防止构造 key 越权删到白名单之外。

能复用的删除 API 全部复用（数据集走 DS.delete、训练 run 走 runs.delete_run、
音色走 voice_bank.remove），保证索引文件（index.json 等）同步更新，
不会留下「目录没了索引还在」的悬空状态。
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from webui_app.config import AppConfig, PROJECT_ROOT
from webui_app.services.monitor import human_size

# ---------------------------------------------------------------------------
# 分类定义
# ---------------------------------------------------------------------------

# (kind, 图标, 分类名, 说明) —— 顺序即界面展示顺序：
# 越靠前的越是「流水线产物」（一键三连 → 评测 → 训练），越靠后的越是
# 零散产物（切片 / 单条合成 / 缓存），用户清磁盘时通常从上往下看。
CATEGORIES: List[Tuple[str, str, str, str]] = [
    ("oneclick",  "🚀", "一键三连产物", "outputs/oneclick/ 下各候选的评选试听与报告"),
    ("eval",      "🧪", "A/B 评测产物", "outputs/eval/ 下的对比评测输出"),
    ("train_run", "🎓", "训练运行",     "training_runs/ 下的 LoRA 运行（adapter + 保险库 checkpoint + 日志）"),
    ("lora",      "📦", "LoRA 导出",    "outputs/lora/ 下导出的独立权重"),
    ("train_int", "⚙️", "训练中间文件", "outputs/train/ 下的训练中间产物"),
    ("dataset",   "🗃", "数据集",       "datasets/ 下的数据集（音频 + 文本 + 已提取特征）"),
    ("voice",     "🎙", "音色库条目",   "voice_bank/ 中保存的音色（含体检记录）"),
    ("lab",       "🔬", "工作台切片",   "outputs/lab/ 下的智能切片与降噪归一文件"),
    ("synth",     "🔊", "合成音频",     "outputs/ 根目录下的单次合成结果"),
    ("batch",     "🧾", "批量任务",     "outputs/tasks/ 下的批量合成产物"),
    ("cache",     "💨", "缓存",         "outputs/cache/，删了下次使用会自动重建"),
]

CATEGORY_INFO: Dict[str, Tuple[str, str, str]] = {k: (i, l, d) for k, i, l, d in CATEGORIES}

# 数据集名前缀：一键三连流水线自动建的带 oneclick_ 前缀（oneclick._auto_dataset_name）
ONECLICK_PREFIX = "oneclick_"


@dataclass
class ArtifactItem:
    """一条可展示 / 可删除的产物。

    key 是跨扫描稳定的标识（"<kind>:<name>"），删除回调只回传 key，
    服务端按 key 重新定位磁盘路径 —— 不信任客户端传来的任何路径。
    """

    key: str
    kind: str
    name: str
    path: str                  # 展示用相对路径
    size: int = 0
    n_files: int = 0
    mtime: float = 0.0
    origin: str = ""           # 一键三连 / 手动 —— 从产物元数据推断
    note: str = ""             # 状态补充（如「训练进行中」「含 3 个 checkpoint」）
    protected: bool = False    # True = 只展示不可删（按钮置灰）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "kind": self.kind, "name": self.name,
            "path": self.path, "size": self.size, "n_files": self.n_files,
            "mtime": self.mtime, "origin": self.origin, "note": self.note,
            "protected": self.protected,
        }


@dataclass
class ScanResult:
    """一次完整盘点。items 按 CATEGORIES 顺序排好。"""

    items: List[ArtifactItem] = field(default_factory=list)
    scanned_at: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return sum(i.size for i in self.items)

    @property
    def deletable_size(self) -> int:
        return sum(i.size for i in self.items if not i.protected)

    def by_kind(self) -> Dict[str, List[ArtifactItem]]:
        out: Dict[str, List[ArtifactItem]] = {k: [] for k, _i, _l, _d in CATEGORIES}
        for it in self.items:
            out.setdefault(it.kind, []).append(it)
        return out

    def find(self, key: str) -> Optional[ArtifactItem]:
        for it in self.items:
            if it.key == key:
                return it
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "scanned_at": self.scanned_at,
            "warnings": list(self.warnings),
            "total_size": self.total_size,
        }


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def scan(cfg: AppConfig) -> ScanResult:
    """盘点全部产物目录。只读磁盘，不加载任何模型。"""
    res = ScanResult(scanned_at=time.time())
    for fn in (_scan_oneclick, _scan_eval, _scan_train_runs, _scan_lora,
               _scan_train_int, _scan_datasets, _scan_voices, _scan_lab,
               _scan_synth, _scan_batch, _scan_cache):
        try:
            res.items.extend(fn(cfg))
        except Exception as e:                     # 单类失败不连累整页
            res.warnings.append(f"{fn.__name__.strip('_')}: {type(e).__name__}: {e}")
    order = {k: n for n, (k, *_r) in enumerate(CATEGORIES)}
    res.items.sort(key=lambda i: (order.get(i.kind, 99), -i.mtime))
    return res


def _walk_stat(path: str) -> Tuple[int, int, float]:
    """(总字节数, 文件数, 最新 mtime)。目录与单文件通用。"""
    size, nfiles, newest = 0, 0, 0.0
    if os.path.isfile(path):
        try:
            st = os.stat(path)
            return st.st_size, 1, st.st_mtime
        except OSError:
            return 0, 0, 0.0
    for root, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
                size += st.st_size
                nfiles += 1
                newest = max(newest, st.st_mtime)
            except OSError:
                pass
    return size, nfiles, newest


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    except ValueError:
        return path


def _dir_children(path: str) -> List[str]:
    """目录下第一层子项（目录或文件），按 mtime 新→旧。"""
    if not os.path.isdir(path):
        return []
    out = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            out.append((os.path.getmtime(p), p))
        except OSError:
            out.append((0.0, p))
    return [p for _t, p in sorted(out, reverse=True)]


def _scan_oneclick(cfg: AppConfig) -> List[ArtifactItem]:
    out = []
    for p in _dir_children(os.path.join(cfg.output_dir, "oneclick")):
        size, nfiles, mtime = _walk_stat(p)
        name = os.path.basename(p)
        has_report = os.path.isfile(os.path.join(p, "report.md"))
        out.append(ArtifactItem(
            key=f"oneclick:{name}", kind="oneclick", name=name,
            path=_rel(p), size=size, n_files=nfiles, mtime=mtime,
            origin="一键三连",
            note=("含评选报告" if has_report else "无报告（可能中途被清理）"),
        ))
    return out


def _scan_eval(cfg: AppConfig) -> List[ArtifactItem]:
    out = []
    for p in _dir_children(os.path.join(cfg.output_dir, "eval")):
        size, nfiles, mtime = _walk_stat(p)
        name = os.path.basename(p)
        out.append(ArtifactItem(
            key=f"eval:{name}", kind="eval", name=name,
            path=_rel(p), size=size, n_files=nfiles, mtime=mtime,
            note="A/B 对比试听与结论",
        ))
    return out


def _active_run_name() -> str:
    """当前正在训练的 run 名（没有则空串）。删除保护用。"""
    try:
        from webui_app.training.runner import get_runner
        r = get_runner()
        if r.running:
            return str(r._log_run or "")
    except Exception:
        pass
    return ""


def _scan_train_runs(cfg: AppConfig) -> List[ArtifactItem]:
    from webui_app.training import runs as RN
    out = []
    active = _active_run_name()
    for ri in RN.list_runs():
        d = RN.run_dir(ri.name)
        size, nfiles, mtime = _walk_stat(d)
        origin = "一键三连" if ri.dataset.startswith(ONECLICK_PREFIX) else "手动训练"
        notes = [f"{ri.arch.upper()} · 数据集 {ri.dataset or '?'}"]
        if ri.n_checkpoints:
            notes.append(f"{ri.n_checkpoints} 个 checkpoint")
        if ri.has_adapter:
            notes.append("含可用 adapter")
        protected = (ri.status == "running") or (ri.name == active)
        if protected:
            notes.append("训练进行中，不可删除")
        elif ri.status == "error":
            notes.append("上次运行出错")
        out.append(ArtifactItem(
            key=f"train_run:{ri.name}", kind="train_run", name=ri.name,
            path=_rel(d), size=size, n_files=nfiles, mtime=mtime,
            origin=origin, note=" · ".join(notes), protected=protected,
        ))
    return out


def _scan_lora(cfg: AppConfig) -> List[ArtifactItem]:
    out = []
    for p in _dir_children(cfg.lora_dir):
        size, nfiles, mtime = _walk_stat(p)
        name = os.path.basename(p)
        out.append(ArtifactItem(
            key=f"lora:{name}", kind="lora", name=name,
            path=_rel(p), size=size, n_files=nfiles, mtime=mtime,
            note="导出的独立权重（合并/分享用）",
        ))
    return out


def _scan_train_int(cfg: AppConfig) -> List[ArtifactItem]:
    out = []
    for p in _dir_children(cfg.train_dir):
        size, nfiles, mtime = _walk_stat(p)
        name = os.path.basename(p)
        out.append(ArtifactItem(
            key=f"train_int:{name}", kind="train_int", name=name,
            path=_rel(p), size=size, n_files=nfiles, mtime=mtime,
            note="训练过程中间文件",
        ))
    return out


def _scan_datasets(cfg: AppConfig) -> List[ArtifactItem]:
    from webui_app.training import dataset as DS
    out = []
    for name in DS.list_datasets():
        d = DS.dir_of(name)
        size, nfiles, mtime = _walk_stat(d)
        n_audio = len(_dir_children(os.path.join(d, DS.AUDIO_SUBDIR))) \
            if hasattr(DS, "AUDIO_SUBDIR") else nfiles
        out.append(ArtifactItem(
            key=f"dataset:{name}", kind="dataset", name=name,
            path=_rel(d), size=size, n_files=nfiles, mtime=mtime,
            origin="一键三连" if name.startswith(ONECLICK_PREFIX) else "手动创建",
            note=f"{n_audio} 条音频 · 音频与特征一并删除",
        ))
    return out


def _scan_voices(cfg: AppConfig) -> List[ArtifactItem]:
    from webui_app.services import voice_bank
    out = []
    for e in voice_bank.list_voices():
        out.append(ArtifactItem(
            key=f"voice:{e.name}", kind="voice", name=e.name,
            path=_rel(e.audio_path) if e.audio else "voice_bank/",
            size=_walk_stat(e.audio_path)[0] if e.audio_path else 0,
            n_files=1, mtime=e.created_at,
            note=f"体检 {e.score:.0f} 分（{e.grade}）"
                 + (f" · {e.duration:.1f}s" if e.duration else ""),
        ))
    return out


def _scan_lab(cfg: AppConfig) -> List[ArtifactItem]:
    out = []
    lab = os.path.join(cfg.output_dir, "lab")
    for p in _dir_children(lab):
        if not os.path.isfile(p):
            continue
        size, _n, mtime = _walk_stat(p)
        name = os.path.basename(p)
        out.append(ArtifactItem(
            key=f"lab:{name}", kind="lab", name=name,
            path=_rel(p), size=size, n_files=1, mtime=mtime,
            origin="音频工作台",
            note="降噪归一切片" if name.endswith("_enh.wav") or "_enh_" in name else "智能切片",
        ))
    return out


def _scan_synth(cfg: AppConfig) -> List[ArtifactItem]:
    """outputs/ 根目录下的散装音频（单次合成的直接落盘）。"""
    out = []
    for p in _dir_children(cfg.output_dir):
        if not os.path.isfile(p) or not p.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            continue
        size, _n, mtime = _walk_stat(p)
        name = os.path.basename(p)
        out.append(ArtifactItem(
            key=f"synth:{name}", kind="synth", name=name,
            path=_rel(p), size=size, n_files=1, mtime=mtime,
            note="单次合成输出",
        ))
    return out


def _scan_batch(cfg: AppConfig) -> List[ArtifactItem]:
    out = []
    for p in _dir_children(cfg.tasks_dir):
        size, nfiles, mtime = _walk_stat(p)
        name = os.path.basename(p)
        out.append(ArtifactItem(
            key=f"batch:{name}", kind="batch", name=name,
            path=_rel(p), size=size, n_files=nfiles, mtime=mtime,
            note="批量任务产物",
        ))
    return out


def _scan_cache(cfg: AppConfig) -> List[ArtifactItem]:
    """缓存按「整类一条」展示：碎片化的小文件逐条列出没有意义。"""
    p = cfg.cache_dir
    if not os.path.isdir(p):
        return []
    size, nfiles, mtime = _walk_stat(p)
    if size == 0 and nfiles == 0:
        return []
    return [ArtifactItem(
        key="cache:__all__", kind="cache", name="推理缓存",
        path=_rel(p), size=size, n_files=nfiles, mtime=mtime,
        note=f"{nfiles} 个文件 · 删除后下次合成自动重建",
    )]


# ---------------------------------------------------------------------------
# 删除
# ---------------------------------------------------------------------------

def _root_of(cfg: AppConfig, kind: str) -> Optional[str]:
    """kind 对应的白名单根目录。"""
    return {
        "oneclick":  os.path.join(cfg.output_dir, "oneclick"),
        "eval":      os.path.join(cfg.output_dir, "eval"),
        "train_run": os.path.join(PROJECT_ROOT, "training_runs"),
        "lora":      cfg.lora_dir,
        "train_int": cfg.train_dir,
        "dataset":   cfg.dataset_dir,
        "voice":     os.path.join(PROJECT_ROOT, "voice_bank"),
        "lab":       os.path.join(cfg.output_dir, "lab"),
        "synth":     cfg.output_dir,
        "batch":     cfg.tasks_dir,
        "cache":     cfg.cache_dir,
    }.get(kind)


def _is_under(path: str, root: str) -> bool:
    """realpath 包含性校验：path 必须严格位于 root 之内（不能等于 root）。"""
    try:
        rp = os.path.realpath(path)
        rr = os.path.realpath(root)
    except OSError:
        return False
    if rp == rr:
        return False
    try:
        return os.path.relpath(rp, rr).split(os.sep, 1)[0] not in ("..", ".")
    except ValueError:
        return False


def delete_items(cfg: AppConfig, keys: List[str]) -> Dict[str, Any]:
    """按 key 删除产物。返回 {ok, deleted, failed, freed_bytes, message}。

    删除的是扫描时刻之后**仍然存在**的项；期间被别处删掉的记为 skip，
    不算失败 —— 清理页的目标是「目录变干净」，不是「每发必中」。
    """
    deleted: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []
    freed = 0

    for key in keys or []:
        kind, _, name = str(key).partition(":")
        if not name or kind not in CATEGORY_INFO:
            failed.append(f"{key}: 无法识别的产物类型")
            continue

        # ---- 各 kind 的专用删除路径（同步索引文件） ----
        try:
            if kind == "dataset":
                from webui_app.training import dataset as DS
                if not DS.exists(name):
                    skipped.append(key)
                    continue
                d = DS.dir_of(name)
                if not _is_under(d, DS.root()):
                    failed.append(f"{key}: 路径越界，已拒绝删除")
                    continue
                size = _walk_stat(d)[0]
                if DS.delete(name):
                    deleted.append(key)
                    freed += size
                else:
                    failed.append(f"{key}: 删除后目录仍在（可能被占用）")
                continue

            if kind == "train_run":
                from webui_app.training import runs as RN
                d = RN.run_dir(name)
                if not os.path.isdir(d):
                    skipped.append(key)
                    continue
                # run_dir 自带 safe_run_name 净化，这里再做一次包含性校验：
                # 删除是 rmtree，纵深防御比单层拦截可靠
                if not _is_under(d, RN.root()):
                    failed.append(f"{key}: 路径越界，已拒绝删除")
                    continue
                if RN.info_of(name).status == "running" or name == _active_run_name():
                    failed.append(f"{key}: 训练进行中，已拒绝删除")
                    continue
                size = _walk_stat(d)[0]
                if RN.delete_run(name):
                    deleted.append(key)
                    freed += size
                else:
                    failed.append(f"{key}: 删除后目录仍在（可能被占用）")
                continue

            if kind == "voice":
                from webui_app.services import voice_bank
                if not voice_bank.remove(name):
                    skipped.append(key)      # 不存在或音频已丢
                else:
                    deleted.append(key)
                continue

            # ---- 通用文件/目录删除（带白名单校验） ----
            if kind == "cache" and name == "__all__":
                target, root, keep_root = cfg.cache_dir, cfg.cache_dir, True
            else:
                root = _root_of(cfg, kind)
                if root is None:
                    failed.append(f"{key}: 该类型不允许删除")
                    continue
                target = os.path.join(root, name)
                keep_root = False          # 删的是子项本身

            if not os.path.exists(target):
                skipped.append(key)
                continue
            if not _is_under(target, root):
                failed.append(f"{key}: 路径越界，已拒绝删除")
                continue

            size = _walk_stat(target)[0]
            ok = _remove(target, keep_root=keep_root)
            if ok:
                deleted.append(key)
                freed += size
            else:
                failed.append(f"{key}: 删除失败（可能被其他程序占用）")
        except Exception as e:
            failed.append(f"{key}: {type(e).__name__}: {e}")

    return {
        "ok": not failed,
        "deleted": deleted,
        "failed": failed,
        "skipped": skipped,
        "freed_bytes": freed,
        "freed_text": human_size(freed),
    }


def _remove(target: str, keep_root: bool = False) -> bool:
    """删除文件或目录。keep_root=True 时只清空内容保留目录本身。"""
    try:
        if keep_root:
            for entry in os.listdir(target):
                p = os.path.join(target, entry)
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return True
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
        return not os.path.exists(target)
    except OSError:
        return False
