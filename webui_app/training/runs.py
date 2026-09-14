"""训练运行的登记簿：目录布局、run.json、adapter 激活与回滚。

所有训练器（GPT / CFM / DPO）共用同一套布局，这样「训练记录」页、
合并页（b7）、评测台（b6）就不用各自约定一遍路径。

目录结构::

    training_runs/
        base_manifest.json          ← BaseGuard 的底座快照（跨 run 共享）
        <run_name>/
            run.json                ← 配置 + 历史 + 结论（唯一事实来源）
            adapter/                ← **推理端唯一入口**，永远是当前生效的那份
            checkpoints/            ← CheckpointVault 的 root
                index.json
                ckpt-e000-s000100/
                active/             ← vault.rollback() 的落点
            log.txt

`adapter/` 与 `checkpoints/` 分开是有意的：checkpoint 是训练过程的历史档位，
会按 keep_checkpoints 被剪枝删掉；而推理端只需要一个稳定的路径。
每次刷新 best 就把那份 checkpoint 同步进 `adapter/`，
于是「回滚」= 换一个 checkpoint 再同步一次，推理端代码完全不用改。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from webui_app.training import guard as GD

ROOT = GD.TRAINING_ROOT

# update_run 是「读-改-写」，训练线程（epoch 末回写状态）与 UI 线程
# （激活 checkpoint）可能同时进來；不加锁会互相丢字段，极端情况下
# 两个线程共用同一个 run.json.tmp 还会写出损坏的 JSON。
_RUN_LOCK = threading.RLock()

RUN_FILE = "run.json"
ADAPTER_DIR = GD.ADAPTER_SUBDIR          # "adapter"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "log.txt"

# 训练目标：决定加载哪个底座、注入哪些层、用哪个前向
ARCHS = ("gpt", "cfm", "dpo")
ARCH_LABELS = {
    "gpt": "GPT(T2S) · 语气与韵律",
    "cfm": "CFM(S2M) · 音色与音质",
    "dpo": "DPO · 偏好对齐",
}

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
STATUS_LABELS = {
    STATUS_RUNNING: ("🏃 进行中", True),
    STATUS_DONE: ("✅ 完成", True),
    STATUS_STOPPED: ("⏸ 已中止", True),
    STATUS_FAILED: ("✖ 失败", False),
}


# ---------------------------------------------------------------------------
# 命名与路径
# ---------------------------------------------------------------------------

def safe_run_name(name: str) -> str:
    """run 名会当目录名用，规则与 dataset.safe_dataset_name 保持一致。

    两边规则不同会造成同一套 UI 里「数据集能叫的名，训练记录不能叫」，
    用户很难理解，所以刻意复刻而不是各写一份。
    """
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "untitled"


def suggest_run_name(arch: str, dataset: str) -> str:
    """给一个不会撞车的默认名：`gpt_角色A_20260913-1420`。"""
    stamp = time.strftime("%Y%m%d-%H%M")
    base = safe_run_name(f"{arch}_{dataset}_{stamp}")
    if not os.path.isdir(run_dir(base)):
        return base
    for i in range(2, 100):                     # 同一分钟内重复点击也不能撞
        cand = f"{base}_{i}"
        if not os.path.isdir(run_dir(cand)):
            return cand
    return f"{base}_{int(time.time())}"


def root() -> str:
    os.makedirs(ROOT, exist_ok=True)
    return ROOT


def run_dir(name: str, create: bool = False) -> str:
    d = os.path.join(ROOT, safe_run_name(name))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def adapter_dir(name: str, create: bool = False) -> str:
    d = os.path.join(run_dir(name), ADAPTER_DIR)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def checkpoints_dir(name: str, create: bool = False) -> str:
    d = os.path.join(run_dir(name), CHECKPOINT_DIR)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def log_path(name: str) -> str:
    return os.path.join(run_dir(name), LOG_FILE)


def vault(name: str, keep: int = 3) -> GD.CheckpointVault:
    """这个 run 的 checkpoint 保险库。mode 固定 min —— 我们盯的是 val loss。"""
    return GD.CheckpointVault(checkpoints_dir(name, create=True), keep=keep, mode="min")


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, obj: Any) -> None:
    """先写 .tmp 再 os.replace —— 训练中途断电不会留下半个 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_run(name: str) -> Dict[str, Any]:
    p = os.path.join(run_dir(name), RUN_FILE)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_run(name: str, data: Dict[str, Any]) -> None:
    _atomic_write_json(os.path.join(run_dir(name, create=True), RUN_FILE), data)


def update_run(name: str, **kw) -> Dict[str, Any]:
    """改 run.json 的几个字段。训练循环里每步都会调，所以是读-改-写。

    频率高的字段（history）由训练器自己攒在内存里，结束时一次性写；
    这里只负责低频的状态字段，避免每步都重写整个文件。
    """
    with _RUN_LOCK:
        d = read_run(name)
        d.update(kw)
        d["updated_at"] = time.time()
        write_run(name, d)
        return d


def list_checkpoints(name: str) -> List[GD.CkptInfo]:
    """列出某个 run 保险库里的档位。**不创建目录** —— UI 下拉框刷新
    会频繁调它，用 vault() 会在每个 run 名下留一个空 checkpoints/。"""
    v = GD.CheckpointVault(checkpoints_dir(name), mode="min")
    return v.list()


def append_log(name: str, line: str) -> None:
    """追加一行训练日志。用 append 模式，训练崩了也能看到最后写到哪。"""
    p = log_path(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    stamp = time.strftime("%H:%M:%S")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {line}\n")


def read_log(name: str, tail: int = 400) -> str:
    p = log_path(name)
    if not os.path.isfile(p):
        return "_还没有日志。_"
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"_读日志失败：{type(e).__name__}: {e}_"
    return "".join(lines[-max(1, int(tail)):]) or "_日志是空的。_"


# ---------------------------------------------------------------------------
# 登记
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """一条训练记录（UI 列表用）。"""
    name: str
    arch: str = "gpt"
    dataset: str = ""
    status: str = STATUS_RUNNING
    created_at: float = 0.0
    finished_at: float = 0.0
    seconds: float = 0.0
    steps: int = 0
    epochs: int = 0
    best_val: Optional[float] = None
    first_val: Optional[float] = None
    replay_dataset: str = ""
    replay_ratio: float = 0.0
    rank: int = 0
    alpha: int = 0
    target_preset: str = ""
    adapter_params: int = 0
    base_params: int = 0
    has_adapter: bool = False
    n_checkpoints: int = 0
    error: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, (self.status, False))[0]

    @property
    def improved(self) -> Optional[float]:
        """val loss 相对第一次评估降了多少（比例）。None = 还没有可比的数。"""
        if self.first_val is None or self.best_val is None or self.first_val <= 0:
            return None
        return 1.0 - self.best_val / self.first_val

    def adapter_path(self) -> str:
        return adapter_dir(self.name)


def info_of(name: str) -> RunInfo:
    """从磁盘拼出一条 RunInfo。run.json 缺失时也能列出目录（标成失败）。"""
    d = read_run(name)
    cfg = d.get("config") or {}
    hist = d.get("history") or []
    vals = [h.get("val") for h in hist if isinstance(h.get("val"), (int, float))]
    ad = adapter_dir(name)
    has_adapter = False
    if os.path.isdir(ad):
        names = os.listdir(ad)
        has_adapter = ("adapter_config.json" in names
                       or any(f.startswith("adapter_model") for f in names))
    return RunInfo(
        name=name,
        arch=d.get("arch", "gpt"),
        dataset=d.get("dataset", ""),
        status=d.get("status", STATUS_RUNNING),
        created_at=float(d.get("created_at") or 0.0),
        finished_at=float(d.get("finished_at") or 0.0),
        seconds=float(d.get("seconds") or 0.0),
        steps=int(d.get("steps") or 0),
        epochs=int(d.get("epochs") or 0),
        best_val=d.get("best_val"),
        first_val=(vals[0] if vals else None),
        replay_dataset=cfg.get("replay_dataset", ""),
        replay_ratio=float(cfg.get("replay_ratio") or 0.0),
        rank=int(cfg.get("rank") or 0),
        alpha=int(cfg.get("alpha") or 0),
        target_preset=cfg.get("target_preset", ""),
        adapter_params=int(d.get("adapter_params") or 0),
        base_params=int(d.get("base_params") or 0),
        has_adapter=bool(has_adapter),
        n_checkpoints=len(vault(name).list()) if os.path.isdir(checkpoints_dir(name)) else 0,
        error=d.get("error", ""),
        note=d.get("note", ""),
    )


def list_runs(arch: Optional[str] = None) -> List[RunInfo]:
    if not os.path.isdir(ROOT):
        return []
    out: List[RunInfo] = []
    for n in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, n)
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, RUN_FILE)):
            continue
        ri = info_of(n)
        if arch and ri.arch != arch:
            continue
        out.append(ri)
    out.sort(key=lambda r: -(r.created_at or 0.0))
    return out


def delete_run(name: str) -> bool:
    d = run_dir(name)
    if not os.path.isdir(d):
        return False
    shutil.rmtree(d, ignore_errors=True)
    return not os.path.isdir(d)


# ---------------------------------------------------------------------------
# adapter 激活 / 回滚
# ---------------------------------------------------------------------------

def sync_adapter(name: str, src_dir: str) -> int:
    """把一个 checkpoint 目录的内容同步进 `<run>/adapter/`，返回文件数。

    先拷到 `adapter.tmp/` 再整目录替换：推理端可能在另一个线程里读 `adapter/`，
    直接往里写会让它读到半份权重（PEFT 的 safetensors + config 是两个文件）。
    """
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"源目录不存在：{src_dir}")
    dst = adapter_dir(name)
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src_dir, tmp)
    n = sum(len(fs) for _r, _d, fs in os.walk(tmp))
    # 记下这份 adapter 是从哪来的，事后排查「为什么音色变了」时全靠它
    _atomic_write_json(os.path.join(tmp, "provenance.json"), {
        "run": name, "from": os.path.abspath(src_dir), "at": time.time(),
    })
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return n


def activate(name: str, which: str = "best") -> Optional[str]:
    """让某个 checkpoint 成为推理端加载的那份。

    which: "best" | checkpoint 名 | 绝对路径。
    走 `CheckpointVault.rollback()` 落到 `checkpoints/active/`，再同步到 `adapter/`。
    多一次拷贝，但换来「推理端只有一个入口路径」，值。
    """
    v = vault(name)
    act = v.rollback(which)
    if not act:
        return None
    sync_adapter(name, act)
    src = v.best() if which == "best" else next(
        (i for i in v.list() if i.name == which or i.path == which), None)
    update_run(name, active_from=os.path.abspath(act),
               active_metric=(src.metric if src else None),
               activated_at=time.time())
    append_log(name, f"已激活 checkpoint：{which} → adapter/")
    return adapter_dir(name)


def best_checkpoint(name: str) -> Optional[GD.CkptInfo]:
    return vault(name).best()


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def runs_markdown(arch: Optional[str] = None, limit: int = 30) -> str:
    runs = list_runs(arch)
    if not runs:
        scope = ARCH_LABELS.get(arch or "", "任何目标")
        return f"_还没有{scope}的训练记录。_"
    L = ["| 记录 | 目标 | 数据集 | 状态 | val（最好/首次） | 步数 | 回放 | adapter |",
         "|---|---|---|---|---|---|---|---|"]
    for r in runs[:max(1, limit)]:
        best = "—" if r.best_val is None else f"{r.best_val:.4f}"
        first = "—" if r.first_val is None else f"{r.first_val:.4f}"
        imp = r.improved
        imp_s = f"（降 {imp * 100:.1f}%）" if imp is not None else ""
        rp = f"{r.replay_ratio:.0%}" + (f" ← `{r.replay_dataset}`" if r.replay_dataset else "")
        L.append(
            f"| `{r.name}` | {ARCH_LABELS.get(r.arch, r.arch)} | `{r.dataset or '—'}` "
            f"| {r.status_label} | {best} / {first} {imp_s} | {r.steps} | {rp} "
            f"| {'✅' if r.has_adapter else '—'} |")
    if len(runs) > limit:
        L += ["", f"> 只显示最近 {limit} 条，共 {len(runs)} 条。"]
    return "\n".join(L)


def run_detail_markdown(name: str) -> str:
    d = read_run(name)
    if not d:
        return f"_找不到训练记录 `{name}`。_"
    r = info_of(name)
    cfg = d.get("config") or {}
    L = [f"### `{name}`", "",
         f"**目标**：{ARCH_LABELS.get(r.arch, r.arch)}  ",
         f"**数据集**：`{r.dataset or '—'}`"
         + (f"（回放 `{r.replay_dataset}`，{r.replay_ratio:.0%}）" if r.replay_dataset else ""),
         f"**状态**：{r.status_label}  ",
         f"**创建**：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.created_at)) if r.created_at else '—'}",
         f"**耗时**：{r.seconds / 60:.1f} 分钟 · {r.steps} 步 · {r.epochs} epoch  ",
         f"**adapter**：{r.adapter_params / 1e6:.2f} M / 底座 {r.base_params / 1e6:.0f} M"
         + (f"（{r.adapter_params / max(1, r.base_params) * 100:.3f}%）" if r.base_params else ""),
         ""]
    if r.best_val is not None:
        L.append(f"**val loss**：最好 `{r.best_val:.5f}`"
                 + (f"，首次 `{r.first_val:.5f}`（降 {r.improved * 100:.1f}%）"
                    if r.improved is not None else ""))
        L.append("")
    if r.error:
        L += [f"> ✖ **失败原因**：{r.error}", ""]
    notes = d.get("notes") or []
    if notes:
        L += ["**体检提示**", ""]
        icon = {"error": "✖", "warn": "⚠️", "info": "ℹ️"}
        for x in notes:
            L.append(f"> {icon.get(x.get('level'), '·')} {x.get('message', '')}")
            L.append(">")
        if L[-1] == ">":
            L.pop()
        L.append("")
    drift = d.get("drift")
    if drift:
        L += ["**底座漂移体检**", "", drift.get("markdown", "_无_"), ""]
    base = d.get("base_verify")
    if base:
        L += ["**底座完整性**", "",
              f"> {'✅ 完好' if base.get('ok') else '✖ 校验失败'}："
              f"{base.get('checked', 0)} 个文件，{base.get('seconds', 0):.1f}s", ""]
    hist = d.get("history") or []
    if hist:
        L += [f"**训练曲线**（{len(hist)} 个记录点）", "",
              "| epoch | step | train | val | lr | s/step |", "|---|---|---|---|---|---|"]
        for h in hist[-40:]:
            vv = "—" if h.get("val") is None else f"{h['val']:.5f}"
            L.append(f"| {h.get('epoch', 0)} | {h.get('step', 0)} "
                     f"| {h.get('train', float('nan')):.5f} | {vv} "
                     f"| {h.get('lr', 0.0):.2e} | {h.get('sec', 0.0):.2f} |")
        L.append("")
    L += ["**配置快照**", "", "```json",
          json.dumps(cfg, ensure_ascii=False, indent=2)[:3000], "```"]
    return "\n".join(L)
