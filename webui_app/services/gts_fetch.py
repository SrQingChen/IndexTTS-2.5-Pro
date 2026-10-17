"""gts_fetch — GTSinger 数据镜像拉取（hf-mirror 优先，断点续传，可停止）。

为什么走 hf-mirror 而不是 GitHub 加速器：GTSinger 托管在 HuggingFace
Datasets（GTSinger/GTSinger），本项目 `webui_app.config` 已把
`HF_ENDPOINT` 默认设为 https://hf-mirror.com（直连 huggingface.co 在
国内网络超时）——镜像 + `huggingface_hub` 的断点续传就是现成组合，
不涉及 GitHub。

设计：
- **逐文件拉取**而不是 snapshot_download 一把梭：GTSinger 是数万个小文件
  （每 phrase 一个 wav+json），逐文件 + 线程池既能给向导页真实的
  「i/N」进度，也支持中途停止（已完成文件复用 hf_hub 缓存续传）；
- **语言子集过滤**：全量约 40GB 量级，zh+en 子集约 15–20GB——
  按仓库文件列表的顶层语言目录过滤，是控制体积的第一手段；
- `plan_files()` 是纯函数（文件列表 → 过滤结果），可单测钉住。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from webui_app.config import PROJECT_ROOT

REPO_ID = "GTSinger/GTSinger"
# 原始数据统一放 datasets/_sources/（gitignore 已覆盖 datasets/；
# list_datasets 只认带 info 文件的目录，_sources 不会混进数据集下拉）
DEFAULT_DEST = os.path.join(PROJECT_ROOT, "datasets", "_sources", "GTSinger")

# GTSinger 的 9 种语言（顶层目录名）
GTS_LANGS = ["zh", "en", "ja", "ko", "de", "fr", "it", "es", "ru"]
_LANG_NAMES = {"zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语",
               "de": "德语", "fr": "法语", "it": "意语", "es": "西语",
               "ru": "俄语"}

_ROOT_META_EXTS = (".json", ".md", ".txt", ".csv")


def lang_label(code: str) -> str:
    return f"{code}（{_LANG_NAMES.get(code, code)}）"


def ensure_endpoint() -> str:
    """HF 镜像端点（在 huggingface_hub 导入前调用才生效）。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return os.environ["HF_ENDPOINT"]


def plan_files(files: List[str], langs: List[str]) -> Dict[str, object]:
    """仓库文件列表 → 过滤计划（纯函数）。

    返回 {"files": [...], "mode": "lang-subset"|"full", "note": str}：
    - 顶层目录命中所选语言 → 语言子集 + 根级元数据文件；
    - 一个都没命中（布局变化/语言码写错）→ 退回全量并说明原因——
      宁可全量也不能静默给空集（空集会让「下载完成」是假的）。
    """
    langs = [l for l in (langs or []) if l]
    if not langs:
        return {"files": list(files), "mode": "full", "note": "未选语言，拉取全量"}
    top = {f.split("/", 1)[0] for f in files if "/" in f}
    hit = [l for l in langs if l in top]
    if not hit:
        return {"files": list(files), "mode": "full",
                "note": f"仓库顶层没有 {langs}（布局可能已变化），退回全量"}
    keep = []
    for f in files:
        if "/" not in f:
            if f.lower().endswith(_ROOT_META_EXTS):
                keep.append(f)
            continue
        if f.split("/", 1)[0] in hit:
            keep.append(f)
    return {"files": keep, "mode": "lang-subset",
            "note": f"语言子集 {hit}（{len(keep)} 个文件）"}


def list_repo_files() -> List[str]:
    ensure_endpoint()
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(REPO_ID, repo_type="dataset")


def fetch(dest: str, langs: List[str],
          progress: Optional[Callable[[float, str], None]] = None,
          should_stop: Optional[Callable[[], bool]] = None,
          workers: int = 6) -> Dict:
    """拉取 GTSinger 到 dest（断点续传）。返回摘要 dict。

    已存在的文件由 hf_hub 本地缓存/本地目录校验跳过——重复调用是
    增量续传，天然支持「下了一半断了再点一次」。
    """
    dest = dest or DEFAULT_DEST
    os.makedirs(dest, exist_ok=True)
    progress = progress or (lambda f, m: None)

    progress(0.01, "获取仓库文件列表（hf-mirror）…")
    try:
        files = list_repo_files()
    except Exception as e:
        return {"ok": False, "error": f"文件列表获取失败：{e}\n"
                "（镜像站偶发抖动可重试；确认网络可达 hf-mirror.com）"}
    plan = plan_files(files, langs)
    todo = list(plan["files"])
    if not todo:
        return {"ok": False, "error": "过滤后没有文件可下载"}

    ensure_endpoint()
    from huggingface_hub import hf_hub_download

    done = skipped = 0
    total = len(todo)
    stop = False
    lock_free_counter = {"n": 0}

    def _one(fname: str) -> bool:
        p = hf_hub_download(REPO_ID, fname, repo_type="dataset",
                            local_dir=dest)
        return os.path.isfile(p)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_one, f): f for f in todo}
        for fut in as_completed(futs):
            if should_stop is not None and should_stop():
                stop = True
                for x in futs:
                    x.cancel()
                break
            try:
                if fut.result():
                    done += 1
                else:
                    skipped += 1
            except Exception:
                # 单文件失败不炸整链（镜像偶发 5xx）：最后汇总报数
                skipped += 1
            lock_free_counter["n"] += 1
            if lock_free_counter["n"] % 25 == 0 or lock_free_counter["n"] == total:
                progress(0.02 + 0.97 * lock_free_counter["n"] / total,
                         f"下载 {lock_free_counter['n']}/{total}"
                         f"（{plan['mode']}）")
    if stop:
        return {"ok": False, "stopped": True, "dest": dest,
                "done": done, "total": total,
                "error": "用户停止（已完成的文件保留，续传即可）"}
    if skipped > total * 0.05:
        return {"ok": False, "dest": dest, "done": done, "skipped": skipped,
                "total": total,
                "error": f"{skipped}/{total} 个文件失败（镜像抖动），"
                         "再点一次下载按钮续传"}
    progress(1.0, f"完成：{done} 个文件 → {dest}")
    return {"ok": True, "dest": dest, "done": done, "skipped": skipped,
            "total": total, "note": plan["note"]}


def source_state(dest: Optional[str] = None) -> Dict[str, object]:
    """本地数据源状态（向导总览用；纯文件系统读取，轻量）。"""
    d = dest or DEFAULT_DEST
    if not os.path.isdir(d):
        return {"exists": False, "path": d, "langs": [], "wavs": 0}
    langs = [n for n in sorted(os.listdir(d))
             if os.path.isdir(os.path.join(d, n)) and n in GTS_LANGS]
    wavs = 0
    for root, _dirs, files in os.walk(d):
        wavs += sum(1 for f in files if f.lower().endswith(".wav"))
        if wavs > 200_000:            # 防御：超大目录提前停
            break
    return {"exists": True, "path": d, "langs": langs, "wavs": wavs}
