"""gts_fetch — GTSinger 数据镜像拉取（hf-mirror 优先，断点续传，可停止）。

为什么走 hf-mirror 而不是 GitHub 加速器：GTSinger 托管在 HuggingFace
Datasets（GTSinger/GTSinger），本项目 `webui_app.config` 已把
`HF_ENDPOINT` 默认设为 https://hf-mirror.com（直连 huggingface.co 在
国内网络超时）——不涉及 GitHub。

**真实仓库布局（2026-10 实测探明）**：
    Chinese/ZH-Alto-1/<Technique>/<Song>/<Group>/<NNNN>.{wav,json,TextGrid,musicxml}
    - 顶层目录是语言全名（Chinese/English/…），不是语言码；
    - Group 有三种：<Tech>_Group（带技巧演唱）/ Control_Group（同句无技巧
      演唱，同一首歌会出现在多个技巧目录下——uid 去重靠导入器）/
      Paired_Speech_Group（配对**语音**，capability 不拉）；
    - json 与 wav 同目录同名（逐词标注，schema 见 gts_import）。

**两个实测坑（本模块的为什么）**：
1. `HfApi().list_repo_files` 的分页「下一页」Link 由 hf-mirror 返回时指向
   huggingface.co 原站，跟着跳就直连超时——所以清单获取用**裸 requests
   固定打镜像端点**、cursor 自己从 Link 里抽参数回填，绝不跟随绝对 URL。
2. 只列**所选语言子树**（tree API 带 path 参数），不列全仓——全仓几十万
   文件要翻几百页，子树只需几页到几十页。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from webui_app.config import PROJECT_ROOT

REPO_ID = "GTSinger/GTSinger"
# 原始数据统一放 datasets/_sources/（gitignore 已覆盖 datasets/；
# list_datasets 只认带 info 文件的目录，_sources 不会混进数据集下拉）
DEFAULT_DEST = os.path.join(PROJECT_ROOT, "datasets", "_sources", "GTSinger")

# 语言码 ↔ 仓库顶层目录名（实测：顶层是全名不是码）
LANG_DIR = {"zh": "Chinese", "en": "English", "ja": "Japanese",
            "ko": "Korean", "de": "German", "fr": "French",
            "it": "Italian", "es": "Spanish", "ru": "Russian"}
DIR_LANG = {v: k for k, v in LANG_DIR.items()}
GTS_LANGS = list(LANG_DIR.keys())
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

    langs 用语言码（zh/en/…）或目录全名（Chinese/…）皆可。返回
    {"files": [...], "mode": "lang-subset"|"full", "note": str}：
    - 顶层目录命中所选语言 → 语言子集 + 根级元数据文件；
    - 一个都没命中 → 退回全量并说明原因——宁可全量也不能静默给空集。
    """
    langs = [l for l in (langs or []) if l]
    if not langs:
        return {"files": list(files), "mode": "full", "note": "未选语言，拉取全量"}
    want_dirs = {LANG_DIR.get(l, l) for l in langs}
    top = {f.split("/", 1)[0] for f in files if "/" in f}
    hit = sorted(want_dirs & top)
    if not hit:
        return {"files": list(files), "mode": "full",
                "note": f"仓库顶层没有 {sorted(want_dirs)}（布局可能已变化），退回全量"}
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


def _tree_page(session, endpoint: str, path: Optional[str], cursor: str,
               recursive: bool):
    """裸 requests 打镜像的 tree API（单页）。

    不跟随响应里的绝对 next 链接（那会指回 huggingface.co）——只抽
    cursor 参数回填到镜像 URL。path 是 URL 路径段（子树），不是查询参数
    （实测查询参数版 400）。
    """
    from urllib.parse import quote
    # URL 形态对齐 huggingface_hub 实测：repo 的斜杠**不**编码，子路径的
    # 斜杠编码（…/tree/main/Chinese%2FZH-Alto-1）。repo 斜杠编码成 %2F 会被
    # hf-mirror 判 400。
    url = f"{endpoint}/api/datasets/{REPO_ID}/tree/main"
    if path:
        url += "/" + quote(path, safe="")
    params: Dict[str, str] = {"recursive": "true" if recursive else "false",
                              "expand": "false"}
    if cursor:
        params["cursor"] = cursor
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    nxt = r.links.get("next", {}).get("url")
    next_cursor = ""
    if nxt:
        q = parse_qs(urlparse(nxt).query)
        next_cursor = (q.get("cursor") or [""])[0]
    return rows, next_cursor


def _walk_tree(session, endpoint: str, path: Optional[str]) -> List[Dict]:
    out: List[Dict] = []
    cursor = ""
    while True:
        rows, cursor = _tree_page(session, endpoint, path, cursor, True)
        out.extend(rows)
        if not cursor:
            return out


def list_repo_files(langs: Optional[List[str]] = None) -> List[str]:
    """列仓库文件路径。给了 langs 时只列所选语言子树 + 根级元数据。

    Paired_Speech_Group（配对语音）默认**不列**——capability 只要歌声，
    语音组体积可观且用途不同（省下载省磁盘）。
    """
    import requests

    endpoint = ensure_endpoint().rstrip("/")
    s = requests.Session()
    files: List[str] = []
    if langs:
        for code in [l for l in langs if l]:
            sub = LANG_DIR.get(code, code)
            for row in _walk_tree(s, endpoint, sub):
                if row.get("type") != "file":
                    continue
                p = row["path"]
                if "Paired_Speech_Group" in p:
                    continue
                files.append(p)
        # 根级元数据（README/许可等小文件）
        root_rows, cursor = _tree_page(s, endpoint, None, "", False)
        while True:
            for row in root_rows:
                if row.get("type") == "file" and "/" not in row["path"]:
                    if row["path"].lower().endswith(_ROOT_META_EXTS):
                        files.append(row["path"])
            if not cursor:
                break
            root_rows, cursor = _tree_page(s, endpoint, None, cursor, False)
        return files
    # 全量：逐顶层目录走（跳过 processed/ 纯元数据打包，能力训练用不到）
    rows, cursor = _tree_page(s, endpoint, None, "", False)
    tops: List[str] = []
    while True:
        for row in rows:
            if row.get("type") == "directory":
                tops.append(row["path"])
        if not cursor:
            break
        rows, cursor = _tree_page(s, endpoint, None, cursor, False)
    files = [r["path"] for r in rows if r.get("type") == "file"
             and r["path"].lower().endswith(_ROOT_META_EXTS)]
    for sub in tops:
        if sub in ("processed",):
            continue
        for row in _walk_tree(s, endpoint, sub):
            if row.get("type") != "file":
                continue
            if "Paired_Speech_Group" in row["path"]:
                continue
            files.append(row["path"])
    return files


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

    langs = [l for l in (langs or []) if l]
    progress(0.01, "获取仓库文件清单（hf-mirror，语言子树）…")
    try:
        files = list_repo_files(langs or None)
    except Exception as e:
        return {"ok": False, "error": f"文件清单获取失败：{e}\n"
                "（镜像站偶发抖动可重试；确认网络可达 hf-mirror.com）"}
    plan = plan_files(files, langs)
    todo = [f for f in plan["files"]
            # TextGrid/musicxml 是标注冗余（json 已含全部信息），不下载
            if not f.lower().endswith((".textgrid", ".musicxml"))]
    if not todo:
        return {"ok": False, "error": "过滤后没有文件可下载"}

    ensure_endpoint()
    from huggingface_hub import hf_hub_download

    done = skipped = 0
    total = len(todo)
    stop = False
    counter = {"n": 0}

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
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == total:
                progress(0.02 + 0.97 * counter["n"] / total,
                         f"下载 {counter['n']}/{total}"
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
    from webui_app.services.gts_fetch import DIR_LANG
    d = dest or DEFAULT_DEST
    if not os.path.isdir(d):
        return {"exists": False, "path": d, "langs": [], "wavs": 0}
    langs = [DIR_LANG.get(n, n.lower()) for n in sorted(os.listdir(d))
             if os.path.isdir(os.path.join(d, n)) and n in DIR_LANG]
    wavs = 0
    for root, _dirs, files in os.walk(d):
        wavs += sum(1 for f in files if f.lower().endswith(".wav"))
        if wavs > 200_000:            # 防御：超大目录提前停
            break
    return {"exists": True, "path": d, "langs": langs, "wavs": wavs}
