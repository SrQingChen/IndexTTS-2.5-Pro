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
import time
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

    # 清单接口也要重试（2026-10-19 实锤：重试下载时 5.6s 就死在
    # "Response ended prematurely"——镜像偶发断流，一次失败就整任务失败，
    # 用户看到的是「文件已下完却报失败」）。指数退避 3 次。
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            rows = r.json()
            break
        except Exception as e:      # noqa: BLE001 - 网络类异常统一重试
            last = e
            if attempt < 2:
                time.sleep(1.5 * (2 ** attempt))
    else:
        raise RuntimeError(f"清单请求失败（已重试 3 次）：{type(last).__name__}: {last}")

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


INCOMPLETE_FILE = "_incomplete.txt"


def _missing_on_disk(dest: str, files: List[str]) -> List[str]:
    """以**磁盘实况**为准判定缺失（不信下载过程的计数）。

    2026-10-19 实锤的假失败：2.35 小时下完 14280 个文件里 12789 个，
    但过程计数把 1491 个镜像抖动算成失败、又用「失败>5% 即整任务失败」
    一刀切——用户看到「下载结束却失败」，而 wav 其实 7133/7139 已到位。
    判定改成一句：文件在不在磁盘上。
    """
    return [f for f in files if not os.path.isfile(os.path.join(dest, f))]


def _run_pass(todo: List[str], dest: str, workers: int,
              progress: Callable[[float, str], None],
              should_stop: Optional[Callable[[], bool]],
              base: float, span: float, label: str):
    """跑一轮下载。返回 (ok_count, failed_files, stopped)。"""
    from huggingface_hub import hf_hub_download

    total = max(1, len(todo))
    ok = 0
    failed: List[str] = []
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
            fname = futs[fut]
            try:
                if fut.result():
                    ok += 1
                else:
                    failed.append(fname)
            except Exception:
                # 单文件失败不炸整链：进重试轮（镜像 429/断流很常见）
                failed.append(fname)
            counter["n"] += 1
            if counter["n"] % 25 == 0 or counter["n"] == total:
                progress(base + span * counter["n"] / total,
                         f"{label} {counter['n']}/{total}")
    return ok, failed, stop


def fetch(dest: str, langs: List[str],
          progress: Optional[Callable[[float, str], None]] = None,
          should_stop: Optional[Callable[[], bool]] = None,
          workers: int = 4) -> Dict:
    """拉取 GTSinger 到 dest（断点续传 + 失败自动重试）。返回摘要 dict。

    三轮：主跑（workers）→ 重试一（半并发）→ 重试二（单线程）。
    完成判定看**磁盘实况**（文件齐=成功）；仍缺的写入
    `_incomplete.txt` 并在返回里给精确数量——再点一次只补这些。
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
                "（镜像站偶发抖动，已自动重试 3 次仍失败；稍后再点一次即可）"}
    plan = plan_files(files, langs)
    todo = [f for f in plan["files"]
            # TextGrid/musicxml 是标注冗余（json 已含全部信息），不下载
            if not f.lower().endswith((".textgrid", ".musicxml"))]
    if not todo:
        return {"ok": False, "error": "过滤后没有文件可下载"}

    ensure_endpoint()
    total = len(todo)
    already = [f for f in todo if os.path.isfile(os.path.join(dest, f))]
    pending = [f for f in todo if f not in set(already)]
    if already:
        progress(0.02, f"本地已有 {len(already)}/{total} 个文件，续传剩余 "
                       f"{len(pending)} 个")

    stop = False
    retried_fail: List[str] = []
    # 三轮：主跑 / 半并发重试 / 单线程收尾（最后 1-2% 通常是一次性抖动）
    plans = [(max(1, workers), 0.02, 0.92, "下载"),
             (max(1, workers // 2), 0.92, 0.97, "重试"),
             (1, 0.97, 0.995, "收尾重试")]
    for workers_now, base, hi, label in plans:
        if not pending:
            break
        ok_n, failed, stop = _run_pass(pending, dest, workers_now, progress,
                                       should_stop, base, hi - base, label)
        if stop:
            break
        if failed:
            retried_fail = failed
            progress(min(hi, 0.999),
                     f"{label}完成：本轮成功 {ok_n}，待重试 {len(failed)}")
            pending = failed
        else:
            retried_fail = []
            pending = []

    if stop:
        return {"ok": False, "stopped": True, "dest": dest,
                "done": total - len(pending), "total": total,
                "error": "用户停止（已完成的文件保留，再点一次续传即可）"}

    # 以磁盘实况收口（进程计数只能参考）
    missing = _missing_on_disk(dest, todo)
    inc_path = os.path.join(dest, INCOMPLETE_FILE)
    if missing:
        with open(inc_path, "w", encoding="utf-8") as f:
            f.write(f"# 缺失文件（{len(missing)}/{total}）"
                    f" · {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n".join(missing))
        sample = "\n".join(missing[:5])
        return {"ok": False, "partial": True, "dest": dest,
                "done": total - len(missing), "missing": len(missing),
                "total": total, "incomplete_file": inc_path,
                "error": f"还有 {len(missing)}/{total} 个文件未完成"
                         f"（已完成 {total - len(missing)}）。名单已写入 "
                         f"{os.path.basename(inc_path)}；再点一次下载只会补"
                         f"这些缺失项。示例：\n{sample}"}
    if os.path.isfile(inc_path):
        try:
            os.remove(inc_path)       # 补齐了就把残缺名单清掉
        except OSError:
            pass
    progress(1.0, f"完成：{total} 个文件齐全 → {dest}")
    return {"ok": True, "dest": dest, "done": total, "missing": 0,
            "total": total, "note": plan["note"],
            "resumed": len(already)}


def source_state(dest: Optional[str] = None) -> Dict[str, object]:
    """本地数据源状态（向导总览用；纯文件系统读取，轻量）。

    2026-10-19 补：不只看 wav——逐目录核对**同名 json 兄弟文件**。
    json 缺失的样本导入后没有文本/技巧标签，训练用不上，状态里必须
    可见（此前只报 wav 数，1487 个缺标注被藏了起来）。
    """
    d = dest or DEFAULT_DEST
    if not os.path.isdir(d):
        return {"exists": False, "path": d, "langs": [], "wavs": 0,
                "jsons": 0, "missing_json": 0, "partials": 0,
                "incomplete": False}
    langs = [DIR_LANG.get(n, n.lower()) for n in sorted(os.listdir(d))
             if os.path.isdir(os.path.join(d, n)) and n in DIR_LANG]
    wavs = jsons = missing_json = partials = 0
    for root, _dirs, files in os.walk(d):
        names = {f.lower(): f for f in files}
        wav_here = [f for f in files if f.lower().endswith(".wav")]
        json_stems = {os.path.splitext(f)[0] for f in files
                      if f.lower().endswith(".json")}
        wavs += len(wav_here)
        jsons += len(json_stems)
        missing_json += sum(1 for f in wav_here
                            if os.path.splitext(f)[0] not in json_stems)
        partials += sum(1 for k in names
                        if k.endswith(".part") or k.endswith(".incomplete"))
        if wavs > 200_000:            # 防御：超大目录提前停
            break
    inc = os.path.join(d, INCOMPLETE_FILE)
    return {"exists": True, "path": d, "langs": langs, "wavs": wavs,
            "jsons": jsons, "missing_json": missing_json,
            "partials": partials, "incomplete": os.path.isfile(inc)}
