"""模型资源管理（包装 tools/model_fetcher，加后台线程与进度轮询）。

下载放到后台线程，UI 通过读 checkpoints/.fetch_status.json 轮询进度，
这样 8GB 的下载不会阻塞 Gradio 的请求线程。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from webui_app.config import PROJECT_ROOT

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools import model_fetcher as MF  # noqa: E402


class DownloadManager:
    """单例式的下载任务管理器。同时只允许一个下载任务。"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._result: Dict[str, Any] = {"ok": False, "message": "", "seconds": 0.0}

    # -- 状态查询 ----------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> Dict[str, Any]:
        """读取 fetcher 写在磁盘上的状态文件。"""
        path = os.path.join(self.model_dir, MF.STATUS_FILENAME)
        data: Dict[str, Any] = {
            "phase": "idle", "message": "", "running": False, "error": "", "items": [],
        }
        if os.path.isfile(path):
            try:
                import json
                with open(path, "r", encoding="utf-8") as f:
                    data.update(json.load(f))
            except Exception:
                pass
        data["busy"] = self.busy
        data["thread_result"] = dict(self._result)
        return data

    def audit(self, version: str = "2.5") -> Dict[str, Any]:
        return MF.audit(self.model_dir, version)

    # -- 下载 --------------------------------------------------------------

    def start(
        self,
        version: str = "2.5",
        include_qwen_emo: bool = True,
        include_examples: bool = True,
        only: Optional[str] = None,
    ) -> Dict[str, Any]:
        """启动后台下载。only 可选 main/aux/qwen_emo/examples，None 表示全部。"""
        with self._lock:
            if self.busy:
                return {"started": False, "message": "已有下载任务在进行中，请等待完成。"}

            self._result = {"ok": False, "message": "下载中…", "seconds": 0.0}
            t = threading.Thread(
                target=self._run,
                args=(version, include_qwen_emo, include_examples, only),
                daemon=True,
                name="model-fetcher",
            )
            self._thread = t
            t.start()
            return {"started": True, "message": "下载任务已启动（后台运行，可关闭本页）"}

    def _run(self, version, include_qwen_emo, include_examples, only):
        t0 = time.time()
        os.makedirs(self.model_dir, exist_ok=True)
        status = MF.FetchStatus(self.model_dir)
        status.running = True
        status.error = ""
        status.flush()
        try:
            if only is None:
                MF.fetch_main(version, self.model_dir, status)
                MF.fetch_aux(self.model_dir, status)
                if include_qwen_emo and version == "2.5":
                    MF.fetch_qwen_emo(self.model_dir, status)
                if include_examples:
                    MF.fetch_examples(self.model_dir, status)
            elif only == "main":
                MF.fetch_main(version, self.model_dir, status)
            elif only == "aux":
                MF.fetch_aux(self.model_dir, status)
            elif only == "qwen_emo":
                MF.fetch_qwen_emo(self.model_dir, status)
            elif only == "examples":
                MF.fetch_examples(self.model_dir, status)
            else:
                raise ValueError(f"未知的下载范围: {only}")
            dt = time.time() - t0
            status.set_phase("done", f"全部完成，耗时 {dt:.0f}s")
            self._result = {"ok": True, "message": f"下载完成，耗时 {dt:.0f}s", "seconds": dt}
        except Exception as e:
            dt = time.time() - t0
            status.error = f"{type(e).__name__}: {e}"
            status.set_phase("error", status.error)
            self._result = {"ok": False, "message": f"下载失败：{e}", "seconds": dt}
        finally:
            status.running = False
            status.flush()

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._thread is not None:
            self._thread.join(timeout)
        return dict(self._result)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

_SIZE_ORDER = ["main", "qwen", "aux", "examples"]


def audit_markdown(report: Dict[str, Any]) -> str:
    """把 audit() 的结果渲染成带进度条的 Markdown。"""
    lines = [
        f"### 资源审计 · IndexTTS-{report.get('version', '2.5')}",
        f"`{report.get('model_dir', '')}`",
        "",
    ]
    total_bytes = 0
    missing_required: List[str] = []

    for g in report.get("groups", []):
        items = g.get("items", [])
        if not items:
            continue
        got = sum(it["size"] for it in items if it["exists"])
        total_bytes += got
        done = sum(1 for it in items if it["ok"])
        ratio = done / len(items) if items else 0
        lines += [
            f"**{g['title']}** &nbsp; {done}/{len(items)}"
            + (" &nbsp;🟢 完整" if ratio == 1 else
               (" &nbsp;🟠 部分缺失" if ratio else " &nbsp;🔴 全缺")),
            "",
            "| 文件 | 状态 | 大小 |",
            "|---|---|---|",
        ]
        for it in items:
            if it["ok"]:
                mark = "✅"
            elif it["required"]:
                mark = "❌ **缺失**"
                missing_required.append(it["name"])
            else:
                mark = "➖ 未下载（可选）"
            expect = f"（应约 {it['expect_mb']} MB）" if it.get("expect_mb") else ""
            lines.append(f"| `{it['name']}` | {mark} | {it['size_human']} {expect} |")
        lines.append("")

    lines += [
        "---",
        f"**已就位总量**：{MF._human(total_bytes)}",
    ]
    if report.get("ready"):
        lines.append("\n🟢 **推理就绪** —— 所有必需文件都已就位。")
    else:
        lines.append(
            "\n🔴 **尚不可推理** —— 缺失以下必需文件：\n\n"
            + "\n".join(f"- `{n}`" for n in missing_required)
        )
    return "\n".join(lines)


def progress_markdown(model_dir: str) -> str:
    """渲染后台下载的实时进度。"""
    path = os.path.join(model_dir, MF.STATUS_FILENAME)
    if not os.path.isfile(path):
        return "_尚未发起过下载任务。_"
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
    except Exception as e:
        return f"_读取状态失败：{e}_"

    from webui_app import theme as T

    phase = st.get("phase", "idle")
    running = st.get("running", False)
    items = st.get("items", []) or []

    title = {
        "idle": "空闲", "main": "主模型", "aux": "辅助模型",
        "qwen_emo": "QwenEmotion", "examples": "示例音频",
        "done": "已完成", "error": "出错",
    }.get(phase, phase)

    done = sum(1 for i in items if i.get("state") in ("done", "cached"))
    failed = [i for i in items if i.get("state") == "failed"]
    skipped = [i for i in items if i.get("state") == "skipped"]
    total = len(items)
    ratio = done / total if total else 0

    state = "load" if running else ("err" if phase == "error" or failed else "ok")
    head = T.statusbar([
        T.chip("阶段", title, state),
        T.chip("进度", f"{done}/{total}", "load" if running else "ok"),
        T.chip("更新于", time.strftime("%H:%M:%S", time.localtime(st.get("updated_at", time.time()))), "idle"),
    ])

    lines = [head, "", T.progress_bar(ratio, f"{ratio*100:.0f}%  ({done}/{total})")]
    if st.get("message"):
        lines += ["", f"> {st['message']}"]
    if st.get("error"):
        lines += ["", T.err(st["error"])]

    if items:
        lines += ["", "| 条目 | 状态 | 来源 |", "|---|---|---|"]
        icon = {"done": "✅ 完成", "cached": "🟢 已存在", "downloading": "⏳ 下载中",
                "failed": "❌ 失败", "skipped": "➖ 跳过"}
        for it in items[-28:]:
            size = MF._human(it["size"]) if it.get("size") else ""
            lines.append(
                f"| `{it.get('name', '')}` {size} | "
                f"{icon.get(it.get('state'), it.get('state', ''))} | "
                f"{it.get('source', '-')} |"
            )
    if failed:
        lines += ["", T.warn(
            f"有 {len(failed)} 个条目下载失败。可重试 —— 本工具是幂等的，"
            "已完成的文件会自动跳过，且大文件支持断点续传。"
        )]
    return "\n".join(lines)


def source_hint_markdown() -> str:
    """说明下载源与回退策略。"""
    from tools.model_fetcher import MS_REPO_MAP

    lines = [
        "### 下载源与回退策略",
        "",
        "每个文件按以下顺序尝试，任一成功即停止：",
        "",
        "1. **ModelScope**（国内源，实测 ~20 MB/s）",
        "2. **hf-mirror.com** 经 `huggingface_hub` SDK（支持断点续传）",
        "3. **hf-mirror.com** 直连 HTTP（带 `.part` 续传）",
        "",
        "已配置的 ModelScope 镜像映射：",
        "",
        "| HuggingFace | ModelScope |",
        "|---|---|",
    ]
    for hf, ms in MS_REPO_MAP.items():
        lines.append(f"| `{hf}` | `{ms}` |")
    lines += [
        "| `IndexTeam/IndexTTS-2.5` | 同名（ModelScope 官方已镜像） |",
        "| `amphion/MaskGCT` | 无镜像 → 走 hf-mirror |",
        "| `nvidia/bigvgan_v2_22khz_80band_256x` | 无镜像 → 走 hf-mirror |",
        "",
        "> `HF_ENDPOINT` 已在本项目中默认设为 `https://hf-mirror.com`，",
        "> 因为直连 `huggingface.co` 在当前网络下超时。",
        "",
        "### 体积参考",
        "",
        "| 内容 | 大小 |",
        "|---|---|",
        "| gpt.pth | 3108.6 MB |",
        "| codec.pth | 579.2 MB |",
        "| s2mel.pth | 395.7 MB |",
        "| w2v-bert-2.0 | 2214.5 MB |",
        "| QwenEmotion | 1136.9 MB |",
        "| bigvgan_generator.pt | 428.4 MB |",
        "| semantic_codec | 169.0 MB |",
        "| campplus | 26.7 MB |",
        "| 其余（tiktoken/feat/config/stats） | ~2 MB |",
        "| **合计** | **~7.9 GB** |",
    ]
    return "\n".join(lines)
