"""
IndexTTS 模型资源拉取器（镜像优先 + 自动回退 + 断点续传 + 状态可查）

设计目标：
1. 镜像优先：ModelScope 国内源 -> hf-mirror.com -> huggingface.co
2. 断点续传：大文件走 huggingface_hub / modelscope SDK，两者都支持续传
3. 状态可查：全程把进度写入 checkpoints/.fetch_status.json，
   WebUI 的「模型管理」页可以直接读这个文件做实时展示
4. 幂等：已存在且大小正确的文件自动跳过，可反复执行

可被 WebUI 直接 import 复用，也可命令行独立运行：
    python tools/model_fetcher.py --all
    python tools/model_fetcher.py --audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# hf-mirror 必须在 import huggingface_hub 之前设置才生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
STATUS_FILENAME = ".fetch_status.json"

# ModelScope 上的镜像仓库映射（HuggingFace repo_id -> ModelScope model_id）
MS_REPO_MAP = {
    "facebook/w2v-bert-2.0": "AI-ModelScope/w2v-bert-2.0",
    "funasr/campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}

MAIN_REPO = {
    "2": "IndexTeam/IndexTTS-2",
    "2.5": "IndexTeam/IndexTTS-2.5",
}

# 主仓库文件清单：(文件名, 是否必需, 预期大小MB或None)
MAIN_FILES = {
    "2.5": [
        ("gpt.pth", True, 3108.6),
        ("s2mel.pth", True, 395.7),
        ("codec.pth", True, 579.2),
        ("multilingual_zh_ja_yue_char_del.tiktoken", True, 0.9),
        ("wav2vec2bert_stats.pt", True, None),
        ("config.yaml", True, None),
        ("feat1.pt", True, 0.1),
        ("feat2.pt", True, 0.4),
        ("pinyin.vocab", False, None),
    ],
    "2": [
        ("gpt.pth", True, None),
        ("s2mel.pth", True, None),
        ("bpe.model", True, None),
        ("wav2vec2bert_stats.pt", True, None),
        ("config.yaml", True, None),
        ("feat1.pt", True, None),
        ("feat2.pt", True, None),
    ],
}

QWEN_EMO_DIR = "qwen0.6bemo4-merge"
QWEN_EMO_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]

# 辅助模型清单：(逻辑名, HF repo, 远端文件, 本地相对路径, 是否整仓库)
AUX_MODELS = [
    ("w2v-bert-2.0", "facebook/w2v-bert-2.0", None, "hf_cache/w2v-bert-2.0", True),
    (
        "campplus",
        "funasr/campplus",
        "campplus_cn_common.bin",
        "hf_cache/campplus_cn_common.bin",
        False,
    ),
    (
        "semantic_codec",
        "amphion/MaskGCT",
        "semantic_codec/model.safetensors",
        "hf_cache/semantic_codec_model.safetensors",
        False,
    ),
    (
        "bigvgan",
        "nvidia/bigvgan_v2_22khz_80band_256x",
        "config.json",
        "hf_cache/bigvgan/config.json",
        False,
    ),
    (
        "bigvgan",
        "nvidia/bigvgan_v2_22khz_80band_256x",
        "bigvgan_generator.pt",
        "hf_cache/bigvgan/bigvgan_generator.pt",
        False,
    ),
]

EXAMPLES_BASE_URLS = [
    "https://hf-mirror.com/spaces/IndexTeam/IndexTTS-2-Demo/resolve/main/examples",
    "https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo/resolve/main/examples",
]


# ---------------------------------------------------------------------------
# 状态追踪
# ---------------------------------------------------------------------------

@dataclass
class FetchStatus:
    """写入磁盘的下载状态，供 WebUI 轮询展示。"""

    updated_at: float = field(default_factory=time.time)
    phase: str = "idle"
    message: str = ""
    running: bool = False
    error: str = ""
    items: list = field(default_factory=list)

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.path = os.path.join(model_dir, STATUS_FILENAME)
        self.updated_at = time.time()
        self.phase = "idle"
        self.message = ""
        self.running = False
        self.error = ""
        self.items = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.phase = data.get("phase", "idle")
                self.message = data.get("message", "")
                self.items = data.get("items", [])
                self.error = data.get("error", "")
            except Exception:
                pass

    def flush(self):
        with self._lock:
            self.updated_at = time.time()
            payload = {
                "updated_at": self.updated_at,
                "phase": self.phase,
                "message": self.message,
                "running": self.running,
                "error": self.error,
                "items": self.items,
            }
            tmp = self.path + ".tmp"
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    def set_phase(self, phase: str, message: str = ""):
        self.phase = phase
        if message:
            self.message = message
        self.flush()

    def upsert_item(self, name: str, **kwargs):
        for it in self.items:
            if it.get("name") == name:
                it.update(kwargs)
                break
        else:
            self.items.append({"name": name, **kwargs})
        self.flush()


# ---------------------------------------------------------------------------
# 底层下载原语
# ---------------------------------------------------------------------------

def _human(size_bytes) -> str:
    if not size_bytes:
        return "?"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _ms_file(model_id: str, file_path: str, local_dir: str) -> str:
    from modelscope.hub.file_download import model_file_download

    return model_file_download(
        model_id=model_id, file_path=file_path, local_dir=local_dir
    )


def _ms_snapshot(model_id: str, local_dir: str, allow_patterns=None) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    kwargs = {}
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    try:
        snapshot_download(model_id=model_id, local_dir=local_dir, **kwargs)
    except TypeError:
        # 老版本 modelscope 不支持 allow_patterns
        snapshot_download(model_id=model_id, local_dir=local_dir)
    return local_dir


def _hf_file(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir, resume_download=True
    )


def _hf_snapshot(repo_id: str, local_dir: str, allow_patterns=None) -> str:
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns
    )


def _http_file(url: str, local_path: str, timeout: int = 120) -> None:
    """无 SDK 依赖的直连下载（支持已下载部分续传）。"""
    import requests

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    headers = {}
    mode = "wb"
    existing = 0
    if os.path.isfile(local_path + ".part"):
        existing = os.path.getsize(local_path + ".part")
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    if resp.status_code == 416:  # Range 超出，文件已完整
        os.replace(local_path + ".part", local_path)
        return
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    if resp.status_code == 200 and mode == "ab":
        mode, existing = "wb", 0  # 服务端不支持 Range，重下

    part = local_path + ".part"
    with open(part, mode) as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
    os.replace(part, local_path)


def fetch_file(
    repo_id: str,
    remote_file: str,
    local_path: str,
    status: FetchStatus | None = None,
    label: str | None = None,
) -> str:
    """
    下载单个文件：ModelScope -> hf-mirror(SDK) -> 直连 hf-mirror。
    已存在则跳过。返回本地路径。
    """
    label = label or f"{repo_id}/{remote_file}"
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        if status:
            status.upsert_item(
                label, state="cached", size=os.path.getsize(local_path)
            )
        return local_path

    local_dir = os.path.dirname(local_path) or "."
    os.makedirs(local_dir, exist_ok=True)
    errors = []

    # 1) ModelScope
    ms_id = MS_REPO_MAP.get(repo_id, repo_id)
    try:
        if status:
            status.upsert_item(label, state="downloading", source="modelscope")
        got = _ms_file(ms_id, remote_file, local_dir)
        if got and os.path.abspath(got) != os.path.abspath(local_path):
            _move_into(got, local_path)
        if os.path.isfile(local_path):
            if status:
                status.upsert_item(
                    label, state="done", source="modelscope",
                    size=os.path.getsize(local_path),
                )
            return local_path
        raise RuntimeError("modelscope 返回路径无效")
    except Exception as e:
        errors.append(f"modelscope: {e}")

    # 2) huggingface_hub 走 hf-mirror（支持续传）
    try:
        if status:
            status.upsert_item(label, state="downloading", source="hf-mirror")
        got = _hf_file(repo_id, remote_file, local_dir)
        if got and os.path.abspath(got) != os.path.abspath(local_path):
            _move_into(got, local_path)
        if os.path.isfile(local_path):
            if status:
                status.upsert_item(
                    label, state="done", source="hf-mirror",
                    size=os.path.getsize(local_path),
                )
            return local_path
        raise RuntimeError("hf-mirror 返回路径无效")
    except Exception as e:
        errors.append(f"hf-mirror: {e}")

    # 3) 直连 hf-mirror HTTP（带续传）
    url = f"https://hf-mirror.com/{repo_id}/resolve/main/{remote_file}"
    try:
        if status:
            status.upsert_item(label, state="downloading", source="http")
        _http_file(url, local_path)
        if status:
            status.upsert_item(
                label, state="done", source="http", size=os.path.getsize(local_path)
            )
        return local_path
    except Exception as e:
        errors.append(f"http: {e}")

    msg = " | ".join(errors)
    if status:
        status.upsert_item(label, state="failed", error=msg)
    raise RuntimeError(f"下载失败 {label}: {msg}")


def _move_into(src: str, dst: str):
    """把 SDK 下载到的文件搬到目标路径（跨盘时退化为复制）。"""
    import shutil

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.isfile(src):
        try:
            os.replace(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def fetch_repo(
    repo_id: str,
    local_dir: str,
    allow_patterns=None,
    status: FetchStatus | None = None,
    label: str | None = None,
) -> str:
    """下载整个仓库快照：ModelScope -> hf-mirror。"""
    label = label or repo_id
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        if status:
            status.upsert_item(label, state="cached")
        return local_dir

    errors = []
    ms_id = MS_REPO_MAP.get(repo_id, repo_id)
    try:
        if status:
            status.upsert_item(label, state="downloading", source="modelscope")
        _ms_snapshot(ms_id, local_dir, allow_patterns)
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            if status:
                status.upsert_item(label, state="done", source="modelscope")
            return local_dir
        raise RuntimeError("modelscope 快照为空")
    except Exception as e:
        errors.append(f"modelscope: {e}")

    try:
        if status:
            status.upsert_item(label, state="downloading", source="hf-mirror")
        _hf_snapshot(repo_id, local_dir, allow_patterns)
        if status:
            status.upsert_item(label, state="done", source="hf-mirror")
        return local_dir
    except Exception as e:
        errors.append(f"hf-mirror: {e}")

    msg = " | ".join(errors)
    if status:
        status.upsert_item(label, state="failed", error=msg)
    raise RuntimeError(f"仓库下载失败 {label}: {msg}")


# ---------------------------------------------------------------------------
# 高层任务
# ---------------------------------------------------------------------------

def fetch_main(version: str, model_dir: str, status: FetchStatus | None = None):
    """下载主模型仓库的必需文件。"""
    repo = MAIN_REPO[version]
    if status:
        status.set_phase("main", f"拉取主模型 {repo}")
    for name, required, _size in MAIN_FILES[version]:
        local = os.path.join(model_dir, name)
        try:
            fetch_file(repo, name, local, status, label=f"main/{name}")
        except Exception as e:
            if required:
                raise
            if status:
                status.upsert_item(f"main/{name}", state="skipped", error=str(e))


def fetch_qwen_emo(model_dir: str, status: FetchStatus | None = None):
    """下载 QwenEmotion（情感描述文本控制所需）。"""
    repo = MAIN_REPO["2.5"]
    if status:
        status.set_phase("qwen_emo", "拉取 QwenEmotion 情感模型")
    target_dir = os.path.join(model_dir, QWEN_EMO_DIR)
    os.makedirs(target_dir, exist_ok=True)
    for name in QWEN_EMO_FILES:
        try:
            fetch_file(
                repo,
                f"{QWEN_EMO_DIR}/{name}",
                os.path.join(target_dir, name),
                status,
                label=f"qwen_emo/{name}",
            )
        except Exception as e:
            # chat_template.jinja 等小文件缺失不致命
            if status:
                status.upsert_item(f"qwen_emo/{name}", state="skipped", error=str(e))


def fetch_aux(model_dir: str, status: FetchStatus | None = None):
    """下载辅助模型（w2v-bert / campplus / semantic_codec / bigvgan）。"""
    if status:
        status.set_phase("aux", "拉取辅助模型")

    # w2v-bert-2.0 是整仓库
    w2v_dir = os.path.join(model_dir, "hf_cache", "w2v-bert-2.0")
    fetch_repo(
        "facebook/w2v-bert-2.0",
        w2v_dir,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "*.json",
        ],
        status=status,
        label="aux/w2v-bert-2.0",
    )

    for name, repo, remote, rel, whole in AUX_MODELS:
        if whole:
            continue
        fetch_file(
            repo,
            remote,
            os.path.join(model_dir, rel.replace("/", os.sep)),
            status,
            label=f"aux/{name}/{os.path.basename(remote)}",
        )


def fetch_examples(model_dir: str = None, status: FetchStatus | None = None):
    """下载 examples/ 下的示例音频。"""
    import requests

    examples_dir = os.path.join(PROJECT_ROOT, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    cases_path = os.path.join(examples_dir, "cases.jsonl")
    names = {"voice_01.wav"}
    if os.path.isfile(cases_path):
        with open(cases_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("prompt_audio", "emo_audio"):
                    if case.get(key):
                        names.add(case[key])

    if status:
        status.set_phase("examples", f"拉取示例音频（{len(names)} 个）")

    for fname in sorted(names):
        local = os.path.join(examples_dir, fname)
        if os.path.isfile(local) and os.path.getsize(local) > 0:
            if status:
                status.upsert_item(f"examples/{fname}", state="cached")
            continue
        for base in EXAMPLES_BASE_URLS:
            try:
                if status:
                    status.upsert_item(f"examples/{fname}", state="downloading")
                _http_file(f"{base}/{fname}", local)
                if status:
                    status.upsert_item(f"examples/{fname}", state="done")
                break
            except Exception as e:
                if status:
                    status.upsert_item(
                        f"examples/{fname}", state="failed", error=str(e)
                    )


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------

def audit(model_dir: str, version: str = "2.5") -> dict:
    """检查所有资源就位情况，返回结构化报告（不触发下载）。"""
    report = {"version": version, "model_dir": model_dir, "groups": [], "ready": True}

    def entry(name, path, required=True, expect_mb=None):
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        ok = exists and size > 0
        item = {
            "name": name,
            "path": os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"),
            "required": required,
            "exists": exists,
            "size": size,
            "size_human": _human(size) if exists else "-",
            "expect_mb": expect_mb,
            "ok": ok,
        }
        if required and not ok:
            report["ready"] = False
        return item

    main_group = {"title": "主模型", "items": []}
    for name, required, size in MAIN_FILES[version]:
        main_group["items"].append(
            entry(name, os.path.join(model_dir, name), required, size)
        )
    report["groups"].append(main_group)

    qwen_dir = os.path.join(model_dir, QWEN_EMO_DIR)
    qwen_group = {"title": "QwenEmotion（情感文本控制·可选）", "items": []}
    key_missing = False
    for name in QWEN_EMO_FILES:
        it = entry(
            f"{QWEN_EMO_DIR}/{name}",
            os.path.join(qwen_dir, name),
            required=False,
            expect_mb=1136.9 if name == "model.safetensors" else None,
        )
        qwen_group["items"].append(it)
        if name in ("model.safetensors", "config.json") and not it["ok"]:
            key_missing = True
    qwen_group["available"] = not key_missing
    report["groups"].append(qwen_group)

    aux_specs = [
        ("w2v-bert-2.0 目录", os.path.join(model_dir, "hf_cache", "w2v-bert-2.0", "config.json")),
        ("w2v-bert-2.0 权重", os.path.join(model_dir, "hf_cache", "w2v-bert-2.0", "model.safetensors")),
        ("campplus_cn_common.bin", os.path.join(model_dir, "hf_cache", "campplus_cn_common.bin")),
        ("semantic_codec_model.safetensors", os.path.join(model_dir, "hf_cache", "semantic_codec_model.safetensors")),
        ("bigvgan/config.json", os.path.join(model_dir, "hf_cache", "bigvgan", "config.json")),
        ("bigvgan/bigvgan_generator.pt", os.path.join(model_dir, "hf_cache", "bigvgan", "bigvgan_generator.pt")),
    ]
    aux_group = {"title": "辅助模型（hf_cache）", "items": []}
    for name, path in aux_specs:
        # w2v-bert 权重也可能是 pytorch_model.bin
        if name.endswith("model.safetensors") and not os.path.isfile(path):
            alt = path.replace("model.safetensors", "pytorch_model.bin")
            if os.path.isfile(alt):
                path = alt
        aux_group["items"].append(entry(name, path, True))
    report["groups"].append(aux_group)

    ex_group = {"title": "示例音频", "items": []}
    examples_dir = os.path.join(PROJECT_ROOT, "examples")
    for name in ("voice_01.wav",):
        ex_group["items"].append(
            entry(name, os.path.join(examples_dir, name), required=False)
        )
    report["groups"].append(ex_group)

    return report


def print_audit(report: dict):
    print(f"\n=== IndexTTS-{report['version']} 资源审计 ===")
    print(f"模型目录: {report['model_dir']}\n")
    for group in report["groups"]:
        print(f"[{group['title']}]")
        for it in group["items"]:
            flag = "OK " if it["ok"] else ("MISS" if it["required"] else "opt ")
            req = "" if it["required"] else " (可选)"
            print(
                f"  {flag} {it['name']:<46} {it['size_human']:>9}{req}"
            )
        print()
    total = sum(
        it["size"]
        for g in report["groups"]
        for it in g["items"]
        if it["exists"]
    )
    print(f"已就位总量: {_human(total)}")
    print(f"推理就绪: {'是' if report['ready'] else '否（缺必需文件）'}\n")


def fetch_all(
    model_dir: str = DEFAULT_MODEL_DIR,
    version: str = "2.5",
    include_qwen_emo: bool = True,
    include_examples: bool = True,
):
    os.makedirs(model_dir, exist_ok=True)
    status = FetchStatus(model_dir)
    status.running = True
    status.error = ""
    status.flush()
    started = time.time()
    try:
        fetch_main(version, model_dir, status)
        fetch_aux(model_dir, status)
        if include_qwen_emo and version == "2.5":
            fetch_qwen_emo(model_dir, status)
        if include_examples:
            fetch_examples(model_dir, status)
        status.set_phase("done", f"全部完成，耗时 {time.time() - started:.0f}s")
    except Exception as e:
        status.error = str(e)
        status.set_phase("error", f"失败: {e}")
        raise
    finally:
        status.running = False
        status.flush()
    return status


def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS 模型资源拉取器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--version", default="2.5", choices=["2", "2.5"])
    parser.add_argument("--all", action="store_true", help="拉取全部资源")
    parser.add_argument("--main", action="store_true", help="仅拉取主模型")
    parser.add_argument("--aux", action="store_true", help="仅拉取辅助模型")
    parser.add_argument("--qwen_emo", action="store_true", help="仅拉取 QwenEmotion")
    parser.add_argument("--examples", action="store_true", help="仅拉取示例音频")
    parser.add_argument("--audit", action="store_true", help="只审计不下载")
    parser.add_argument(
        "--no-qwen-emo", action="store_true", help="配合 --all 使用，跳过 QwenEmotion"
    )
    args = parser.parse_args()

    if args.audit or not (
        args.all or args.main or args.aux or args.qwen_emo or args.examples
    ):
        print_audit(audit(args.model_dir, args.version))
        return 0

    os.makedirs(args.model_dir, exist_ok=True)
    status = FetchStatus(args.model_dir)
    status.running = True
    status.flush()
    try:
        if args.all:
            fetch_all(
                args.model_dir,
                args.version,
                include_qwen_emo=not args.no_qwen_emo,
            )
        else:
            if args.main:
                fetch_main(args.version, args.model_dir, status)
            if args.aux:
                fetch_aux(args.model_dir, status)
            if args.qwen_emo:
                fetch_qwen_emo(args.model_dir, status)
            if args.examples:
                fetch_examples(args.model_dir, status)
            status.running = False
            status.set_phase("done", "完成")
    except Exception as e:
        status.running = False
        status.error = str(e)
        status.set_phase("error", str(e))
        print(f"\n!! 下载失败: {e}", file=sys.stderr)
        return 1

    print_audit(audit(args.model_dir, args.version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
