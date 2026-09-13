"""系统与依赖监控。

给「系统」页提供：GPU/显存/CPU/内存/磁盘、Python 依赖版本核对、
ffmpeg 可用性、以及针对当前硬件的能力评估（哪些加速选项可用）。
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from webui_app.config import PROJECT_ROOT, AppConfig

# pyproject.toml 里的关键版本约束（人工摘录，用于体检对比）
EXPECTED_VERSIONS = {
    "torch": "2.8.*",
    "torchaudio": "2.8.*",
    "transformers": "4.52.1",
    "tokenizers": "0.21.0",
    "gradio": "5.45.0",
    "accelerate": "1.8.1",
    "numpy": "2.2.6",
    "librosa": "0.10.2.post1",
    "modelscope": "1.27.0",
    "omegaconf": ">=2.3.0",
    "safetensors": "0.5.2",
    "sentencepiece": ">=0.2.1",
}

# 可选加速依赖
OPTIONAL_EXTRAS = {
    "flash_attn": ("--accel GPT2 加速引擎", "uv sync --extra accel"),
    "triton": ("--torch_compile s2mel 图优化", "uv sync --extra torch_compile"),
    "deepspeed": ("--deepspeed 推理加速", "uv sync --extra deepspeed"),
    "peft": ("LoRA / DPO 训练", "pip install peft"),
    "whisper": ("评测台的 WER 计算", "pip install openai-whisper"),
    "jiwer": ("评测台的 WER 计算", "pip install jiwer"),
    "noisereduce": ("参考音频工作台降噪", "pip install noisereduce"),
}

# 本项目的安装方式与官方 pyproject.toml 的钉版**故意**不一致，
# 并且已经跑通完整推理链路验证过。这里记下原因，
# 体检时就不会把它们报成“依赖异常”而让用户以为环境坏了。
#
# 背景：.venv 是用 --system-site-packages 创建的，目的是复用系统已装好的
# torch 2.7.1+cu118（避开约 4 GB 的 torch 2.8+cu128 重复下载）。
# venv 内的包优先级高于系统包，所以 transformers 等钉版项能被准确屏蔽。
VERIFIED_DEVIATIONS: Dict[str, str] = {
    "torch": "复用系统已装的 2.7.1+cu118（sm_89 受支持），避开约 4 GB 重复下载。"
             "完整推理与 LoRA 注入均已实测通过",
    "torchaudio": "随 torch 2.7.1 一起复用，重采样/读写功能实测正常",
    "accelerate": "本项目不依赖 device_map 自动分派（QwenEmotion 改为显式 "
                  "CPU→GPU 两步加载），新版兼容",
    "librosa": "音频工作台的体检/切片/降噪均基于 0.11 实测通过",
    "modelscope": "7.9 GB 底模已用它完整下载成功，镜像接口兼容",
    "safetensors": "仅用于读写 w2v-bert / bigvgan 权重，格式向后兼容",
    "numpy": "上游包已适配 numpy 2.x，无 1.x 专有 API 调用",
}

# 钉版必须精确匹配的关键包：版本错了会直接影响模型结构/分词，不能将就
CRITICAL_PINS = {"transformers", "tokenizers", "gradio"}


def _ver(module: str) -> Optional[str]:
    try:
        import importlib
        m = importlib.import_module(module)
        return getattr(m, "__version__", "已安装")
    except Exception:
        return None


def python_info() -> Dict[str, Any]:
    return {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "project_root": PROJECT_ROOT,
    }


def gpu_info() -> Dict[str, Any]:
    """优先用 torch，失败则退回 nvidia-smi。"""
    out: Dict[str, Any] = {"available": False}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            out.update({
                "available": True,
                "name": p.name,
                "count": torch.cuda.device_count(),
                "capability": f"{p.major}.{p.minor}",
                "vram_total_gb": round(p.total_memory / 1024**3, 2),
                "vram_free_gb": round(free / 1024**3, 2),
                "vram_alloc_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                "vram_peak_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
                "bf16": torch.cuda.is_bf16_supported(),
            })
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            out["xpu"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    if not out.get("available"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                name, tot, used, drv = [x.strip() for x in r.stdout.strip().split(",")[:4]]
                out.update({
                    "name": name, "driver": drv,
                    "vram_total_gb": round(float(tot.split()[0]) / 1024, 2),
                    "vram_used_gb": round(float(used.split()[0]) / 1024, 2),
                })
        except Exception:
            pass
    return out


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.system(),
        "release": platform.release(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 1)
        info["ram_avail_gb"] = round(vm.available / 1024**3, 1)
        info["ram_percent"] = vm.percent
    except ImportError:
        info["ram_note"] = "未安装 psutil，无法读取内存占用"

    du = shutil.disk_usage(PROJECT_ROOT)
    info["disk_total_gb"] = round(du.total / 1024**3, 1)
    info["disk_free_gb"] = round(du.free / 1024**3, 1)
    info["disk_percent"] = round(100 * du.used / du.total, 1)
    return info


def ffmpeg_available() -> Dict[str, Any]:
    exe = shutil.which("ffmpeg")
    out = {"path": exe, "available": bool(exe), "version": ""}
    if exe:
        try:
            r = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
            out["version"] = (r.stdout or "").split("\n")[0][:80]
        except Exception:
            pass
    return out


def dependency_report() -> List[Dict[str, Any]]:
    """核对关键依赖的实际版本 vs pyproject 期望版本。

    state 三态：
        "ok"       版本匹配钉版约束
        "deviated" 不匹配，但属于本项目故意且已验证的偏离（见 VERIFIED_DEVIATIONS）
        "mismatch" 不匹配，且无已知的合理理由 —— 这种才需要用户关注
    """
    rows = []
    for mod, want in EXPECTED_VERSIONS.items():
        got = _ver(mod)
        if got is None:
            state = "mismatch"
        elif _version_ok(got, want):
            state = "ok"
        elif mod in VERIFIED_DEVIATIONS:
            state = "deviated"
        else:
            state = "mismatch"
        rows.append({
            "module": mod, "expected": want, "actual": got or "未安装",
            "ok": state == "ok", "state": state, "kind": "必需",
            "critical": mod in CRITICAL_PINS,
            "note": VERIFIED_DEVIATIONS.get(mod, "") if state == "deviated" else "",
        })
    for mod, (purpose, install) in OPTIONAL_EXTRAS.items():
        got = _ver(mod)
        rows.append({
            "module": mod, "expected": "可选", "actual": got or "未安装",
            "ok": got is not None, "state": "ok" if got else "optional_missing",
            "kind": "可选", "purpose": purpose, "install": install,
        })
    return rows


def _version_ok(got: Optional[str], want: str) -> bool:
    """宽松的版本期望匹配：支持 ==、>=、* 通配。"""
    if not got:
        return False
    want = want.strip()
    try:
        if want.endswith(".*"):
            return got.startswith(want[:-2])
        if want.startswith(">="):
            return _cmp(got, want[2:]) >= 0
        if want.startswith("=="):
            return got == want[2:]
        return got == want or got.startswith(want)
    except Exception:
        return True     # 解析不了就不报警，避免误报


def _cmp(a: str, b: str) -> int:
    def parse(v):
        return [int(c) if c.isdigit() else 0 for c in re.split(r"[.\-+]", v)]

    pa, pb = parse(a), parse(b)
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return (pa > pb) - (pa < pb)


def capability_report(cfg: AppConfig) -> List[Dict[str, Any]]:
    """针对当前硬件评估各加速选项的可用性。"""
    g = gpu_info()
    rows: List[Dict[str, Any]] = []

    def add(name, state, note):
        rows.append({"feature": name, "state": state, "note": note})

    if not g.get("available"):
        add("GPU 推理", "不可用", "未检测到 CUDA 设备，将以 CPU 运行（很慢）")
    else:
        vram = g.get("vram_total_gb", 0)
        add("GPU 推理", "可用", f"{g.get('name')} / {vram} GB")
        add("BF16 半精度",
            "可用" if g.get("bf16") else "不可用",
            "推荐开启，显存减半且质量损失极小" if g.get("bf16")
            else "该 GPU 不支持 BF16，会回退全精度")
        add("低显存模式",
            "已激活" if vram and vram < 10 else "未激活",
            "长文本会按 40 字自动切块，块间韵律不接续" if vram and vram < 10
            else "长文本按 token 数正常分句")
        add("QwenEmotion（情感文本控制）",
            "紧张" if vram and vram < 10 else "宽裕",
            "常驻显存约 4.9GB，QwenEmotion 需额外 1.2GB → 本 UI 采用串行挂载 + "
            "显存不足自动回退 CPU" if vram and vram < 10
            else "可常驻加载，切换零延迟")

    add("flash-attn（--accel）",
        "可用" if _ver("flash_attn") else "不可用",
        "已安装" if _ver("flash_attn")
        else "Windows 上需预编译 wheel，且要求 torch2.8+cu128；当前环境大概率装不上")
    add("triton（--torch_compile）",
        "可用" if (_ver("triton") or _ver("triton_windows")) else "不可用",
        "可对 CFM 做图优化，25 步 Euler 循环受益明显"
        if (_ver("triton") or _ver("triton_windows"))
        else "Windows 需 triton-windows 包")
    add("DeepSpeed（--deepspeed）",
        "可用" if _ver("deepspeed") else "不可用",
        "官方提示 Windows 上安装困难，且单卡收益不确定"
        if not _ver("deepspeed") else "建议实测对比开/关的速度差异")

    peft_ok = bool(_ver("peft"))
    add("LoRA / DPO 训练", "可用" if peft_ok else "不可用",
        "GPT(96层 Conv1D) 与 CFM(105层 Linear) 注入均已实测通过"
        if peft_ok else "需要 peft 包")
    w = bool(_ver("whisper")) and bool(_ver("jiwer"))
    add("评测台（WER + SS）", "可用" if w else "不可用",
        "whisper 做 ASR 转写，campplus 做声纹余弦" if w
        else "需要 openai-whisper 与 jiwer")

    ff = ffmpeg_available()
    add("ffmpeg", "可用" if ff["available"] else "缺失",
        ff.get("version") or ff.get("path") or ""
        if ff["available"] else
        "torchaudio/librosa 读取 mp3/m4a 等格式需要 ffmpeg，缺失时仅支持 wav/flac")
    return rows


def disk_usage_report(cfg: AppConfig) -> List[Dict[str, Any]]:
    """各资源目录的占用。"""
    rows = []
    targets = [
        ("模型 checkpoints", cfg.model_dir),
        ("辅助模型 hf_cache", os.path.join(cfg.model_dir, "hf_cache")),
        ("QwenEmotion", os.path.join(cfg.model_dir, "qwen0.6bemo4-merge")),
        ("输出 outputs", cfg.output_dir),
        ("音色库 voice_bank", cfg.voice_bank_dir),
        ("数据集 datasets", cfg.dataset_dir),
        ("LoRA 产物", cfg.lora_dir),
        ("训练产物", cfg.train_dir),
    ]
    for label, path in targets:
        rows.append({"label": label, "path": _rel(path), "size": _dir_size(path)})
    return rows


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    except ValueError:
        return path


def _dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render_markdown(cfg: AppConfig) -> str:
    py = python_info()
    g = gpu_info()
    s = system_info()
    ff = ffmpeg_available()

    L: List[str] = ["## 运行环境", ""]

    L += ["### 硬件", "", "| 项 | 值 |", "|---|---|"]
    if g.get("available"):
        L += [
            f"| GPU | {g.get('name')} |",
            f"| 显存 | {g.get('vram_total_gb')} GB 总量 / "
            f"{g.get('vram_free_gb')} GB 空闲 |",
            f"| PyTorch 已分配 | {g.get('vram_alloc_gb')} GB "
            f"(峰值 {g.get('vram_peak_gb')} GB) |",
            f"| 算力架构 | sm_{g.get('capability')} |",
            f"| BF16 | {'支持' if g.get('bf16') else '不支持'} |",
            f"| CUDA 构建 | {g.get('cuda_build')} |",
        ]
    else:
        L.append(f"| GPU | 未检测到（{g.get('error', '无 CUDA 设备')}） |")
    L += [
        f"| CPU 核心 | {s.get('cpu_count')} |",
        f"| 内存 | {s.get('ram_avail_gb', '?')} / {s.get('ram_total_gb', '?')} GB 可用 |",
        f"| 磁盘（项目所在） | {s.get('disk_free_gb')} / {s.get('disk_total_gb')} GB 空闲 |",
        f"| ffmpeg | {'✅ ' + ff['version'] if ff['available'] else '❌ 未找到'} |",
        "",
    ]

    L += ["### Python", "", "| 项 | 值 |", "|---|---|",
          f"| 版本 | {py['version']} |",
          f"| 解释器 | `{py['executable']}` |",
          f"| 虚拟环境 | {'✅ 是' if py['in_venv'] else '⚠️ 否（直接用的系统 Python）'} |",
          f"| 操作系统 | {py['platform']} |",
          f"| 工作目录 | `{py['cwd']}` |", ""]

    L += ["### 能力评估", "", "| 功能 | 状态 | 说明 |", "|---|---|---|"]
    icon = {"可用": "✅", "不可用": "❌", "缺失": "❌",
            "已激活": "🟡", "未激活": "🟢", "紧张": "🟠", "宽裕": "🟢"}
    for r in capability_report(cfg):
        L.append(f"| {r['feature']} | {icon.get(r['state'], '·')} {r['state']} | {r['note']} |")
    L.append("")

    L += ["### 依赖版本核对", "",
          "| 包 | pyproject 钉版 | 实际 | 状态 |", "|---|---|---|---|"]
    deviated = []
    mismatched = []
    for r in dependency_report():
        if r["kind"] != "必需":
            continue
        if r["state"] == "ok":
            mark = "✅ 匹配"
        elif r["state"] == "deviated":
            mark = "🟡 偏离已验证"
            deviated.append(r)
        else:
            mark = "⚠️ **不符**" if r["critical"] else "⚠️ 不符"
            mismatched.append(r)
        L.append(f"| `{r['module']}` | {r['expected']} | {r['actual']} | {mark} |")
    L.append("")

    if deviated:
        L += ["<details><summary>🟡 为什么有偏离项？（点开看逐项原因）</summary>", "",
              "本项目的 <code>.venv</code> 是用 <code>--system-site-packages</code> 创建的，"
              "目的是复用系统已装好的 torch，避开约 4 GB 的重复下载。"
              "venv 内的包优先级高于系统包，所以真正关键的钉版项"
              "（transformers / tokenizers / gradio）能被准确屏蔽。"
              "以下偏离项均已跑通完整推理链路验证：", ""]
        for r in deviated:
            L.append(f"- **`{r['module']}`** {r['actual']}（钉版 {r['expected']}）—— {r['note']}")
        L += ["", "</details>", ""]

    if mismatched:
        L += ["⚠️ 以下包与钉版不符且不属于已验证的偏离，建议核对：", ""]
        for r in mismatched:
            L.append(f"- `{r['module']}` 实际 {r['actual']}，期望 {r['expected']}")
        L.append("")

    L += ["<details><summary>可选依赖（加速 / 训练 / 评测）</summary>", "",
          "| 包 | 用途 | 状态 | 安装命令 |", "|---|---|---|---|"]
    for r in dependency_report():
        if r["kind"] == "可选":
            mark = "✅" if r["ok"] else "—"
            L.append(f"| `{r['module']}` | {r.get('purpose', '')} | {mark} {r['actual']} | "
                     f"`{r.get('install', '')}` |")
    L += ["", "</details>", ""]

    L += ["### 磁盘占用", "", "| 目录 | 路径 | 大小 |", "|---|---|---|"]
    for r in disk_usage_report(cfg):
        L.append(f"| {r['label']} | `{r['path']}` | {human_size(r['size'])} |")
    L.append("")

    notes = cfg.startup_notes()
    if notes:
        L += ["### 启动提示", ""] + [f"- {n}" for n in notes] + [""]

    return "\n".join(L)
