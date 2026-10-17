"""separation — 人声分离 / 去混响（audio-separator 封装，可选 extra）。

设计（dev-plan D8）：
- `audio-separator` 为可选依赖（pip install "audio-separator[gpu]"），未装时
  available()=False，UI 优雅降级（与 audiosr/noisereduce 同模式）。
- 权重自动下载到 checkpoints/separators/（checkpoints 是 artifacts 盘点的
  明文排除区——模型资源归模型页，不进清理页产物盘点）。
- **国内镜像**：audio-separator 自身只认 GitHub 直连（元数据
  download_checks.json + Releases 权重），国内不可达。本模块在调用
  Separator 之前自行经镜像链预取（默认 ghproxy.net → gh-proxy.com → 直连，
  可用环境变量 INDEXTTS_GH_MIRRORS 覆盖，逗号分隔，末尾留空项表示直连）。
  支持断点续传（.part + Range）、大小校验、多镜像自动回退。
- 分离不改时长（与源曲同时间轴），是混音对齐的前提。
- 显存：roformer 2-4GB，与引擎互斥——调用方负责先卸引擎（cover_pipeline
  统一编排），本模块镜像 audio_sr 的显式释放模式。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "separators")

# 模型选择（调研报告 §3.1/3.2：社区默认 + UVR DeEcho 三件套之一）
VOCALS_MODELS: Dict[str, str] = {
    "标准（BS-Roformer ViperX，SDR 12.98）": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "轻量（Mel-Band Roformer，SDR 11.44）": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "人声专用（vocals_mel_band_roformer）": "vocals_mel_band_roformer.ckpt",
}
DEREVERB_MODELS: Dict[str, str] = {
    "标准（UVR-DeEcho-DeReverb）": "UVR-DeEcho-DeReverb.pth",
    "轻柔（UVR-DeEcho-Normal）": "UVR-DeEcho-Normal.pth",
}
DEFAULT_VOCALS = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_DEREVERB = "UVR-DeEcho-DeReverb.pth"

# ---------------------------------------------------------------------------
# 模型来源与镜像（audio-separator 只认 GitHub，国内需前置镜像预取）
# ---------------------------------------------------------------------------

UVR_RELEASE = ("https://github.com/TRvlvr/model_repo/releases/download/"
               "all_public_uvr_models")
NOMAD_RELEASE = ("https://github.com/nomadkaraoke/python-audio-separator/"
                 "releases/download/model-configs")
UVR_CHECKS_URL = ("https://raw.githubusercontent.com/TRvlvr/application_data/"
                  "main/filelists/download_checks.json")
# VR/MDX 权重加载时要按 MD5 查参数表（audio-separator 会直连 raw.githubusercontent）。
# 远端名 model_data_new.json，本地落名不同——必须按它的期望命名预置。
UVR_DATA_FILES = {
    "vr_model_data.json": ("https://raw.githubusercontent.com/TRvlvr/"
                           "application_data/main/vr_model_data/model_data_new.json"),
    "mdx_model_data.json": ("https://raw.githubusercontent.com/TRvlvr/"
                            "application_data/main/mdx_model_data/model_data_new.json"),
}

# 环境变量覆盖示例：INDEXTTS_GH_MIRRORS="https://ghproxy.net/,"
# （末尾空项 = 最后再试直连；海外用户可设为 "," 即仅直连）
_DEFAULT_MIRRORS = "https://ghproxy.net/,https://gh-proxy.com/,"

# 每个模型的完整文件对（权重 + 配置）。顺序与 audio-separator 的
# download_files 一致；YAML 只在 nomadkaraoke release 有全量。
MODEL_FILES: Dict[str, List[str]] = {
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": [
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.yaml",
    ],
    "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": [
        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
    ],
    "vocals_mel_band_roformer.ckpt": [
        "vocals_mel_band_roformer.ckpt",
        "vocals_mel_band_roformer.yaml",
    ],
    "UVR-DeEcho-DeReverb.pth": ["UVR-DeEcho-DeReverb.pth"],
    "UVR-DeEcho-Normal.pth": ["UVR-DeEcho-Normal.pth"],
}

# Release 资产名与模型标识不一致的别名（实锤：UVR 仓库资产带连字符，
# audio-separator 的模型名不带；直接拼 URL 会 404）。
FILE_URL_ALIASES: Dict[str, str] = {
    "UVR-DeEcho-Normal.pth": "UVR-De-Echo-Normal.pth",
}

# 已知体积（GitHub Release 资产表，2026-10 核对）；-1 = 未知（YAML 小文件）
KNOWN_SIZES: Dict[str, int] = {
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": 639331213,
    "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt": 1007816988,
    "UVR-DeEcho-DeReverb.pth": 223650277,
    "vocals_mel_band_roformer.ckpt": 913106900,
    "UVR-DeEcho-Normal.pth": 127139365,
    # YAML 运行时用 Range 探测远端大小；探测值异常（<1MB 的权重）
    # 一律按未知处理，避免"大小不符→删除重试"死循环。
}

_LOCK = threading.Lock()
_active: List = []          # 持有 Separator 实例，release_all 用
_download_lock = threading.Lock()


class DownloadCancelled(Exception):
    """用户停止下载（.part 保留，下次续传）。"""


def mirrors() -> List[str]:
    raw = os.environ.get("INDEXTTS_GH_MIRRORS", _DEFAULT_MIRRORS)
    return [m.strip() for m in raw.split(",")]


def _mirrored(url: str) -> List[str]:
    """给 GitHub URL 生成镜像候选（顺序即优先级）。"""
    out = []
    for m in mirrors():
        out.append(f"{m}{url}" if m else url)
    return out


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# YAML 配置可能只有几百字节（vocals_mel_band 的 944B < 旧阈值 1024B，
# 曾被误判为缺失导致每次推理重下——2026-10-17 实锤）
_MIN_YAML = 200
_MIN_WEIGHT = 1 << 20          # 1 MiB


def _ready(path: str) -> bool:
    n = _size(path)
    return n > (_MIN_YAML if path.lower().endswith(".yaml") else _MIN_WEIGHT)


def _http_size(url: str, timeout: float = 20.0) -> int:
    """远端大小（Range 探测，兼容不支持 HEAD 的镜像）。"""
    import requests
    try:
        r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                         timeout=timeout, allow_redirects=True)
        cr = r.headers.get("Content-Range", "")
        r.close()
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        return int(r.headers.get("Content-Length", 0) or 0)
    except Exception:
        return 0


def _download_once(url: str, dest: str, expected: int,
                   progress=None, label: str = "", attempts: int = 3,
                   should_stop=None) -> bool:
    """单 URL 下载（.part + Range 续传；同源失败重试）。

    实测坑（2026-10-17）：ghproxy 链路偶发中断，若直接换镜像会落到不支持
    Range 的节点（200 全量重传），把已下进度清零；因此中断后**先在同源
    重试并续传**，服务端忽略 Range 且已有较大进度时放弃该镜像。
    """
    import requests
    part = dest + ".part"
    for _attempt in range(attempts):
        pos = _size(part)
        if expected > 0 and pos > expected:
            try:
                os.remove(part)
            except OSError:
                pass
            pos = 0
        if expected > 0 and pos == expected > 0:
            os.replace(part, dest)
            return True
        headers = {"Range": f"bytes={pos}-"} if pos > 0 else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 60),
                              allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    if r.status_code in (404, 410):
                        # 镜像冷缓存会偶发 404（实锤：UVR-DeEcho-Normal 第
                        # 一次 404、重试即 206），给两次机会再判缺失。
                        time.sleep(2.0)
                        continue
                    if r.status_code == 403:
                        return False
                    time.sleep(1.5)
                    continue
                ctype = (r.headers.get("Content-Type") or "").lower()
                if "text/html" in ctype:
                    # 镜像错误页：绝不能写进 .part（续传会拼进垃圾）
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                    return False
                if pos > 0 and r.status_code == 200:
                    # 服务端忽略 Range。小进度可接受重来，大进度换镜像。
                    if pos > 8 * 1024 * 1024:
                        return False
                    pos = 0
                total = expected
                cl = int(r.headers.get("Content-Length", 0) or 0)
                if total <= 0 and cl:
                    total = cl + pos
                mode = "ab" if r.status_code == 206 and pos > 0 else "wb"
                done = pos
                last = 0.0
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if should_stop is not None and should_stop():
                            raise DownloadCancelled()
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if progress is not None and now - last > 0.5:
                            last = now
                            try:
                                if total > 0:
                                    progress(done / total,
                                             f"{label} {done/1e6:.0f}/"
                                             f"{total/1e6:.0f}MB")
                                else:
                                    progress(0.0, f"{label} {done/1e6:.0f}MB")
                            except Exception:
                                pass
            got = _size(part)
            if expected > 0 and got != expected:
                if got > expected:
                    try:
                        os.remove(part)
                    except OSError:
                        pass
                    continue
                time.sleep(1.0)
                continue                     # 中断：同源续传重试
            if got <= 0:
                continue
            os.replace(part, dest)
            return True
        except DownloadCancelled:
            raise
        except Exception:
            time.sleep(1.0)
            continue                         # 网络异常：同源续传重试
    return False


def _fetch(filename: str, progress=None, rel_frac: float = 0.0,
           should_stop=None) -> bool:
    """按镜像链抓取单个文件。先查本地；YAML 优先 nomadkaraoke，权重先 UVR。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, filename)
    if _ready(dest):
        return True
    expected = KNOWN_SIZES.get(filename, -1)
    if filename.endswith(".yaml"):
        urls = _mirrored(f"{NOMAD_RELEASE}/{filename}") + \
               _mirrored(f"{UVR_RELEASE}/mdx_model_data/mdx_c_configs/{filename}")
    else:
        remote = FILE_URL_ALIASES.get(filename, filename)
        urls = _mirrored(f"{UVR_RELEASE}/{remote}") + \
               _mirrored(f"{NOMAD_RELEASE}/{remote}")

    for u in urls:
        # 首次提示用远端大小（慢，但只在需要下载时发生）
        if expected <= 0:
            probed = _http_size(u)
            # 权重不可能 <1MB：镜像的畸形 Content-Range 常返回几十字节
            if probed > (1 << 20) or filename.endswith(".yaml"):
                expected = probed
            if expected > 0:
                KNOWN_SIZES[filename] = expected
        if _download_once(u, dest, expected, progress,
                          label=filename, should_stop=should_stop):
            return True
    return False


def ensure_metadata(progress=None, should_stop=None) -> bool:
    """预置 audio-separator 依赖的 UVR 元数据（它只会走 GitHub 直连）。

    - download_checks.json：模型清单（list_supported_model_files 必需）
    - vr_model_data.json / mdx_model_data.json：VR/MDX 权重→参数表
      （load_model_data_using_hash 必需，缺了 VR 模型加载会直接网络失败）
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    jobs = [("download_checks.json", UVR_CHECKS_URL, 1024)]
    jobs += [(local, url, 1024) for local, url in UVR_DATA_FILES.items()]
    ok = True
    for local, url, min_size in jobs:
        dest = os.path.join(MODEL_DIR, local)
        if _size(dest) > min_size:
            continue
        got = False
        for u in _mirrored(url):
            if _download_once(u, dest, 0, progress, label=local,
                              should_stop=should_stop):
                got = True
                break
        if not got:
            ok = False
    return ok


def ensure_model(model_filename: str, progress=None,
                 should_stop=None) -> Dict:
    """推理前保证模型**完整可用**；缺文件自动经镜像下载。

    返回 {ok, downloaded: [...], missing: [...]}。progress(frac,msg) 可空。
    """
    files = MODEL_FILES.get(model_filename, [model_filename])
    missing = [f for f in files if not _ready(os.path.join(MODEL_DIR, f))]
    if not missing:
        return {"ok": True, "downloaded": [], "missing": []}
    with _download_lock:
        # 双检（并发调用只下载一次）
        missing = [f for f in files if not _ready(os.path.join(MODEL_DIR, f))]
        if not missing:
            return {"ok": True, "downloaded": [], "missing": []}
        try:
            meta_ok = ensure_metadata(progress, should_stop)
        except DownloadCancelled:
            return {"ok": False, "cancelled": True, "downloaded": [],
                    "missing": missing, "error": "用户停止（模型清单下载中断）"}
        if not meta_ok:
            return {"ok": False, "downloaded": [],
                    "missing": missing + ["download_checks.json"],
                    "error": "无法获取 UVR 模型清单（镜像均失败）"}
        total = sum(max(KNOWN_SIZES.get(f, 0), 1) for f in missing)
        done_before = 0
        downloaded: List[str] = []
        for f in missing:
            base = done_before / total
            span = max(KNOWN_SIZES.get(f, 0), 1) / total

            def _p(frac, msg, _b=base, _s=span):
                if progress is not None:
                    try:
                        progress(_b + _s * frac, msg)
                    except Exception:
                        pass

            try:
                fetched = _fetch(f, _p, should_stop=should_stop)
            except DownloadCancelled:
                return {"ok": False, "cancelled": True,
                        "downloaded": downloaded,
                        "missing": [x for x in missing if x not in downloaded],
                        "error": "用户停止（模型下载中断，.part 已保留可续传）"}
            if fetched:
                downloaded.append(f)
            else:
                return {"ok": False, "downloaded": downloaded,
                        "missing": [x for x in missing if x not in downloaded],
                        "error": f"{f} 下载失败（已尝试镜像：{mirrors()}）；"
                                 "可设 INDEXTTS_GH_MIRRORS 或手动放入 "
                                 f"{MODEL_DIR}"}
            done_before += max(KNOWN_SIZES.get(f, 0), 1)
        return {"ok": True, "downloaded": downloaded, "missing": []}


def ensure_all_exposed(progress=None, should_stop=None) -> Dict:
    """下载 UI 暴露的全部模型（翻唱页 3 个人声模型 + 2 个去混响模型）。"""
    names = list(dict.fromkeys(list(VOCALS_MODELS.values()) +
                               list(DEREVERB_MODELS.values())))
    out = {"ok": True, "models": {}}
    for i, n in enumerate(names):
        def _p(frac, msg, _i=i, _n=len(names)):
            if progress is not None:
                try:
                    progress((_i + frac) / _n, msg)
                except Exception:
                    pass

        r = ensure_model(n, _p, should_stop)
        out["models"][n] = r
        if not r.get("ok"):
            out["ok"] = False
    return out


def installed_models() -> List[Dict]:
    """已就绪模型清单（模型页/UI 提示用）。"""
    rows = []
    for label, fn in {**VOCALS_MODELS, **DEREVERB_MODELS}.items():
        size = sum(_size(os.path.join(MODEL_DIR, f))
                   for f in MODEL_FILES.get(fn, [fn]))
        rows.append({"label": label, "file": fn,
                     "ready": all(_ready(os.path.join(MODEL_DIR, f))
                                  for f in MODEL_FILES.get(fn, [fn])),
                     "mb": round(size / 1e6, 1)})
    return rows


def available() -> bool:
    try:
        from importlib.util import find_spec
        return find_spec("audio_separator") is not None
    except Exception:
        return False


def _vram_free_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def release_all():
    """显式释放分离模型显存（镜像 audio_sr.release_all）。"""
    global _active
    with _LOCK:
        for inst in _active:
            # audio-separator 把权重挂在实例的 model_instance 上；只清列表
            # 仍可能被内部引用拖住，显式置空再 GC。
            try:
                inst.model_instance = None
            except Exception:
                pass
        _active.clear()
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _make_separator(out_dir: str):
    from audio_separator.separator import Separator
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    # 0.47 的签名没有 output_bitdepth（实锤：传了会 TypeError）��
    # WAV 默认 16bit；需要更高位深可在调用侧改 output_bitrate。
    return Separator(
        model_file_dir=MODEL_DIR,
        output_dir=out_dir,
        output_format="WAV",
    )


def _stem_path(files: List[str], keys: List[str], out_dir: str,
               model_name: str = "", exclude_keys: Optional[List[str]] = None
               ) -> Optional[str]:
    """从 separate() 返回里定位 stem，返回**绝对路径**。

    两个实锤坑（2026-10-17 用户翻唱失败）：
    1. audio-separator 0.47 部分路径返回**相对文件名**（common_separator.py:530
       的 os.path.join 只有一个参数）。直接 os.path.isfile 会以 CWD 判定，
       于是“人声明明写出来了却报未找到”。
    2. 输出名形如 `000_(other)_vocals_mel_band_roformer.wav`——模型名里
       带 "vocals"，用子串匹配会把 other/伴奏轨当成人声。必须先从文件名里
       剥掉模型名再匹配；人声候选还要排除 instrumental 类关键词。
    """
    ex = [k.lower() for k in (exclude_keys or [])]
    model_l = (model_name or "").lower()
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(out_dir, os.path.basename(f))
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p).lower()
        stem_area = base[:-4] if base.endswith(".wav") else base
        if model_l:
            stem_area = stem_area.replace(model_l, "")
        if any(k in stem_area for k in ex):
            continue
        if any(k.lower() in stem_area for k in keys):
            return p
    return None


def separate_vocals(src_path: str, out_dir: str,
                    model_label: str = "标准（BS-Roformer ViperX，SDR 12.98）",
                    progress=None, should_stop=None) -> Dict:
    """人声/伴奏分离。返回 {ok, vocals, instrumental, seconds, model}。"""
    model_name = VOCALS_MODELS.get(model_label, DEFAULT_VOCALS)
    return _run_sep(src_path, out_dir, model_name,
                    vocal_keys=["vocals", "vocal"],
                    inst_keys=["instrumental", "accompaniment", "no_vocals", "other"],
                    progress=progress, both=True, should_stop=should_stop)


def dereverb(src_path: str, out_dir: str,
             model_label: str = "标准（UVR-DeEcho-DeReverb）",
             progress=None, should_stop=None) -> Dict:
    """去混响/去回声（干声化）。返回 {ok, dry, ...}。"""
    model_name = DEREVERB_MODELS.get(model_label, DEFAULT_DEREVERB)
    return _run_sep(src_path, out_dir, model_name,
                    vocal_keys=["no_reverb", "noreverb", "dry", "vocals", "vocal"],
                    inst_keys=["reverb"],
                    progress=progress, both=False, should_stop=should_stop)


def _run_sep(src_path: str, out_dir: str, model_name: str,
             vocal_keys: List[str], inst_keys: List[str],
             progress=None, both: bool = True, should_stop=None) -> Dict:
    if not available():
        return {"ok": False, "error": "audio-separator 未安装（pip install \"audio-separator[gpu]\"）"}
    if not os.path.isfile(src_path):
        return {"ok": False, "error": f"文件不存在: {src_path}"}
    # 引擎互斥由调用方保证；这里再做一次显存预检（<1.5GB 提示但继续——audio-separator 会自己回退 CPU）
    free_gb = _vram_free_gb()
    t0 = time.perf_counter()

    # 推理前自动部署：缺模型/配置就经镜像下载（含断点续传与大小校验）。
    # audio-separator 自身只走 GitHub 直连，国内必失败——必须在建 Separator 前做。
    if progress is not None:
        try:
            progress(0.01, f"检查分离模型 {model_name}（缺失则自动下载）")
        except Exception:
            pass
    def _dl_progress(frac, msg):
        if progress is not None:
            try:
                progress(0.01 + 0.04 * float(frac), msg)
            except Exception:
                pass

    dl = ensure_model(model_name, _dl_progress, should_stop=should_stop)
    if not dl.get("ok"):
        if dl.get("cancelled"):
            return {"ok": False, "cancelled": True, "error": dl.get("error"),
                    "model": model_name}
        return {"ok": False,
                "error": "模型自动下载失败: " + str(dl.get("error", "未知原因")),
                "model": model_name, "missing": dl.get("missing", [])}

    try:
        with _LOCK:
            if progress is not None:
                try:
                    progress(0.05, f"加载分离模型 {model_name}（显存空闲 {free_gb:.1f}GB）")
                except Exception:
                    pass
            sep = _make_separator(out_dir)
            _active.append(sep)
            sep.load_model(model_filename=model_name)
            if progress is not None:
                try:
                    progress(0.08, "分离推理中（模型已加载；不改时长）")
                except Exception:
                    pass
            files = sep.separate(src_path)
    except Exception as exc:
        release_all()
        return {"ok": False, "error": f"分离失败: {exc}", "model": model_name}

    seconds = time.perf_counter() - t0
    # 排除规则只用于“人声 vs 伴奏”（both=True）。去混响的干声叫
    # no_reverb，若拿 inst_keys=["reverb"] 去排除会把干声误杀。
    vocals = _stem_path(files, vocal_keys, out_dir, model_name,
                        exclude_keys=inst_keys if both else None)
    instrumental = _stem_path(files, inst_keys, out_dir, model_name) if both else None
    ok_v = vocals is not None and os.path.isfile(vocals)
    if not ok_v:
        release_all()
        return {"ok": False, "error": f"未找到人声输出（产物: {files}）",
                "model": model_name, "seconds": round(seconds, 1)}
    return {"ok": True, "vocals": vocals, "instrumental": instrumental,
            "seconds": round(seconds, 1), "model": model_name,
            "all": files}


def model_dir_size_mb() -> float:
    """已下载分离权重的体积（模型页/清理提示用）。"""
    if not os.path.isdir(MODEL_DIR):
        return 0.0
    total = 0
    for root, _dirs, names in os.walk(MODEL_DIR):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total / (1024 ** 2)
