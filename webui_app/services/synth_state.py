"""services/synth_state.py —— 合成页参数的自动记忆。

回答一个用户痛点：合成页调好的参数，重启之后全都回到出厂默认。
这一层把「上次离开时的参数」落盘，下次启动时直接作为控件初始值。

存什么 / 不存什么（边界刻意画得很清楚）：

    记住：音色来源（含上传音频的副本）、语言、时长系数、种子、
          文本归一化、分句与静音、情感控制全部（模式/权重/8 维向量/
          描述文本/随机采样）、GPT 采样 8 参数、LoRA 选择与强度、
          输出后处理 3 参数、音色库下拉选择。
    不记：合成文本（内容不是配置，且可能很长/敏感）、读音纠正工具区
          的查询状态、QwenEmotion 设备偏好（引擎级运行期状态）。

两个关键实现选择：

    · 恢复走「渲染时初始值」而不是页面加载事件回填 —— render() 在进程
      启动时只跑一次，把上次的值直接塞进控件构造参数，首屏 payload
      就是正确的，没有闪烁、没有事件竞态、也不依赖 Tab.select 在
      首屏是否触发（Gradio 对此行为无保证）。
    · 上传的参考音频是 Gradio 临时文件（TEMP/gradio/...），进程退出
      就没了。所以 save 时把**项目外/临时目录**的音频复制一份到
      outputs/state/ 下（固定两个文件名，滚动覆盖，不会堆积）；
      音色库与 examples 里的路径本来就稳定，原样引用即可。

配置档（命名存档）不走这一层 —— 它复用官方 indextts/utils/presets.py
（存在 outputs/presets/<名称>/，与官方 webui.py 互通），本文件只提供
「合成页实时值 → 官方预设字典」的纯映射 live_to_preset_data()。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from webui_app.config import LANGUAGES_V25, PROJECT_ROOT

# 落盘位置：outputs/state/（配置目录，不是产物目录 —— 清理页不列它）
STATE_DIR = os.path.join(PROJECT_ROOT, "outputs", "state")
STATE_PATH = os.path.join(STATE_DIR, "synthesis_state.json")
PROMPT_COPY = "last_prompt.wav"          # 相对 STATE_DIR，滚动覆盖
EMO_COPY = "last_emo_ref.wav"

# 版本号：将来字段变更时可以做迁移而不是硬崩
STATE_VERSION = 1

# -------------------------------------------------------------------------
# 记忆字段清单（顺序即文档顺序；UI 侧按需取用）
# -------------------------------------------------------------------------

# 数值/布尔/文本类的语义键 —— 与官方预设字段同名的部分可直接互通
VALUE_KEYS: List[str] = [
    "lang", "duration_factor", "seed", "text_normalization",
    "max_text_tokens_per_segment", "interval_silence",
    "emo_alpha", "emo_text", "use_random",
    "emo_vec_0", "emo_vec_1", "emo_vec_2", "emo_vec_3",
    "emo_vec_4", "emo_vec_5", "emo_vec_6", "emo_vec_7",
    "do_sample", "top_p", "top_k", "temperature", "num_beams",
    "repetition_penalty", "length_penalty", "max_mel_tokens",
]
# 枚举/选择类的 UI 值（存 UI 原值，恢复前由调用方校验仍在合法集合内）
CHOICE_KEYS: List[str] = ["emo_mode_label", "voice_name",
                          "lora_run", "lora_ckpt"]
FLOAT_KEYS = {"duration_factor", "emo_alpha", "top_p", "temperature",
              "repetition_penalty", "length_penalty", "interval_silence"} | {
                  f"emo_vec_{i}" for i in range(8)}
INT_KEYS = {"top_k", "num_beams", "max_mel_tokens",
            "max_text_tokens_per_segment"}
BOOL_KEYS = {"text_normalization", "use_random", "do_sample"}
# 本项目自有的、官方预设格式之外的配置（记忆机制保存，配置档不保存）
EXTRA_KEYS: List[str] = ["lora_scale", "polish_on", "polish_presence",
                         "polish_exciter"]
FLOAT_KEYS |= {"lora_scale", "polish_presence", "polish_exciter"}

# 数值范围钳制（与合成页控件一致，防手改 json 注入离谱值）
_CLAMP: Dict[str, Tuple[float, float]] = {
    "duration_factor": (0.5, 2.0), "emo_alpha": (0.0, 1.0),
    "top_p": (0.0, 1.0), "top_k": (0, 100), "temperature": (0.1, 2.0),
    "num_beams": (1, 10), "repetition_penalty": (1.0, 20.0),
    "length_penalty": (-5.0, 5.0), "max_mel_tokens": (50, 1815),
    "max_text_tokens_per_segment": (20, 600),
    "lora_scale": (0.0, 1.5), "polish_presence": (0.0, 6.0),
    "polish_exciter": (0.0, 0.3), "interval_silence": (0.0, 1000.0),
}


# -------------------------------------------------------------------------
# 读写
# -------------------------------------------------------------------------

def _atomic_write(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)              # 同目录 rename，Windows 上也原子


def _stable_audio(path: Optional[str]) -> Optional[str]:
    """音频路径是否重启后仍在（音色库 / examples / presets / state 内的才稳定）。"""
    if not path or not os.path.isabs(path):
        return None
    try:
        rp = os.path.realpath(path)
    except OSError:
        return None
    if not os.path.isfile(rp):
        return None
    stable_roots = (
        os.path.realpath(os.path.join(PROJECT_ROOT, "voice_bank")),
        os.path.realpath(os.path.join(PROJECT_ROOT, "examples")),
        os.path.realpath(os.path.join(PROJECT_ROOT, "outputs", "presets")),
        os.path.realpath(STATE_DIR),
    )
    for root in stable_roots:
        if rp.startswith(root + os.sep):
            return rp
    return None


def _persist_audio(path: Optional[str], copy_name: str) -> Optional[str]:
    """把音频变成「重启后仍存在」的路径：稳定路径原样返回，
    临时上传复制进 STATE_DIR（滚动覆盖，最多两份，不会堆积）。"""
    stable = _stable_audio(path)
    if stable:
        return stable
    if not path or not os.path.isfile(path):
        return None
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        dst = os.path.join(STATE_DIR, copy_name)
        shutil.copy2(path, dst)
        return dst
    except OSError:
        return None


def sanitize(values: Dict[str, Any]) -> Dict[str, Any]:
    """清洗外部（磁盘 json）来的值：类型纠正 + 范围钳制 + 白名单过滤。

    纯函数，探针直接测。任何字段的任何畸形都静默丢弃该字段 ——
    记忆机制坏了不该连累界面起不来。
    """
    out: Dict[str, Any] = {}
    for k in VALUE_KEYS + CHOICE_KEYS + EXTRA_KEYS + ["emo_mode_index"]:
        if k not in values:
            continue
        v = values[k]
        try:
            if k in BOOL_KEYS:
                out[k] = bool(v)
            elif k in FLOAT_KEYS:
                x = float(v)
                lo, hi = _CLAMP.get(k, (-1e12, 1e12))
                out[k] = min(hi, max(lo, x))
            elif k in INT_KEYS:
                x = int(float(v))
                lo, hi = _CLAMP.get(k, (-10 ** 9, 10 ** 9))
                out[k] = int(min(hi, max(lo, x)))
            elif k == "seed":
                out[k] = None if v in (None, "",) else int(float(v))
            elif k == "emo_mode_index":
                out[k] = max(0, min(3, int(v)))
            elif k == "lang":
                s = str(v or "").strip().upper()
                if s in LANGUAGES_V25:      # 只认合法语言，其余丢弃
                    out[k] = s
            elif isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
        except (TypeError, ValueError):
            continue                     # 该字段畸形 → 丢弃
    return out


def save_state(values: Dict[str, Any],
               prompt_audio: Optional[str] = None,
               emo_audio: Optional[str] = None) -> bool:
    """把合成页当前参数落盘。values 是 UI 原始值（sanitize 负责清洗）。"""
    try:
        clean = sanitize(values)
        payload = {
            "version": STATE_VERSION,
            "_remember": bool(values.get("_remember", True)),
            "saved_at": time.time(),
            "values": clean,
            "prompt_audio": _persist_audio(prompt_audio, PROMPT_COPY),
            "emo_audio": _persist_audio(emo_audio, EMO_COPY),
        }
        _atomic_write(STATE_PATH, payload)
        return True
    except Exception:
        return False                    # 记忆失败不影响合成主流程


def load_state() -> Dict[str, Any]:
    """读回上次的参数。任何异常（缺文件/坏 json/字段畸形）都静默降级
    为「无记忆」，界面回到注册表默认值。

    返回 {} 表示没有可用记忆。音频路径在读取时做存在性校验，
    失效的丢弃（UI 侧显示一条提示即可）。
    """
    if not os.path.isfile(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {}
        values = sanitize(payload.get("values") or {})
        if not values:
            return {}
        for key in ("prompt_audio", "emo_audio"):
            p = payload.get(key)
            payload[key] = p if (p and os.path.isfile(p)) else None
        payload["values"] = values
        return payload
    except (OSError, ValueError):
        return {}


def remember_enabled(payload: Dict[str, Any]) -> bool:
    """记忆开关（存在 state 文件自身里，默认开）。"""
    return bool(payload.get("_remember", True)) if payload else True


def forget() -> bool:
    """清掉记忆文件与音频副本（「恢复默认」按钮用）。"""
    ok = True
    for p in (STATE_PATH,
              os.path.join(STATE_DIR, PROMPT_COPY),
              os.path.join(STATE_DIR, EMO_COPY)):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            ok = False
    return ok


def saved_at_text(payload: Dict[str, Any]) -> str:
    ts = payload.get("saved_at") if payload else None
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))


# -------------------------------------------------------------------------
# 配置档映射：合成页实时值 → 官方预设字典
# -------------------------------------------------------------------------

def live_to_preset_data(live: Dict[str, Any]) -> Dict[str, Any]:
    """把合成页控件的实时值映射成官方 save_preset 的 data 字典。

    官方格式 = inference.PRESET_FIELDS 的 19 个语义键
    （emo_control_method 存索引、emo_vector 存 8 元列表）。
    live 里缺失的字段用官方默认值补齐，保证产物预设始终完整可用。
    纯函数，探针直接测。
    """
    def f(key: str, dflt: float) -> float:
        try:
            return float(live.get(key, dflt))
        except (TypeError, ValueError):
            return dflt

    def i(key: str, dflt: int) -> int:
        try:
            return int(float(live.get(key, dflt)))
        except (TypeError, ValueError):
            return dflt

    vec = []
    for n in range(8):
        try:
            vec.append(float(live.get(f"emo_vec_{n}", 0.0)))
        except (TypeError, ValueError):
            vec.append(0.0)

    mode = live.get("emo_mode_index")
    if not isinstance(mode, int) or not (0 <= mode <= 3):
        mode = 0

    seed = live.get("seed")
    try:
        seed = None if seed in (None, "") else int(float(seed))
    except (TypeError, ValueError):
        seed = None

    return {
        "emo_control_method": mode,
        "emo_alpha": f("emo_alpha", 0.65),
        "emo_vector": vec,
        "emo_text": str(live.get("emo_text") or ""),
        "use_random": bool(live.get("use_random", False)),
        "max_text_tokens_per_segment": i("max_text_tokens_per_segment", 120),
        "duration_factor": f("duration_factor", 1.0),
        "interval_silence": f("interval_silence", 0.3),
        "text_normalization": bool(live.get("text_normalization", True)),
        "lang": str(live.get("lang") or "ZH")[:8],
        "seed": seed,
        "do_sample": bool(live.get("do_sample", True)),
        "top_p": f("top_p", 0.8),
        "top_k": i("top_k", 30),
        "temperature": f("temperature", 0.8),
        "num_beams": i("num_beams", 3),
        "repetition_penalty": f("repetition_penalty", 10.0),
        "length_penalty": f("length_penalty", 0.0),
        "max_mel_tokens": i("max_mel_tokens", 1500),
    }
