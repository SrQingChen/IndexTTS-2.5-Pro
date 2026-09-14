"""参数元数据注册表 —— 全 UI 的单一事实来源。

设计意图：
    官方 webui.py 把 label / info / 默认值散落在几百行 gr.xxx() 调用里，
    改一个提示要翻代码，也无法统一输出「参数手册」。

    这里把每个参数抽成 Param 对象集中登记：
        · UI 控件由 Param 自动生成（label/info/范围/步长/默认值全部来自这里）
        · 「参数手册」Tab 直接把注册表渲染成可搜索文档
        · 预设保存/加载用同一份默认值，不会出现两处不一致

    detail_md / pitfall_md 里的行为描述均来自对 infer_v2_5.py、model_v2.py、
    flow_matching.py、diffusion_transformer.py 的代码核实，而非二手文档。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Param:
    """一个可调参数的完整元数据。"""

    key: str
    label: str
    kind: str = "slider"      # slider|number|checkbox|dropdown|text|textarea|audio
    default: Any = None
    group: str = "general"

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    precision: Optional[int] = None   # 仅 kind="number"：小数位数，0 = 只允许整数
    choices: Optional[Sequence[Any]] = None
    lines: int = 3

    info: str = ""            # 控件下方一行简短提示（界面直接可见）
    unit: str = ""

    summary: str = ""         # 一句话说明它是什么
    affects: str = ""         # 影响维度
    detail_md: str = ""       # 详细行为说明
    tuning_md: str = ""       # 调优建议
    pitfall_md: str = ""      # 常见坑

    version: str = "all"      # all | 2.5 | 2
    experimental: bool = False
    readonly: bool = False

    def applies_to(self, is_v25: bool) -> bool:
        return self.version == "all" or self.version == ("2.5" if is_v25 else "2")


GROUP_TITLES: Dict[str, str] = {
    "voice":    "音色与参考音频",
    "text":     "文本与语言",
    "emotion":  "情感控制",
    "sampling": "GPT 采样（T2S 自回归）",
    "segment":  "分句与时长",
    "engine":   "引擎与精度",
    "memory":   "显存策略",
    "oneclick": "一键三连（全自动）",
    "dataset":  "训练 · 数据集",
    "lora":     "训练 · LoRA",
    "optim":    "训练 · 优化器与调度",
    "dpo":      "训练 · DPO 偏好优化",
    "reward":   "评测 · 奖励指标",
}

REGISTRY: Dict[str, Param] = {}


def _reg(p: Param) -> Param:
    REGISTRY[p.key] = p
    return p


def get(key: str) -> Param:
    return REGISTRY[key]


def by_group(group: str, is_v25: bool = True) -> List[Param]:
    return [p for p in REGISTRY.values() if p.group == group and p.applies_to(is_v25)]


def all_params(is_v25: bool = True) -> List[Param]:
    return [p for p in REGISTRY.values() if p.applies_to(is_v25)]


def group_order(is_v25: bool = True) -> List[str]:
    seen = []
    for p in REGISTRY.values():
        if p.applies_to(is_v25) and p.group not in seen:
            seen.append(p.group)
    return seen


def defaults(group: Optional[str] = None, is_v25: bool = True) -> Dict[str, Any]:
    return {
        p.key: p.default
        for p in all_params(is_v25)
        if group is None or p.group == group
    }


# 情感原型候选库大小（config.yaml: emo_num）
EMO_LIB_SPLITS = [3, 17, 2, 8, 4, 5, 10, 24]


def register_inference_params() -> None:
    """登记推理相关参数。拆成函数以便分文件维护。"""
    from webui_app.params_inference import build
    build(_reg, Param)


def register_training_params() -> None:
    """登记训练相关参数（阶段 2）。"""
    try:
        from webui_app.params_training import build
        build(_reg, Param)
    except ImportError:
        pass


register_inference_params()
register_training_params()
