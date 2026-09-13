"""训练子系统。

阶段 2 的四层：
    L1  SFT      dataset / features / gpt_lora / cfm_lora
    L2  偏好优化  reward / dpo
    L3  评测      evaluate（复用 reward）
    出口          merge（LoRA 权重合并回 gpt.pth / s2mel.pth）

横向贯穿所有层：
    guard        泛化保护 —— 底座只读、参数校验、漂移体检、早停、
                 checkpoint 保险库、推理期 adapter 强度旋钮、回放混合。
                 训练器只消费它，不各自实现一套。

runner.py 统一负责后台执行、进度上报、取消与断点续训，UI 只调它。
"""

from __future__ import annotations

__all__ = ["dataset", "guard", "features", "runner"]
