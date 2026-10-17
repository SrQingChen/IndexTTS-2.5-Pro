"""E1-T · 音高/技巧条件化 TimbreConditioner（歌声转换第二曲线）。

设计依据（docs/dev-plan-e1t-2026-10.md §5.2/5.3 + 2026-10 先进性调研结论）：

    style_t(t) = style + α · MLP([MLP_f0([z̃(t); uv(t)]); Emb_tech(tech(t)); style])

- **逐帧残差替换全局 style 复制**（YingMusic-SVC 的 F0-aware timbre adaptor 同构，
  arXiv 2512.04793）：FiLM/AdaLN 逐层改造在 SVCC2025 的证据里收益集中在风格相似度
  且集中在 AR 侧，对音高外推无对口收益，还要逐块扰动 98M DiT 的预训练平衡——不做。
- **残差 MLP 末层零初始化 + α=0.5 可学习**：训练起点严格等于原模型
  （residual≡0），这是比"输出恰好接近"更强的 TTS 回归护栏（ControlNet 惯例）。
- **F0 表示**：连续 speaker 归一化 log2-F0（清音位填 0）+ **独立 uv 二值通道**
  （区分"清音"与"均值音高"，so-vits/MaskVCT 实践）+ ``4·tanh(z/4)`` 软截断
  （外推保险：保序保梯度，超界高音不会被推成无方向的 OOD 输入）。
  不启用官方 length_regulator 的 512-bin 量化死代码——量化 bin 对训练范围外
  音高只能 clamp 饱和，没有外推语义（Seed-VC 量化路线被迫 ±12st 平移规避外推）。
- **technique**：7 类（0=control/直音，1..6=GTSinger 六技巧）帧级 embedding。
- **null/bypass 语义**：条件丢弃（训练 p≈0.1，CosyVoice 式只丢 {f0,tech}）与
  CFG null 分支都实现为**旁路 conditioner**（输出=纯 style 广播），而不是把
  f0/tech 输入置零——置零会得到 ``style + α·MLP([0;emb(0);style]) ≠ style``，
  与预训练的零分支不等价（调研 Q3 结论）。预训练自带的 class_dropout
  （x_in[..., 80:] 整段置零）会连带清掉 style_t 切片，无需重复处理。

硬约束：本模块只在 singing config（``DiT.technique_condition: true``）下构造；
默认 TTS config 不实例化，state_dict 键集合与上游完全一致。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

# technique 词表（与 docs/dev-plan-e1t-2026-10.md §5.2 一致；GTSinger 六技巧 + control）
TECHNIQUE_CLASSES = (
    "control",     # 0：直音/无技巧（GTSinger 无标签帧与普通语音都落这里）
    "mixed",       # 1
    "falsetto",    # 2
    "breathy",     # 3
    "pharyngeal",  # 4
    "vibrato",     # 5
    "glissando",   # 6
)
NUM_TECHNIQUES = len(TECHNIQUE_CLASSES)

# F0 软截断宽度（σ 单位）：4·tanh(z/4)。4σ 之外仍有梯度、保序。
F0_SOFT_CLIP = 4.0


def normalize_f0(f0: torch.Tensor, f0_mean: Optional[torch.Tensor],
                 f0_std: Optional[torch.Tensor],
                 eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """原始 F0(Hz, 0=清音) → (软截断归一化 log2-F0, uv)。

    f0:      (B, T) float，Hz，清音=0
    f0_mean/f0_std: 广播到 (B, T) 的浊音 log2-F0 统计——接受 (B,)（整段同源，
             推理常见）或 (B, T)（训练配对 [prompt|target] 两侧不同说话人，
             各用各的统计，防止跨说话人统计污染条件）；None 时用 batch 内
             该样本自身浊音均值兜底（推理端总该给统计，这里是防御）
    返回 z̃: (B, T)，清音位 0；uv: (B, T) float 0/1
    """
    B, T = f0.shape
    voiced = (f0 > 1e-6).to(f0.dtype)
    logf0 = torch.where(f0 > 1e-6, torch.log2(f0.clamp(min=1e-6)),
                        torch.zeros_like(f0))
    if f0_mean is None:
        # 逐样本浊音均值/标准差兜底（B,)
        denom = voiced.sum(dim=1).clamp(min=1.0)
        mu = (logf0 * voiced).sum(dim=1) / denom
        var = (((logf0 - mu[:, None]) ** 2) * voiced).sum(dim=1) / denom
        f0_mean, f0_std = mu, torch.sqrt(var + eps)
    if f0_mean.dim() == 1:
        f0_mean = f0_mean[:, None].expand(B, T)
    if f0_std is not None and f0_std.dim() == 1:
        f0_std = f0_std[:, None].expand(B, T)
    f0_std = f0_std.clamp(min=0.1)      # σ 下限：防止统计退化把 z 撑爆
    z = (logf0 - f0_mean) / f0_std
    z = F0_SOFT_CLIP * torch.tanh(z / F0_SOFT_CLIP)
    z = z * voiced                        # 清音位回到 0（uv 通道另给信息）
    return z, voiced


class TimbreConditioner(nn.Module):
    """逐帧音高/技巧 → style 残差。只服务 singing config。

    维度（默认 D=256，config ``DiT.timbre_dim`` 可调）：
        f0_proj:  Linear(2, D)        输入 [z̃; uv]
        tech_emb: Embedding(7, D)
        style_mlp: Linear(192 + 2D, D) → SiLU → Linear(D, 192)
        alpha:    Parameter(0.5)      残差全局门控（可学习）
        out_proj 末层零初始化 → 初始 residual≡0，等价原模型。
    """

    def __init__(self, style_dim: int = 192, dim: int = 256,
                 num_techniques: int = NUM_TECHNIQUES):
        super().__init__()
        self.style_dim = int(style_dim)
        self.dim = int(dim)
        self.num_techniques = int(num_techniques)
        # 条件丢弃概率：训练期只丢 {f0,tech}（旁路本模块），保 prompt/cond/style 常在。
        # 与预训练 class_dropout（整段清零）语义不同、互不冲突。
        self.cond_drop_prob = 0.1

        self.f0_proj = nn.Linear(2, dim)
        self.tech_emb = nn.Embedding(num_techniques, dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim + 2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, style_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 末层零初始化：训练起点严格等于原模型（见模块 docstring）
        nn.init.zeros_(self.style_mlp[-1].weight)
        nn.init.zeros_(self.style_mlp[-1].bias)

    def forward(self, style: torch.Tensor, f0: torch.Tensor,
                technique: torch.Tensor,
                f0_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """算逐帧 style_t。

        style:     (B, 192)   全局声纹
        f0:        (B, T)     Hz，清音=0
        technique: (B, T)     long 0..6
        f0_stats:  ((B,), (B,)) 浊音 log2-F0 均值/标准差；None=逐样本兜底
        cond_mask: (B,) bool/0-1；False 的样本旁路（输出=纯 style 广播）。
                   CFG null 分支与训练期条件丢弃都走这里。
        返回 (B, T, 192)
        """
        B, T = f0.shape
        if f0_stats is not None:
            f0_mean, f0_std = f0_stats
        else:
            f0_mean = f0_std = None
        z, uv = normalize_f0(f0, f0_mean, f0_std)
        f0_in = torch.stack([z, uv], dim=-1)                 # (B, T, 2)
        f0_emb = self.f0_proj(f0_in)                          # (B, T, D)
        tech_emb = self.tech_emb(technique.long().clamp(
            0, self.num_techniques - 1))                      # (B, T, D)
        s = style[:, None, :].expand(B, T, self.style_dim)
        residual = self.style_mlp(torch.cat([s, f0_emb, tech_emb], dim=-1))
        out = s + self.alpha * residual
        if cond_mask is not None:
            keep = cond_mask.to(out.dtype).reshape(B, 1, 1)
            out = s + keep * (out - s)                         # 旁路样本回纯 style
        return out
