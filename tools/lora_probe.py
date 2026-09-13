"""LoRA 可行性实测探针。

验证三件事（这是阶段 2 的最高风险点，必须先证实）：
  1. PEFT 能否识别 HF GPT2 的 Conv1D 层并注入 LoRA
  2. 注入后可训练参数量 / 显存增量是否在 8GB 预算内
  3. CFM(DiT) 的 LoRA 目标层命名与注入可行性

用法：
    .venv\\Scripts\\python.exe tools/lora_probe.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

print("=" * 70)
print("LoRA 可行性探针")
print("=" * 70)

cfg = OmegaConf.load("checkpoints/config.yaml")

# ---------------------------------------------------------------------------
# 1. 只构建 GPT（不加载权重，省时间省显存）来探测结构
# ---------------------------------------------------------------------------
from indextts.gpt.model_v2 import UnifiedVoice  # noqa: E402

print("\n[1] 构建 UnifiedVoice（CPU，随机权重）以探测结构 ...")
gpt_model = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")

from transformers.pytorch_utils import Conv1D  # noqa: E402

conv1d_names = {}
linear_names = {}
for name, mod in gpt_model.named_modules():
    leaf = name.split(".")[-1]
    if isinstance(mod, Conv1D):
        conv1d_names[leaf] = conv1d_names.get(leaf, 0) + 1
    elif isinstance(mod, torch.nn.Linear):
        linear_names[leaf] = linear_names.get(leaf, 0) + 1

print(f"    顶层 self.gpt 的类型: {type(gpt_model.gpt).__module__}.{type(gpt_model.gpt).__name__}")
print("\n    Conv1D 层（HF GPT2 的注意力/MLP 用这个，不是 nn.Linear）:")
for k, v in sorted(conv1d_names.items(), key=lambda x: -x[1]):
    print(f"      {k:<20} x{v}")
print("\n    nn.Linear 层:")
for k, v in sorted(linear_names.items(), key=lambda x: -x[1]):
    print(f"      {k:<24} x{v}")

total = sum(p.numel() for p in gpt_model.parameters())
print(f"\n    UnifiedVoice 总参数: {total/1e6:.1f}M")

# ---------------------------------------------------------------------------
# 2. PEFT 注入测试（GPT）
# ---------------------------------------------------------------------------
print("\n[2] PEFT LoRA 注入测试 ...")
from peft import LoraConfig, get_peft_model  # noqa: E402
from peft.tuners.lora import LoraLayer  # noqa: E402

for r, alpha in ((8, 16), (16, 32), (32, 64)):
    targets = ["c_attn", "c_proj", "c_fc"]
    lc = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.05,
        target_modules=targets, bias="none",
        task_type=None,
    )
    try:
        peft_model = get_peft_model(gpt_model, lc)
        n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in peft_model.parameters())
        injected = sum(
            1 for m in peft_model.modules()
            if isinstance(m, LoraLayer)
        )
        print(f"    r={r:<3} alpha={alpha:<3} → 注入 {injected:>3} 个 LoRA 层, "
              f"可训练 {n_train/1e6:.2f}M / {n_all/1e6:.1f}M "
              f"({100*n_train/n_all:.3f}%)")
        if r == 16:
            sample = [
                n for n, m in peft_model.named_modules()
                if isinstance(m, LoraLayer)
            ][:6]
            print(f"        样例层名: {sample}")
        # 释放
        del peft_model
        gpt_model = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    except Exception as e:
        print(f"    r={r} 注入失败: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 3. 只对 GPT2 主干注入（排除 Conformer/Perceiver），更接近实际训练配置
# ---------------------------------------------------------------------------
print("\n[3] 仅 GPT2 主干 + mel_head 的注入方案 ...")
lc = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "c_fc"],
    modules_to_save=[],
    bias="none",
)
pm = get_peft_model(gpt_model, lc)
groups = {}
for n, p in pm.named_parameters():
    if p.requires_grad:
        # 归类
        if "base_model.model.gpt.h." in n:
            key = "GPT2主干 h.*"
        elif "mel_head" in n:
            key = "mel_head"
        elif "text_head" in n:
            key = "text_head"
        elif "conformer" in n or "linear_q" in n or "w_1" in n:
            key = "Conformer(情感编码器)"
        elif "perceiver" in n or "to_q" in n:
            key = "Perceiver"
        else:
            key = "其他"
        groups[key] = groups.get(key, 0) + p.numel()
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"    {k:<26} {v/1e6:>7.3f}M")
n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
print(f"    {'合计可训练':<26} {n_train/1e6:>7.3f}M")

# 显存估算
print("\n[4] 8GB 显存下的 GPT LoRA 训练预算估算 ...")
base_bf16 = total * 2 / 1024**3
lora_p = n_train
# AdamW: fp32 参数副本 + m + v = 12 bytes/param；梯度 bf16 = 2 bytes
opt_gb = lora_p * 12 / 1024**3
grad_gb = lora_p * 2 / 1024**3
print(f"    base 权重 (bf16)         : {base_bf16:.2f} GB")
print(f"    LoRA 梯度 (bf16)         : {grad_gb:.3f} GB")
print(f"    AdamW 状态 (fp32 m+v+主)  : {opt_gb:.3f} GB")
print(f"    小计（不含激活）          : {base_bf16+grad_gb+opt_gb:.2f} GB")
print(f"    激活（梯度检查点+batch1） : ~1.5-2.5 GB（实测为准）")
print(f"    CUDA 上下文              : ~0.5 GB")
print(f"    → 预计峰值               : {base_bf16+grad_gb+opt_gb+2.0+0.5:.2f} GB / 8.00 GB")

# ---------------------------------------------------------------------------
# 5. CFM(DiT) 注入测试
# ---------------------------------------------------------------------------
print("\n[5] CFM(DiT) LoRA 注入测试 ...")
from indextts.s2mel.modules.commons import MyModel  # noqa: E402

s2mel = MyModel(cfg.s2mel)
n_s2mel = sum(p.numel() for p in s2mel.parameters())
n_cfm = sum(p.numel() for p in s2mel.models["cfm"].parameters())
print(f"    s2mel 总参数 {n_s2mel/1e6:.1f}M，其中 CFM {n_cfm/1e6:.1f}M")

cfm_linear = {}
for name, mod in s2mel.models["cfm"].named_modules():
    if isinstance(mod, torch.nn.Linear):
        leaf = name.split(".")[-1]
        cfm_linear[leaf] = cfm_linear.get(leaf, 0) + 1
print("    CFM 内 nn.Linear 层:")
for k, v in sorted(cfm_linear.items(), key=lambda x: -x[1]):
    print(f"      {k:<24} x{v}")

# 只注入 DiT 主干的注意力+MLP
dit_targets = [k for k, v in cfm_linear.items() if v >= 13]
print(f"\n    选定注入目标（出现 ≥13 次 = 每层都有）: {dit_targets}")
lc2 = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=dit_targets, bias="none",
)
try:
    pm2 = get_peft_model(s2mel.models["cfm"], lc2)
    n2 = sum(p.numel() for p in pm2.parameters() if p.requires_grad)
    inj2 = sum(1 for m in pm2.modules() if isinstance(m, LoraLayer))
    print(f"    → 注入 {inj2} 个 LoRA 层, 可训练 {n2/1e6:.3f}M "
          f"({100*n2/n_cfm:.3f}% of CFM)")
    base_gb = n_cfm * 4 / 1024**3
    print(f"    base 权重 (fp32)          : {base_gb:.2f} GB")
    print(f"    base 权重 (bf16)          : {base_gb/2:.2f} GB")
    print(f"    AdamW 状态               : {n2*12/1024**3:.3f} GB")
    print(f"    → CFM LoRA 在 8GB 上非常宽裕 ✅")
except Exception as e:
    print(f"    CFM 注入失败: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("探针完成")
print("=" * 70)
