"""Tab：参数手册与架构说明。

用户要求的「所有参数的提示信息都做好」在这里集中兑现。
内容分六部分：

    ① 架构总览 —— 数据流、音色/情感的注入点、各阶段职责
    ② 参数手册 —— REGISTRY 全量渲染，支持关键词搜索与按组浏览
    ③ 场景配方 —— 常见需求下推荐怎么调
    ④ 故障排查 —— 症状 → 原因 → 处理
    ⑤ 读音纠正 —— <字|拼音> 标注语法（文档与合成页共用同一份定义）
    ⑥ 泛化保护 —— LoRA 微调防退化的十道防线

所有行为描述均来自对 infer_v2_5.py / model_v2.py / flow_matching.py /
diffusion_transformer.py / config.yaml 的代码核实，不是二手文档转述。
读音标注部分另由 tools/pinyin_probe.py 实测验证（1728 条拼音全量往返无损）。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import params as P
from webui_app import theme as T
from webui_app import widgets as W
from webui_app.config import (EMO_BIAS, EMO_MATRIX_SPLITS, EMO_SUM_LIMIT,
                              EMO_VECTOR_LABELS, LOW_VRAM_THRESHOLD_GB,
                              OUTPUT_SAMPLE_RATE, REF_AUDIO_IDEAL)
from webui_app.context import AppContext
from webui_app.services import pronunciation as PR
from webui_app.training import guard as GD

# ---------------------------------------------------------------------------
# ① 架构总览
# ---------------------------------------------------------------------------

ARCH_FLOW = """
### 端到端数据流

```
                    ┌── TextNormalizer → tiktoken → lang_emb ──┐
   输入文本 ────────┤                                          │
                    └── 分句（max_text_tokens_per_segment）────┤
                                                               │
   音色参考音频 ──┬── CAMPPlus ──→ style(192) ──spk_emb_proj──→ │  ①音色注入
                  │                                    (1280)  │
                  └── w2v-BERT h17 → spk_cond_emb(1024) ─────→ │
                                                               │
   情感参考音频 ────── w2v-BERT h17 → Conformer×4               │
                        → Perceiver(num_latents=1) → emo_vec ──┤  ②情感注入
                                       (1280)                  │
                                                               ↓
                                    conds = spk_emb + emo_vec
                                                               ↓
        ╔══════════════════════════════════════════════════════════════╗
        ║  T2S · GPT2  24 层 / 1280 维 / 20 头 / ≈812M 参数            ║
        ║  自回归生成语义 token（25 Hz，codebook 8192）                 ║
        ╚══════════════════════════════════════════════════════════════╝
                                                               ↓
                    EnhancedCodec.decode → 语义特征(1024)
                                                               ↓
                    InterpolateRegulator（×1.72×duration_factor）→ mu(512)
                                                               ↓
   ref_mel(prompt 80×T) + style(192) ────────────────────────→ │  ③音色注入
                                                               ↓
        ╔══════════════════════════════════════════════════════════════╗
        ║  S2M · CFM/DiT  13 层 / 512 维 / ≈98M 参数                   ║
        ║  流匹配（flow matching），默认 25 步 Euler                    ║
        ╚══════════════════════════════════════════════════════════════╝
                                                               ↓
                                    mel(80) → BigVGAN → 波形 22.05 kHz
```
"""

ARCH_NOTES = [
    ("音色注入有两处，不是一处",
     "①<b>CAMPPlus 声纹 → GPT 条件 token</b>：影响韵律、停顿、语速这些「人味」层面的特征；"
     "②<b>ref_mel + style → CFM/DiT</b>：影响音色本身（频谱包络、共振峰、气声质感）。"
     "<br><b>结论：声学音色的主战场在 CFM，不在 GPT。</b>"
     "这也是为什么本项目把 LoRA 做成 GPT + CFM 双目标 —— 只训 GPT 的话，"
     "「神似」会提升，但「形似」（听起来就是那个人）改善有限。"),
    ("情感注入只有一处，且完全不进 CFM",
     "<code>emo_vec</code> 与 <code>spk_emb</code> 相加后送进 GPT，"
     "<b>CFM/DiT 完全不接收情感条件</b>。这正是官方说的「音色与情感解耦」的实现方式："
     "情感只改变说什么、怎么说（token 序列），不改变音质本身。"
     "<br>副作用：情感对音色的影响是<b>间接</b>的 —— 通过改变生成的 token 序列，"
     "再由 CFM 把这些 token 渲染成声学特征。"),
    ("语义 token 是 25 Hz 而不是 50 Hz",
     "config.yaml 里 <code>downsample_scale: 2</code>，"
     "把 codec 的 50 Hz 语义特征降到 25 Hz 再喂给 GPT。"
     "序列长度减半 → GPT 的自回归步数减半 → 显存与耗时都显著下降。"),
    ("duration_factor 是真·时长控制，不是变速",
     "代码里是 <code>target_lengths = int(S_infer.shape[1] * 1.72 * duration_factor)</code>，"
     "作用在 <b>InterpolateRegulator 输出的 mel 帧数</b>上。"
     "帧数变了但采样率不变 → 语速改变而<b>音高不变</b>。"
     "这与「后期用 librosa 变速」有本质区别（后者要么变调，要么引入相位失真）。"),
    ("官方在 GPT 上做过 GRPO 强化学习",
     "docs/README2.5_ZH.md 的评测表里有 <code>IndexTTS2.5-RL</code> 一行："
     "在 T2S 阶段用 GRPO，reward = ASR 的 WER，每条采样 4 个候选。"
     "效果是 WER 4.36% → 3.93%，说话人相似度 77.10% → 77.92%。"
     "<br><b>这对我们的意义：</b>GPT 侧已经被官方优化过一轮，"
     "再在上面做 LoRA 的边际收益有限；<b>CFM 侧没有做过 RL</b>，是更大的空间。"
     "8 GB 显存跑不动 GRPO（要同时驻留 policy + reference + reward 模型 + 在线采样），"
     "所以本项目用 <b>DPO</b> 做离线等效替代 —— 只需要 policy + reference，"
     "且不需要在线采样。"),
]

EMO_DOC = f"""
### 8 维情感向量：从滑块值到生效值

顺序固定为 **{' / '.join(EMO_VECTOR_LABELS)}**，
对应 config.yaml 的 `emo_num: {EMO_MATRIX_SPLITS}`（合计 {sum(EMO_MATRIX_SPLITS)} 条情感原型）。

**归一化流程**（`normalize_emo_vec`）：

1. 每一维乘上偏置系数 `{EMO_BIAS}`
2. 若 8 维之和 > **{EMO_SUM_LIMIT}**，整体**等比压缩**到 {EMO_SUM_LIMIT}

```python
vec = [v * b for v, b in zip(vec, EMO_BIAS)]
total = sum(vec)
if total > {EMO_SUM_LIMIT}:
    scale = {EMO_SUM_LIMIT} / total
    vec = [v * scale for v in vec]
```

> ⚠️ 第 2 步是**静默**的 —— 界面上滑块显示 0.9，实际生效可能只有 0.3。
> 合成页的情感向量区域有一个**实时仪表**，直接显示「滑块值 → 生效值」，
> 超限时会标红。调参时请盯着生效值，不是滑块值。

**四种情感控制方式的本质区别**：

| 模式 | 输入 | emo_vec 从哪来 | 音色还原度 | 可控性 |
|---|---|---|---|---|
| 0 跟随音色音频 | 只有音色参考音频 | 同一段音频过 w2v-BERT+Perceiver | ⭐⭐⭐⭐⭐ 最高 | 无 |
| 1 情感参考音频 | 音色音频 + 情感音频 | 情感音频过 w2v-BERT+Perceiver | ⭐⭐⭐⭐ | 中 |
| 2 8 维向量 | 8 个滑块 | 用户直接指定 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ 最高 |
| 3 情感描述文本 | 一句话描述 | QwenEmotion 推理出 8 维 → **走模式 2** | ⭐⭐⭐ | ⭐⭐⭐ |

**模式 3 的实现细节**：官方 `qwen_emo` 分支本质是「文本 → 8 维向量 → 模式 2 的路径」，
下游完全等价。本项目因为 8 GB 显存装不下 QwenEmotion(1.1 GB) + 主引擎(4.94 GB)，
改成**串行执行**：挂载 QwenEmotion → 算出向量 → **立即卸载** → 走 emo_vector 路径。
结果与官方一致，代价是首次调用多约 5~12 秒（有结果缓存，重复调用 0 开销）。
"""

VRAM_DOC = f"""
### 显存预算（本机 RTX 4060 Laptop 8 GB 实测）

| 项目 | 占用 |
|---|---|
| 引擎常驻（GPT + CFM + codec + BigVGAN + w2v-BERT + CAMPPlus） | **4.94 GB** |
| 推理峰值（含 KV cache 与中间激活） | **5.24 GB** |
| 推理结束后整卡剩余空闲 | **1.34 GB** |
| QwenEmotion（Qwen2.5-0.6B FP16） | **≈1.1 GB** |
| GPT LoRA r=16 训练峰值（96 层 Conv1D，7.86M 可训参数） | **≈4.12 GB** |
| CFM LoRA r=16 训练峰值（105 层 Linear，2.90M 可训参数，base bf16 仅 0.18 GB） | **很宽裕** |

**低显存模式**（显存 < {LOW_VRAM_THRESHOLD_GB:.0f} GB 自动激活）会做两件事：

1. 把部分模块放到 CPU，按需搬到 GPU
2. `infer()` 对**超过 40 字**的文本用 `split_text_by_punctuation(max_chars=40)`
   二次粗切

> ⚠️ 第 2 点是很多「长句读起来怪」问题的根源：切分点由标点决定，
> 40 字的硬上限可能把一句话从中间断开。
> **对策**：在合成页把「分句最大 Token 数」调小（如 60~80），
> 让分句在你控制的位置发生，而不是等官方兜底。

**OOM 应急顺序**（代价从小到大）：

1. 关掉浏览器其他占显存的标签页 / 其他 AI 程序
2. 「系统监控」页 → 清空 CUDA 缓存（只归还缓存块，通常几百 MB）
3. 降低 `max_mel_tokens`（KV cache 显存与它成正比）
4. 降低 `num_beams`（beam search 显存 ≈ num_beams 倍）
5. 缩短单次文本长度
6. 卸载引擎（归还全部 4.94 GB）
"""


# ---------------------------------------------------------------------------
# ③ 场景配方
# ---------------------------------------------------------------------------

RECIPES: List[Dict[str, Any]] = [
    {
        "name": "🎯 最高音色还原度（克隆某人）",
        "when": "想让输出听起来「就是那个人」",
        "params": "情感模式 0（跟随音色音频）· duration_factor 1.0 · "
                  "num_beams 3~5 · temperature 0.7~0.8 · top_p 0.8",
        "key": "参考音频质量 > 一切。先用「参考音频工作台」体检，"
               f"确保时长 ≥{REF_AUDIO_IDEAL['min_sec']:.0f}s"
               f"（甜区 {REF_AUDIO_IDEAL['ideal_min_sec']:.0f}~"
               f"{REF_AUDIO_IDEAL['ideal_max_sec']:.0f}s）、"
               f"SNR ≥{REF_AUDIO_IDEAL['min_snr_db']:.0f}dB、无削波、无 BGM。",
        "why": "模式 0 下 spk_emb 与 emo_vec 来自<b>同一段音频</b>，"
               "配对关系与训练时一致。任何其他模式都会引入不同程度的偏移。",
    },
    {
        "name": "🎭 保留音色但换情感",
        "when": "同一个人，要读出开心/悲伤/愤怒等不同情绪",
        "params": "情感模式 2（8 维向量）· emo_alpha 0.6~0.8 · 只给 1~2 维非零值",
        "key": "盯着「生效值」仪表，8 维总和别超过 "
               f"{EMO_SUM_LIMIT}，否则会被静默等比压缩。",
        "why": "情感只进 GPT 不进 CFM，所以音色底色不会被动摇 —— "
               "这正是解耦设计的好处。但 emo_alpha 过高会让韵律夸张到失真。",
    },
    {
        "name": "⚡ 最快出结果（草稿/试听）",
        "when": "只想快速听听效果，不追求质量",
        "params": "num_beams 1 · max_mel_tokens 1200 · temperature 0.8 · "
                  "分句 Token 上限 120",
        "key": "合成页右栏有「快速 / 均衡 / 质量」三档一键切换。",
        "why": "beam search 的耗时与显存都近似与 num_beams 成正比。"
               "从 3 降到 1 通常能快 2~3 倍，质量损失在草稿场景可接受。",
    },
    {
        "name": "📚 长篇有声书",
        "when": "几千字的连续文本，要求风格统一、不断裂",
        "params": "duration_factor 0.95~1.05 · 分句 Token 上限 80~120 · "
                  "seed 固定 · num_beams 3",
        "key": "<b>固定 seed</b> 是保证风格统一的关键。"
               "批量合成时用 JSONL，把相同音色的任务排在一起 —— "
               "换参考音频会触发一次声纹重算（约 1~2 秒）。",
        "why": "低显存模式下官方对 >40 字文本会二次粗切，"
               "主动把分句上限设小可以让切分点落在你期望的位置。",
    },
    {
        "name": "🗣 播报/新闻腔",
        "when": "要求字正腔圆、节奏稳定、少即兴",
        "params": "temperature 0.5~0.7 · top_p 0.6~0.7 · top_k 20 · "
                  "num_beams 5 · repetition_penalty 10",
        "key": "参考音频本身要选播报风格的素材 —— "
               "模型会模仿参考音频的说话方式，不只是音色。",
        "why": "低温度 + 低 top_p 收窄采样分布，减少「发挥」；"
               "高 num_beams 让序列更平滑稳定。",
    },
    {
        "name": "🎬 角色配音（夸张表演）",
        "when": "动画/游戏角色，需要强烈的情绪起伏",
        "params": "情感模式 1（独立情感参考音频）· emo_alpha 0.8~1.0 · "
                  "temperature 0.9~1.1 · duration_factor 1.05~1.2",
        "key": "情感参考音频可以用<b>另一个人的</b>表演素材 —— "
               "音色来自 spk 音频，情感来自 emo 音频，两者独立。",
        "why": "模式 1 把情感特征来源与音色来源分开，"
               "比模式 2 的手调向量更自然（因为是真实表演的特征）。",
    },
    {
        "name": "🔁 结果可复现（调参对比）",
        "when": "做 A/B 对比，或想稳定复现某条好结果",
        "params": "seed 设为固定整数（如 42）· 其余参数每次只改一个",
        "key": "seed 会同时播种 <code>random</code> / <code>numpy</code> / "
               "<code>torch</code> / <code>torch.cuda</code> 四个源。",
        "why": "GPT 是自回归采样，CFM 的流匹配从随机噪声起步 —— "
               "两处都吃随机数。不固定 seed 的话，同样的参数每次结果都不同，"
               "根本没法判断改动是否有效。",
    },
]


# ---------------------------------------------------------------------------
# ④ 故障排查
# ---------------------------------------------------------------------------

TROUBLE: List[Dict[str, str]] = [
    {"sym": "合成结果音色不像参考音频",
     "cause": "① 参考音频太短（<3s，CAMPPlus 声纹统计不稳定）；"
              "② 参考音频有 BGM/混响/噪声；③ 参考音频开头质量差 —— "
              "官方 <code>_load_and_cut_audio(prompt, 15)</code> 是<b>取前 15 秒</b>而不是择优；"
              "④ 情感模式不是 0，破坏了 spk/emo 配对",
     "fix": "去「参考音频工作台」体检并做智能切片，选评分最高的 8~15s 片段；"
            "先切情感模式 0 验证音色上限，再考虑加情感控制"},
    {"sym": "开了情感随机采样后音色也变了",
     "cause": "<code>use_random</code> 会从 emo_matrix 里<b>随机取一行</b>，"
              "破坏了它与 spk_matrix 的成对关系。"
              f"feat1.pt / feat2.pt 里是 {sum(EMO_MATRIX_SPLITS)} 条<b>成对</b>的情感-声纹原型，"
              "随机打乱配对后，声纹条件也跟着偏了",
     "fix": "要音色稳定就<b>关掉 use_random</b>。"
            "想要情感多样性，改用模式 2 手动给不同的向量组合"},
    {"sym": "情感滑块调到 0.9 但听起来没什么变化",
     "cause": f"8 维之和超过 {EMO_SUM_LIMIT} 时被<b>静默等比压缩</b>了。"
              f"另外每一维还要先乘偏置系数 {EMO_BIAS}",
     "fix": "看合成页的「生效值」实时仪表，按生效值调而不是按滑块值调。"
            "想要强情感，只给 1~2 维非零，别 8 维全开"},
    {"sym": "长句读起来在奇怪的地方断开",
     "cause": f"显存 <{LOW_VRAM_THRESHOLD_GB:.0f}GB 时激活了 low_vram，"
              "官方对 >40 字文本用 <code>split_text_by_punctuation(max_chars=40)</code> 二次粗切，"
              "切分点只看标点不看语义",
     "fix": "把「分句最大 Token 数」调小到 60~80 主动控制切分；"
            "或在原文里补上标点，让切分点落在自然位置"},
    {"sym": "CUDA out of memory",
     "cause": "8 GB 卡上引擎常驻 4.94 GB，推理峰值 5.24 GB，"
              "再叠加 QwenEmotion(1.1GB) 或高 num_beams / 高 max_mel_tokens 就会溢出",
     "fix": "按本页「显存预算」小节的 6 步应急顺序处理。"
            "注意 <code>empty_cache()</code> 只归还缓存块，不释放仍被引用的权重"},
    {"sym": "情感模式 3（文本描述）很慢或报显存不足",
     "cause": "QwenEmotion 是 0.6B 的 LLM，需要额外约 1.1 GB 显存",
     "fix": "本项目已改为串行执行 + 显存预检 + CPU 自动回退（CPU 约 11.8s，显存增量 0）。"
            "在合成页把「QwenEmotion 设备」改成 <b>cpu</b> 可强制走 CPU 路径，"
            "虽然慢但绝不 OOM。相同文本有结果缓存，第二次 0 开销"},
    {"sym": "重复字、复读机、说到一半崩掉",
     "cause": "<code>repetition_penalty</code> 太低或 <code>temperature</code> 太高，"
              "自回归采样陷入循环；也可能是 <code>max_mel_tokens</code> 太小把句子截断了",
     "fix": "repetition_penalty 保持官方默认 10；temperature 降到 0.7~0.8；"
            "max_mel_tokens 提到 1500~1815。合成页「质量档」一键设好这些"},
    {"sym": "声音发闷、有水声、金属感",
     "cause": "参考音频降噪过度。<code>noisereduce</code> 的 "
              "<code>prop_decrease</code> 过高会把语音细节一起削掉",
     "fix": "「参考音频工作台」里把降噪强度降到 0.3~0.5；"
            "如果原素材本身够干净，直接关掉降噪"},
    {"sym": "语速变快/变慢但音调也跟着变了",
     "cause": "你可能在用外部工具（如 librosa/Audacity）做变速，那是时域伸缩，会变调",
     "fix": "用 <code>duration_factor</code>。它作用在 InterpolateRegulator 的 mel 帧数上"
              "（<code>target_lengths = S_infer.shape[1] * 1.72 * duration_factor</code>），"
              "改的是时长不是采样率，所以<b>不变调</b>"},
    {"sym": "加载模型时报缺少文件",
     "cause": "checkpoints/ 下缺必需文件。完整推理需要约 7.9 GB",
     "fix": "去「模型资源」页点「重新审计」看缺哪些，然后选对应范围下载。"
            "下载是幂等的，已就位的文件会自动跳过"},
    {"sym": "transformers 版本不匹配报错",
     "cause": "本项目要求 transformers 4.52.1，系统环境可能装了 5.x",
     "fix": "用项目自带的 <code>.venv</code>（<code>--system-site-packages</code> 创建，"
            "venv 内的 4.52.1 会屏蔽系统的 5.x，且不污染系统环境）："
            "<code>.venv\\Scripts\\python.exe webui_pro.py</code>"},
    {"sym": "中文数字/单位读法不对（如「2024年」）",
     "cause": "文本归一化（TextNormalizer）被关掉了",
     "fix": "保持 <code>text_normalization=True</code>。"
            "它会处理数字、日期、单位、符号的中文读法。"
            "关掉后这些内容会原样进 tokenizer，读法不可控"},
]


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _recipes_md() -> str:
    L = ["每个配方都给出「什么时候用 / 怎么设 / 关键注意 / 为什么」。", ""]
    for r in RECIPES:
        L += [
            f"<details><summary><b>{r['name']}</b></summary>",
            "",
            f"**适用场景**：{r['when']}",
            "",
            f"**推荐参数**：{r['params']}",
            "",
            f"**关键**：{r['key']}",
            "",
            f"**原理**：{r['why']}",
            "",
            "</details>",
            "",
        ]
    return "\n".join(L)


def _trouble_md() -> str:
    L = ["| 症状 | 原因 | 处理 |", "|---|---|---|"]
    for t in TROUBLE:
        L.append(f"| <b>{t['sym']}</b> | {t['cause']} | {t['fix']} |")
    return "\n".join(L)


def _group_md(group: str, is_v25: bool) -> str:
    ps = P.by_group(group, is_v25)
    if not ps:
        return ""
    return "\n\n".join(W.help_markdown(p) for p in ps)


def _all_manual_md(is_v25: bool) -> str:
    """导出用的完整 Markdown（纯 md，不含 HTML 卡片）。"""
    L = [
        "# IndexTTS-2.5 参数手册",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 参数总数：{len(P.all_params(is_v25))}",
        "",
        "## 一、架构总览",
        "",
        "```",
        "文本 → [归一化+tiktoken] ─┐",
        "音色音频 → [CAMPPlus 192] → spk_emb_proj(1280) ─┤",
        "音色音频 → [w2v-BERT h17] → spk_cond_emb(1024) ─┤",
        "情感音频 → [w2v-BERT+Conformer+Perceiver] → emo_vec(1280) ─┤",
        "                                     conds = spk_emb + emo_vec",
        "                                              ↓",
        "              T2S · GPT2 24层/1280维/20头/≈812M  →  语义token(25Hz)",
        "                                              ↓",
        "              EnhancedCodec.decode → InterpolateRegulator → mu(512)",
        "                                              ↓",
        "   ref_mel + style → S2M · CFM/DiT 13层/512维/≈98M（流匹配25步）",
        "                                              ↓",
        "                        mel(80) → BigVGAN → 波形 22.05kHz",
        "```",
        "",
    ]
    for title, body in ARCH_NOTES:
        L += [f"### {title}", "", _strip_html(body), ""]

    L += ["## 二、参数详解", ""]
    for g in P.group_order(is_v25):
        ps = P.by_group(g, is_v25)
        if not ps:
            continue
        L += [f"### {P.GROUP_TITLES.get(g, g)}", ""]
        for p in ps:
            L += [f"#### `{p.key}` — {p.label}", ""]
            if p.summary:
                L += [p.summary, ""]
            bits = []
            if p.minimum is not None:
                bits.append(f"最小 {p.minimum}")
            if p.maximum is not None:
                bits.append(f"最大 {p.maximum}")
            if p.step is not None:
                bits.append(f"步长 {p.step}")
            if p.choices:
                bits.append(f"可选 {' / '.join(str(c) for c in p.choices)}")
            bits.append(f"默认 **{p.default}**")
            if p.unit:
                bits.append(f"单位 {p.unit}")
            L += [" · ".join(bits), ""]
            if p.info:
                L += [f"> {p.info}", ""]
            if p.affects:
                L += [f"**影响维度**：{p.affects}", ""]
            for head, body in (("工作原理", p.detail_md), ("调优建议", p.tuning_md),
                               ("常见坑", p.pitfall_md)):
                if body:
                    L += [f"**{head}**", "", _strip_html(body), ""]
            flags = []
            if p.experimental:
                flags.append("实验功能")
            if p.readonly:
                flags.append("只读")
            if p.version != "all":
                flags.append(f"仅 v{p.version}")
            if flags:
                L += [f"_{ ' / '.join(flags) }_", ""]
            L.append("---")
            L.append("")

    L += ["## 三、场景配方", ""]
    for r in RECIPES:
        L += [f"### {r['name']}", "",
              f"- **适用场景**：{r['when']}",
              f"- **推荐参数**：{r['params']}",
              f"- **关键**：{_strip_html(r['key'])}",
              f"- **原理**：{_strip_html(r['why'])}", ""]

    L += ["## 四、故障排查", "", "| 症状 | 原因 | 处理 |", "|---|---|---|"]
    for t in TROUBLE:
        L.append(f"| {t['sym']} | {_strip_html(t['cause'])} | {_strip_html(t['fix'])} |")
    L.append("")

    # 这两份文档本身就是 Markdown，直接拼进去（SYNTAX_DOC 里的表格
    # 竖线已经转义过，不会被当作列分隔符）
    L += ["## 五、读音纠正（拼音 / 音素 / 假名标注）", "", PR.SYNTAX_DOC.strip(), ""]
    L += ["## 六、LoRA 微调的泛化保护", "", GD.PROTECTION_DOC.strip(), ""]
    return "\n".join(L)


def _strip_html(s: str) -> str:
    """导出纯 Markdown 时把内联 HTML 标签去掉，保留文字。"""
    import re
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</?(b|strong)>", "**", s)
    s = re.sub(r"<code>", "`", s)
    s = re.sub(r"</code>", "`", s)
    s = re.sub(r"<[^>]+>", "", s)
    return s


def render(ctx: AppContext):
    cfg = ctx.cfg
    is_v25 = cfg.is_v25
    total = len(P.all_params(is_v25))

    gr.HTML(T.section(
        "参数手册", "📖",
        f"全部 <b>{total}</b> 个参数的完整文档 · 架构原理 · 场景配方 · 故障排查 · "
        "读音纠正 · 微调保护。"
        "所有行为描述均来自对源码与 config.yaml 的逐行核实。"))

    with gr.Tabs():
        # =================================================================
        # 架构总览
        # =================================================================
        with gr.Tab("🏗 架构总览"):
            gr.Markdown(ARCH_FLOW)
            for title, body in ARCH_NOTES:
                with gr.Column(elem_classes=["ix-manual-card"]):
                    gr.Markdown(f"#### {title}")
                    gr.HTML(f'<div style="font-size:13px">{body}</div>')
            gr.Markdown(EMO_DOC)
            gr.Markdown(VRAM_DOC)

        # =================================================================
        # 参数手册
        # =================================================================
        with gr.Tab(f"🔧 参数详解（{total}）"):
            with gr.Row():
                search_tb = gr.Textbox(
                    label="🔍 搜索参数", scale=3,
                    placeholder="输入关键词，如 情感 / temperature / 显存 / 分句 / emo_alpha",
                )
                search_btn = gr.Button("搜索", variant="primary", scale=1)
                reset_btn = gr.Button("显示全部", scale=1)
            search_out = gr.HTML("")
            with gr.Accordion("📑 按分组浏览（全部展开在下面）", open=True):
                group_area = gr.HTML("")

            # 预先渲染所有分组，搜索时替换 group_area 的内容
            all_groups_md = "".join(
                f'<details><summary style="cursor:pointer;font-weight:700;'
                f'font-size:15px;margin:8px 0">'
                f'{P.GROUP_TITLES.get(g, g)} '
                f'<span style="opacity:.6;font-weight:400">'
                f'({len(P.by_group(g, is_v25))})</span></summary>'
                f'{_group_md(g, is_v25)}</details>'
                for g in P.group_order(is_v25) if P.by_group(g, is_v25)
            )

            def do_search(kw: str):
                kw = (kw or "").strip().lower()
                if not kw:
                    return gr.update(value=all_groups_md), ""
                hits: List[P.Param] = []
                for p in P.all_params(is_v25):
                    blob = " ".join([
                        p.key, p.label, p.info, p.summary, p.affects,
                        p.detail_md, p.tuning_md, p.pitfall_md,
                        P.GROUP_TITLES.get(p.group, p.group),
                    ]).lower()
                    if kw in blob:
                        hits.append(p)
                if not hits:
                    return (gr.update(value=all_groups_md),
                            T.warn(f"没有匹配「{kw}」的参数。"
                                   "试试更短的关键词，或直接在下方分组里找。"))
                cards = "".join(W.help_markdown(p) for p in hits)
                head = (f'<div class="ix-tip">🔍 匹配到 <b>{len(hits)}</b> 个参数'
                        f'（关键词「{kw}」）</div>')
                grouped: Dict[str, List[P.Param]] = {}
                for p in hits:
                    grouped.setdefault(p.group, []).append(p)
                body = "".join(
                    f'<h4 style="margin:14px 0 6px 0">'
                    f'{P.GROUP_TITLES.get(g, g)}</h4>'
                    + "".join(W.help_markdown(p) for p in grouped[g])
                    for g in grouped
                )
                return gr.update(value=head + body), ""

            search_btn.click(do_search, inputs=[search_tb],
                             outputs=[group_area, search_out])
            search_tb.submit(do_search, inputs=[search_tb],
                             outputs=[group_area, search_out])
            reset_btn.click(lambda: (gr.update(value=all_groups_md), ""),
                            inputs=[], outputs=[group_area, search_out])

        # =================================================================
        # 场景配方
        # =================================================================
        with gr.Tab("🍳 场景配方"):
            gr.HTML(T.section("按需求选配方", "🍳",
                              "每个配方都是从架构原理推出来的，不是拍脑袋的「玄学调参」。"))
            gr.Markdown(_recipes_md())

        # =================================================================
        # 故障排查
        # =================================================================
        with gr.Tab("🚑 故障排查"):
            gr.HTML(T.section("症状 → 原因 → 处理", "🚑",
                              "每条都标注了具体的代码依据，方便你自己去验证。"))
            gr.Markdown(_trouble_md())
            gr.HTML(T.tip(
                "排查前先做两件事：<br>"
                "① 去「系统监控」页跑一次<b>完整体检</b>（硬件/依赖/能力评估/磁盘）<br>"
                "② 去「模型资源」页跑一次<b>重新审计</b>（确认文件齐全且大小达标）<br>"
                "八成的「跑不起来」都是这两处的问题。"))

        # =================================================================
        # 微调保护 + 读音纠正
        # =================================================================
        # 两者放同一个子页是有意的：它们回答的是同一个问题 ——
        # 「输出不对的时候该改什么」。先试零风险的读音标注，
        # 确实需要改音色/语气时才上微调，而微调就要看保护机制。
        with gr.Tab("🛡 微调保护"):
            gr.HTML(T.section(
                "LoRA 泛化保护", "🛡",
                "微调最常见的翻车不是「学不像」，而是<b>学会了角色、忘了怎么正常说话</b>。"
                "下面是本机<b>能控制</b>的十道防线，以及它们的代码依据。"))
            gr.Markdown(GD.PROTECTION_DOC)
            gr.HTML(T.section(
                "读音纠正", "🔤",
                "语法在合成页「🔤 读音纠正」里实时生效，这里只是完整文档。"
                "1728 条拼音已用 tools/pinyin_probe.py 逐条实测往返无损。"))
            gr.Markdown(PR.SYNTAX_DOC)
            gr.HTML(T.tip(
                "遇到读错字，<b>先试读音标注，再考虑微调</b>。"
                "标注是零风险的精确修正（不改任何权重），"
                "微调用来改音色与语气，不用来纠读音。"))

        # =================================================================
        # 导出
        # =================================================================
        with gr.Tab("💾 导出手册"):
            gr.HTML(T.section("导出为 Markdown 文件", "💾",
                              "把上面全部内容（含每个参数的完整文档）导出成一个 .md 文件，"
                              "方便离线查阅或分享给同事。"))
            with gr.Row():
                export_btn = gr.Button("📄 生成并导出", variant="primary", scale=1)
                export_file = gr.File(label="下载", scale=2, visible=False)
            export_out = gr.HTML("")

            def on_export():
                md = _all_manual_md(is_v25)
                out_dir = os.path.join(cfg.output_dir, "docs")
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(
                    out_dir, f"IndexTTS-{cfg.version}_参数手册_"
                             f"{time.strftime('%Y%m%d-%H%M%S')}.md")
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(md)
                except OSError as e:
                    return T.err(f"写入失败：{e}"), gr.update()
                n_params = len(P.all_params(is_v25))
                return (T.tip(f"✅ 已导出 <b>{n_params}</b> 个参数的完整文档，"
                              f"{len(md)/1024:.1f} KB<br><code>{path}</code>"),
                        gr.update(value=path, visible=True))

            export_btn.click(on_export, inputs=[], outputs=[export_out, export_file])

    return {"components": {}}
