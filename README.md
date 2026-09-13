<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/indextts_icon_dark.png"/>
  <img src="assets/indextts_icon_light.png" width="260"/>
</picture>

# IndexTTS-2.5 Pro

**在 IndexTTS2 之上搭起来的一整套「可视化控制台 + 自训练体系」**

从参考音频体检、单条合成、批量生产，到数据集构建、LoRA 自训练、
DPO 偏好对齐、A/B 自动评测、权重合并部署 —— 全部在一个网页里完成。

[![Author](https://img.shields.io/badge/作者-SrQingChen-blue?logo=github)](https://github.com/SrQingChen)
[![Repo](https://img.shields.io/badge/GitHub-IndexTTS--2.5--Pro-181717?logo=github)](https://github.com/SrQingChen/IndexTTS-2.5-Pro)
[![Base](https://img.shields.io/badge/基于-IndexTTS2-orange)](https://github.com/index-tts/index-tts)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](#-快速开始)
[![License](https://img.shields.io/badge/许可-分层（见下）-yellow)](#-许可)

[界面一览](#-界面) · [新增能力](#-相比上游新增了什么) · [快速开始](#-快速开始) · [训练体系](#-训练体系本项目的核心) · [验证结果](#-验证结果) · [许可](#-许可)

</div>

---

## 📌 这是什么

这是 **[IndexTTS2](https://github.com/index-tts/index-tts) 的一个增强分支**，作者
[SrQingChen](https://github.com/SrQingChen)。

上游提供了业界领先的零样本语音克隆**模型**；本项目**不改模型**，而是围绕它补齐工程侧最缺的两块：

1. **一个真正好用的可视化控制台** —— 上游的 `webui.py` 是单页长表单，参数挤在一起、
   没有提示、没有工作流。本项目把它拆成 **11 个职责清晰的 Tab**，每个参数都带说明，
   并补上了参考音频工作台、批量生产、预设管理、模型下载、系统监控和一本**参数手册**。
2. **一条完整的自训练流水线** —— 上游只有推理。本项目实现了
   **LoRA 自训练（GPT 语气 + CFM 音色）→ DPO 偏好对齐 → 自动 A/B 评测 → 合并部署**，
   并配了**九道泛化保护防线**，专门解决「微调完读错字、语气发飘」这类翻车。

> **红线**：官方 `webui.py` **一行未改**，`indextts/` 推理引擎**原样使用**。
> 本项目的全部改动都是**新增文件**，两个入口可以并存、互不干扰。

### 目录结构一眼看懂

```
webui.py              ← 官方入口，原样保留
webui_pro.py          ← 本项目入口（11 Tab 控制台）
start.bat             ← Windows 一键启动（环境自检 + 补装 + 启动）
webui_app/            ← 新增：控制台（tabs / services / theme / widgets）
webui_app/training/   ← 新增：训练体系（数据集 / LoRA / DPO / 评测 / 合并 / 泛化保护）
tools/                ← 新增：回归探针 + 模型下载器 + 截图工具
indextts/             ← 上游推理引擎（未改动）
docs/verification/    ← 真机验收报告
```

---

## 🖼 界面

### 总览

![总览](assets/ui/00_overview.png)

顶栏是实时状态条：GPU、引擎加载状态、显存占用、低显存模式、QwenEmotion 状态 —— 一目了然。
左侧主区是全部参数，右侧是可折叠的**参数详解**，鼠标悬停还有逐项 tooltip。

### 11 个 Tab

<table>
<tr>
<td width="50%"><img src="assets/ui/01_synthesis.png"/><br/><b>🎙 合成</b> — 单条合成全参数：参考音频 / 文本语言 / 情感控制 / GPT 采样 / 分句时长，支持 <code>&lt;字|拼音&gt;</code> 读音标注</td>
<td width="50%"><img src="assets/ui/02_audio_lab.png"/><br/><b>🔬 音频工作台</b> — 参考音频体检打分、智能切片、降噪归一、音色库管理</td>
</tr>
<tr>
<td><img src="assets/ui/03_batch.png"/><br/><b>📦 批量</b> — 多行文本 / JSONL 驱动，进度实时可见，结果可打包下载</td>
<td><img src="assets/ui/04_presets.png"/><br/><b>💾 预设</b> — 参数快照管理，与官方 <code>webui.py</code> 互通</td>
</tr>
<tr>
<td><img src="assets/ui/05_dataset.png"/><br/><b>🗂 数据集</b> — 建集 / 导入 / 补文本 / train-val 划分 / 特征离线提取（断点续提）</td>
<td><img src="assets/ui/06_training.png"/><br/><b>🎓 训练</b> — GPT / CFM / DPO 三目标 LoRA 训练，预检 → 训练 → 保险库 → 记录</td>
</tr>
<tr>
<td><img src="assets/ui/07_alignment.png"/><br/><b>⚖️ 对齐</b> — 用当前模型批量合成候选、奖励打分、构造 DPO 偏好对</td>
<td><img src="assets/ui/08_eval_deploy.png"/><br/><b>🏁 评测/部署</b> — A/B 对比、adapter 挂载与强度旋钮、合并成独立权重、泛化保护面板</td>
</tr>
<tr>
<td><img src="assets/ui/09_models.png"/><br/><b>📥 模型</b> — 资源审计 + 镜像优先三级回退下载，实时进度、可断点续传</td>
<td><img src="assets/ui/10_system.png"/><br/><b>🖥 系统</b> — 显存监控、环境体检、事件日志、缓存维护</td>
</tr>
<tr>
<td colspan="2"><img src="assets/ui/11_manual.png"/><br/><b>📖 手册</b> — 架构原理、全部参数逐项说明、场景配方、故障排查。参数说明与控件 tooltip <b>共用同一份注册表</b>，不会两处不一致</td>
</tr>
</table>

> 截图由 `tools/ui_screenshots.py` 自动生成（走 Chrome CDP，无需额外依赖），
> 界面改版后重跑一次即可刷新。

---

## ✨ 相比上游新增了什么

| 能力 | 上游 IndexTTS2 | 本项目 |
|---|:---:|:---:|
| 零样本语音克隆 / 情感控制 / 多语言 | ✅ | ✅（原样使用） |
| 移动端友好的模块化界面 | ❌ 单页长表单 | ✅ **11 Tab 控制台** |
| 逐参数提示 + 参数手册 | ❌ | ✅ 共用一份参数注册表 |
| 参考音频体检 / 切片 / 降噪 / 音色库 | ❌ | ✅ **音频工作台** |
| 批量生产 + JSONL 驱动 | 部分 | ✅ 进度可视 + 打包 |
| 模型完整性审计 + 镜像下载 | ❌ | ✅ **模型资源页** |
| **LoRA 自训练（GPT / CFM）** | ❌ | ✅ **完整训练器** |
| **DPO 偏好对齐** | ❌ | ✅ **偏好对构造 + 训练器** |
| **训练泛化保护** | ❌ | ✅ **九道防线** |
| **自动 A/B 评测（WER / 声纹相似 / reward）** | ❌ | ✅ **评测台 + 试听** |
| **LoRA 合并回独立权重** | ❌ | ✅ **合并 + 可调强度挂载** |
| 拼音纠音辅助 | 手写标注 | ✅ **候选音列表 + 词表过滤** |

---

## 🎓 训练体系（本项目的核心）

上游只有推理，没有训练。本项目补上了完整四层：

```
L1  SFT LoRA      GPT(T2S) 学「怎么说」—— 语气、节奏、停顿
                  CFM(S2M) 学「像谁」  —— 音色、音质、频谱细节
                       ↓
L2  DPO 偏好对齐   同一句话合成多个候选 → 打分 → 好的当 chosen、差的当 rejected
                       ↓
L3  评测与落地     A/B 自动对比（WER / 声纹相似 / reward）→ 调强度 → 合并部署
```

### 两个训练目标为什么要分开

| | GPT (T2S) | CFM (S2M) |
|---|---|---|
| 决定 | 「怎么说」语气、韵律 | 「像谁」音色、音质 |
| 参数量 | 813 M | 98 M |
| 前向 | teacher-forcing 交叉熵 | flow matching（L1 速度场） |
| 只练它的后果 | 说话像但音质发飘 | 音色对但语气平 |

**想真的像本人，两个都得练。**

### 九道泛化保护防线

微调最怕的不是训不动，而是**训得动却把底座带坏了**（读错字、语气发飘）。为此每一层都设了防线：

| # | 防线 | 做什么 |
|---|---|---|
| 1 | 底座只读快照 | 训练前后 SHA-256 / size+mtime 校验，底座被动过一个字节就报警 |
| 2 | 参数可配置校验 | 60+ 项交叉检查（显存 / 数据量 / 超参匹配度），开工前全部摆出来 |
| 3 | 注入面收敛 | 从真实模型扫描可用层，而非硬编码；死模块自动排除 |
| 4 | 权重漂移体检 | `‖ΔW‖/‖W‖` 全局 + 逐层，分五档给建议 |
| 5 | 早停 | 盯 val 曲线，连续不改善即停 —— 升上去的部分全是过拟合 |
| 6 | checkpoint 保险库 | 只留 top-K，原子写入，可回滚；**永不写 `checkpoints/`** |
| 7 | 回放抗遗忘 | 混入底座蒸馏样本，钉住通用能力 |
| 8 | adapter 强度旋钮 | 推理期 `0~1.5` 连续调节，**不用重训**就能在「像」和「稳」之间折中 |
| 9 | 一键回滚 | 保险库里的任意档位可激活，未改善的评估不占名额 |

### 训练相关的工程细节

这些是踩过坑之后固化下来的，写在 `TODO.md` 里：

- **离线特征预提取**：w2v-BERT / campplus / codec / mel / mu 全部离线算好缓存，
  训练时完全不碰这些大模型 —— 8 GB 卡才训得动 813 M 的底座
- **前向只有一份**（`forward.py`）：绕开了上游三个会让训练**静默失效**的陷阱
  （漏 `lang_embedding`、bf16 崩溃、`mask_content` 把条件全清零）
- **val 必须确定性**：固定噪声 + 固定配对，否则 val 曲线的抖动比训练带来的改善还大
- **续训必须恢复权重**：只恢复优化器状态会让续训从纯底座重来，而且**不报错**
- **显存预检硬拦**：Windows WDDM 下显存溢出不报 OOM，而是静默降速 20~30 倍

---

## 🚀 快速开始

### Windows 一键启动

```bat
:: 双击根目录的 start.bat
start.bat
```

`start.bat` 会依次自检 **虚拟环境 → Python 依赖 → 模型文件 → CUDA**，
缺什么就明确告诉你，依赖缺失时可直接用 uv 自动补装，全部就绪后启动并自动打开浏览器。
支持参数透传：`start.bat --lazy`、`start.bat --port 7861`、`start.bat --host 0.0.0.0`。

### 手动安装

```bash
# 1) 环境（首次；约 5~8 GB：Python + torch/CUDA + 依赖）
uv sync --extra webui

# 2) 下载模型（约 7.9 GB，走国内镜像，可断点续传）
uv run tools/model_fetcher.py --version 2.5 --all

# 3) 启动
uv run webui_pro.py          # 本项目：11 Tab 控制台
uv run webui.py              # 上游：单页界面（原样保留）
```

> **为什么是 `--extra webui` 而不是 `--all-extras`**：后者还会拉
> `deepspeed` / `flash-attn` / `torch_compile`，需要 CUDA 工具链且本项目并不依赖。
> `peft`（LoRA 训练）与 `pypinyin`（拼音纠音）已写进基础依赖，`uv sync` 就会带上。

### 常用参数

```bash
uv run webui_pro.py --lazy              # 不预加载模型，秒开界面
uv run webui_pro.py --host 0.0.0.0      # 局域网访问
uv run webui_pro.py --port 7861         # 换端口
uv run webui_pro.py --help              # 全部参数
```

---

## ✅ 验证结果

本项目所有功能都有**可复现的回归测试**，不是「跑通了就完事」。
每个探针自报通过/失败项数，失败时以非 0 退出码结束。

| 测试套件 | 覆盖内容 | 结果 |
|---|---|---|
| `tools/guard_test.py` | 泛化保护九道防线 | **194 / 194** |
| `tools/gpt_train_probe.py` | GPT LoRA 训练器 | **148 / 148** |
| `tools/cfm_train_probe.py` | CFM LoRA 训练器（含全部陷阱回归钉） | **163 / 163** |
| `tools/dpo_probe.py` | DPO 训练器（含 ★ln2 不变量） | **70 / 70** |
| `tools/reward_probe.py` | 奖励打分（whisper + campplus） | **49 / 49** |
| `tools/eval_probe.py` | A/B 评测台 | **28 / 28** |
| `tools/merge_probe.py` | 合并 / 挂载（双向数学对账） | **25 / 25** |
| `tools/stage2_ui_probe.py` | UI + 后台执行器集成 | **20 / 20** |
| `tools/build_check.py` | 11 Tab 构建校验 | ✅ |
| `tools/stage2_acceptance.py` | **真机完整闭环**（见下） | **33 / 33** |

### 真机端到端验收

在 **RTX 4060 Laptop 8 GB** 上跑通完整闭环，耗时 4.0 分钟：

```
引擎合成 28 句建数据集 → 特征提取 → 划分 21/7
  → GPT LoRA 真训练（峰值 1.87 GB，漂移 0.000423，底座逐字节未变）
  → CFM LoRA 真训练（漂移 0.00319）
  → DPO 偏好对真机构造（8 候选 → 1 对成对，margin 不足的 3 对正确丢弃）
  → A/B 评测（真合成 + whisper 打分 + 试听文件）
  → 强度旋钮 48 层 @ 0.7 生效
  → 合并 GPT/CFM 权重，产物校验 missing/unexpected = 0
```

报告存档在 [`docs/verification/`](docs/verification)：

- [阶段 2 验收报告](docs/verification/stage2_report.md) · [GPT 训练报告](docs/verification/gpt_report.md)
- [CFM 训练报告](docs/verification/cfm_report.md) · [A/B 评测报告](docs/verification/ab_report.md)

进度与踩坑记录见 [`TODO.md`](TODO.md)。

---

## ⚠️ 已知限制与环境陷阱

- **8 GB 显存必须错峰**：引擎推理（4.9~5.7 GB）与训练不能并存，
  训练前必须先卸载引擎 —— 控制台会在你点「开始训练」时主动拦下。
- **`uv sync` 会就地覆盖 `.venv`**：如果你之前用 `--system-site-packages` 借过系统 torch，
  直接 `uv sync` 会把 gradio / peft / pypinyin 一起清掉（表现为 `No module named 'gradio'`）。
  修复：`uv sync --extra webui`。
- **Windows 显存溢出不报 OOM**：WDDM 下会静默降速 20~30 倍，所以训练前强制做显存预检。
- **JA / ES 文本归一化**需要 `nemo-text-processing`（依赖 `pynini`，Windows 无官方 wheel）。
  不装也能用，会在日志里明确提示已跳过归一化。
- **理论单条音频上限 72.6 s**（1815 语义 token ÷ 25 Hz），参考音频推理时被截到前 15 s。

---

## 📄 许可

**这是一个衍生作品，因此两层许可同时生效。**

| 适用范围 | 许可证 |
|---|---|
| **模型权重**与**上游代码**（`indextts/`、`webui.py`） | [bilibili 模型使用许可协议](LICENSE) · [中文](LICENSE_ZH.txt) —— **强制、不可替换**，且约束一切衍生品 |
| [SrQingChen](https://github.com/SrQingChen) 的**原创新增文件**（见 [NOTICE](NOTICE) §3） | [GNU GPL v3](LICENSE-ADDITIONS.txt) |

简单说：

- **可以自由使用、研究、修改、再分发**，包括做优化和二次开发；
- **如果你分发修改版**，必须沿用同样的条款：新增部分保持 GPL v3，模型部分遵守
  bilibili 协议（该协议本身要求你把条款继续传递给**你的**下游用户）；
- **请保留署名**：保留 [NOTICE](NOTICE) 与版权声明，同时标注原始权利人
  （bilibili · IndexTTS2）与修改作者（SrQingChen）；你在此基础上继续开发时，
  请把自己的贡献也加进 `NOTICE`；
- 两层条款冲突时，**以 bilibili 协议为准**，GPL v3 不延伸至模型。

> 上游协议 §4.1(a) 要求本分发必须声明：*该衍生品对原模型所作的任何改动
> 与原模型原始权利人无关，原始权利人对该衍生品不背书、不担保、不承担责任。*
> 完整归属链与修改清单见 [NOTICE](NOTICE)。本节不构成法律意见。

---

## 🙏 致谢与上游

本项目只是**外壳与训练工具**，真正了不起的是上游的模型工作：

- **[IndexTTS2](https://github.com/index-tts/index-tts)** — bilibili IndexTTS Team
  （模型权重、推理引擎、`webui.py`、原版 README 存档见
  [`docs/README_UPSTREAM.md`](docs/README_UPSTREAM.md)）
- [tortoise-tts](https://github.com/neonbjb/tortoise-tts) ·
  [XTTSv2](https://github.com/coqui-ai/TTS) ·
  [BigVGAN](https://github.com/NVIDIA/BigVGAN) ·
  [wenet](https://github.com/wenet-e2e/wenet) ·
  [icefall](https://github.com/k2-fsa/icefall) ·
  [maskgct](https://github.com/open-mmlab/Amphion) ·
  [seed-vc](https://github.com/Plachtaa/seed-vc)
- 本项目训练体系用到的开源组件：**PEFT**（LoRA）、**OpenAI Whisper**（WER 评测）、
  **CAMPPlus**（声纹相似度）、**Gradio**（界面）

### 引用上游

```bibtex
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng and Siyi Zhou and Jingchen Shu and Jinchao Wang and Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025},
  doi={10.48550/arXiv.2502.05512},
  url={https://arxiv.org/abs/2502.05512}
}
```

<div align="center">
<sub>IndexTTS-2.5 Pro · 由 <a href="https://github.com/SrQingChen">SrQingChen</a> 维护 ·
官方 <code>webui.py</code> 保持不变，本界面为独立实现 ·
参数行为说明均经源码核实</sub>
</div>
