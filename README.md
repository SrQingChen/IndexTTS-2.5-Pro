<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/indextts_icon_dark.png"/>
  <img src="assets/indextts_icon_light.png" width="240"/>
</picture>

# IndexTTS-2.5 Pro

**零样本语音克隆 · 可视化控制台 · LoRA 自训练 · DPO 偏好对齐 · A/B 自动评测 · 权重合并部署**

基于 [IndexTTS2](https://github.com/index-tts/index-tts) 构建的增强工程：模型与推理引擎一行不改，
为它补上一整套**网页控制台**和一条**完整的自训练流水线**。
从参考音频体检、单条合成、批量生产，到数据集构建、LoRA 训练、DPO 对齐、
自动评测、合并部署 —— 全部在一个浏览器页面里完成。

[![Author](https://img.shields.io/badge/作者-SrQingChen-blue?logo=github)](https://github.com/SrQingChen)
[![Repo](https://img.shields.io/badge/GitHub-IndexTTS--2.5--Pro-181717?logo=github)](https://github.com/SrQingChen/IndexTTS-2.5-Pro)
[![Base](https://img.shields.io/badge/基于-IndexTTS2-orange)](https://github.com/index-tts/index-tts)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](#4-环境要求)
[![License](https://img.shields.io/badge/许可-分层（见下）-yellow)](#13-许可证)

[功能特性](#2-核心特性) · [界面预览](#3-界面总览) · [安装](#5-安装) · [快速上手](#6-快速上手) · [功能详解](#7-功能详解) · [自训练指南](#8-自训练指南从数据集到部署) · [测试验证](#9-测试与验证) · [FAQ](#10-常见问题-faq) · [许可证](#13-许可证)

</div>

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心特性](#2-核心特性)
- [3. 界面总览](#3-界面总览)
- [4. 环境要求](#4-环境要求)
- [5. 安装](#5-安装)
- [6. 快速上手](#6-快速上手)
- [7. 功能详解](#7-功能详解)
- [8. 自训练指南（从数据集到部署）](#8-自训练指南从数据集到部署)
- [9. 测试与验证](#9-测试与验证)
- [10. 常见问题（FAQ）](#10-常见问题-faq)
- [11. 已知限制](#11-已知限制)
- [12. 项目结构](#12-项目结构)
- [13. 许可证](#13-许可证)
- [14. 致谢](#14-致谢)
- [15. 引用](#15-引用)

---

## 1. 项目简介

IndexTTS-2.5 Pro 是 [IndexTTS2](https://github.com/index-tts/index-tts) 的增强分支，由 [SrQingChen](https://github.com/SrQingChen) 开发维护。

上游提供了业界领先的零样本语音克隆**模型**。本项目不改模型，围绕它补齐工程侧最缺的两块能力：

1. **一个真正好用的可视化控制台。** 上游的 `webui.py` 是单页长表单，参数挤在一起、没有提示、没有工作流。本项目将其重构为 **11 个职责清晰的功能页**：每个参数都带悬停提示与参数手册，并新增参考音频工作台、批量生产、预设管理、模型下载、系统监控等页面。
2. **一条完整的自训练流水线。** 上游只有推理。本项目实现了 **LoRA 自训练（GPT 语气 + CFM 音色）→ DPO 偏好对齐 → 自动 A/B 评测 → 合并部署** 的完整闭环，并配有九道泛化保护防线，专门解决「微调完读错字、语气发飘」这类常见翻车。

**工程红线：** 官方 `webui.py` 一行未改，`indextts/` 推理引擎原样使用。本项目的全部改动都是新增文件，两个入口可以并存、互不干扰。

### 与上游的能力对照

| 能力 | 上游 IndexTTS2 | 本项目 |
|---|:---:|:---:|
| 零样本语音克隆 / 情感控制 / 多语言 | ✅ | ✅（引擎原样使用） |
| 模块化界面（11 个功能页） | ❌ 单页长表单 | ✅ |
| 逐参数提示 + 参数手册 | ❌ | ✅ 共用同一份参数注册表 |
| 参考音频体检 / 切片 / 降噪 / 音色库 | ❌ | ✅ 音频工作台 |
| 批量生产 + JSONL 驱动 | 部分 | ✅ 进度可视 + 打包下载 |
| 模型完整性审计 + 镜像优先下载 | ❌ | ✅ 模型资源页 |
| LoRA 自训练（GPT / CFM） | ❌ | ✅ 完整训练器 |
| DPO 偏好对齐 | ❌ | ✅ 偏好对构造 + 训练器 |
| 训练泛化保护 | ❌ | ✅ 九道防线 |
| 自动 A/B 评测（WER / 声纹相似 / reward） | ❌ | ✅ 评测台 + 逐条试听 |
| LoRA 合并回独立权重 | ❌ | ✅ 合并 + 强度可调挂载 |
| 拼音纠音辅助 | 手写标注 | ✅ 候选音列表 + 词表过滤 |

---

## 2. 核心特性

### 🎙 推理与生产

- **全参数可视化合成** —— 音色来源、文本语言（ZH / EN / JA / AR / ES）、情感控制、GPT 采样参数、分句与时长，全部图形化，每个参数都有悬停提示与手册说明。
- **情感控制** —— 支持 8 维情感向量（喜 / 怒 / 哀 / 惧 / 厌恶 / 低落 / 惊喜 / 平静）连续调节，也支持情感参考音频与情感文本描述。
- **读音标注纠音** —— 多音字用 `<字|拼音>` 标注即可精确控制读音（如 `银<行|HANG2>`），合成页可一键列出候选读音；英文支持音素标注、日文支持假名标注。
- **参考音频工作台** —— 体检打分、智能切片、降噪归一、音色库管理，把「选对参考音频」这件投入产出比最高的事做成可视化流程。
- **批量生产** —— 多行文本或 JSONL 任务文件驱动，每条任务可独立设置参考音频 / 情感 / 语言 / 参数；进度实时可见，结果打包下载。JSONL 格式与官方 `examples/batch/*.jsonl` 兼容。
- **预设管理** —— 参数快照保存 / 加载 / 删除，与官方 `webui.py` 互通。

### 🎓 自训练体系（本项目核心）

- **数据集构建** —— 创建 / 导入音频（拷贝入库，不怕源文件被挪）、补文本、train / val 可复现划分（同 seed 同结果）。
- **特征离线预提取** —— w2v-BERT / campplus / codec / mel / mu 全部离线算好缓存，训练时不再加载这些大模型，8 GB 显卡也能训练 813 M 参数的 GPT 底座；支持断点续提。
- **GPT（T2S）LoRA 训练器** —— 学「怎么说」：语气、节奏、停顿。
- **CFM（S2M）LoRA 训练器** —— 学「像谁」：音色、音质、频谱细节。
- **DPO 偏好对齐** —— 用当前模型批量合成候选 → whisper WER + campplus 声纹相似度自动打分 → 最优 / 最差成对构造偏好数据 → DPO 训练；margin 不足的对自动丢弃，不学噪声。
- **自动 A/B 评测** —— 两个配置同文本同种子各合成一遍，WER / 声纹相似 / reward 三指标胜负表 + 逐条试听，「像不像」用数据说话。
- **挂载与合并部署** —— adapter 可挂载到引擎并用**强度旋钮**（0 ~ 1.5）实时调节，不用重训就能在「像」和「稳」之间折中；试听满意后可把 ΔW 合并进独立的 `gpt.pth` / `s2mel.pth`，脱离训练目录使用。
- **九道泛化保护防线** —— 底座只读快照、参数预检、注入面收敛、权重漂移体检、早停、checkpoint 保险库、回放抗遗忘、强度旋钮、一键回滚（详见 [§8.8](#88-泛化保护九道防线)）。

### 🛡 工程质量

- **全部功能有可复现的回归测试**：11 套回归与静态校验、**730 项断言全部通过**，另有一份真机端到端闭环验收报告，均见 [§9 测试与验证](#9-测试与验证)。
- **Windows 一键启动**：`start.bat` 自检虚拟环境 → 依赖 → 模型 → CUDA，缺什么明确提示，依赖缺失可一键补装。
- **镜像优先的模型下载**：hf-mirror 优先 + 三级回退，实时进度、断点续传、资源完整性审计。

---

## 3. 界面总览

![总览](assets/ui/00_overview.png)

顶栏是实时状态条：GPU 型号、引擎加载状态、显存占用、低显存模式、QwenEmotion 状态一目了然。主区上方为功能页导航，右侧提供可折叠的参数详解，鼠标悬停任意参数都有逐项提示。

### 11 个功能页

<table>
<tr>
<td width="50%"><img src="assets/ui/01_synthesis.png"/><br/><b>🎙 合成</b> —— 单条合成全参数：参考音频 / 文本语言 / 情感控制 / GPT 采样 / 分句时长，支持 <code>&lt;字|拼音&gt;</code> 读音标注</td>
<td width="50%"><img src="assets/ui/02_audio_lab.png"/><br/><b>🔬 音频工作台</b> —— 参考音频体检打分、智能切片、降噪归一、音色库管理</td>
</tr>
<tr>
<td><img src="assets/ui/03_batch.png"/><br/><b>📦 批量</b> —— 多行文本 / JSONL 驱动，进度实时可见，结果可打包下载</td>
<td><img src="assets/ui/04_presets.png"/><br/><b>💾 预设</b> —— 参数快照管理，与官方 <code>webui.py</code> 互通</td>
</tr>
<tr>
<td><img src="assets/ui/05_dataset.png"/><br/><b>🗂 数据集</b> —— 建集 / 导入 / 补文本 / train-val 划分 / 特征离线提取（断点续提）</td>
<td><img src="assets/ui/06_training.png"/><br/><b>🎓 训练</b> —— GPT / CFM / DPO 三目标 LoRA 训练：预检 → 训练 → 保险库 → 记录</td>
</tr>
<tr>
<td><img src="assets/ui/07_alignment.png"/><br/><b>⚖️ 对齐</b> —— 用当前模型批量合成候选、奖励打分、构造 DPO 偏好对</td>
<td><img src="assets/ui/08_eval_deploy.png"/><br/><b>🏁 评测/部署</b> —— A/B 对比、adapter 挂载与强度旋钮、合并成独立权重、泛化保护面板</td>
</tr>
<tr>
<td><img src="assets/ui/09_models.png"/><br/><b>📥 模型</b> —— 资源审计 + 镜像优先三级回退下载，实时进度、可断点续传</td>
<td><img src="assets/ui/10_system.png"/><br/><b>🖥 系统</b> —— 显存监控、环境体检、事件日志、缓存维护</td>
</tr>
<tr>
<td colspan="2"><img src="assets/ui/11_manual.png"/><br/><b>📖 手册</b> —— 架构原理、全部参数逐项说明、场景配方、故障排查。参数说明与控件提示<b>共用同一份注册表</b>，永不错位</td>
</tr>
</table>

> 截图由 `tools/ui_screenshots.py` 自动生成（走 Chrome CDP，无额外依赖），界面改版后重跑一次即可刷新。

---

## 4. 环境要求

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows 10 / 11（主要测试平台）；Linux 亦可 | `start.bat` 仅限 Windows，Linux 用 `uv` 手动流程 |
| GPU | NVIDIA，显存 ≥ 8 GB（推荐） | 已在 RTX 4060 Laptop 8 GB 上完整验收；无 CUDA 时回退 CPU，速度慢 |
| 磁盘 | ≥ 25 GB 可用空间 | 环境 5~8 GB + 模型 7.9 GB + 数据 / 输出 |
| Python | 3.10 或 3.11 | 由 `uv` 自动管理，无需手动安装 |
| 包管理 | [uv](https://docs.astral.sh/uv/) | `pip install uv` 即可安装 |

---

## 5. 安装

### 5.1 Windows 一键启动（推荐）

```bat
:: 双击根目录的 start.bat，或在本目录执行：
start.bat
```

`start.bat` 会依次自检 **虚拟环境 → Python 依赖 → 模型文件 → CUDA**：

- 缺虚拟环境 / 依赖时，给出明确的修复命令，依赖缺失时可直接用 uv 一键补装；
- 缺模型时提示先去 WebUI「模型」页下载，或用命令行下载；
- 全部就绪后启动控制台并自动打开浏览器（`http://127.0.0.1:7860`）。

支持参数透传：`start.bat --lazy`、`start.bat --port 7861`、`start.bat --host 0.0.0.0`。

### 5.2 手动安装（Windows / Linux 通用）

```bash
# 1) 克隆仓库
git clone https://github.com/SrQingChen/IndexTTS-2.5-Pro.git
cd IndexTTS-2.5-Pro

# 2) 创建环境（首次约 5~8 GB：Python + torch/CUDA + 依赖）
uv sync --extra webui

# 3) 下载模型（约 7.9 GB，镜像优先，可断点续传）
uv run tools/model_fetcher.py --version 2.5 --all

# 4) 启动
uv run webui_pro.py          # 本项目：11 页控制台
uv run webui.py              # 上游：单页界面（原样保留，可对照使用）
```

> **为什么是 `--extra webui` 而不是 `--all-extras`**：后者还会拉 `deepspeed` / `flash-attn` / `torch_compile`，需要 CUDA 工具链且本项目并不依赖。`peft`（LoRA 训练）与 `pypinyin`（拼音纠音）已写进基础依赖，`uv sync` 就会带上。

### 5.3 模型下载

三种方式任选：

1. **WebUI「模型」页**（推荐）—— 资源审计 + 实时进度 + 断点续传，缺什么一目了然；
2. **命令行** —— `uv run tools/model_fetcher.py --version 2.5 --all`；
3. **手动放置** —— 解压到 `checkpoints/` 目录，模型页会自动审计完整性。

下载走 `hf-mirror.com` 镜像优先并自动回退官方源，国内网络友好。

### 5.4 常用启动参数

```bash
uv run webui_pro.py --lazy              # 不预加载模型，秒开界面（首次合成时再加载）
uv run webui_pro.py --host 0.0.0.0      # 局域网访问
uv run webui_pro.py --port 7861         # 换端口
uv run webui_pro.py --help              # 查看全部参数
```

---

## 6. 快速上手

第一次使用，三分钟即可出声：

1. **启动** —— 双击 `start.bat`（或 `uv run webui_pro.py`），浏览器自动打开控制台。
2. **选音色** —— 进入「🎙 合成」页，在「音色来源」上传一段参考音频（建议 5~15 秒、干净人声、无背景音乐），或从音色库选择已有音色。
3. **输入文本** —— 在文本框输入要合成的内容，语言选 `auto` 自动检测即可。
4. **生成** —— 点击「生成」，等待数秒，试听并下载。

进阶提示：

- 想控制情感 → 展开「情感控制」，拖动 8 维情感向量，或上传一段「情感参考音频」；
- 遇到多音字读错 → 用 `<字|拼音>` 标注，如 `他在银<行|HANG2>里工作`；合成页的「读音助手」可列出候选读音一键插入；
- 想批量生产 → 去「📦 批量」页，粘贴多行文本或上传 JSONL 任务文件；
- 想系统性了解每个参数 → 读内置「📖 手册」，架构原理、逐参数说明、场景配方、故障排查都在里面。

---

## 7. 功能详解

### 7.1 语音合成

单条合成的主战场，参数分五组：

| 参数组 | 内容 |
|---|---|
| 音色来源 | 上传参考音频 / 音色库选择；参考音频推理时取**前 15 秒**，开头质量最关键 |
| 文本与语言 | 文本输入 + 语言选择（auto / ZH / EN / JA / AR / ES）；支持 `<字\|拼音>`、`<word\|phoneme>`、`<漢字\|かな>` 读音标注 |
| 情感控制 | 8 维情感向量（喜 / 怒 / 哀 / 惧 / 厌恶 / 低落 / 惊喜 / 平静）/ 情感参考音频 / 情感文本描述三种方式 |
| GPT 采样 | temperature、top_p、top_k、重复惩罚等，控制稳定性与多样性 |
| 分句与时长 | 自动分句开关、时长因子（语速快慢） |

示例文本（内置「示例」区可直接载入）：

```
中文 · 多音字标注：他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。
英文 · 音素标注：He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.
日语 · 假名标注：彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。
```

### 7.2 读音标注（拼音纠音）

- 输入 `<字|拼音>` 后合成时按指定读音发音，解决多音字 / 生僻字 / 专有名词读错的问题；
- 「读音助手」基于词表给出候选读音列表，点击即插入，不用手查拼音；
- 英文支持 ARPABET 音素标注，日文支持假名标注，规则同官方引擎。

### 7.3 参考音频工作台

IndexTTS-2.5 是零样本 TTS，音色几乎完全由参考音频决定（CAMPPlus 声纹 + w2v-BERT 情感特征 + ref_mel 声学模板三路注入）。把参考音频选对、处理好，往往比训练 LoRA 更有效、更零风险。

流程：**上传 → 体检打分 → 智能切片 → 增强处理 → 入库 → 合成页直接调用**

- **体检打分**：每个指标对应推理链路里的一个真实约束（并非泛泛的「音质好不好」），例如官方推理只取前 15 秒，所以开头静音 / 噪声会直接吃掉有效信息；
- **智能切片**：从长音频里自动切出质量最佳的片段；
- **降噪归一**：一键增强处理；
- **音色库**：处理好的参考音频入库管理，合成页 / 批量页 / JSONL 任务均可引用。

### 7.4 批量合成

两种输入：

1. **多行文本** —— 一行一条，共用界面上的全局参数；
2. **JSONL 任务文件** —— 一行一个 JSON 对象，**只有 `text` 必填**，其余字段缺省时继承全局设置：

```jsonl
{"text": "第一段文本"}
{"text": "第二段", "lang": "EN", "spk_audio_prompt": "voice_bank/audio/host_a.wav"}
{"text": "第三段", "emo_control_method": 2, "emo_vector": [0,0,0.7,0,0,0,0,0]}
{"text": "第四段", "duration_factor": 1.2, "temperature": 0.7, "seed": 42}
```

进度实时可见，完成后可打包下载全部音频。格式与官方 `examples/batch/*.jsonl` 兼容。

### 7.5 预设管理

把当前合成参数存成命名快照，随时载入 / 删除。预设文件与官方 `webui.py` 互通——在官方界面保存的预设，本项目同样能用。

### 7.6 模型管理

- **资源审计**：逐项检查主模型与辅助模型（w2v-BERT、campplus、semantic codec、BigVGAN 等）是否就位；
- **镜像优先下载**：hf-mirror 优先 + 三级回退，实时进度、断点续传；
- 缺模型时明确告诉你缺哪个，不会到合成时才报错。

### 7.7 系统监控与参数手册

- **🖥 系统**：显存实时仪表、环境体检、事件日志、缓存维护；引擎可一键加载 / 卸载（8 GB 显存错峰必备）。
- **📖 手册**：架构原理（GPT / CFM / 情感三路注入）、全部参数逐项说明、场景配方、故障排查。手册与控件提示共用同一份参数注册表，界面与文档永不打架。

---

## 8. 自训练指南（从数据集到部署）

### 8.1 流程总览

```
L1  SFT LoRA       GPT(T2S) 学「怎么说」—— 语气、节奏、停顿
                   CFM(S2M) 学「像谁」  —— 音色、音质、频谱细节
                        ↓
L2  DPO 偏好对齐    同一句话合成多个候选 → 打分 → 好的当 chosen、差的当 rejected
                        ↓
L3  评测与落地      A/B 自动对比（WER / 声纹相似 / reward）→ 调强度 → 合并部署
```

对应到界面就是四个页面的接力：**🗂 数据集 → 🎓 训练 → ⚖️ 对齐 → 🏁 评测/部署**。

### 8.2 第一步：构建数据集（数据集页）

1. **创建数据集** —— 命名并创建（落盘在 `datasets/<名称>/`）；
2. **导入音频** —— 支持从任意路径导入，文件会**拷贝**进数据集目录，训练几小时也不怕源文件被挪；
3. **补文本** —— 每条音频补上对应文本（特征提取的硬前提：没有文本就没有 text_tokens / mu）；
4. **音频体检** —— 每条音频自动体检，全部 ready 才建议进入下一步；
5. **train / val 划分** —— 按比例划分，同 seed 结果可复现。

数据来源建议：本人录音、引擎批量合成（「批量」页正好用上）、或两者混合。

### 8.3 第二步：特征预提取（数据集页）

点击「特征提取」，后台线程自动完成 w2v-BERT / campplus / codec / mel / mu 的离线提取与缓存：

- 训练时**完全不加载**这些大模型 —— 这是 8 GB 显卡训得动 813 M 底座的关键；
- 支持进度显示、中断、**断点续提**；
- 提取需要引擎里的前向模块，引擎未加载时会自动加载。

### 8.4 第三步：LoRA 训练（训练页）

**先想清楚练哪个目标**——两个目标决定的东西不同：

| | GPT (T2S) | CFM (S2M) |
|---|---|---|
| 决定 | 「怎么说」语气、韵律 | 「像谁」音色、音质 |
| 参数量 | 813 M | 98 M |
| 前向 | teacher-forcing 交叉熵 | flow matching（L1 速度场） |
| 只练它的后果 | 说话像但音质发飘 | 音色对但语气平 |

**想真的像本人，两个都得练。**

操作步骤：

1. 「训练目标」选择 GPT / CFM / DPO 三者之一；
2. 选择数据集、加载预设（内置多档超参预设）、按需微调；
3. 点击「开始训练」—— **训练前必须卸载引擎**（8 GB 卡上二者显存不能并存，后台任务管理器会在门口自动拦截并提示）；
4. 训练过程中：预检报告 → 实时进度 → val 曲线与早停 → checkpoint 自动进保险库（只留 top-K，原子写入）；
5. 结束后训练记录与 adapter 自动同步到 `training_runs/`，评测页可直接选用。

### 8.5 第四步：DPO 偏好对齐（对齐页 → 训练页）

1. 「⚖️ 对齐」页选择数据集，配置候选数与合成参数；
2. 引擎用**当前策略**（可先在评测页挂载某个 adapter）逐条合成多个候选；
3. 自动打分（whisper WER + campplus 声纹相似），最优 / 最差成对，**margin 不足的对直接丢弃**——学噪声不如不学；
4. 偏好对写入数据集的 `pairs.jsonl`；
5. 回「🎓 训练」页选 DPO 目标，用偏好对训练。

### 8.6 第五步：A/B 评测（评测/部署页）

1. 选择 A / B 两个配置（如：底座 vs 训练后的 adapter）；
2. 同文本、**同种子**各合成一遍（保证公平对比）；
3. 自动产出 WER / 声纹相似 / reward 三指标胜负表 + 逐条试听文件 + `report.json` / `report.md`；
4. 「像不像」用数据说话，不靠感觉。

### 8.7 第六步：部署（评测/部署页）

- **挂载 + 强度旋钮**：把某个 run 的 adapter 挂上引擎，强度 0~1.5 连续调节（0 = 纯底座，0.6~0.8 是常见折中），**不用重训**即可在「像」和「稳」之间找平衡，试听实时生效；
- **合并**：把 adapter 的 ΔW 烘进一份独立的 `gpt.pth` / `s2mel.pth`，可脱离训练目录使用；合并前自动备份，合并产物做 missing / unexpected 键校验。**合并不可逆，先评测再合并**。

### 8.8 泛化保护（九道防线）

微调最怕的不是训不动，而是训得动却把底座带坏了（读错字、语气发飘）。每道训练流程都内置防线：

| # | 防线 | 做什么 |
|---|---|---|
| 1 | 底座只读快照 | 训练前后 SHA-256 / size+mtime 校验，底座被动过一个字节就报警 |
| 2 | 参数预检 | 60+ 项交叉检查（显存 / 数据量 / 超参匹配度），开工前全部摆出来 |
| 3 | 注入面收敛 | 从真实模型扫描可用层而非硬编码，死模块自动排除 |
| 4 | 权重漂移体检 | `‖ΔW‖/‖W‖` 全局 + 逐层统计，分五档给建议 |
| 5 | 早停 | 盯 val 曲线，连续不改善即停——升上去的部分全是过拟合 |
| 6 | checkpoint 保险库 | 只留 top-K，原子写入，可回滚；**永不写 `checkpoints/`** |
| 7 | 回放抗遗忘 | 混入底座蒸馏样本，钉住通用能力 |
| 8 | adapter 强度旋钮 | 推理期 0~1.5 连续调节，不重训就能折中 |
| 9 | 一键回滚 | 保险库里任意档位可激活，未改善的评估不占名额 |

### 8.9 训练相关的工程细节

这些是踩过坑之后固化下来的实现，详细记录见 [`TODO.md`](TODO.md)：

- **训练前向只有一份**（`webui_app/training/forward.py`）：绕开了上游代码里三个会让训练**静默失效**的陷阱（漏 `lang_embedding`、bf16 崩溃、`mask_content` 把条件全清零）；
- **val 必须确定性**：固定噪声 + 固定配对，否则 val 曲线的抖动比训练改善还大；
- **续训必须恢复 adapter 权重**：只恢复优化器状态会让续训从纯底座重来且不报错（已修 + 回归钉）；
- **显存预检硬拦**：Windows WDDM 下显存溢出不报 OOM 而是静默降速 20~30 倍，所以训练前强制预检。

---

## 9. 测试与验证

所有功能都有**可复现的回归测试**，不是「跑通了就完事」。每个探针自报通过 / 失败项数，失败时以非 0 退出码结束；下表数字为最近一次实跑结果，合计 **730 项断言全部通过**。

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
| `tools/build_check.py` | 11 页构建校验 | ✅ |
| `tools/ui_output_check.py` | 事件回调返回值 vs 输出组件（源码级静态检查） | ✅ |
| `tools/stage2_acceptance.py` | **真机完整闭环**（见下） | **33 / 33** |

### 真机端到端验收

在 **RTX 4060 Laptop 8 GB** 上跑通完整闭环，总耗时 4.0 分钟：

```
引擎合成 28 句建数据集 → 特征提取 → 划分 21/7
  → GPT LoRA 真训练（峰值显存 1.87 GB，权重漂移 0.000423，底座逐字节未变）
  → CFM LoRA 真训练（漂移 0.00319）
  → DPO 偏好对真机构造（8 候选 → 1 对成对，margin 不足的 3 对正确丢弃）
  → A/B 评测（真合成 + whisper 打分 + 试听文件）
  → 强度旋钮 48 层 @ 0.7 生效
  → 合并 GPT/CFM 权重，产物校验 missing/unexpected = 0
```

训练效果示例：GPT val loss 5.706 → 1.723（降 69.8%）；CFM val loss 1.338 → 1.164（降 13.0%）；DPO loss 0.693 → 0.568、acc → 1.0。

报告存档在 [`docs/verification/`](docs/verification)：

- [阶段 2 验收报告](docs/verification/stage2_report.md) · [GPT 训练报告](docs/verification/gpt_report.md)
- [CFM 训练报告](docs/verification/cfm_report.md) · [A/B 评测报告](docs/verification/ab_report.md)

开发进度与踩坑记录见 [`TODO.md`](TODO.md)。

---

## 10. 常见问题（FAQ）

**Q1：启动报 `No module named 'gradio'`？**
之前跑过普通 `uv sync` 把 webui 依赖清掉了。修复：`uv sync --extra webui`。

**Q2：为什么不要用 `uv sync --all-extras`？**
它会额外拉 deepspeed / flash-attn / torch_compile，需要 CUDA 工具链，且本项目不依赖。

**Q3：点「开始训练」被拦下，提示要先卸载引擎？**
8 GB 显存上引擎推理（4.9~5.7 GB）与训练不能并存。先在「系统」页（或状态条）卸载引擎再训练；反之，特征提取 / 偏好对构造 / A/B 评测需要引擎加载。

**Q4：Windows 下训练突然变得极慢？**
WDDM 显存溢出不报 OOM，而是静默降速 20~30 倍。训练预检会硬拦显存不足；若绕过了预检，请减小 batch / 序列长度。

**Q5：合成断句奇怪 / 长文本出问题？**
单条音频理论上限 72.6 秒（1815 语义 token ÷ 25 Hz）；超长文本请开启自动分句或用批量页分段。

**Q6：参考音频明明 30 秒，为什么只用了前 15 秒？**
官方推理链路固定取参考音频**前 15 秒**。请在音频工作台做好切片，把最佳片段放到开头。

**Q7：日语 / 西语文本归一化提示跳过？**
JA / ES 归一化需要 `nemo-text-processing`（依赖 pynini，Windows 无官方 wheel）。不装也能用，日志会明确提示已跳过归一化。

**Q8：下载模型很慢 / 失败？**
下载器默认 `hf-mirror.com` 镜像优先并自动三级回退；支持断点续传，重跑即续传。也可在「模型」页查看缺哪些文件。

---

## 11. 已知限制

- **8 GB 显存必须错峰**：引擎推理与训练不能同时进行（控制台会自动拦截提示）；
- **单条音频理论上限 72.6 秒**，参考音频推理时取前 15 秒；
- **JA / ES 文本归一化**在 Windows 上依赖缺失时自动跳过（见 FAQ Q7）；
- 训练体系的界面与流程按**单机单卡**设计，未做多卡 / 多机。

---

## 12. 项目结构

```
webui.py              # 官方入口，原样保留（可对照使用）
webui_pro.py          # 本项目入口（11 页控制台）
start.bat             # Windows 一键启动（环境自检 + 补装 + 启动）
webui_app/            # 控制台本体
├── app.py            #   装配与启动
├── params*.py        #   参数注册表（控件提示与手册共用）
├── theme.py          #   主题与通用组件
├── services/         #   引擎生命周期 / 推理 / 音频工作台 / 音色库 / 纠音 / 监控
├── tabs/             #   11 个功能页（每页一个文件）
└── training/         #   训练体系
    ├── dataset.py    #     数据集与元数据
    ├── features.py   #     离线特征预提取
    ├── forward.py    #     唯一的训练前向（绕开上游陷阱）
    ├── trainer_base.py / gpt_lora.py / cfm_lora.py
    ├── guard.py      #     泛化保护九道防线
    ├── replay.py     #     底座蒸馏回放（抗遗忘）
    ├── reward.py / dpo.py / evaluate.py
    ├── merge.py      #     合并 / 挂载 / 强度旋钮
    └── runs.py / runner.py
indextts/             # 上游推理引擎（未改动）
tools/                # 回归探针 + 静态校验 + 模型下载器 + 截图工具
docs/verification/    # 真机验收报告存档
datasets/  voice_bank/  training_runs/  outputs/   # 运行期数据目录
```

---

## 13. 许可证

**这是一个衍生作品，两层许可同时生效。**

| 适用范围 | 许可证 |
|---|---|
| **模型权重**与**上游代码**（`indextts/`、`webui.py`） | [bilibili 模型使用许可协议](LICENSE) · [中文](LICENSE_ZH.txt) —— **强制、不可替换**，约束一切衍生品 |
| [SrQingChen](https://github.com/SrQingChen) 的**原创新增文件**（清单见 [NOTICE](NOTICE) §3） | [GNU GPL v3](LICENSE-ADDITIONS.txt) |

简单说：

- 可以自由使用、研究、修改、再分发，包括做优化和二次开发；
- 如果分发修改版，必须沿用同样条款：新增部分保持 GPL v3，模型部分遵守 bilibili 协议（该协议要求把条款继续传递给你的下游用户）；
- 请保留署名：保留 [NOTICE](NOTICE) 与版权声明，同时标注原始权利人（bilibili · IndexTTS2）与修改作者（SrQingChen）；继续开发时请把自己的贡献加进 `NOTICE`；
- 两层条款冲突时，以 bilibili 协议为准，GPL v3 不延伸至模型。

> 上游协议 §4.1(a) 要求本分发声明：*该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、不担保、不承担责任。* 完整归属链与修改清单见 [NOTICE](NOTICE)。本节不构成法律意见。

---

## 14. 致谢

本项目只是外壳与训练工具，真正了不起的是上游的模型工作：

- **[IndexTTS2](https://github.com/index-tts/index-tts)** —— bilibili IndexTTS Team（模型权重、推理引擎、`webui.py`；上游原版 README 存档见 [`docs/README_UPSTREAM.md`](docs/README_UPSTREAM.md)）
- [tortoise-tts](https://github.com/neonbjb/tortoise-tts) · [XTTSv2](https://github.com/coqui-ai/TTS) · [BigVGAN](https://github.com/NVIDIA/BigVGAN) · [wenet](https://github.com/wenet-e2e/wenet) · [icefall](https://github.com/k2-fsa/icefall) · [maskgct](https://github.com/open-mmlab/Amphion) · [seed-vc](https://github.com/Plachtaa/seed-vc)
- 本项目训练体系用到的开源组件：**PEFT**（LoRA）、**OpenAI Whisper**（WER 评测）、**CAMPPlus**（声纹相似度）、**Gradio**（界面）

---

## 15. 引用

如果你使用了本项目或在其基础上工作，请引用上游模型与本仓库：

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

---

<div align="center">
<sub>IndexTTS-2.5 Pro · 由 <a href="https://github.com/SrQingChen">SrQingChen</a> 维护 ·
官方 <code>webui.py</code> 保持不变，本界面为独立实现 · 参数行为说明均经源码核实</sub>
</div>
