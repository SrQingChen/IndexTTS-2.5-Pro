# 高音 / 戏腔 / 音色外推 · 独立交接文档

**日期：** 2026-10-17
**用途：** 本文件是「AI 翻唱高音嘶哑、音色外推、IndexTTS 原生改造」这一专题的**自包含交接包**。新窗口只需要读这一份，就能接着细化/推进，不依赖之前的对话上下文。
**项目：** `E:\AI\indextts2.5\index-tts-2.5.0`（IndexTTS-2.5 Pro，分支 `main`）
**约束：** 用户希望**只用本模型**（不引入 RVC/Seed-VC/HQ-SVC 等外部 SVC 权重）；可以接受很大的工作量，包括改模型训练。

---

## 0. 一页速览（先读这个）

### 0.1 当前结论

1. **高音嘶哑不是“音高上不去”**：源曲 F0 中位 390.8Hz、p99 589Hz、最高 930Hz；模型转换输出（未移植）p99 564Hz、最高 1054Hz。音高能到，**歌声音色出不来**。
2. **核心短板**：IndexTTS 的 CFM/BigVGAN 是语音训练、22.05kHz、**没有显式 F0 条件**；内容特征（w2v-BERT + RVQ）基本不带音高。歌声高音需要高音区的声道/声门源，模型没学过。
3. **最近一次跑通**：`variant=B`（实验档）+ 超分开，最终 SS 0.459；A/B 实测 **variant A 全面优于 B**；超分让 SS 再降。→ 先做 **A（安全预设）**。
4. **B（预降调/整首降调）已实现但默认关闭**：它只降低高音外推压力，**不改善音色**；实测“预降 5 再升回”SS 0.639→0.664→0.637，净收益≈0。
5. **只给说话音色推断高音**：有专门技术，核心不是移调，而是**音高条件化的音色生成**（YingMusic-SVC 的 F0-aware timbre adaptor；Everyone-Can-Sing 的 speech-reference zero-shot SVS/SVC；SPA-SVC 的 pitch augmentation）。IndexTTS 原生可实现，但需要训练（E1-R/E2）。
6. **“推断”是先验猜测，不是本人真实高音**：有 10–30min 角色歌声会更像；没有就是“population prior + 角色身份锚点”。戏腔必须有戏腔数据。

### 0.2 现在能做什么

- **A（已实现）**：翻唱页「🎵 歌声安全预设」→ variant A、steps 25、cfg 0.7、超分关、F0 auto、音域适配 off。**基本纯增益**（关超分牺牲 48k，但歌声 SS 更好）。
- **B（已实现，默认 off）**：音域适配 off/auto/manual；两种输出：
  - `整首降调（推荐音质）`：人声+伴奏一起降 K，不升回；
  - `仅人声升回原调`：伴奏原调，人声用相位声码器升回（有伪影，建议 K≤4）。
- **B2（未实现，可选）**：用输出反馈自动搜索“最高但不破音”的 K；见 §7。
- **E1-R（未实现，推荐的原生训练线）**：F0 条件 + 音高条件化音色残差 + 歌声数据 + 音高感知声码器；见 §6。
- **E2（未实现）**：44.1kHz 原生 SVC 重建；见 §6.6。

### 0.3 关键数字（本次实测）

| 指标 | 值 |
|---|---|
| 源干声 F0 中位 / p99 / 最高 | 390.8 / 589.0 / 930.2 Hz |
| 03 转换（B）SS / p99 | 0.548 / 591.5 Hz |
| 04 F0 移植 SS / 移植后 r / 50 音分内 | 0.510 / 0.865 / 91.7% |
| 最终 48k（含超分）SS | **0.459** |
| A vs B（同一 30s 高音段，原始 SS） | **0.618 vs 0.576** |
| A vs B（F0 移植后 SS） | **0.543 vs 0.508** |
| 预降 5 半音→转换→升回 SS | 0.639 → 0.664 → **0.637** |
| 低音段 SS（参考说话） | 0.354 |
| 参考上移 +8st 后：对原参考 / 对移调参考 SS | 0.649→0.483 / 0.654 |
| B 链路（30s，−5，整首降调） | F0 后 r=0.850，50 音分内 90.9% |
| B 链路（20s，−5，仅人声升回） | 最终中位 400Hz、p99 596Hz |

---

## 1. 项目当前状态（代码 / 已实现）

### 1.1 最近提交

```
ba5e53e Research: pitch/timbre extrapolation from a speech reference (E1-R)
7d970ad Add A singing-safe preset and B key-range adaptation (default off)
d086fa0 Research: high-pitch singing/戏腔 forensics and plans
f8e9c42 Fix F0 transplant wiring: direction-aware envelopes, no window averaging, key-preserving transpose
94a3737 Fix cover song-state abort: resolve relative separator outputs + stem matching
da07479 Fix 8GB OOM in cover VC: bound encoder/CFM windows, offload GPT, lease tokens
```

### 1.2 新增/关键模块

| 文件 | 作用 |
|---|---|
| `webui_app/services/svc_core.py` | 原生 VC 核心：源 16k→w2v-BERT→RVQ→length_regulator→CFM→BigVGAN；离线分块 + 流式 `StreamConverter`；变体 A/B；F0 移植前的音频分块 |
| `webui_app/services/svc_f0.py` | F0 管线：harvest 全曲/分窗、八度映射、轮廓相关、WORLD 移植（方向感知）、局部修复、音域适配 `plan_key_shift`、探针指标 |
| `webui_app/services/cover_pipeline.py` | 翻唱编排：输入路由（原歌曲/分离/干净）→预处理→音域适配→VC→F0→后处理→超分→BGM 混合→报告 |
| `webui_app/services/separation.py` | audio-separator 封装 + 镜像自动下载（ghproxy）+ 模型部署 |
| `webui_app/services/vc_realtime.py` | 实时变声（sounddevice，分块，3 档） |
| `webui_app/services/svc_state.py` | 翻唱/变声参数记忆（独立 json） |
| `webui_app/services/engine.py` | 引擎生命周期；**VC 期间 GPT 移到 CPU**；租约 token 化；`load/unload` 拦截 |
| `webui_app/params_cover.py` | 翻唱/变声参数注册（含 `cover_key_*`） |
| `webui_app/tabs/cover_tab.py` | 翻唱页；歌声安全预设、音域适配 UI、日志/报告 |
| `webui_app/tabs/vc_tab.py` | 变声页（实时/文件） |
| `tools/fetch_separator_models.py` | 分离模型镜像下载 CLI |

### 1.3 已实现的工程保护（不要拆掉）

- **分块**：8GB 卡自动压到 30s；块间 `gc+empty_cache`；OOM 自动减半重试一次（30→15s）。
- **编码器分窗**：w2v-BERT 按 24s 窗 + 0.5s 重叠提取，避免 60s 输入的 O(T²) OOM。
- **GPT 让出**：VC 会话期间 `gpt.to("cpu")`，8GB 约释放 1.5–1.7GB；结束/异常恢复。
- **租约**：`begin_svc(label)->token`；`unload(bypass_lease=自己的 token)` 只绕自己；VC 期间 `engine.infer`/`mount_qwen` 拒绝。
- **F0 修复**：
  - WORLD 包络**方向感知**：上行用合成-F0 分析，下行用输出自身 F0 分析；4s 探针不收敛会换策略/回退；
  - `harvest_f0` **不再对分窗结果做线性平均**；≤12min 整段 harvest，更长只取窗中心帧；
  - 自动移调：源/参考中位差 ≤8 半音返回 0（保调），跨音区才 ±12；
  - 报告 `f0_r` 为**移植后探针**，另有 `f0_r_pre`/`within_50c`/局部八度错误。
- **分离链路**：audio-separator 0.47 已装；5 个模型已下载到 `checkpoints/separators/`；缺模型自动经镜像下载；`_stem_path` 解析相对路径 + 剥模型名匹配。
- **参考音频**：翻唱/变声三处与合成页一致，音色库选择自动填入 Audio 组件；可直接上传覆盖。
- **A 安全预设**：一键设置 A/25/0.7/超分关/F0 auto/音域 off；variant B 与长超分有警告。

### 1.4 当前默认参数

| 参数 | 默认 |
|---|---|
| variant | A 量化重构 |
| steps / cfg | 25 / 0.7 |
| 分块 | 30s（8GB 自动） |
| F0 | auto；自动移调 0（±8 半音内）；±12 跨音区 |
| 超分 | 关 |
| 音域适配 | off |
| 参考 | 15s；音色库/上传 |
| 后处理 | 高通 70Hz、间隙 de-esser、浊音 HF 限幅；频谱对齐/气声软化默认关 |

### 1.5 运行 / 测试

```bash
# WebUI
start.bat
# 或
.venv\Scripts\python.exe webui_pro.py --lazy

# 测试（当前 58 项通过）
.venv\Scripts\python.exe -m pytest tests/test_svc.py tests/test_separation_fetch.py tests/test_webui_syntax.py -q

# 分离模型状态/预下载
.venv\Scripts\python.exe tools/fetch_separator_models.py --list
.venv\Scripts\python.exe tools/fetch_separator_models.py --all
```

产物：`outputs/cover/<时间戳>/`（02 干声、03 转换、04 F0、05 成品、06 混音、`report.json`）；日志：`outputs/logs/indextts.log` / `error.log`。

---

## 2. 最近一次跑通取证（2026-10-17 12:24）

### 2.1 配置

- 输入：原歌曲（分离+去混响）
- **variant B（实验）**，steps 30，cfg 0.7
- 分块请求 60s→运行 30s（8GB cap）
- 参考：达妮娅说话 15s（中位 257Hz）
- **超分开**（AudioSR→48k，469.5s）
- 总耗时 899.8s

### 2.2 分阶段（真实文件）

| 文件 | SS(对参考) | F0 中位 | p99 | 备注 |
|---|---|---|---|---|
| 02 分离干声 | 0.319 | 390.8 | 589 | 源歌手 |
| 03 转换（B） | 0.548 | 372.2 | 591 | 音高大致跟，音色差 |
| 04 F0 移植 | 0.510 | 387.6 | 732 | 音准修好，音色再降；少量局部八度 |
| 05 后处理 22k | 0.494 | 387.5 | 690 | de-esser/限幅 |
| 05 超分 48k | **0.459** | 388.0 | 742 | 超分再降 SS |
| 06 混音 | 0.335 | 337.1 | 878 | 混入伴奏后测量 |

### 2.3 对照实验

- **A vs B（30s 高音段，25 步，cfg 0.7）**
  - 原始：A SS 0.618 / 12.2% within50；B 0.576 / 10.2%
  - F0 后：A 0.543 / 82.8%；B 0.508 / 82.6%
  - 高音段自相关峰：A_f0 0.744，B_f0 0.853（B 更周期，但音色更差）
- **CFG/步数**：cfg 1.0 比 0.7 差（0.60 vs 0.637）；steps 40/cfg1.0 0.657（收益小）。
- **预降调**：高音段原样 0.639；预降 5 0.664；升回 0.637；低音段 0.354。
- **参考移调**：参考 +8st 后，输出对原参考 SS 0.483（原 0.649），对移调参考 0.654。→ **不要移调参考**。
- **B 链路**：
  - 30s，−5，整首降调：F0 后 r=0.850，within50 90.9%，混音 44.1k。
  - 20s，−5，仅人声升回：最终中位 400Hz、p99 596Hz（源 576/589），有伪影警告。

### 2.4 结论

- 低 SS + 高音嘶哑的主因是**歌声域音色转换**，不是音高上限。
- B 的预移调**不能修复音色**；只作为音域工具。
- 超分与 variant B 会进一步降低 SS；先关。
- F0 后处理把音准从 12% 修到 83–92%，但高音仍像“被拉上去的说话声”。

---

## 3. 已试过的设置与结论（避免重复踩坑）

| 设置 | 结果 | 建议 |
|---|---|---|
| variant B | SS 比 A 低 0.04；更易嘶哑 | **默认 A**；B 仅实验 |
| 超分 AudioSR | 最终 SS 0.494→0.459；耗时 43–55min/3min 人声 | 歌声默认关 |
| 预降 5 + 升回 | 净 SS≈0；两次移调伪影 | 只做音域工具，不做音色修复 |
| 参考移调 +8 | 身份 SS 0.649→0.483 | 不要移调参考 |
| cfg 1.0 | SS 更低 | 保持 0.7 |
| steps 40 | SS +0.02（0.657） | 可选，收益有限 |
| WORLD 全曲移植 | 修音准但降 SS；局部八度/粗糙 | 保留 auto（0 移调时也可只做局部修复） |
| 60s 分块 | 第二块 OOM | 30s cap + 块间回收 |
| 变体 B 连续特征 | 分布外，嘶哑 | 默认 A |

---

## 4. 为什么会这样（架构事实）

- `s2mel.pth` **284 个参数键，F0 相关 0 个**；`f0_condition=false`；`CFM.inference(..., f0)` 恒传 None。
- CFM 98M、BigVGAN 22.05kHz，均在语音数据上训练。
- 内容链路：w2v-BERT（0.58B）→ RVQ 25Hz → decode → length_regulator → CFM；RVQ 后音高信息大幅丢失（隐式旋律 r≈0.8），精确旋律必须外置 F0。
- 参考三路：ref_mel（80 维）、campplus style（192）、prompt_condition（512）；都是说话参考。
- 高音不是“F0 数值”问题，而是**高音区的谱包络/声门源/假声**没有生成；WORLD 只能搬包络，不能造新音色。
- 商业级高音翻唱：按角色训练 RVC（训练集含高音/戏腔）或 SVS（DiffSinger/ACE Studio/Synthesizer V），不是后处理。

---

## 5. 调研：音色/音域外推（核心）

### 5.1 两条路线

1. **信号级外推**：保共振峰移调（WORLD/STRAIGHT/PSOLA/相位声码器）。
   - 只搬已有谱包络；不生成新音色；高音仍像“拉上去的说话声”。
   - 我们的参考 +8st 实验证明会丢身份（0.649→0.483）。
2. **学习式外推（音高→音色先验）**：在歌声数据上训练，让模型学“F0 变化时，谱包络/声道/声门源怎么变”；推理时参考音频提供身份，F0 条件选择音高，模型生成“该身份在该音高下最可能”的音色。
   - 这是用户问的“推测高音样子”的正解；商业级高音多来自这条路线（且通常有角色真实高音数据）。

### 5.2 关键论文/机制

| 工作 | 关键点 | 对本项目的用法 |
|---|---|---|
| **YingMusic-SVC**（arXiv 2512.04793） | F0-aware fine-grained timbre adaptor：`hτ(t)=eτ(global)+ατ·MLP([hf(t);eτ])`；静态歌手嵌入变成随音高变化的细粒度音色图；energy-balanced flow matching 加权高频；RVC timbre shifter 120 歌手；Flow-GRPO。作者称可让高音更亮/头声、低音更胸声，并“避免极端音高的音色失真”（定性，无单独高音指标）。 | **E1-R 的核心公式**：给 style/prompt 加 F0 条件残差；不要只把 F0 塞进内容 |
| **Everyone-Can-Sing**（ICASSP 2025，arXiv 2501.13870；demo everyone-can-sing.github.io） | 用**说话样本**做零样本 SVS/SVC；混合语音+歌声训练；推理期有 Adjust Pitch 消融处理语音参考与歌曲音域不匹配；diffusion generator；demo 含歌剧风格。 | 证明“说话参考→唱歌”可行；给 E1-R 的数据混合/音高调整思路 |
| **SPSinger**（ICASSP 2025，IEEE 10888907） | 5s 短参考；LPAM（Transformer-decoder）从全局音色+曲谱隐序列**预测需要的移调量**，解决“目标歌手与曲谱音域不匹配”；短参考的局部音色预测。 | 可作为 **B2 自动找调**的学习式版本；也可用于“参考音色→高音”的条件 |
| **SPA-SVC**（Interspeech 2024，ISCA bai24） | cycle pitch-shifting 训练 + SSIM loss；专治“源音域远高于目标、输出嘶哑”的跨域 SVC；不需要额外数据/参数。 | E1-R 数据增广/损失；有限角色数据时的关键技巧 |
| **Perrotin et al.**（Interspeech 2021，HAL hal-03338483） | 神经声码器**极端音高外推**评测；WaveGlow 外推好，LPCNet/Parallel WaveGAN 会塌回训练集中位 F0。 | E1-R/E2 必须做“音域受控”声码器验证；BigVGAN 要测 |
| **SpSiVC**（2025，J-STAGE e24.47） | 自适应 F0 损失：歌声按八度约束、语音按说话人均值；解决单一模型语音/歌声切换的八度偏移。 | 自动移调/八度约束的参考 |
| **Interspeech 2020 语音/歌声条件信号**（Thu-3-4-8） | 只在语音上训练的网络，在显式 pitch/loudness/phoneme 条件下可泛化到歌声，且音高可在解码时独立操纵。 | “语音-only + 显式条件”可外推到歌声的证据 |
| **AlignSTS**（arXiv 2305.04476） | 语音转歌声；节奏适配器 + 跨模态对齐；不强制细粒度目标 F0。 | 如果要从说话直接“唱”，这是路线参考 |
| **Speech-to-Singing encoder-decoder**（arXiv 2002.06595；code ericwudayi/speech2singing） | F0 来自哼唱/参考歌声；paired data；mel 时间拉伸。 | 参考旋律/时间对齐的工程参考 |
| **Seed-VC**（arXiv 2411.09943；GitHub Plachtaa/seed-vc） | 44.1k SVC，F0 条件，Whisper 内容 + CAMPPlus 风格 + BigVGAN；30–50 步；BigVGAN 更新改善高音歌声。 | 外部基线/对照；若放开“只用本模型”，这是最快路线（GPL-3.0） |
| **HQ-SVC**（AAAI 2026，GitHub ShawnPi233/HQ-SVC） | FACodec + EVA（F0/能量/相位 FiLM）+ DDSP→Mel→扩散 + NSF-HiFiGAN 44.1k；<80h 训练；训练码未放。 | E2 的架构参考；NSF 声码器 |
| **DiffSinger / KunquSinger** | 曲谱 SVS；44.1k；HiFi-GAN 对假声/高音吃力→BigVGAN；昆曲 7.68h 数据。 | 戏腔路线；SVS 重唱，不适合“保表演”的翻唱 |
| **Diff-Pitcher / StylePitcher** | 扩散音准修正/神经移调，保音色；+12 仍有音准漂移。 | 后移调兜底；不能当高音生成主路线 |

> 注：部分数字（如 Everyone-Can-Sing 的“一个八度”调整、SPA-SVC 的 6–18 半音）来自搜索摘要，未在论文主页逐字确认；新窗口引用前请回源 PDF 核实。

### 5.3 “模型怎么会推断高音音色”

- 训练时看到大量歌手的 **F0→谱包络** 关系（高音时声道缩短、头声/咽音、高频能量变化）。
- 推理时参考音频给 **身份**（音色嵌入），F0 条件给 **目标音高**；模型用学到的关系生成“这个身份在这个音高下最可能”的包络/声门源。
- 有角色高音样本 → 个性化先验；没有 → population prior + 身份锚点。
- 这不是“真实高音”，是**最可能的猜测**；越极端的音高/戏腔，先验越不可靠。

---

## 6. 方案（重点：E1-R / E2）

### 6.1 A（已实现）

- 一键预设：variant A、steps 25、cfg 0.7、超分关、F0 auto、音域 off。
- 预期：SS 0.54–0.62；排除自伤；高音仍不自然。
- 这是当前**唯一基本纯增益**的动作（关超分牺牲 48k，但歌声 SS 更好）。

### 6.2 B（已实现，默认 off）

- `cover_key_mode` = off/auto/manual；`cover_key_semitones` −12..0；`cover_key_output` = lower_key/original_key；`cover_key_ceiling` 默认 600Hz。
- 预降：WORLD（保共振峰），失败回退 librosa；F0 目标基于降调后源。
- lower_key：伴奏同步降（librosa），不升回；original_key：人声升回（librosa），>4 半音警告。
- **不是纯增益**：改调/二次移调伪影；实测净 SS≈0。默认关闭。
- 代码：`cover_pipeline.py` 的 `shift_vocal_key`/`shift_audio_librosa`/`plan_key_shift`；UI 在 `cover_tab.py`。

### 6.3 B2（未实现，可选）

**目标**：自动找“最高但不破音”的整首 K，避免降过头/降不够。

1. 选 10–15s 高音最密集探针（按目标浊音密度）。
2. K=0,−1,…−8 逐个短转换（10–15 步）。
3. 破音分数：
   - F0 50 音分内比例、局部八度错误率（对齐后 |cents|>600）；
   - 高音帧 HNR/自相关峰、谱平坦度；
   - 削波、能量突降、F0 跳变 >2 半音；
   - SS（对角色参考）——防止降过头。
4. 选**第一个通过阈值的 K**（降得最少）；K=0 通过则不降；−8 仍不过则报告“无解”。
5. 长音加权：只统计 ≥200ms 的高音，忽略一闪而过的尖叫。
6. 报告搜索曲线；整首用该 K。
- 成本：5–8 次短探针，约 1–3 分钟。
- 注意：破音可能不是音高，而是音色；B2 只能找“指标上最正常”的调。

### 6.4 E1-R（推荐的原生训练线，4–8 周）

**目标**：让模型从“说话音色 + 目标 F0”生成“该角色在高音下最可能的音色”，不再靠 WORLD 后处理。

**架构改动**
1. **F0 条件**：log-F0 → 256/512 维嵌入；FiLM 注入 `cond`/DiT 时间嵌入；`flow_matching.py` 的 `f0` 形参要真正被 estimator 消费。
2. **音高条件化音色残差**（核心，参考 YingMusic）：
   - 现：`style` 是全局 192 维。
   - 改：`style_t = style + α·MLP([F0_emb_t; style])`，按帧/按块；把 `style_t` 注入 CFM 的 cross-attention/FiLM。
   - 只加 F0 到内容、不加到音色，仍会“音高对了但音色是说话色”。
3. **内容**：w2v-BERT 语音训练；可先冻结，用 LoRA/adapter 适配歌声；不建议直接换 Whisper（除非允许外部权重）。
4. **声码器**：BigVGAN 22k 在歌声上微调（优先）；或 E2 换 44.1k NSF-HiFiGAN；用 Perrotin 式音域受控测试验证极端 F0。
5. **训练目标**：flow matching + F0 辅助/条件 + SSIM（SPA-SVC）+ 音高条件音色对比损失（同歌手不同音高、同音高不同歌手）。

**数据**
- 1–5h 公开歌声（E1-R 起步）：Opencpop ~5h、M4Singer ~30h、GTSinger 80+h、OpenSinger；戏腔 KunquDuoVox 7.68h（如要做戏腔）。
- 角色说话 10–60s（参考音色）；可选角色歌声 10–30min（强烈建议）。
- 数据增广：cycle pitch-shifting（SPA-SVC 式）；对角色说话做保共振峰移调生成伪高音样本（辅助，不是真高音）。
- 数据量换算：1h 22k mono 16-bit ≈158MB；44.1k ≈318MB；10h 22k ≈1.6GB；50h 44.1k ≈16GB。

**算力**
- E1-R：8GB 可做 LoRA + gradient checkpointing（22k，10–20s 片段，batch 1–2）；全参/44.1k 建议 24GB 云。
- E2：24GB+；训练 1–3 个月。

**推理**
- 目标 F0 + 角色说话参考 → CFM（带 F0 条件/音色残差）→ mel → 声码器；**不再 WORLD 移植**。
- 保留 A/B 作为兜底；E1-R 未过闸门前不替换主链。

**评估/闸门**
- G1：22k、1–5h 公开歌声 + 角色说话；SS≥0.70、F0 50 音分内≥85%、高音段 HNR 不低于基线；否则停。
- G2：44.1k/新声码器；与 Seed-VC 同参考 A/B，SS≥0.80、高音 MOS 不劣；否则回 22k/补数据。
- G3：角色歌声 10–30min + 戏腔数据；角色 SS≥0.85、戏腔人工 A/B 通过。
- 指标：SS、F0 50 音分内/p90/局部八度错误、HNR/CPP、高音子集（F0>500/600）、破音率、MOS/A/B。

### 6.5 E2（44.1k 原生 SVC 重建，2–3 个月）

- 按 HQ-SVC 配方：FACodec/内容解耦 + EVA（F0/能量/相位 FiLM）+ DDSP→Mel→扩散 + NSF-HiFiGAN 44.1k；或保留 CFM，改 mel 配置到 44.1k 重训 + 44.1k 声码器。
- 数据 10–50h 公开歌声 + 角色；24GB 云。
- 预期：接近 Seed-VC/HQ-SVC；“超过”取决于数据/训练/DPO；戏腔需戏腔数据。

### 6.6 C/D（如果放开“只用本模型”）

- C：Seed-VC 44.1k SVC（1–2 周，GPL-3.0，零样本，高音改善明显）。
- D：RVC 角色歌声模型（2–4 周，MIT，10–30min 角色歌声，商用级；戏腔需戏腔数据）。
- 仅作对照；当前约束下不实施。

### 6.7 F（SVS，数月）

- DiffSinger/OpenUTAU + 角色歌声库 + 戏腔数据；高音/戏腔上限最高，但重唱、丢源表演。

---

## 7. 代码触点（E1-R / E2）

| 位置 | 改什么 |
|---|---|
| `indextts/infer_v2_5.py` | VC 路径/参考条件；`length_regulator`、`cfm.inference`、`bigvgan` 调用 |
| `indextts/s2mel/modules/flow_matching.py` | `inference(..., f0, ...)` 形参已有；让 estimator 消费 f0；加 F0 条件 |
| `indextts/s2mel/modules/length_regulator.py` | `f0_condition` 分支（当前 false/死代码） |
| `indextts/s2mel/modules/diffusion_transformer.py` | DiT 无 f0；加 FiLM/条件通道 |
| `checkpoints/config.yaml` | `s2mel.f0_condition`、mel 参数（44.1k 时 n_fft/hop/sr） |
| `webui_app/training/features.py` | 特征池；加 F0 特征缓存 |
| `webui_app/training/cfm_lora.py` / `trainer_base.py` | LoRA/训练循环；加 F0 条件/损失 |
| `webui_app/services/svc_core.py` | 推理时传 F0/音色残差；去掉 WORLD 依赖（可选） |
| `webui_app/services/svc_f0.py` | F0 提取/RMVPE/FCPE、音域规划、探针指标 |
| `webui_app/services/cover_pipeline.py` | 编排；E1-R 开关；报告 |
| `checkpoints/hf_cache/bigvgan` | 22k 声码器；微调或替换 |
| `webui_app/services/engine.py` | 显存调度/租约/GPT offload |

**注意**：项目有“上游引擎零改动”的偏好；E1-R 若要改 `indextts/`，建议用**旁路/开关**，保留原 TTS 路径；更新 NOTICE 记录修改。

---

## 8. 数据、算力、许可证

### 8.1 数据清单

| 数据 | 用途 | 许可/注意 |
|---|---|---|
| Opencpop ~5h（中文流行） | E1-R 起步 | 研究用途；核对官网 |
| M4Singer ~30h（多歌手） | E1-R/E2 | 研究用途 |
| GTSinger 80+h（多语言，含技巧/音域标签） | E1-R/E2 | 核对许可 |
| OpenSinger | 补充 | 核对 |
| KunquDuoVox 7.68h（昆曲） | 戏腔 | 研究用途 |
| 角色说话 10–60s | 参考身份 | 游戏角色声音权利 |
| 角色歌声 10–30min | 个性化高音 | 游戏/声优/歌曲版权 |
| 角色戏腔 1–5h | 戏腔 | 最难获取 |

### 8.2 算力

- 本地 RTX 4060 Laptop 8GB：E1-R LoRA/22k 可试；E2/44.1k 不够，租 24GB 云。
- 现有工程已有 8GB 保护：30s 分块、24s 编码窗、GPT offload、OOM 重试。

### 8.3 许可证（bilibili 模型使用许可协议）

- **允许微调/再训练/LoRA/PEFT/衍生品**（1.5/1.6/2.1）；这是项目已有训练的法律基础。
- 必须保留 `LICENSE`/`LICENSE_ZH.txt`/`NOTICE`；下游同约束（3.4a/b）；发布衍生品加 4.1(a) 免责声明。
- **3.4(c)**：不得用 indextts2 或其衍生品改进**其他** AI 模型，除非是 indextts2 自身/衍生品或**非商业** AI 模型。
  - 微调 IndexTTS 自己：允许。
  - 用 IndexTTS 输出训练 RVC/Seed-VC：非商业可以，商业不允许。
  - 用真实角色歌声训练 RVC：3.4(c) 不直接管辖，但声音/歌曲/数据集权利独立。
- **商业门槛**：中文 2.2 是 MAU >1 亿 **或年收入 >1 亿人民币**；英文写 10 亿，中文管辖（NOTICE），按更严口径。
- 4.1(b)：第三方数据/权重授权自负；4.1(a) 免责声明；5.2 不得用商标（归属描述除外）；6.1 中国法/上海仲裁。
- 修改模型后**更新 NOTICE**（当前写“上游权重未修改”，F0 手术/歌声微调后不再成立）；衍生权重受 bilibili 协议，新增代码 GPL-3.0。
- 角色声音 likeness、歌曲版权、平台翻唱规则是独立风险；个人本地研究 vs 公开发布/商用不同。

---

## 9. 建议的推进顺序

1. **A**：立刻用安全预设跑 30–60s，确认基础质量（0.5 天）。
2. **B 试验**：manual −3~−5，整首降调输出，30s A/B；确认是否减轻高音破音（3–5 天已实现，直接试）。
3. **E1-R G1**（推荐）：F0 条件 + 音高条件化音色残差 + 1–5h 公开歌声 + 角色说话参考；22k LoRA；4–6 周；按 G1 闸门决定是否继续。
4. **E2**：G1 通过后，44.1k + 10–50h + 24GB 云；2–3 个月。
5. **B2**（可选）：如果继续用 B，做输出反馈的自动调搜索；1–2 天。
6. **C/D**（仅当放开“只用本模型”）：Seed-VC 1–2 周 / RVC 2–4 周，作为质量对照。

**决策点**：
- 只要“合理的高音” → E1-R。
- 要“角色真实高音/戏腔” → 必须有角色高音/戏腔数据；否则任何模型都是猜。
- 要“尽快商用级” → C/D（但违反当前“只用本模型”约束）。

---

## 10. 新窗口启动提示（可直接粘贴）

> 请先完整阅读 `docs/handoff-high-pitch-singing-2026-10.md`（本文件），然后：
> 1. 基于 §6.4 的 E1-R 给出**可执行的代码级实施计划**（F0 条件、音高条件化音色残差、LoRA/训练、声码器、评测闸门），标注要改的文件/函数与风险；
> 2. 先设计 G1 最小验证（22k、1–5h 公开歌声、角色说话参考），给出数据准备、训练命令、指标与判定线；
> 3. 不要引入 RVC/Seed-VC/HQ-SVC 权重；如需借鉴架构，只用论文思想并在本项目内实现；
> 4. 遵守 §8.3 许可证：微调允许、保留 NOTICE、3.4(c) 商业限制、第三方数据授权；
> 5. 输出：分阶段计划 + 验收标准 + 预估算力/周期 + 失败回退方案。

---

## 11. 参考链接

**本项目**
- `docs/research-ai-cover-voice-changer-2026-10.md`（前置调研，附录 A 接线）
- `docs/research-high-pitch-singing-2026-10.md`（高音取证 + 方案 A–F + 本专题 §9）
- `docs/dev-plan-ai-cover-vc-2026-10.md`（开发规划/实施修正）
- `outputs/logs/indextts.log`、`outputs/cover/<ts>/report.json`

**SVC / 外推**
- YingMusic-SVC：https://arxiv.org/abs/2512.04793
- Everyone-Can-Sing：https://arxiv.org/abs/2501.13870 ；https://everyone-can-sing.github.io/
- SPSinger：https://ieeexplore.ieee.org/abstract/document/10888907
- SPA-SVC：https://www.isca-archive.org/interspeech_2024/bai24_interspeech.html
- Perrotin (vocoder extrapolation)：https://hal.science/hal-03338483v1/file/Perrotin_IS2021.pdf
- SpSiVC：https://www.jstage.jst.go.jp/article/ast/46/1/46_e24.47/_article
- Interspeech 2020 speech/singing conditioning：http://www.interspeech2020.org/uploadfile/pdf/Thu-3-4-8.pdf
- AlignSTS：https://arxiv.org/abs/2305.04476
- Speech-to-Singing：https://arxiv.org/abs/2002.06595 ；https://github.com/ericwudayi/speech2singing
- Seed-VC：https://github.com/Plachtaa/seed-vc
- HQ-SVC：https://github.com/ShawnPi233/HQ-SVC
- DiffSinger：https://github.com/openvpi/DiffSinger
- KunquSinger：https://hualizhou167.github.io/KunquSinger-demo/
- Diff-Pitcher：https://engineering.jhu.edu/lcap/data/uploads/pdfs/waspaa2023_hai.pdf
- StylePitcher：https://arxiv.org/abs/2510.21685
- WORLD：https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
- RubberBand：https://breakfastquay.com/rubberband/

**RVC / 社区**
- RVC issue #697：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/697
- RVC 高音指南：https://begin0808.github.io/studio0808_video/rvc_guide.html
- AICoverGen：https://github.com/blane187gt/AiCovergen

**许可证**
- `LICENSE`、`LICENSE_ZH.txt`、`NOTICE`、`DISCLAIMER`（仓库根）
- 中文 2.2 商业门槛 1 亿/1 亿；3.4(c) 不得用其产物改进其他 AI 模型（非商业除外）；4.1(b) 第三方数据授权。

> 本文不是法律意见；商用/公开发布前请让法务按中文协议、游戏二创政策、歌曲版权逐项确认。
