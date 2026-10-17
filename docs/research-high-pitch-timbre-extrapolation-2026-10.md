# 高音音色外推：2025–2026 先进方法调研

**日期：** 2026-10-17
**前置：** `docs/handoff-high-pitch-singing-2026-10.md`、`docs/research-high-pitch-singing-2026-10.md`
**问题：** 只给已有人声（角色说话、少量歌声），怎样尽可能自然地外推“高音区的音色行为”，用于翻唱？
**边界：** 调研报告。不引入 RVC / Seed-VC / HQ-SVC 权重，不改代码、不跑模型。外部系统只作为对照与上限参考。

---

## 0. 结论速览（TL;DR）

1. **2025–2026 的前沿已经从“移调”转到“音高条件化的音色生成”**。核心不是把说话声升调，而是让模型学习 `P(音色 | 身份, F0, 音区/技巧, 元音, 源演唱行为)`。
2. **最直接的三条线**：
   - **F0-aware timbre adaptor**（YingMusic-SVC，2025-12）：把静态歌手嵌入变成随 F0 变化的细粒度音色；公式 `hτ(t)=eτ(global)+ατ·MLP([hf(t);eτ])`，α=0.5；44.1 kHz；作者定性称能“避免极端音高的音色失真”，但**没有单独的高音量化指标**。
   - **Pitch/energy-aware zero-shot SVC**（HQ-SVC，AAAI 2026）：EVA 把 speaker + F0 + style + volume + phase 融合成 1024 维条件，FiLM 注入内容；SFP 从 speaker embedding 预测目标 F0 的均值和方差；44.1 kHz NSF-HiFiGAN；<80 h 数据、6 GB 显存可训（官方训练码未放）。
   - **技巧/音区条件化**（SVCC 2025 及其冠军系统 S²Voice / S4）：SVCC 2025 的 style 明确包含 **falsetto、mixed voice、pharyngeal、breathy、glissando、vibrato**；这是目前最接近“高音音色行为”和“戏腔”的任务定义。但最难的 breathy/glissando/vibrato 风格准确率只有 **37.3% / 42.6% / 43.9%**，没有任何系统达到人类自然度。
3. **说话参考零样本 SVS/SVC 已经可行，但高音仍是短板**：Everyone-Can-Sing（5 s 语音参考；把曲谱移进参考的一个八度内；相似度升、质量略降；跨性别质量明显下降）、TCSinger 2、Vevo2、SoulX-Singer、OneVoice。它们解决“没有歌声参考”，不解决“参考里没有高音音色”。
4. **支撑技术不能少**：SPSinger（预测该移多少调）、SPA-SVC（6–18 半音 cycle pitch-shifting + SSIM，专治跨音域嘶哑）、SpSiVC（八度约束）、Prompt-Singer（range–melody 解耦）、R2-SVC（F0 扰动 + NSF）、DDSP/SawSing/NSF-HiFiGAN/BigVGAN v2 44.1 kHz、Perrotin 的声码器音高外推测试、Sundberg 的 formant tuning。
5. **关键判断：高音音色不是单一 F0 的函数**。它同时取决于音区/技巧（胸声、混声、假声、咽音）、元音修饰、声门源（H1–H2、气声）、谱包络（F1 是否被 F0 超过）和声码器的稀疏谐波渲染。只加 F0 条件，会得到“音高对了，但音色仍是说话色”；必须同时建模 **音区/技巧**，并让源演唱的技巧作为行为模板。
6. **只给说话参考时，上限是“population prior + 身份锚点”**，不是本人真实高音。要有“像本人”的高音，必须有目标的高音/戏腔数据（10–30 min 歌声是现实下限）；戏腔还需要 pharyngeal/falsetto 数据。
7. **对 IndexTTS 的推荐：把 E1-R 升级为 E1-T**——F0 条件 + 音高条件化音色残差 + **technique/register embedding** + pitch-invariant speaker encoder + 训练期 pitch augmentation；G1 用 22 kHz + range-controlled split；G2 再上 44.1 kHz / NSF；DPO/GRPO 或 BoN 做后训练。细节见 §4。

---

## 1. 为什么高音不是“F0 数值”问题

### 1.1 高音时到底发生了什么

| 层面 | 变化 | 对生成模型的要求 |
|---|---|---|
| 声源 | 声带闭合/张力改变；modal → mixed → falsetto/head；H1–H2、气声、subglottal pressure 变化 | 需要 register/technique 条件，不能只给 F0 |
| 声道 | F1 可能低于 F0；歌手会抬 F1 去追 F0（formant tuning）；张口、舌位下降、咽腔加宽 | 需要音高依赖的谱包络/共振峰轨迹 |
| 元音 | 高音区元音空间压缩，`/i/` 可能变 `[y]`、`/a/` 变 `[ɑ]` | 需要 vowel modification，不能固定元音模板 |
| 频谱 | 谐波变稀疏；出现 interharmonics；F0 泄漏到包络会产生“双音高”伪影 | 需要 harmonic/noise 分离、pitch-synchronous 分析、NSF/DDSP |
| 声码器 | 训练集音域外的 F0 可能塌回训练集中位；LPCNet/PWG 有这个问题，WaveGlow 外推较好 | 必须做“音域受控”的声码器验证（Perrotin 2021） |

权威依据：
- **Sundberg (1975)**：职业女高音在 F0 超过正常 F1 时会调整下颌/舌位，使 F1 跟随 F0；这是“formant tuning”，可带来 20–30 dB 的声压提升。副作用是元音变形。
- **Titze (2004)**：精确把 F1 对准 F0 未必最优；F0 或第 2/3 谐波略低于 F1 中心时，声门能量转换可能更好。说明“高音音色”不是简单的 F1=F0 规则。
- **Drugman et al. (2020)**：高音歌声的谐波稀疏、interharmonics、MGC 包络携带 F0 信息，会导致噪声化/双音高伪影；建议 pitch-synchronous 分析和可生成 interharmonics 的声码器。
- **Perrotin et al. (Interspeech 2021)**：在训练集刻意挖掉某段 F0 后，WaveGlow 外推最好，WaveRNN 出现大量 F0 错误，LPCNet/Parallel WaveGAN 系统性地生成训练集中位 F0。**匹配测试集几乎看不出这些失败**。

### 1.2 对“音色外推”的形式化

把问题写成：

```
音色(t) = f(身份 embedding, F0(t), 音区/技巧(t), 元音(t), 源演唱行为(t), 噪声)
```

- **身份**：参考音频提供，要求 pitch-invariant；不能把参考的 F0 也带进来（否则会“复制参考的音高”，即 CoMelSinger 说的 prosody leakage）。
- **F0(t)**：源歌曲提供，是连续条件。
- **音区/技巧(t)**：源演唱提供“行为模板”；目标角色是否有该技巧的数据，决定能否“像本人”。
- **元音(t)**：内容特征提供；高音区需要 vowel modification。
- **源演唱行为(t)**：vibrato、glissando、气声、撕裂、戏腔的咽音等；当前管线最容易丢的就是这一层。

**关键推论：** 翻唱场景里，源歌曲已经提供了高音的“行为”（音高、颤音、滑音、假声/混声/咽音）。模型真正要学的是：**“这个身份，在这种音高和技巧下，最可能的谱包络/声门源是什么”**。这比“从说话声凭空推断本人高音”容易，也是 YingMusic-SVC / SVCC 2025 正在做的事。

### 1.3 对当前 IndexTTS 管线的再诊断（可检验假设）

当前链路：`w2v-BERT → RVQ 25 Hz → length_regulator → CFM → BigVGAN 22k`，F0 通过 WORLD 后处理移植。

已确认的事实：
- `s2mel.pth` 284 个参数键，F0 相关 0 个；`CFM.inference(..., f0)` 恒传 None；`length_regulator.f0_condition=false`。
- RVQ 25 Hz 主要保音素/内容，音高信息大幅丢失（隐式旋律 r≈0.8）；精确旋律必须外置 F0。
- BigVGAN 22k 在语音数据上训练；高音歌声的声门源/稀疏谐波没有学过。
- WORLD 移植能把音准修到 83–92%，但用的是输出自身的谱包络；它**不能恢复源演唱的技巧/音区行为**。

**新假设（本次调研提出，待验证）：** 当前高音“像被拉上去的说话声”，除了“目标高音音色没学过”之外，还有一层是**源演唱的技巧/音区信息在 RVQ + WORLD 链路里被丢掉了**。验证方法：
1. 在 GTSinger 上训练一个 6 类 technique 分类器（mixed/falsetto/breathy/pharyngeal/vibrato/glissando + control）。
2. 对源干声、VC 输出、F0 移植后输出分别打分。
3. 如果输出 technique 准确率显著低于源，说明“技巧丢失”成立；E1-T 必须加 technique 条件，而不只是 F0。

这个假设不影响 §0 的结论，但会改变 E1-T 的优先级：**先保源技巧，再谈目标音色外推**。

---

## 2. 五类先进方法

### 2.1 音高条件化音色（F0-aware timbre adaptation）

这是最直接回答“音高变化时音色怎么变”的一类。

#### YingMusic-SVC（arXiv 2512.04793，2025-12；MIT，代码+权重）

**做法**
- 基于 Seed-VC 的 DiT（17 blocks、12 heads、768 dim、3072 FFN），44.1 kHz、128 mel、BigVGAN。
- 三阶段：continuous pre-training → robust SFT → Flow-GRPO（奖励含音色 SIM、可懂度、美学）。
- **F0-aware fine-grained timbre adaptor**：

```
Δeτ(t) = MLP([hf(t); eτ(global)])
hτ(t) = eτ(global) + ατ · Δeτ(t)
ατ = 0.5
```

其中 `hf(t)` 是当前帧 F0 嵌入，`eτ(global)` 是全局歌手嵌入。直观解释：让模型按当前音高“加亮度/头声，或加胸声”，而不是用一张静态音色图。

- 跨性别时对 F0 做 ±12 半音移调，避免八度错误/不稳定；SFT 期做 F0 扰动（Jitter/Glide/Jump）。
- 数据：Emilia 1000 h 英文 + 1000 h 中文语音；6 个公开歌声集约 **230 h** 干净独唱；内部 **500 h** 专业多轨（lead + backing）；RL 只用公开干净歌声。
- 作者报告在“带伴奏/和声污染”输入上收益最大；F0 adaptor 让“极端音符”的音色失真更少——**这是定性描述，论文没有单独的高音 F1/HNR/SS 指标**。

**评价**
- 优点：与问题最同构；公式简单，可直接移植到 IndexTTS 的 style/prompt 条件；44.1 kHz；真实歌曲鲁棒；MIT，代码和权重可用。
- 缺点：500 h 内部数据不可得；极端高音只有定性证据；“F0-aware”不等于“register-aware”，它不知道当前是假声还是混声。
- 对项目的用法：**作为 E1-T 的核心公式**；把 `hf(t)` 换成 `[F0_emb; technique_emb]`，并保留全局身份锚点。

#### HQ-SVC（AAAI 2026；arXiv 2511.08496；推理码+权重已放，训练码未放）

**做法**
- FACodec 解耦内容/说话人；RMVPE F0；DDSP + diffusion + NSF-HiFiGAN 44.1 kHz。
- **EVA（Enhanced Voice Adaptation）**：
  - speaker 特征过 MLP 得 `e_spk`；F0 取 `x_f0=log_e(f0/700+1)` 后过 MLP 得 `e_f0`；再加 style/volume/phase 嵌入。
  - `e_s = Concat(e_spk + e_f0, e_sty, e_vol, e_pha)`，1D conv 压到 256 维。
  - 内容特征 `e_c` 通过 FiLM：`FiLM(e_c,e_s)=f_α(e_s)·e_c+f_β(e_s)`，再过 8 头 Conformer。
  - **F0 与 speaker 刻意相加**，因为“音色与音高强相关”；volume/phase 保持独立。
- **SFP（Speaker-F0 Predictor）**：从 `e_spk` 预测目标 F0 的均值/方差，L1 监督。动机是“音色与音域强耦合，但零样本拿不到目标音高统计”。
- 损失：`L_total = L_ddsp + L_diff + L_spk + L_f0`；speaker 用 InfoNCE（τ=0.1）。
- 数据：OpenSinger + M4Singer，<80 h；44.1 kHz；6 GB 显存、单卡 3090 约 11 h。

**评价**
- 优点：把“音高统计”显式塞进 speaker 条件；EVA 的 FiLM 融合很适合小数据；44.1 kHz NSF 声码器对高音友好；训练成本低。
- 缺点：作者自己承认 **zero-shot 音色匹配仍弱于 CAM++/ContentVec 分离编码器**；SECS 与 SMOS 有张力——F0 一致性提高会让人耳觉得“音色变化小”，但目标音域差很大时反而降低感知相似度；训练码未放。
- 对项目的用法：EVA 的 `e_spk + e_f0` 与 FiLM 结构可直接借鉴；SFP 在“只有参考、没有目标 F0 轮廓”的场景有用，翻唱场景已有源 F0，优先级较低。

#### 相关但次一级的工作

- **DAFMSVC**（Interspeech 2025，arXiv 2508.05978）：dual cross-attention 融合 speaker / melody / content；用目标音频中最相似的 SSL 特征替换源特征，减少音色泄漏。
- **PFlow-VC**（ICASSP 2025，arXiv 2502.05471）：离散 pitch token（speaker-mean normalized log-F0）+ masked pitch-conditioned flow matching；证明“pitch 作为独立条件通道”在表达性 VC 里有效（语音，不是歌声）。
- **Vevo2**（TASLP 2026，arXiv 2508.16332）：统一语音/歌声的 prosody tokenizer + content-style tokenizer，AR + flow matching；支持 synthesis/conversion/editing，style-preserved 与 style-converted SVC 是其任务之一；开源在 `microsoft/Vevo2` / Amphion。

**这一类方法的总评：** 方向正确、工程可落地，但**没有任何一篇给出“训练音域外的高音子集”上的量化证明**。YingMusic 的“extreme notes”是定性；HQ-SVC 的高音也不是单独评测。把它们当作“最强先验”，不要当作“已解决”。

---

### 2.2 技巧/音区条件化（technique / register conditioning）

这是本次调研最重要的增量：**高音音色行为不能只由 F0 决定，必须由“音区/技巧”这个隐变量决定。**

#### SVCC 2025：任务定义已经把“高音行为”拆成 style

- 任务 1：in-domain singing style conversion；任务 2：zero-shot（源歌手训练时未见）。
- 七种 style：**breathy、falsetto、mixed voice、pharyngeal、glissando、vibrato** + control（只作为源 style）。
- 评价：33 个系统；top systems 的 singer identity 已接近 ground truth；但 **style modeling 和 naturalness 仍是难点**。
- 最难的 style 准确率：breathy **37.3%**、glissando **42.6%**、vibrato **43.9%**；没有系统达到人类自然度。
- 客观指标：VERSA toolkit；chroma-alignment 与 speaker embedding 和主观分最相关（峰值约 0.8），但仍不能替代主观测试。

**启示：** “高音”在竞赛里不是作为 F0 区间，而是作为 falsetto / mixed voice / pharyngeal 等 style 来评。**你要的“高音音色行为”，本质上是一个 style/technique 转换问题。**

#### S²Voice（ICASSP 2026，arXiv 2601.13629；SVCC 2025 双赛道冠军，代码未放）

- 基于 Vevo 两阶段：AR LLM 生成 content-style token；flow-matching transformer 渲染 mel。
- 三个关键件：
  1. **FiLM-style layer-norm conditioning**：`(1+γℓ)⊙LN(hiℓ)+βℓ`，把全局 style 注入每个 transformer block。
  2. **Style-aware cross-attention**：style embedding 作 query，LLM latent 作 key/value，做时间局部 style 条件。
  3. **全局 speaker embedding**（预训练说话人验证网络）注入 flow transformer，压 style 泄漏、提 singer similarity。
- 数据：官方 SVCC 2025 约 70 h + 自建约 500 h 清洗歌声；GTSinger 未用于训练。
- 训练：SFT + DPO；DPO 用“平均指标略降”换“更少低质量离群样本”（早停、重复、断句、抖动）。
- 结果：作者称双赛道第一；style similarity / singer similarity 领先；但论文没有按音高区间拆分。

**评价：** 目前最接近“高音行为转换”的公开系统设计；但代码/权重未放，只能借鉴结构：**FiLM-LN + style cross-attention + speaker embedding + DPO**。

#### S4（SVCC 2025 系统，arXiv 2604.05526）

- **Boundary-aware Whisper bottleneck**：按音素跨度池化，压源 style 泄漏，保语言内容。
- **Frame-level technique matrix + 推理期定向 F0 处理**：稳定、可区分的动态 style。
- **High-band completion**：用一个辅助 48 kHz SVC 模型补高频，解决数据少、又不想过拟合的问题。
- 结果：官方主观评测中 **naturalness 所有提交中最好**，speaker similarity 与 technique control 有竞争力；额外歌声数据显著少于其他 top 系统。

**评价：** “高音听起来闷/缺亮度”的一个直接工程解：**22 kHz 主模型 + 48 kHz 高频补全**。对 8 GB / 22 kHz 的 IndexTTS 很有参考价值，比整体上 44.1 kHz 便宜。

#### TechSinger（AAAI 2025，arXiv 2502.12572；代码+权重）

- 技术可控的多语言 SVS；7 种 technique，明确包含 **mixed voice、falsetto**（另有 intensity、bubble、breathy 等）。
- **FMPP（Flow Matching Pitch Predictor）**：用 flow matching 建模“technique → F0 轮廓”的复杂映射，替代纯 L1。
- **CFGFMP（Classifier-Free Guidance Flow Matching Postnet）**：粗 mel → 细化；CFG scale 控制 technique 强度。
- **Technique detection model** 自动给数据打 phoneme-level technique 标签；**prompt-based technique prediction** 用自然语言指定技巧。
- 数据：GTSinger；预训练 checkpoint 只支持中/英；多语言需自己训练。
- 限制：SVS（需要曲谱/音符），不是 SVC；代码无显式开源 license，只有“未经本人同意不得生成他人歌声”的声明。

**评价：** 证明“technique label 可以显式控制 F0 轮廓和音色”；对 E1-T 的 technique embedding / auxiliary loss 设计最直接。缺点是 SVS 重唱，丢源表演。

#### FABYOL（PMLR v303，2026；Hsu & Yang）

- 基于 FACodec 的 **vocal timbral technique conversion**：whisper / **falsetto** / vocal fry scream；保说话人身份。
- BYOL-TT 监督对比学习得到 technique embedding；embedding 通过 **AdaLN** 同时调制 timbre 和 prosody。
- 数据：JVS（modal/whisper/falsetto）、EMVD、新 EMO（706 clips，vocal fry scream）。
- 结果（据论文/检索）：MCD 7.59、HNR −1.27、technique similarity AFO 379 / SEC 0.79、TSMOS 3.64；消融显示去掉 prosody modulation 对 falsetto 的谱保真伤害明显。
- 限制：**不做旋律/歌声**（作者把 melodic/singing 列为 future work）；更像“说话技巧转换”的证据。

**评价：** 证明“falsetto 可以作为独立 embedding 被转换，且 prosody 调制不能少”；但还不能直接用于翻唱。

#### 其他相关

- **StarGAN-based singing technique conversion**（arXiv 2308.10021，2023）：WORLD 特征 + 卷积自编码器，falsetto/chest/raspy/whistle 四类；瓶颈宽度影响咬字清晰度与目标技巧相似度。
- **VibE-SVC2**（IEEE TASLP 2026，arXiv 2606.17126）：独立控制 vibrato extent/rate；**Subharmonic Correction** 修正 vocal fry 的 F0 提取；说明“音高风格”和“音色风格”可以分开建模。
- **OneVoice**（arXiv 2601.18094，2026）：一个模型统一 LVC / EVC / SVC；MoE + scenario-aware routing + LoRA experts；说明“按场景/技巧分专家”是可行路线。

**这一类方法的总评：** 对“高音/戏腔”最对症，因为高音不是单一音高，而是 **falsetto / mixed / pharyngeal / belt** 等音区技巧。但它需要 **technique 标注数据**，且现有系统在 falsetto/mixed 上的准确率远未到“完美”。**E1-T 必须把 technique 作为一等条件，而不是可选后处理。**

---

### 2.3 说话参考零样本 SVS/SVC

这类方法直接回答“只有角色说话，没有角色歌声”的前置问题。

#### Everyone-Can-Sing（ICASSP 2025，arXiv 2501.13870）

- 一个 SVS + 两个 SVC；内容用预训练 embedding，生成用 diffusion；训练混合歌声 + 语音，支持 **5 秒语音参考** 的音色克隆。
- **Adjust Pitch**：把目标曲谱移进语音参考的“一个八度以内”；SVS 相似度 3.78 vs 3.60，SVC(b) 3.61 vs 3.50，但质量略降（SVS 3.97 vs 4.02；SVC(b) 3.96 vs 4.03）。
- 作者结论：移调“对齐源/目标音域，提升相似度，但有时降低歌唱质量”；跨性别 SVC 质量“显著下降”，因为移调更大。
- 数据：6262 h 歌声、5050 歌手、3 语言；语音用 LibriTTS-R 585585 h、2456 说话人；1:1 混合。
- 限制：zero-shot SVS 略逊于非零样本 ExpressiveSinger；歌剧弱于流行；跨性别质量下降；对齐歌词 SVC 依赖标注。

**评价：** 证明“5 s 说话 → 唱歌”可行；也证明 **移调是相似度/质量的 trade-off**，不是免费午餐。对 E1-T 的启示：不要指望“把参考移调”解决高音；要在模型内部用 F0/technique 条件，而不是改参考。

#### TCSinger 2（ACL 2025 Findings，arXiv 2505.14910）

- 零样本多语言 SVS；style 可由 **singing / speech / text prompt** 控制。
- Custom Audio Encoder 用对比学习对齐三种 prompt；BBC Encoder 处理边界；Flow-based Custom Transformer + Cus-MOE + F0 supervision。
- 支持风格迁移/控制；摘要未提 falsetto/vocal range。
- 评价：证明“speech prompt 可以驱动歌声风格”；但高音/音区不是它的显式目标。

#### Vevo2（IEEE TASLP 2026，arXiv 2508.16332；`microsoft/Vevo2` / Amphion）

- 统一语音/歌声：music-notation-free prosody tokenizer + content-style tokenizer（timbre disentanglement）；AR 阶段控 text/prosody/style，flow-matching 阶段控 timbre。
- 任务覆盖：zero-shot TTS、text-to-singing、SVS、style-preserved / style-converted VC/SVC、语音/歌声编辑、humming-to-singing、instrument-to-singing。
- **Pitch Region Control**：在 token 提取前把源 F0 移到参考音域；表显示 SIM 提升，但 WER 略升（train-inference mismatch）。
- 多目标后训练：可懂度 + prosody 相似度，扩展 GRPO。
- 评价：目前最完整的“统一语音/歌声 + 风格转换”开源框架；S²Voice 就是基于 Vevo 系列。对项目：如果未来允许引入外部框架，Vevo2 是最值得对照的基线；在当前“只用 IndexTTS”约束下，主要借鉴其 **content-style tokenizer + timbre disentanglement + style-preserved SVC** 思想。

#### SoulX-Singer（arXiv 2602.07803，2026）

- 开源 SVS；42,000+ 小时歌声数据；支持 MIDI/旋律条件；zero-shot；提出 SoulX-Singer-Eval 基准（严格训练-测试解耦）。
- 评价：数据规模和工程化很强，但仍是 SVS（需要曲谱/旋律），不是“保留源表演”的 SVC。

#### OneVoice（arXiv 2601.18094，2026）

- 单一模型统一 LVC / EVC / SVC；VAE-free next-patch diffusion；MoE 分离共享转换知识与场景特定表达；两阶段预训练 + LoRA 专家；2 步快速解码。
- SVC 细节：RMVPE F0 量化成 256-bin 离散序列；推理时按目标说话人的 pitch 统计/均值**自适应平移** F0；F0 相关 0.898、speaker similarity 0.738、CER 3.92（作者对比 YINGSVC/SeedVC-Sing）。
- 限制：歌声数据仅约 400 h（语音约 100,000 h）；冻结 backbone 限制 prosody/pitch 建模；patch size r=10 时 F0cor 降到 0.748。
- 评价：证明“场景/技巧分专家”可以统一在一个模型里；对 E1-T 的 register/technique MoE 设计有参考。

**这一类方法的总评：** 解决“参考只有说话”的问题，但没有解决“参考里没有高音音色”。它们的高音质量取决于训练数据是否覆盖高音；`adjust pitch` 只是在相似度和质量之间换挡。**对项目：可作为 G1 的对照基线，不作为主路线。**

---

### 2.4 音域适配 / 音高增广

这类方法不生成新音色，但能显著减少“跨音域嘶哑”，是必备支撑。

| 工作 | 机制 | 证据/数字 | 对项目的用法 |
|---|---|---|---|
| **SPSinger**（ICASSP 2025） | 5 s 参考；LPAM 从全局音色 + 曲谱隐序列**预测需要的移调量**；分类头 | 避免 naive pitch shift；短参考音域不匹配 | B2 自动找调的“学习式版本” |
| **SPA-SVC**（Interspeech 2024） | 训练期 cycle pitch-shifting，随机 6–18 半音；`L_cyc=1−SSIM`；`L_total=L_cyc+L_diff` | M4Singer；Soprano-3/Tenor-1 未见；MOS 3.76 vs 3.61；高音 Alto 的 MOS-A 增益**边际** | 训练期增广 > 推理期移调；E1-T 必加 |
| **SpSiVC**（2025） | 自适应 F0 损失：歌声按八度约束、语音按说话人均值；解决整数倍八度偏移 | 统一语音/歌声；歌声余弦相似度 0.969 | 自动移调/八度约束的参考 |
| **R2-SVC**（arXiv 2510.20677，2025-10） | 随机 F0 扰动 + 分离伪影模拟（混响/回声）+ NSF 显式谐波/噪声 | 真实噪声场景 SOTA | 数据增广 + NSF 声码器 |
| **Prompt-Singer**（NAACL 2024） | **range–melody 解耦的 pitch 表示**；自然语言控制 gender/range/volume；引入语音数据缓解数据少 | 可在保持旋律的同时控制音域 | “音域”作为独立条件，而不是改参考 |
| **2309.00284**（2023） | melody-unsupervised 多歌手预训练 → 单歌手微调；phone predictor + speaker encoder + 直接 F0 估计 | 目标：唱到训练音域外仍不降音色相似度 | 小数据角色微调时的预训练策略 |
| **2607.13278**（DAFx 2026） | Pitch range adaptation：半音域均值平移 `p_t^tar=p_t^src−μ_src+μ_tar` | 女→男转换会出现“高音、女声化”伪影；均值平移是必要预处理 | 音域适配的简单强基线 |
| **Vevo2**（TASLP 2026） | **Pitch Region Control**：token 提取前把源 F0 移到参考音域 | SIM 提升、WER 略升（train-inference mismatch） | 音域适配 + style-preserved SVC 的组合 |
| **Everyone-Can-Sing** | 把曲谱移进参考的一个八度内 | 相似度升、质量略降；跨性别质量降 | 推理期兜底 |

**总评：** 这些方法是“把问题搬到模型舒适区”，不是“让模型学会高音”。**训练期 pitch augmentation（SPA-SVC）最值得做**；推理期移调参考/源要谨慎，项目实测参考移调会丢身份。

---

### 2.5 声码器 / 源-滤波（renderer side）

高音的最后一步是“把稀疏谐波渲染对”。模型预测的 mel 再好，声码器在训练音域外塌掉也会前功尽弃。

| 工作 | 机制 | 证据 | 对项目的用法 |
|---|---|---|---|
| **DDSP**（ICLR 2020） | 谐波振荡器 + 滤波噪声；可解释、可独立控制 pitch/loudness | 摘要明确声称 “realistic extrapolation to pitches not seen during training”；pitch/loudness 独立控制 | 物理先验；可作为辅助损失/声码器候选 |
| **SawSing**（ISMIR 2022） | 锯齿波源 + 时变 FIR 滤波；相位连续 | 3 段录音、3 小时训练即超过 GAN/diffusion 基线；数据效率极高 | 小数据角色微调的声码器参考 |
| **NSF-HiFiGAN** | 神经源-滤波：显式谐波/噪声激励 | HQ-SVC、R2-SVC 使用；高音更稳 | G2 声码器首选之一 |
| **BigVGAN v2 44.1 kHz** | 128 mel、512× 上采样、122M；多尺度 sub-band CQT 判别器 + mel 损失；大规模多样音频训练 | Seed-VC 官方 changelog：换 BigVGAN 后 “large improvement to high-pitched singing voices” | G2 首选；G1 可先微调 22k BigVGAN |
| **SnakeGAN**（2023） | DDSP 先验 + periodic inductive bias + anti-aliasing | 只在语音训练也能泛化到歌声/乐器；提升 unseen 场景外推 | 通用声码器的参考 |
| **Perrotin 2021** | 训练集挖掉 F0 区间，测全局/局部外推 | WaveGlow 好；WaveRNN 大量 F0 错误；LPCNet/PWG 塌回训练中位 | **必须做 range-controlled 声码器测试** |
| **Drugman 2020** | 高音歌声的参数化声码器分析 | 谐波稀疏、interharmonics、F0 泄漏→双音高/噪声化 | 高音声码器评测指标：HNR、谐波缺失、双音高 |

**总评：** 声码器是“必要不充分”。22 kHz 是当前 IndexTTS 的硬上限；G1 可以先用 22k 微调 + S4 式高频补全，G2 再上 44.1k / NSF。

---

### 2.6 后训练 / 推理期择优

| 方法 | 代表 | 作用 | 对项目的用法 |
|---|---|---|---|
| **DPO** | S²Voice | 用偏好对压掉重复、断句、抖动；平均指标略降换更少离群 | 有高质量/低质量成对样本时可用 |
| **Flow-GRPO** | YingMusic-SVC | 奖励 = 音色 SIM + 可懂度 + 美学；真实歌曲鲁棒 | 8 GB 训练成本高；可作为后期 |
| **BoN + 高音奖励** | 项目已有 BoN | 推理期多候选，按高音 HNR/音准/音色/技巧选 | **最低成本增益**；建议先做 |
| **高音专用 reward** | 本次建议 | `HNR/CPP`、局部八度错误、F1–F0 tuning 偏差、technique 分类置信度、双音高检测 | 让 BoN 直接优化“高音不嘶哑” |

**总评：** 后训练能把“可接受的候选”挑出来，但不能凭空创造模型没有的高音先验。**先做 BoN 高音奖励，再考虑 DPO/GRPO。**

---

## 3. 横向对比

| 方法族 | 代表 | 能否从说话参考外推高音音色 | 证据强度 | 代码/权重 | 对 IndexTTS 的适配 | 主要风险 |
|---|---|---|---|---|---|---|
| F0-aware timbre | YingMusic-SVC | **部分能**（F0 条件化音色；极端音符定性） | 中（无高音量化） | MIT，代码+权重 | **高**，公式可直接移植 | 500h 内部数据不可得；register 不感知 |
| Pitch/energy-aware SVC | HQ-SVC | **部分能**（EVA+SFP） | 中（zero-shot SVC 强；高音未单测） | 推理+权重；训练码未放 | 高，EVA/FiLM 可借鉴 | SECS/SMOS 张力；zero-shot 音色弱 |
| Technique/style conversion | S²Voice / S4 / TechSinger / FABYOL | **最接近“音色行为”** | 中（SVCC 2025 有 style 指标；falsetto 仍难） | S²Voice 无码；TechSinger 有码；FABYOL 有码 | **最高**（尤其戏腔） | 需要 technique 标注；SVS/SVC 任务不同 |
| Speech-reference zero-shot | Everyone-Can-Sing / TCSinger 2 / Vevo2 | **能唱，但高音弱** | 中高（多人主观） | Everyone 无码；Vevo2 开源 | 中（可作为基线） | 参考无高音；移调 trade-off |
| Range adaptation/augmentation | SPSinger / SPA-SVC / SpSiVC / R2-SVC | **不生成新音色** | 高（工程有效） | 多数有码/可复现 | **必做支撑** | 只搬问题；推理期移调伤身份 |
| Source-filter/vocoder | DDSP / SawSing / NSF / BigVGAN v2 | **渲染层外推** | 高（Perrotin 有明确证据） | 开源 | **G2 必做** | 22k 上限；声码器不能造音色 |
| Post-training | DPO / GRPO / BoN | **只择优** | 中 | 视实现 | 低成本增益 | 奖励设计难；不能替代数据 |

---

## 4. 对 IndexTTS 的落地方案：E1-T

> 把交接文档的 E1-R（F0 条件 + 音高条件化音色残差）升级为 **E1-T**：在 F0 之外显式加入 **technique/register**，并让源演唱的技巧作为行为模板。

### 4.1 目标（G1 最小验证）

在 22 kHz、8 GB 可承受的范围内，验证：
1. 模型能否在 **训练音域外**（held-out 高音子集）保持目标身份相似度与可懂度；
2. 高音段的 HNR/周期性、局部八度错误是否优于当前 A 基线；
3. 源演唱的 technique（falsetto/mixed/pharyngeal/breathy/vibrato/glissando）是否被保留；
4. 加入 technique 条件是否比“只加 F0”显著更好。

**不承诺**：仅凭说话参考就生成“本人真实高音”；不承诺戏腔。

### 4.2 架构改动

在现有 CFM / DiT 上做最小侵入：

1. **F0 条件**
   - log-F0 → 256/512 维嵌入；FiLM 注入 DiT 的 time embedding / cond。
   - `flow_matching.py` 的 `f0` 形参要真正被 estimator 消费；`length_regulator.f0_condition` 打开或旁路。
   - 归一化：按说话人/句子做 speaker-mean normalized log-F0，避免绝对音高泄漏身份。

2. **音高条件化音色残差（核心）**
   - 现有 `style`（192 维）/`prompt_condition`（512 维）保持全局身份。
   - 新增：
     ```
     Δe(t) = MLP([F0_emb(t); technique_emb(t); e_style])
     e_style_t = e_style + α · Δe(t)
     ```
     α 初始 0.5（YingMusic 的设置）；训练中可学习/退火。
   - 把 `e_style_t` 注入 CFM 的 cross-attention / FiLM，而不是只拼到内容上。

3. **Technique/register 条件**
   - 6 类 GTSinger 标签 + control：`mixed / falsetto / breathy / pharyngeal / vibrato / glissando / control`。
   - 训练时用 ground-truth phoneme-level 标签；推理时：
     - **auto**：用源音频跑 technique 分类器（在 GTSinger 上训练）；
     - **manual**：用户选 falsetto/mixed/pharyngeal 等（戏腔场景必须可手动）。
   - 可选：把 technique 做成 **MoE / 多 LoRA 专家**（胸声/混声/假声/咽音），按 F0 + 分类器路由；8 GB 下比全参更现实。

4. **Pitch-invariant speaker encoder**
   - 对同一说话人做 pitch-shift 增广，加 **identity consistency loss**（同人不同音高 embedding 靠近；同音高不同人远离）。
   - 目的：防止参考的 F0/音域泄漏到身份条件（CoMelSinger 的 prosody leakage；VaSAB 的 F0 解耦）。
   - 参考音频在推理时**不要移调**（项目实测身份会崩）。

5. **SFP 可选**
   - HQ-SVC 式 `μ̂,σ̂² = f_sfp(e_spk)`；翻唱场景已有源 F0 轮廓，SFP 主要用于“零样本目标音域统计”，G1 可先不做。

6. **声码器**
   - G1：22k BigVGAN 在 GTSinger 歌声上微调；加 S4 式 **高频补全**（辅助 48k SVC 或高频带宽扩展）作为可选分支。
   - G2：44.1k BigVGAN v2 或 NSF-HiFiGAN；用 Perrotin 式 range-controlled 测试验证不塌。

### 4.3 数据

| 数据 | 用途 | 规模/要点 | 许可/风险 |
|---|---|---|---|
| **GTSinger** | G1 主训练集 | 80.59 h、20 专业歌手、9 语言、4 音域、6 technique（mixed/falsetto/breathy/pharyngeal/vibrato/glissando）、16.16 h 配对语音、48 kHz/24-bit、phoneme-level 标注、technique/control 对照 | 检索显示 **CC BY-NC-SA 4.0**；商用需另找授权/自录 |
| **M4Singer / Opencpop / OpenSinger** | 扩规模/多歌手 | M4Singer ~30 h；Opencpop ~5 h；OpenSinger | 研究用途；逐项核对 |
| **SVCC 2025 数据** | style 转换/评测 | 约 70 h 官方训练集；7 style | 竞赛条款；商用需核对 |
| **SingNet** | 大规模预训练/声码器 | ~3000 h；BigVGAN/NSF-HiFiGAN 预训练 | 核对许可 |
| **角色说话 10–60 s** | 身份参考 | 干净、无混响；不要移调 | 声音权利 |
| **角色歌声 10–30 min（可选）** | 个性化高音 | 覆盖目标音域/技巧；44.1/48 kHz | 游戏/声优/歌曲版权 |
| **角色戏腔 1–5 h（可选）** | 戏腔 | 最难获取；KunquDuoVox 7.68 h 可作研究起点 | 研究用途 |

**数据策略：**
- G1 用 GTSinger 的中/英子集 + M4Singer/Opencpop 扩歌手；角色说话作为 identity anchor。
- **range-controlled split**：
  - 按歌手划分 train/val/test（unseen singer）；
  - 每个歌手留出最高 10% F0 的片段作为 high-pitch test；
  - 留出至少一种 technique（如 pharyngeal）作为 unseen technique；
  - 不能随机按帧划分（会泄漏同一句的相邻帧）。
- **technique-balanced sampling**：避免 falsetto/mixed 样本被 control 淹没；GTSinger 有 technique/control 对照，可按 item 配对采样。
- **pitch augmentation**：SPA-SVC 式 cycle pitch-shift（6–18 半音）+ SSIM；训练期做，推理期不做源/参考移调。

### 4.4 训练

**阶段 0：基线复现**
- 当前 A 档、22k、无 F0 条件；在 GTSinger high-pitch test 上记录 SS/F0/HNR/technique accuracy。

**阶段 1：F0 条件（E1-R 原计划）**
- 只加 F0 embedding + FiLM；LoRA/Adapter；冻结内容编码器。
- 目标：验证“F0 条件本身”能否改善高音；如果 technique accuracy 仍低，进入阶段 2。

**阶段 2：F0 + technique（E1-T 核心）**
- 加 technique embedding + 音高条件化音色残差 + identity consistency loss。
- 损失：
  ```
  L = L_flow + λ_f0·L_f0 + λ_tech·L_tech_ce
      + λ_id·L_contrastive + λ_ssim·L_ssim(cycle)
      + λ_spk·L_spk_consistency
  ```
- 数据：GTSinger + 角色说话；technique 标签用 phoneme-level；pitch augmentation 在线做。
- 8 GB：22k、10–20 s 片段、batch 1–2、gradient checkpointing、LoRA rank 8–32；先只训 CFM，不训 w2v-BERT/BigVGAN。
- 预计：单卡 4060 Laptop 8 GB，80 h 数据、LoRA，约 1–2 周（取决于步数/片段长度；以实际吞吐为准）。

**阶段 3：声码器/后训练**
- 22k BigVGAN 在 GTSinger 上微调；Perrotin 式 range-controlled 测试。
- BoN 高音奖励（HNR、八度错误、F1–F0、technique 置信度）；可选 DPO/GRPO。
- G2：44.1k / NSF，24 GB 云；2–3 个月。

### 4.5 推理

```
源干声 → F0 (RMVPE/FCPE) + technique 分类器
       → 内容编码 (w2v-BERT/RVQ)
       → CFM(F0_emb, technique_emb, ref identity)
       → mel → 22k/44.1k 声码器
       → 可选 BoN 高音奖励择优
```

- **不要**再用 WORLD 移植作为主路径；F0 作为条件进入模型。
- 保留 WORLD/局部 F0 修复作为兜底（音准微调、八度错误修正）。
- technique 选择：
  - `auto`：源分类器；
  - `manual`：用户指定 falsetto/mixed/pharyngeal（戏腔必须可手动）；
  - `preserve`：尽量保留源技巧（默认，翻唱场景）。
- 参考音频保持原样，不移调；身份与音高在模型内部解耦。

### 4.6 评测 / G1 闸门

**必须新增：range-controlled 评测**（Perrotin 式）。随机划分会掩盖外推失败。

| 指标 | 定义 | G1 建议线 |
|---|---|---|
| 身份 SS | campplus/ECAPA 对角色参考；高音子集单独统计 | ≥0.70，或较 A 基线 **+0.08** 且 ABX≥60% |
| 音准 | F0 50 音分内比例；p90 音分；局部八度错误率 | 50c ≥85%；八度错误 <1%（高音子集） |
| 高音周期性 | HNR/CPP；F0>500/600 子集 | 不低于 A 基线（最好 +1 dB HNR） |
| 技巧保真 | 6 类 technique 分类器在输出上的准确率/F1 | 较 A 基线显著提升；falsetto/mixed ≥70% |
| 音色行为 | F1–F0 tuning 偏差；谱倾斜/H1–H2；双音高检测 | 相对基线改善；人工抽听 |
| 自然度 | MOS/ABX；高音段单独 | ABX≥60%；MOS 不低于 A |
| 内容 | CER/歌词可懂度 | 不显著劣化 |
| 声码器外推 | held-out F0 区间的中位 F0 塌陷/错误率 | 不塌回训练中位 |

**闸门判定：**
- 通过：高音子集 SS/F0/HNR/technique 四项中至少三项优于 A，且 ABX≥60%。
- 不通过：回到阶段 1/2 调参或补数据；不要直接上 G2。

### 4.7 代码触点（基于交接文档，不改代码，仅标注）

| 文件 | 改动 |
|---|---|
| `indextts/s2mel/modules/flow_matching.py` | 让 estimator 消费 `f0`；加 technique embedding/FiLM |
| `indextts/s2mel/modules/diffusion_transformer.py` | DiT 加 F0/technique 条件通道（FiLM/AdaLN） |
| `indextts/s2mel/modules/length_regulator.py` | `f0_condition` 分支启用或旁路 |
| `checkpoints/config.yaml` | `s2mel.f0_condition`、technique 配置、mel 参数（44.1k 时） |
| `webui_app/training/features.py` | F0 特征缓存、technique 标签、pitch-shift 增广 |
| `webui_app/training/cfm_lora.py` | LoRA 目标模块、technique embedding 表、损失项 |
| `webui_app/training/trainer_base.py` | technique-balanced sampler、range-controlled validation、grad checkpoint |
| `webui_app/services/svc_core.py` | 推理传 F0/technique；可选去掉 WORLD 主路径 |
| `webui_app/services/svc_f0.py` | RMVPE/FCPE、technique 分类器、音域分析 |
| `webui_app/services/cover_pipeline.py` | 编排、technique 选择、报告 |
| `checkpoints/hf_cache/bigvgan` | 22k 微调；G2 44.1k/NSF |
| `webui_app/services/engine.py` | LoRA 挂载、显存调度、租约 |

**注意：** 上游引擎“零改动”偏好 → 用旁路/开关，保留原 TTS 路径；更新 `NOTICE`（F0/technique 手术后“上游权重未修改”不再成立）。

### 4.8 算力 / 周期

| 阶段 | 硬件 | 预计 |
|---|---|---|
| G1 阶段 1（F0 LoRA） | 4060 Laptop 8 GB | 3–5 天 |
| G1 阶段 2（F0+technique） | 8 GB LoRA / 可选 24 GB | 1–2 周 |
| G1 声码器微调 | 8 GB | 2–4 天 |
| G1 评测/调参 | 8 GB | 1 周 |
| G2 44.1k | 24 GB 云 | 2–3 个月 |
| 戏腔（G3） | 24 GB+；需数据 | 取决于数据 |

### 4.9 风险 / 回退

| 风险 | 影响 | 回退 |
|---|---|---|
| GTSinger 是 NC 许可 | 商用不可用 | 自录/授权数据；研究阶段先用 |
| 8 GB 不够全参 | 训练慢/不稳 | LoRA/Adapter + grad checkpoint + 10–20 s |
| 22k 声码器上限 | 高音亮度/空气感不足 | S4 式高频补全；G2 44.1k |
| technique 标签噪声/共现 | 条件不稳 | 只用高置信标签；control 配对；多任务辅助 |
| 只加 F0 仍“说话色” | G1 失败 | 加 technique + identity consistency；检查 F0 是否真被消费 |
| 源技巧丢失 | 高音不自然 | 加 technique 条件；用源 F0/energy/vibrato 特征；不要过度依赖 RVQ |
| SS 在高音子集不可靠 | 误判 | 加 ABX/MOS、HNR、technique accuracy；不要只看 SS |
| 过拟合 GTSinger 歌手 | 泛化差 | unseen singer/technique split；M4Singer/Opencpop 扩歌手 |
| 外部权重/许可 | 无法发布 | 只用论文思想；保留 NOTICE；商用前法务 |

### 4.10 E1-T 与 E1-R 的差异

| 维度 | E1-R（交接文档） | E1-T（本次建议） |
|---|---|---|
| 条件 | F0 条件 + 音高条件化音色残差 | **F0 + technique/register** 条件 |
| 核心假设 | 高音音色主要由 F0 决定 | 高音音色由 **F0 + 音区/技巧 + 元音** 共同决定 |
| 数据 | 公开歌声 + 角色说话 | **GTSinger（6 technique + paired speech）** + 公开歌声 + 角色说话 |
| 评测 | SS/F0/HNR | 加 **technique 保真、range-controlled split、F1–F0 tuning** |
| 声码器 | 22k 微调 → 44.1k | 22k + 高频补全 → 44.1k/NSF |
| 预期 | 高音“合理” | 高音“合理 + 技巧对” |
| 风险 | 音高对了但音色仍是说话色 | technique 数据/标签依赖；更复杂 |

---

## 5. 如果放开“只用本模型”：外部对照与上限

> 当前约束下不实施；仅作为质量上限/对照。

| 系统 | 定位 | 许可/可用性 | 高音相关 |
|---|---|---|---|
| **YingMusic-SVC** | 零样本 SVC，44.1k，真实歌曲鲁棒 | MIT；代码+权重 | **最接近本问题**；F0-aware timbre；极端音符定性 |
| **Seed-VC 44.1k** | 零样本 SVC 强基线 | GPL-3.0 | BigVGAN 更新明确改善高音歌声；SVC 模型 F0 条件 |
| **HQ-SVC** | AAAI 2026 零样本 SVC | 推理码+权重；训练码未放 | EVA/SFP；44.1k NSF；高音友好 |
| **Vevo2** | 统一 SVS/SVC/风格转换 | 开源（microsoft/Vevo2） | style-preserved/converted SVC；S²Voice 基础 |
| **RVC + 角色歌声** | 个性化上限 | MIT | 有 10–30 min 角色高音/戏腔数据时最强 |
| **SVS（DiffSinger/ACE/Synthesizer V）** | 重唱上限 | 商业/开源 | 高音/戏腔最稳，但丢源表演 |
| **TechSinger / FABYOL** | 技巧控制/转换 | TechSinger 代码+权重；FABYOL 代码+权重 | falsetto/mixed/pharyngeal 参考 |

**结论：** 如果目标是“尽快听到像样的高音翻唱”，YingMusic-SVC / Seed-VC 44.1k 是最快对照；如果目标是“该角色完美高音/戏腔”，仍然是 **RVC/SVS + 角色高音/戏腔数据**。IndexTTS 原生路线要接近它们，必须做 E1-T + G2。

---

## 6. 待验证清单

1. **YingMusic-SVC 的极端高音是否真的更好？** 用其权重在 F0>500/600 Hz 子集上测 SS/F0/HNR/technique，和 Seed-VC/HQ-SVC 对比；论文只有定性。
2. **HQ-SVC 的 EVA/SFP 在“给定目标 F0 轮廓”时是否仍有增益？** SFP 可能只在没有目标 F0 时有用。
3. **Technique 条件是否比 F0-only 显著提升高音自然度？** 需要 G1 消融：F0-only vs F0+technique vs F0+technique+identity consistency。
4. **源技巧是否真的在 RVQ/WORLD 链路丢失？** 用 GTSinger technique 分类器测源/输出；这是 E1-T 的关键假设。
5. **哪种声码器在 F0>600 Hz 最稳？** 22k BigVGAN vs 44.1k BigVGAN vs NSF-HiFiGAN；Perrotin 式 range-controlled。
6. **GTSinger 许可**：CC BY-NC-SA 4.0 是否允许本项目用途；商用需另找数据。
7. **SVCC 2025 数据/模型许可**：训练集、S²Voice、S4 的可用性。
8. **YingMusic-SVC 权重许可**：代码 MIT，权重是否同一许可；HuggingFace model card 核对。
9. **IndexTTS CFM 98M 的容量**：F0+technique 条件是否会被容量限制；LoRA rank/Adapter 层数。
10. **8 GB 训练吞吐**：GTSinger 80 h、10–20 s、batch 1–2 的实际步/秒；是否需要云。
11. **Technique 分类器可靠性**：GTSinger 6 类 + control 的混淆矩阵；falsetto/mixed 共现怎么处理。
12. **评测指标**：SS 在高音子集是否可靠；需要 ABX/MOS + HNR + technique accuracy 组合。
13. **戏腔数据**：KunquDuoVox 7.68 h、Yue Opera 2026 基准、Peking Opera ACV-003 的许可与音域覆盖；pharyngeal 标签是否可用。
14. **“完美”边界**：只给说话参考时，任何模型都只是先验；需要向用户明确预期。

---

## 7. 参考链接

**核心论文/系统**
- YingMusic-SVC：https://arxiv.org/abs/2512.04793 ；https://github.com/GiantAILab/YingMusic-SVC
- HQ-SVC：https://arxiv.org/abs/2511.08496 ；https://github.com/ShawnPi233/HQ-SVC
- S²Voice：https://arxiv.org/abs/2601.13629
- SVCC 2025 分析：https://arxiv.org/abs/2509.15629
- S4（SVCC 2025）：https://arxiv.org/abs/2604.05526
- TechSinger：https://arxiv.org/abs/2502.12572 ；https://github.com/gwx314/TechSinger
- FABYOL：https://proceedings.mlr.press/v303/hsu26a.html
- Everyone-Can-Sing：https://arxiv.org/abs/2501.13870 ；https://everyone-can-sing.github.io/
- TCSinger 2：https://arxiv.org/abs/2505.14910
- Vevo2：https://arxiv.org/abs/2508.16332 ；https://github.com/microsoft/Vevo2
- SoulX-Singer：https://arxiv.org/abs/2602.07803
- OneVoice：https://arxiv.org/abs/2601.18094
- SPSinger：https://ieeexplore.ieee.org/abstract/document/10888907
- SPA-SVC：https://www.isca-archive.org/interspeech_2024/bai24_interspeech.html ；https://ar5iv.labs.arxiv.org/html/2406.05692
- SpSiVC：https://www.jstage.jst.go.jp/article/ast/46/1/46_e24.47/_article
- R2-SVC：https://arxiv.org/abs/2510.20677
- DAFMSVC：https://arxiv.org/abs/2508.05978
- PFlow-VC：https://arxiv.org/abs/2502.05471
- Prompt-Singer：https://arxiv.org/abs/2403.11780
- 2309.00284（vocal range pretraining）：https://arxiv.org/abs/2309.00284
- 2607.13278（DAFx 2026 pitch range adaptation）：https://arxiv.org/abs/2607.13278
- CoMelSinger：https://arxiv.org/abs/2509.19883
- VaSAB：https://arxiv.org/abs/2310.03444
- VibE-SVC2：https://arxiv.org/abs/2606.17126
- StarGAN technique conversion：https://arxiv.org/abs/2308.10021

**声码器/物理建模**
- DDSP：https://arxiv.org/abs/2001.04643
- SawSing：https://arxiv.org/abs/2208.04756
- BigVGAN：https://arxiv.org/abs/2206.04658 ；https://github.com/NVIDIA/BigVGAN
- SnakeGAN：https://arxiv.org/abs/2309.07803
- Perrotin 2021：https://hal.science/hal-03338483
- Drugman 2020：https://arxiv.org/abs/2006.04142
- Sundberg 1975：Formant technique in a professional female singer, Acustica 32(2):89–96
- Titze 2004：F0–F1 interaction in speaking/singing（J. Voice 相关）

**数据/评测**
- GTSinger：https://arxiv.org/abs/2409.13832 ；https://huggingface.co/datasets/GTSinger/GTSinger
- SingNet：https://arxiv.org/abs/2505.09325
- KunquSinger/KunquDuoVox：https://hualizhou167.github.io/KunquSinger-demo/
- KunquDuoVox 论文（2026）：DOI 10.11684/j.issn.1000-310X.2026.04.001
- Seed-VC：https://github.com/Plachtaa/seed-vc
- DiffSinger：https://github.com/openvpi/DiffSinger

> 注：论文数字以 arXiv/DOI 摘要与正文为准；部分系统（S²Voice、Everyone-Can-Sing）无公开代码；YingMusic-SVC 代码 MIT，权重许可以 HuggingFace model card 为准；GTSinger 检索显示 CC BY-NC-SA 4.0，商用前必须回源确认。本文不是法律意见。
