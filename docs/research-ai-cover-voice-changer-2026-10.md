# AI 翻唱 / AI 变声 技术调研报告

**日期：** 2026-10-16  
**范围：** 基于本项目（IndexTTS-2.5 Pro）与上游 IndexTTS-2.5 体系，调研「AI 翻唱」（歌曲人声 → 角色音色）与「AI 变声」（实时、重音色相似度）的技术路线、SOTA 现状与落地路径。  
**方法：** 本项目源码一手核验（引擎 / 训练栈 / 工具链，含 file:line 证据）+ 网络调研（GitHub 仓库一手 README / Release、arXiv 论文、MVSEP 榜、社区实测）。少数无法直连的论文数字来自搜索摘要，文中已标注「二手来源」。

**两个操作级附录（2026-10-16 补）：** [附录 A](#附录-aindextts-25-原生参数打通可能的接线级调研2026-10-16-补)＝「能不能用这套模型打通参数」（接线级，含权重实锤与验证清单）；[附录 B](#附录-b怎么做操作级配方2026-10-16-补)＝「每一步具体怎么做」（参数速查 / 三档训练配方 / 失败模式对策 / 验收口径）。

---

## 0. 结论速览（TL;DR）

1. **两个功能确实是同一内核**：都是「保留语言学内容 + 音高旋律 + 节奏，替换音色」。翻唱 = 离线高质、输入是成品歌；变声 = 在线低延迟、输入是麦克风。**一套模型、两种运行时**的工程判断成立。
2. **主引擎推荐 RVC 系**：MIT 许可、显式 F0 条件、同时具备实时与离线两条运行时、官方 2026-07 仍在发版（2.3.260718：VST3 插件、CUDA Graph 加速、FCPE 音高、PyMSS 分离后端），模型生态最大。零样本场景补充 **Seed-VC / HQ-SVC / Vevo2（Amphion）**。
3. **IndexTTS-2.5 本体不适合直接改造成 SVC/VC**（一手核验）：s2mel/CFM 的 `f0` 形参存在但**恒传 None**，`length_regulator` 与 DiT 的 `f0_condition=false`——**全链无显式音高建模**，歌声旋律无保障；输出上限 22050 Hz；GPT(AR) + 25 步 CFM 的算力结构决定了它做不了 <400ms 的实时。**但它的外围工程层恰恰是翻唱链路最缺的**（数据集/切片/降噪/评分器/BoN/DPO/高频治理/超分/清理/日志）——复用后本项目可以做出市面工具没有的「**带自动评测与择优的翻唱工作台**」。
4. **分离侧 2026 默认栈**：BS-Roformer(124 band) → karaoke 模型（主唱/和声）→ dereverb Mel-Band Roformer；用 UVR5 / MSST / python-audio-separator 落地，全链 MIT 家族。
5. **变声实时延迟的现实区间是 150–800ms**，不是论文里的 20–100ms：RVC 官方宣称端到端 170ms（ASIO 90ms），独立实测 RVC 817–1034ms、Beatrice v2 427ms；VCClient 官方口径 ≈ `2×chunk + fade + 10ms`，约 500ms 现实预期。研究级方案（StreamVC 70.8ms 闭源、LLVC 19.7ms MIT、DualVC2/3）多为原型。
6. **数据门槛**：RVC ≥10 分钟干净干声（10–50 分钟最佳）。本项目已有角色数据集、切片链与择优体系，但**训练集被重采样到 22050**（`webapp/training/oneclick.py:833`），SVC 训练需要新增 44.1k 分支。
7. **合规红线在「声音来源」与少数模型**：RVC/Amphion/Mel-Band RoFormer 为 MIT 家族；**Seed-VC 是 GPL-3.0**；A2SB 仅非商用；部分去混响权重许可未声明；bilibili 模型协议含百 MAU / 10 亿元营收门槛，且**禁止用模型产物改进其他 AI 模型（非商用除外）**——用 IndexTTS 合成数据训 SVC 模型前要读这一条。建议产品定位「个人研究 / 本地使用」。

---

## 1. 本项目现状：与两个功能的距离

### 1.1 引擎事实（一手核验，file:line 见括注）

| 环节 | 事实 |
|---|---|
| 语义 token | w2v-BERT 2.0（24 层，取 layer17 连续特征，50Hz）→ EnhancedCodec/ResidualVQ（码本 8192，downsample ×2）→ **25Hz**（`indextts/infer_v2_5.py:280,174`；`indextts/codec/models.py:39,93-99`） |
| s2mel / CFM | EnhancedCodec.decode（codes 升 ×2 → 50Hz）→ length_regulator → DiT Flow Matching（**25 步** Euler，cfg 0.7）；条件 = mu（参考连续特征 + 目标 token 条件）、ref_mel **前缀**（生成后丢弃）、style（campplus 192d）（`infer_v2_5.py:829-845`；`s2mel/modules/flow_matching.py:31,79-104`） |
| **F0 / 音高** | **无显式建模**：`CFM.inference(..., f0)` 恒传 None；`length_regulator`/DiT `f0_condition=false`（配置里留有休眠的 f0 开关与死代码）（`flow_matching.py:43`；`config.yaml:73,88`；`length_regulator.py:15-26`） |
| 声码器 | BigVGAN，输出 **22050 Hz**；mel 80 bins / hop 256 / 86.13 fps（`infer_v2_5.py:228-232,740`） |
| 参考音频 | 三路注入：w2v-BERT 连续特征（**不量化**）、22k ref_mel、campplus 192d style（`infer_v2_5.py:619-661`） |
| audio→token | 组件齐全但无封装：`get_emb` → `EnhancedCodec.quantize`（`get_scode` 未被调用）；MaskGCT 遗留工具 `s2mel/wav2vecbert_extract.py` import 已坏（`codec/models.py:179,291`） |
| 续写钩子 | GPT `inference_speech(input_tokens=...)` 支持 token 续写，`infer_v2_5.py` 从不传（`gpt/model_v2.py:776-791`） |
| 流式 | 仅**段级**（每段 yield + 段间静音），段内无流式；低显存自动分块（`infer_v2_5.py:508,859-889`） |
| 加速栈 | `accel/`：Triton 融合注意力、CUDA Graph、paged KV、torch.compile；`backends/trt/`：**完整 TensorRT/TensorRT-LLM/Triton 加速与流式管线（TTFA）**，9 个 ONNX 导出脚本——**与 webui_app 零接线**，且其验证环境是 4090 24GB（`backends/trt/README.md`） |
| 显存 | 常驻 4.94GB / 峰值 5.24GB；与训练、AudioSR(5.5GB) 三向互斥（`webui_app/services/engine.py:10-12,227-232`） |

**一句话**：引擎的「内容 token 化」能力现成，但「音高控制」与「实时流式」是结构性缺口——这决定了自研 SVC 只能走研究线，产品线必须引入专门的 SVC 引擎。

### 1.2 可直接复用的资产（对翻唱/变声的用法）

| 现有资产 | 在翻唱 / 变声中的用法 |
|---|---|
| 一键三连切片 / 优化 / SenseVoice 对齐 | 目标角色 **SVC 训练集构建**（需新增：44.1k 分支、去 BGM/去混响前置、音域覆盖体检） |
| 音频工作台（体检 / 智能切片 / 分频段降噪归一 / 音色库） | 训练素材与参考音频治理；分频段降噪对 SVC 素材同样优于整带降噪 |
| WORLD 链（harvest F0 统计、F0 音域恢复、韵尾延长） | 歌声**音域分析**、F0 相关评测指标、后处理（`audio_lab.py:1620-1649,2498-2565,2257-2426`） |
| 频谱画像对齐（`match_band_profile`，浊音门控，±4~5dB） | 翻唱输出向角色本人频谱画像靠拢（治「换声后频谱不贴」） |
| HF 治理（间隙湿爆冲 de-esser / 浊音撕裂限幅） | SVC 输出的齿音、爆冲、撕裂伪影治理（SVC 常见伪影，本项目的既有阈值可直接迁移） |
| 呼吸库（检测 + 按概率插入） | 翻唱气口真实感（当 SVC 把源气口削掉时兜底） |
| AudioSR 超分（48k；本项目实测 16.37s 音频耗时 291s ≈ **17.8× 实时**） | 翻唱**成品**高频重建（离线可接受）★ 变声实时**不可用** |
| reward.py（SenseVoice WER / campplus SS / emotion2vec / pause） | 翻唱候选评分基座（需扩展唱歌指标，见 §5.1.3） |
| BoN / DPO / A-B 评测 / checkpoint 锦标赛 | SVC 候选（checkpoint、音高偏移、参数组）的**自动择优与偏好对齐**——与 SVCC 2025 冠军的 SFT→DPO 思路同构 |
| 模型下载器 / 清理页 / 调试日志 / 泛化保护 | 新模型（分离、RVC、RMVPE）纳入既有治理与错峰排程 |
| **仓库内 RMVPE ONNX 代码**（`indextts/s2mel/modules/rmvpe.py`，当前是死代码） | SVC / 变声 F0 提取器的现成起点（`assets/rmvpe/rmvpe.pt` 即可用） |

### 1.3 缺口（需要新增的工程量）

1. **分离 / 去混响模块**（BGM、和声、混响三件事，本项目现在一件都没有）；
2. **SVC 推理引擎**（RVC / Seed-VC / …）与显存排程接入；
3. **SVC 训练器**（RVC 训练封装）+ 44.1k 数据集分支（`oneclick.py:833` 现在重采样到 22050）；
4. **实时音频 I/O 与路由**（变声：设备枚举、回环、虚拟声卡指引、延迟测量）；
5. **混音 / 对齐 / 湿声**（转换后人声贴回伴奏：干湿比、对齐、母线响度）；
6. **唱歌适配的评测指标**（F0 轮廓一致性、八度错误、稳定段比例；ASR 歌词可懂度只可作相对比较）。

---

## 2. 技术本质拆解（为什么两个功能是一件事）

任何 VC/SVC 系统都在做四要素的「留三换一」：

| 要素 | 载体（技术） | 翻唱 | 变声 |
|---|---|---|---|
| 语言学内容 | 内容编码器（ContentVec / HuBERT / Whisper / w2v-BERT）或 ASR+文本 | 保留 | 保留 |
| **音高旋律（F0）** | RMVPE / FCPE / CREPE 显式条件（RVC 系、Seed-VC SVC、HQ-SVC），或隐式存在于 token（IndexTTS、Vevo 系） | **必须保留**（含颤音/滑音） | 保留（语调连续性） |
| 节奏时值 | 帧级对齐 / 时长调节 | 保留（与伴奏对拍） | 保留 |
| **音色** | 说话人嵌入（campplus / ECAPA / 参考 mel / 检索索引） | **替换** | **替换** |

差别只在约束：翻唱离线（质量优先、可多候选择优）、变声在线（**延迟预算 <300~400ms**、稳定优先）、且变声对「源音色泄漏」更敏感（实时无法用长上下文）。

**流派分类学**（后文按此展开）：

- **A. 内容编码器 + 声码器**（判别式）：so-vits-svc、**RVC**、DDSP-SVC —— 训练便宜、可实时、F0 显式；
- **B. Codec-LM**（自回归 token）：Vevo/Vevo2、OneVoice、S²Voice —— 质量上限高、可承载风格，重且多为离线（IndexTTS 属同族）；
- **C. 扩散 / Flow Matching**：Seed-VC、HQ-SVC、YingMusic-SVC —— 零样本能力最强、可 F0 条件、步数是延迟旋钮；
- **D. SVS 合成**（DiffSinger / OpenUTAU / Vevo2 Text-to-Singing）：不是「转换」而是「给歌词+旋律重唱」，丢表演，只适合原创向。

---

## 3. AI 翻唱链路调研

### 3.1 人声分离（第一步，决定上限）

2026-10 的梯队（数字来源已标注；不同测试集不可直接比大小）：

| 模型 | 关键数字 | 许可 | 备注 |
|---|---|---|---|
| **BS-Roformer 124 bands（2026-07 新）** | MVSEP 榜 vocals SDR **12.3339**（镜像页二手来源） | MIT 家族 | 当前 MVSEP 第一梯队 |
| BS-Roformer ViperX-1297（`model_bs_roformer_ep_317_sdr_12.9755`） | 社区 40 首中位 vocal SDR ≈ **12.10** | MIT 家族 | **UVR 社区事实默认**，稳、快、对无损输入最优 |
| Mel-Band RoFormer（L=9） | MUSDB18-HQ vocals SDR **11.60**（论文 Table 1） | MIT 家族 | 人声最优格的论文基线 |
| SCNet-L | MUSDB18-HQ vocals SDR **10.86** | MIT | 42M 参数，CPU 便宜 |
| MDX23C | MSST 表 vocals SDR **10.17** | MIT 家族 | 老而稳（UVR 默认档） |
| HTDemucs（f.t.） | MUSDB18-HQ vocals SDR **9.20** | MIT（仓库已归档） | 便宜；`--segment 8` 可 3GB 显存跑 |
| BSMamba2（2025-08） | 论文称比 BS-Roformer uSDR +0.41 | — | 新架构实验（Mamba2 替 Roformer 块） |

**和声处理**：再用一个 karaoke 模型（Gabox/Aufr33 Mel-Roformer Small Karaoke）分主唱 / 和声，主唱走转换、和声单独处理或保留。

**落地工具（免自研）**：
- **UVR5**（MIT，GUI）：v5.6 + Roformer beta，社区推荐链 = `BS-Roformer ViperX` → `Karaoke` → `anvuew dereverb` → `denoise Mel-Band Roformer`；
- **MSST**（ZFTurbo，MIT）：模型动物园 + 训练，`docs/pretrained_models.md` 有全部权重直链；
- **python-audio-separator**（MIT）：CLI/API 封装、自动下载权重——**最适合本项目做服务化集成**；
- **Hyper-RVC**（MIT）：把整条链（分离→karaoke→dereverb→deecho→denoise→RVC→reverb→merge）写成代码的参考实现。

### 3.2 去混响 / 去回声（第二步，决定「干声」质量）

| 方案 | 类型 | 现状 | 许可 |
|---|---|---|---|
| **anvuew dereverb Mel-Band Roformer**（`dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt`） | Roformer | **社区首选**（UVR issue #1700 推举）；训练数据用随机参数 ValhallaVintageVerb 生成；附带能剥掉非居中混入（和声残留） | **权重许可未声明（红线，需自查）** |
| UVR DeEcho 三件套（Normal / Aggressive / DeReverb） | VR 架构 `.pth` | UVR5 自带；DeReverb 最狠也最慢（≈2×）；高频 plate 混响吃力 | UVR 仓库 MIT；**权重许可未声明** |
| Sucial Dereverb-Echo MB-Roformer | Roformer | MSST 表 SDR **10.0169** | 未声明 |
| DPDFNet（Ceva-IP） | 因果 DPRNN+DeepFilterNet2 | **Apache-2.0**、`pip install dpdfnet`、ONNX、CPU 流式 | 实为降噪/增强，无显式去混响训练声明 |
| WPE（nara_wpe） | 线性预测 | 多通道/阵列向，**不适合单声道混音** | MIT |
| TIGER-DnR（2025） | 分离+去混响联合 | Apache-2.0，语音向，EchoSet 数据 | Apache-2.0 |

### 3.3 歌声转换（SVC）核心：四流派与 2026 战况

**最权威坐标：SVCC 2025（Singing Voice Conversion Challenge 2025，33 个系统）**
——赛道从「歌手身份转换」升级为「**演唱风格转换**」；结论：最难的是 **气声 37.3% / 滑音（glissando）42.6% / 颤音 43.9%** 三类风格；**冠军 S²Voice**（ASLP@NPU + 字节）建立在 **Vevo1.5** 之上：FiLM 式 layer-norm 风格条件 + style-aware 交叉注意力 + 全局音色嵌入 + **SFT→DPO 多阶段训练**（~500h 清洗歌声；DPO 只作用于 AR LLM）。值得注意的细节：**DPO 阶段平均分略降但失败模式（重复、突断、抖动）显著减少**——与本项目「BoN 择优 → DPO 蒸馏」的经验完全互证。

| 系统 | 流派 | 零样本 | F0 条件 | 实时 | 数据门槛 | 许可 | 备注 |
|---|---|---|---|---|---|---|---|
| **RVC**（Retrieval-based VC） | A | 否（需训练，≥10min） | ✅（RMVPE/FCPE） | ✅ 成熟 | 10–50min 干声 | **MIT** | ContentVec + 检索索引（faiss）+ NSF-HiFiGAN；2026-07 仍在发版：VST3 插件、实时 GUI CUDA Graph（算法延迟最多 **4.7×** 加速、显存 −20~30%，4090D 实测）、FCPE、PyMSS 分离 |
| so-vits-svc / -fork | A | 否 | ✅（CREPE） | ✅（fork） | 10min+ | AGPL-3.0 | **已停止维护**（fork README 明确），新项目不建议 |
| **Seed-VC** | C | ✅（1–30s 参考） | ✅（SVC 需 `--f0-condition True`） | 部分 | 0（微调可 1 句/100 步/2 分钟 T4） | **GPL-3.0** | v1：25M/22050 实时款、98M/22050 离线、**200M/44.1k SVC 款**；v2：hubert-BSQ-VAE（ASTRAL 量化）+ 67M CFM + 90M AR，**源音色抑制最好**；实时 GUI 无 F0 条件（**仅语音**）；实时延迟见 §4.1 |
| **HQ-SVC**（AAAI 2026） | C | ✅ | ✅（RMVPE） | ✗ | 训练 <80h / 单卡 3090 | **未声明**（无 LICENSE 文件） | 冻结 FACodec 解耦 + EVA（F0/能量/相位 FiLM 融合 + 说话人对比损失）+ DDSP→mel→扩散→NSF-HiFiGAN **44.1k**；推理码+权重已放（2025-12-24），**训练码未放**；自带 16k→44.1k 超分 |
| **Vevo1.5 / Vevo2**（Amphion） | B | ✅ | ✅（旋律控制） | ✗ | 0（预训练） | 代码 MIT；**权重许可需自查** | Vevo1.5：AR 780M + FM 350M + Vocos 250M；Vevo2：统一 TTS/SVS/**风格保留与风格转换的 VC/SVC**/编辑/哼唱转唱；**显存**：fp16 参考 15s ≈6GB、30s ≈10GB、45s ≈12GB（HF 镜像口径，二手） |
| **OneVoice**（IJCAI 2026） | B | ✅ | ✅（F0 条件，sigmoid 门控融合） | ✗（明确非流式） | 0 | 代码待放 | 统一 LVC/EVC/SVC：因果 LM + VAE-free next-patch 扩散（DiTAR 式），**条件 MoE**（1 共享 + 6 LoRA 专家，top-2），patchify 到 10Hz；**2 步 MeanFlow、A100 RTF 0.3** |
| **YingMusic-SVC**（2025-12） | C | ✅ | ✅ | ✗ | 0 | 待查 | **基于 Seed-VC**，改用 rectified flow + **Flow-GRPO 强化学习后训练**（奖励 = 音色余弦 SIM + WER 可懂度 + Audiobox 美学），在「带伴奏/和声污染」输入上显著优于基线——与「输入不完美」的真实场景对齐 |
| S²Voice（SVCC'25 冠军） | B | ✅ | ✅ | ✗ | 500h 清洗歌声 | 未开源 | Vevo1.5 + 上述四件套；示范了「**基座 + LoRA/全参微调 + DPO**」在 SVC 的性价比 |
| so-vits-svc 系云端残党 / Utai | 混合 | 混合 | ✅ | ✗（Utai 离线） | — | 混合 | **Utai**（Windows 离线管线）：113 个分离模型 + 改造版 RVC + Score2HuBERT + OpenUTAU 集成 |

**选型判断（8GB 显存 + Windows + 已有角色素材）**：
- **主力 = RVC**：唯一同时满足「可训练到高相似度 + 可离线高质 + 可实时 + MIT + 维护活跃」的选项；训练数据本项目已经有（角色数据集），只需 44.1k 分支与去混响前置。
- **零样本快速通道 = Seed-VC v1 200M（离线 SVC，44.1k，F0 条件，30–50 步）**：10 秒参考即可试跑，适合「先听效果再决定训不训」；注意 GPL-3.0 与实时仅语音。
- **对标上限 = HQ-SVC / Vevo2**：质量/零样本上限，但 8GB 卡上 Vevo2 勉强、HQ-SVC 训练码未放——列为观察项。

### 3.4 后处理（决定「像成品」而不是「像 demo」）

| 环节 | 方案 | 备注 |
|---|---|---|
| 高频重建 | AudioSR（MIT，6GB，本项目实测 17.8× 实时）/ AP-BWE（MIT，4090 292× 实时，含相位扩展）/ NovaSR·LavaSR（Apache-2.0，KB~MB 级，作者称 5000× 实时）/ A2SB（**仅非商用**） | RVC/Seed-VC 多为 44.1k；IndexTTS 链是 22050——翻唱混音建议统一 44.1/48k |
| 音准 / 节奏 | RVC 自带 F0 跟踪已处理大部分；残留可用 WORLD（本项目已有 harvest/cheaptrick/d4c 全链）做**F0 轮廓移植**兜底，或外部 Auto-Tune | 本项目差的只是「轮廓映射」那一小段代码 |
| 混音 | 与伴奏对齐（分离时保留相位/位移；或互相关对齐）+ 干湿比 + 人声 EQ/齿音削减 + 母线限幅 | Audacity 手工流程可自动化进管线 |
| 频谱一致性 | **本项目的 `match_band_profile`（浊音门控）**向角色画像对齐 | 市面工具没有这一层 |

### 3.5 现成端到端工作流（对标与借鉴）

社区标准链（Hyper-RVC 的 `full_inference_program()` 即此）：

```
分离(BS/Mel-Roformer) → 主唱/和声(Karaoke) → 去混响(dereverb) → 去回声 → 降噪
→ RVC 转换(可带 pitch shift) → 和声处理 → 加混响 → 音准修正 → 合轨
```

本项目做这套的差异化不在链路本身，而在：**页内可视化 + 自动评分 + 多候选择优 + 角色音色档案 + 产物治理**——即已有的一切工程能力。

---

## 4. AI 变声（实时 VC）调研

### 4.1 延迟：论文口径 vs 实测口径

| 系统 | 分块 | 延迟（声称） | 延迟（实测/现实） | 硬件 | 状态/许可 |
|---|---|---|---|---|---|
| RVC 官方实时 GUI | 可变（128–256ms 档） | **端到端 170ms；ASIO 90ms** | VCClient+RVC 独立实测 **817–1034ms**；官方口径 ≈`2×chunk+fade+10ms` | RTX 级 GPU | MIT，2026-07 发版；**CUDA Graph 加速 4.7×** |
| RVC VST2/VST3 插件 | DAW 缓冲级 | —（宿主内接近零算法延迟路径） | 依赖 DAW 缓冲设置 | GPU | MIT（2.3.260718 新增） |
| VCClient（w-okada） | 多档 | `2×chunk+fade+10ms` | **≈500ms 现实**；Beatrice v2 **427ms**（0.05s chunk） | CUDA/ONNX/CPU | MIT 家族；v2 支持 RVC + Beatrice；RVC **ONNX 比 .pth 更快** |
| Seed-VC 实时 | block 0.18s + fade 0.04s + 左上下文 2.5s | **430ms**（≈算法 300 + 设备 100） | 每块推理 150ms（10 步） | RTX 3060 Laptop | GPL-3.0；**仅语音（无 F0 条件）** |
| StreamVC（Google） | 20ms 帧 + 2 帧前瞻 | **70.8ms**（算法 60 + 计算 10.8） | — | 手机 CPU | **闭源** |
| LLVC | 16 样本前瞻 | **19.7ms** | — | i9 CPU（RTF≈2.8× 实时口径） | **MIT**，预训练 `f_8312`；质量较旧 |
| DualVC2 / DualVC3 | 160ms / 20ms | 186.4ms / ~50ms | — | 单核 CPU | 论文原型 |
| Conan（ASRU'25） | 80ms（20ms 快档） | 未给总数 | — | — | 论文；Emformer + 因果 shuffle 声码器 |
| Beatrice v2 | 0.05s chunk | — | **427ms** | **CPU** | **专有、免费使用**（非 MIT） |

**结论**：给「游戏/直播」用的实时变声，**200ms 以内才「无感」**；RVC + 优化（ONNX/CUDA Graph/chunk 128/F0 轻量档）在 4060 上现实预期 **≈200–450ms**；要 <100ms 只有流式专用模型（LLVC/StreamVC 类）或 VST3-in-DAW 路线，代价是音质与相似度。

### 4.2 流式架构模式（自研实时运行时要用到的工程要点）

1. **分块因果**是基线，但纯因果卷积在块边界出 click——DualVC2 明确改用「非因果卷积 + 动态 chunk mask」；**工程落地普遍用 SOLA 对齐 + 交叉淡化**（RVC 系即此）。
2. **内容编码器**需流式化（因果 HuBERT / Emformer / k-means 软单元），否则每块重算历史。
3. **声码器**：因果 HiFi-GAN 可行；**因果 BigVGAN 被报告不收敛**（DLL-APNet, 2025）——不要指望把 BigVGAN 直接因果化。
4. **F0 必须跨块连续**：歌声尤其（颤音在边界会「抖断」）。RMVPE 90.4M、RTF 0.033（4090）；**FCPE 10.6M、RTF 0.0062（快 ~5×）**，精度略低（MIR-1K RPA 96.79% vs 97.77%）——**RMVPE 仍是实时稳妥默认**（仓库里已有现成 ONNX 代码）。
5. **检索索引是隐藏成本**：RVC faiss `index_rate>0` 时 nprobe 与延迟成正比（有 4090 上 +1s 的 issue 报告）；实时建议 `index_rate 0–0.3`、nprobe=1，或干脆关索引。
6. **输入侧**：宽带噪声会被 HuBERT 当内容（fork README 明示），实时链路前挂降噪（RTX Voice / 轻量 NS；注意自建 NS 的延迟预算 <10ms）。

### 4.3 Windows 落地（音频路由与工具）

- **VCClient**（w-okada）：`std` / `cuda_win` / `onnx_win` 三版；v1.5.3.7 起「服务器设备模式」分出 Output（→ VoiceMeeter / VB-Cable 供 Discord/OBS）与 Monitor（→ 直连耳机），**三设备必须同一采样率（通常 48k）**；WASAPI loopback 可捕桌面声。
- **延迟测量**：双通道拍手回环法（物理麦 + 虚拟麦同时录），或 so-vits-svc-fork 的「passthrough 原声对照」；**VCClient GUI 的 ms 只是算法延迟**（issue #497：GUI 500ms vs 实际 4s），自建 UI 必须自带回环测量。
- **自建 Runner 的库选择**：`rvc-next`（PyPI，重写为单一应用：文件转换 + 实时 + 分离 + 训练，含 level meter / monitor / 设备恢复）或 `uziproj/rvc`（Python API + ONNX 导出，2026 仍在提交）——都比直接嵌 RVC WebUI 干净。
- **VST3**：RVC 2.3.260718 的官方 VST 插件让「DAW 内变声」成为最省延迟的一条路（宿主缓冲即延迟）。

### 4.4 实时音色相似度手段

1. **每角色训练模型**（RVC）——相似度与稳定性最强，且可离线/实时共用；本项目角色数据集是现成弹药。
2. **零样本参考**（Seed-VC realtime，430ms）——免训练、但仅语音；**变声场景如果只要「换音色说话」，它是唯一免训练的成品级选项**。
3. faiss 检索增强相似度但加延迟（§4.2-5）；speaker embedding 每参考只算一次（缓存即可，非瓶颈）。

---

## 5. 落地到本项目：融合方案

### 5.0 总体架构（一套内核、两个页面、共享治理）

```
                    ┌──────────────── 新增（薄） ────────────────┐
翻唱台 🎤  歌曲 → 分离 → 去混响 → [降噪] → SVC(离线·多候选) → 后处理 → 混音 → 成品
变声台 🎧  麦克风 ───────────→ SVC(实时·单遍·低延迟) ────→ 监听/虚拟声卡输出
                    └──────────────────┬──────────────────────┘
                                       │  共用
        SVC 内核(RVC 优先) · 角色 SVC 模型库 · F0 提取(RMVPE, 仓库已有)
                                       │
   本项目已有治理层：数据集/切片/降噪 · reward 评分器 · BoN/DPO · A-B · 模型下载
                    产物清理 · 调试日志 · 泛化保护 · CPU 并行 · 进度/心跳/停跑
```

**为什么这是本项目的差异化**：市面工具（UVR5/RVC/Utai）没有「**自动评分 + 多候选择优 + 角色档案 + 全量治理**」。把 SVC 的随机/可调维度（checkpoint、pitch shift、index_rate、扩散步数、种子）交给现有 BoN 框架，用「campplus 相似度 + F0 轮廓一致 + 伪影指标」当 reward——这就是把「以时间换质量」和「SFT→DPO」的成功经验平移到翻唱（SVCC 2025 冠军的同款打法）。

### 5.1 方案 A（推荐）：RVC 内核 + Pro 外壳

**A-1 新增服务（`webui_app/services/`）**

| 模块 | 职责 | 复用 |
|---|---|---|
| `separation.py` | BS/Mel-Roformer → karaoke → dereverb（走 `python-audio-separator` CLI 或直接 torch；权重入 `checkpoints/separators/`） | model_fetcher 审计/镜像/续传；progress_hub 进度；cleanup 登记 |
| `svc_engine.py` | RVC 推理双运行时：离线整首（长上下文 + index）与实时分块（chunk 128–256、SOLA、F0 连续） | engine 生命周期与显存互斥 guard；日志体系 |
| `cover_pipeline.py` | 链路编排（分离→去混响→SVC→后处理→混音） | orchestrator 的 runner/停跑/心跳/旁车审计模式 |
| `mixdown.py` | 伴奏对齐、干湿比、齿音削减、母线限幅/响度 | audio_lab（滤波器/响度归一已有） |

**A-2 训练侧（`webui_app/training/`）**

- `svc_dataset.py`：数据集**44.1k 分支**（`oneclick.py:833` 现在统一重采样到 22050——SVC 训练必须旁路保留高采样率）+「SVC 就绪」体检（去混响程度、单说话人、音域覆盖 2 个八度以上）；
- `rvc_trainer.py`：封装 RVC 训练 CLI（子进程 + 进度解析），沿用 guard 的**保险库/漂移体检/早停语义**与 runner 事件回调；
- `svc_tournament.py`（或并入 checkpoint_tournament）：用**新唱歌 reward** 对 checkpoint 排名——与现有锦标赛同构。

**A-3 评测扩展（新 `reward_sing` 指标组）**

| 指标 | 实现 | 注意 |
|---|---|---|
| 音色相似 SS | campplus 余弦（已有） | 跨域可用，主指标 |
| **旋律一致性** | 源/输出各提 F0（RMVPE/pyworld），八度对齐后算对数误差、相关系数、稳定段比例 | **新代码**，约一天量级 |
| 歌词可懂度 | whisper/SenseVoice 对输出做 CER，与源文本比 | 歌声 ASR 有系统性偏差，**只作候选间相对比较** |
| 自然度 | UTMOS（可选新模型） | 需新增权重与下载登记 |
| 伪影 | 撕裂率 / 湿爆冲（已有实现，直接复用） | 阈值需对 SVC 输出重新标定 |

**A-4 页面与登记**

- 新页 `tabs/cover_tab.py`（🎤 翻唱台：上传歌 → 链路选项 → 候选表 → 试听/导出/择优/导入 DPO）、`tabs/vc_tab.py`（🎧 变声台：设备、模型、chunk、延迟测量、路由指引）；
- 参数注册表（`params_cover.py`）纳入悬停提示与手册；模型页/清理页/系统页按「新产物三处登记」模式补全（参见阶段 26 的既定模式）。

**A-5 数据与素材**

- 目标角色训练数据 = 现有角色数据集（10–50 分钟干声即可）；**已有 IndexTTS 合成音色的角色**还可直接用合成语音扩量（**注意 bilibili 协议 3.4(c)：用其产物改进其他 AI 模型仅限非商用**）；
- 音域覆盖：对素材做 F0 分布体检（已有 harvest 统计），必要时用现有「F0 音域恢复」思路准备高音区补充素材。

### 5.2 方案 B（零样本快速通道）

| 选项 | 8GB 卡可行性 | 用途 | 红线 |
|---|---|---|---|
| Seed-VC v1 200M（44.1k SVC，F0 条件，30–50 步） | ✅（约 4–6GB） | 「10 秒参考先听效果」；训练前试音色 | **GPL-3.0**（集成方式需隔离或接受传染） |
| Seed-VC v2（22050，源音色抑制最好） | ✅ | 说话稿件的强去源音色 | 同上 |
| HQ-SVC（44.1k，零样本） | ✅ 推理 | 质量对照 | **权重/代码许可未声明**（无 LICENSE 文件） |
| Vevo2（SVC 走 AR+FM） | ⚠️ 勉强（参考 15s ≈6GB，30s ≈10GB） | 上限对照 / 风格转换研究 | 权重许可自查 |

### 5.3 方案 C（研究长线）：让 IndexTTS 自己会唱 / 能转

三条具体路线（按可行性排序），均建立在已核验的代码事实上：

1. **C1 F0 移植后处理（工程量最小）**：输出音频 WORLD 分析（sp/ap）+ 源歌 F0 轮廓八度对齐后映射 → 重合成。项目已有 harvest/cheaptrick/d4c 全链（`audio_lab.py:2498-2565` 即此结构），缺的只是「轮廓映射」；可给 RVC 输出做**兜底修准**，但整曲 WORLD 重合成有「塑料味」。
2. **C2 F0 注入解锁（中等）**：配置里 `length_regulator.f0_condition` 与 DiT 的 f0 开关**已存在（休眠）**，`CFM.inference` 的 f0 形参也在——给 s2mel 接通 F0 输入并用现有 **CFM-LoRA 训练器**在少量歌声上微调（特征链 mel/mu 现成、CFM-LoRA 本就只需音频不需文本，`features.py:279-294`）。风险：动上游代码（与本项目「引擎零改动」红线冲突，只能做**实验开关 + 旁路**）。
3. **C3 SoVITS 式 VC（研究）**：用引擎现成组件拼 VC——`get_emb` 提源音频连续特征 →（稳妥：`EnhancedCodec.quantize`→`decode`，25Hz 在分布内）→ CFM 以目标 ref_mel + campplus style 为 prompt → BigVGAN。优势：内容/音色与 IndexTTS 生态同源（角色音色可复用现有 LoRA 资产）；劣势：**22050 上限 + 无 F0 保证**，需小规模适配训练验证。

**判断**：C 线**不作为产品主线**（RVC 已解决 90% 问题），但 C1 可作为「音准兜底」进产品，C2/C3 作为差异化研究项（若成功，「同一角色音色从说到唱全在本项目内闭环」是独有卖点）。

> **接线级细节见 [附录 A](#附录-aindextts-25-原生参数打通可能的接线级调研2026-10-16-补)**：内容/音色/时长三轴是「只接线」问题（含精确到行号的通路与伪代码），音高轴是「训练或后处理」问题（s2mel.pth 权重实锤：284 个参数键中 F0 相关 0 个）。

### 5.4 变声实时选型（推荐配置）

| 项 | 推荐 | 依据 |
|---|---|---|
| 内核 | **RVC**，ONNX 或 CUDA Graph 路径 | ONNX 在 VCClient 实测更快；**CUDA Graph 最高 4.7× 算法加速**（官方 4090D 数据） |
| chunk | **128–256ms**（先 256 起调） | 延迟 ∝ chunk；128 以下收益被固定开销吃掉 |
| F0 | RMVPE（默认）→ FCPE 备选 | 仓库已有 RMVPE ONNX；FCPE 快 5× 精度略低 |
| 索引 | `index_rate 0–0.3`、nprobe=1 | 索引是延迟大项 |
| 输入 | 轻量 NS / RTX Voice | HuBERT 会把噪声当内容 |
| 输出 | VoiceMeeter/VB-Cable（应用）/ 耳机 Monitor 直连 | VCClient 同款双路 |
| 懒加载 | 模型常驻 + 参数热更新 | 参考 RVC 官方实时 GUI 的做法 |
| 延迟预算 | 捕获 10–30 + chunk 128–256 + 推理 20–80 + SOLA 20–40 + 输出 10–30 ≈ **200–450ms** | 与实测区间一致 |

备选：**Beatrice v2**（CPU、427ms、专有免费）作低配退路；**LLVC**（MIT、19.7ms）作「极限低延迟」研究档。**不要把 IndexTTS 接进实时链路**（AR+25 步 CFM+22050，结构上不可能）。

### 5.5 不推荐的路线（附原因）

- **so-vits-svc / -fork 作内核**：官方与主要 fork 均已停维护（fork README 明示）；AGPL 传染。
- **用 IndexTTS-2.5 直接当 SVC**：无 F0、22050、算力结构不支持实时（§1.1）。
- **DiffSinger 类 SVS 作翻唱主路**：需歌词+旋律重唱，丢源表演（换气/语气/自由处理），仅适合原创向。
- **Vevo2 作 8GB 主力**：显存与速度都不匹配（§5.2）。

---

## 6. 路线图

| 阶段 | 内容 | 粗估 | 验收 |
|---|---|---|---|
| **P0 离线闭环**（先跑通再页面化） | 分离/去混响服务 + 模型下载登记；脚本级 分离→dereverb→RVC（社区预训练模型）→AudioSR→手混 | 1–2 周 | 一首完整歌产出 48k 成品；对照样张（原声/干声/转换/成品四件套） |
| **P1 训练闭环 + 翻唱台** | 44.1k 数据集分支；RVC 训练器封装（保险库语义）；唱歌 reward（SS+F0 轮廓+伪影）；翻唱台页面；BoN 多候选择优；角色 SVC 档案 | 2–4 周 | 为一个角色训练出模型：campplus SS 与 F0 轮廓相关达标（初标：SS≥0.8、F0 相关≥0.9，视素材上下浮动）；人工 A/B 通过 |
| **P2 变声台** | 实时音频 I/O、路由指引、回环延迟测量工具；ONNX/CUDA Graph 路径；零样本模式（Seed-VC realtime/离线） | 2–4 周 | 端到端延迟 ≤300ms（非 ASIO 现实目标）；连续 30 分钟稳定；启停/热切换模型不炸 |
| **P3 研究与增强** | C1 音准兜底（F0 轮廓移植）；C2 F0 注入实验（旁路开关）；混音自动化；Vevo2/HQ-SVC 上限对照 | 持续 | 每项独立验收；不达标不合并 |

---

## 7. 风险与合规

1. **bilibili 模型使用许可协议**（本项目 LICENSE）：2.2 条 —— 单月 MAU > 1 亿或年营收 > 10 亿元需单独授权；**3.4(c) —— 不得用本模型或其衍生物改进任何 AI 模型（非商用 AI 模型除外）**：用 IndexTTS 合成语音扩量训 RVC 属该条管辖，商用场景必须回避或获授权。衍生物须传递协议并保留署名。
2. **角色声音 likeness**：达妮娅 / 爱弥斯 / 锁暝等游戏角色音色有权利人（游戏厂商 + 声优），翻唱发行于公开平台需自行评估；建议产品内置「仅供个人研究/本地使用」定位与提示。
3. **歌曲版权**：词曲 + 录音制作者权；分离 → 转换 → 发布 = 完整复制链，个人使用与公开发行法律状态不同。
4. **开源许可**：RVC/Amphion/Mel-Band RoFormer/Demucs/UVR = MIT 家族（安全）；**Seed-VC = GPL-3.0**（隔离集成或接受传染）；**A2SB = 非商用**（默认不引入）；**anvuew dereverb / HQ-SVC / 部分社区 Roformer 权重许可未声明**（引入前自查，或在文档中降级为「可选外部组件」）。
5. **工程风险**：8GB 显存全链互斥（分离 8GB 舒适 / AudioSR 5.5GB / 引擎 4.94GB → 必须排程，复用现有 guard）；歌声 ASR 评分可靠性有限（只做相对比较）；SVC 输出的音区跳变与音色漂移（本项目已有 BoN 决胜与频谱画像对齐的成熟经验，可直接迁移）；实时链路的 F0 边界抖动（颤音）需要专项调参。
6. **数据前置失真**：一键三连当前把素材重采样到 22050——若原库只有 22050 副本，SVC 训练/推理的上限会被钉死在 11kHz 带宽；**P1 的 44.1k 分支是硬前提**（源素材尽量保留 44.1/48k 原件）。

---

## 8. 参考来源

**本项目 / 上游（一手核验）**
- 本项目仓库与 README（IndexTTS-2.5 Pro）· `indextts/infer_v2_5.py` · `indextts/codec/models.py` · `indextts/s2mel/modules/flow_matching.py` · `webui_app/services/audio_lab.py` · `webui_app/training/reward.py` · `backends/trt/README.md`
- IndexTTS-2.5 官方：https://github.com/index-tts/index-tts · 论文 https://arxiv.org/abs/2601.03888
- Faster IndexTTS-2（TRT 加速栈来源）：https://arxiv.org/abs/2607.21042
- 本项目 LICENSE（bilibili 模型使用许可协议）/ LICENSE_ZH.txt

**SVC / 变声**
- SVCC 2025 分析论文：https://arxiv.org/abs/2509.15629 · 主页 https://honee-w.github.io/SVC-Challenge-Demo/
- S²Voice：https://arxiv.org/abs/2601.13629
- OneVoice：https://arxiv.org/abs/2601.18094
- YingMusic-SVC：https://arxiv.org/abs/2512.04793
- HQ-SVC：https://github.com/ShawnPi233/HQ-SVC · https://huggingface.co/shawnpi/HQ-SVC · AAAI DOI 10.1609/aaai.v40i36.40249
- Vevo1.5 / Vevo2 / Amphion：https://huggingface.co/RMSnow/Vevo2 · https://github.com/open-mmlab/Amphion · https://arxiv.org/abs/2508.16332
- Seed-VC：https://github.com/Plachtaa/seed-vc · https://arxiv.org/abs/2411.09943
- RVC：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI （Release 2.3.260718）· RMVPE https://github.com/Dream-High/RMVPE · FCPE https://arxiv.org/abs/2509.15140
- so-vits-svc-fork（停维护声明）：https://github.com/voicepaw/so-vits-svc-fork
- VCClient：https://github.com/w-okada/voice-changer （延迟口径 issue #848、#497）
- StreamVC：https://arxiv.org/abs/2401.03078 · LLVC：https://arxiv.org/abs/2311.00873 · DualVC2：https://arxiv.org/abs/2309.15496 · DualVC3：https://arxiv.org/abs/2406.07846 · Conan：https://arxiv.org/abs/2507.14534
- rvc-next：https://pypi.org/project/rvc-next/ · uziproj/rvc：https://github.com/uziproj/rvc

**分离 / 去混响 / 超分**
- Mel-RoFormer 论文：https://arxiv.org/abs/2310.01809 · MVSEP 榜（镜像）：https://mirror.mvsep.com/quality_checker/entry/10197
- HTDemucs：https://arxiv.org/abs/2211.08553 · SCNet：https://arxiv.org/abs/2401.13276 · BSMamba2：https://arxiv.org/abs/2508.14556
- MSST：https://github.com/ZFTurbo/Music-Source-Separation-Training · UVR5：https://github.com/Anjok07/ultimatevocalremovergui · python-audio-separator：https://github.com/nomadkaraoke/python-audio-separator
- anvuew dereverb：https://huggingface.co/anvuew/dereverb_mel_band_roformer · DPDFNet：https://github.com/Ceva-IP/DPDFNet
- AudioSR：https://github.com/haoheliu/versatile_audio_super_resolution · AP-BWE：https://github.com/yxlu-0102/AP-BWE · A2SB：https://arxiv.org/abs/2501.11311
- Hyper-RVC：https://github.com/BF667/Hyper-RVC · Utai：https://github.com/splinter21/Utai

> 标注「二手来源」的数字（MVSEP 镜像 SDR、Vevo2 显存表、部分论文延迟）来自间接索引，落地前应回源复核；其余数字均自仓库 README / Release / 本项目实测。

---

# 附录 A：IndexTTS-2.5 原生「参数打通」可能的接线级调研（2026-10-16 补）

**问题**：不换引擎，能否在 IndexTTS-2.5 内部把「源音频（歌唱/说话）→ 内容 / 音高 / 音色」的参数通路接出来，让这套模型自己做转换？  
**方法**：逐行核验 `indextts/` 推理链 + 直接检查官方权重（`checkpoints/s2mel.pth`）的张量构成。**本节所有结论都有代码行号或权重级证据支撑；未做真机实验，需实验验证的点已单列（A.5 验证清单）。**

## A.1 四轴总表：哪条轴可以「接线」，哪条不行

| 参数轴 | 现实状态 | 改动性质 | 置信度 |
|---|---|---|---|
| **内容**（说什么/唱什么） | 组件齐全：`extract_features`→`get_emb`→`semantic_codec.quantize`→`.decode` 全链在册，且 `get_scode` 已封装 | **只接线**（旁路 GPT，约 10 行） | 高（代码级） |
| **音色**（像谁） | 槽位现成：`ref_mel` 前缀 + campplus 192 维 `style` + `prompt_condition` 三路同源，全部由「参考音频」决定 | **只接线**（换参考音频即可） | 高（代码级） |
| **时长/节奏** | `target_lengths` 是显式入参，可直接喂「源音频 mel 帧数」实现精确时长 | **只接线** | 高（`infer_v2_5.py:832`） |
| **音高（F0/旋律）** | **权重里零 F0 参数**：`s2mel.pth` 共 284 个参数键，f0 相关 **0 个**；DiT 内无任何 f0 代码；`length_regulator` 的 f0 机器有代码但模块**未被构建**（配置 `f0_condition: false`） | **训练新模块**（或后处理兜底） | 高（权重级） |

**一句话**：说/唱什么、像谁、多长——**能打通，且是纯接线**；唱得准不准（旋律）——**打不通，必须训练或后处理**。这也解释了为什么 RVC/Seed-VC 都非要把 F0 显式塞进条件里：只靠自监督特征承载音高，对唱歌不够（对说话够用，见 A.5 判定线）。

## A.2 内容通路：可接线（现有链路 + 替换点）

现有 TTS 路径（行号自 `indextts/infer_v2_5.py`）：

```
参考音频 → extract_features(:630) → get_emb(:635, w2v-BERT layer17, 50Hz, 已做 mean/std 归一)
文本 → GPT → codes(25Hz, :770-790, max_mel_tokens=1815 即 72.6s 上限)
codes → semantic_codec.decode(:831) → S_infer(50Hz, 1024 维连续特征)
S_infer → length_regulator(:834, ylens=target_lengths, content_in_proj 1024→512)
      → cond(512 维) ；prompt_condition 同函数对参考特征做(:650)
mu = cat([prompt_condition, cond], dim=1)            (:839)
mel = cfm.inference(mu, [mu_len], ref_mel, style, None, 25 步, cfg=0.7)  (:840-844)
mel = mel[:, :, ref_mel_len:]                        (:845)
wav = bigvgan(mel)                                    (:849)
```

**接线就是把 `codes` 的来源换掉**（伪代码，6 行）：

```python
feats = engine.get_emb(*engine_features_from(source_16k))     # 源音频 → 归一化 50Hz 特征
codes = engine.semantic_codec.quantize(feats)[0]              # 变体 A（在分布内）：量化
S_infer = engine.semantic_codec.decode(codes)                 # 回到 50Hz 连续特征
# 变体 B（更细、但分布外）：S_infer = feats 直接用（与 prompt 同域，维数/归一化一致）
S_infer = S_infer.transpose(1, 2) if S_infer.shape[1] != feats.shape[1] else S_infer  # 形状自查
cond = engine.s2mel.models['length_regulator'](S_infer, ylens=[源 mel 帧数], n_quantizers=3, f0=None)[0]
mu = torch.cat([prompt_condition_of_character_ref, cond], dim=1)
mel = engine.s2mel.models['cfm'].inference(mu, [mu.size(1)], ref_mel, style, None, 25, inference_cfg_rate=0.7)[:, :, ref_mel.size(-1):]
wav = engine.bigvgan(mel.float())
```

关键事实与边界（全部已核验）：
- `quantize` 返回 `(indices, quantized_out)`；batch=1 时 indices 会被 squeeze（`codec/models.py:179-203`），decode 接受 `[B,T]` 并 ×2 上采样回 50Hz（`codec/models.py:205-226`）。
- `length_regulator` 对连续输入走 `content_in_proj`（Linear 1024→512）+ 4 层 Conv1d 提升 + `F.interpolate` 到目标帧数（`length_regulator.py:85-141`）；**它同时服务于 prompt 和 target 两个槽位**，两槽输入是同域特征——所以变体 B 在维数/归一化上与 prompt 完全兼容，风险只在「原始特征比 decode 重构更"细"」这一点分布差。
- **时长天然对齐**：`target_lengths` 直接给源音频的 mel 帧数（mel 帧率 86.13 = 22050/256），输出时长即源时长（帧级精度 ≈11.6ms）；引擎自带的启发式是 `int(S_infer.shape[1] × 1.72 × duration_factor)`，其中 1.72 = 86.13/50，走我们的接线时这个系数用不上。
- **单次上下文上限**：CFM 的 KV cache 按 `max_seq_length=8192` 帧建立（`infer_v2_5.py:200`），mel 帧率 86.13 → **约 95 秒（含参考段）**；整首歌需分块（带重叠与交叉淡化，做法与 Seed-VC 实时分块同构）。好消息：**VC 模式绕过 GPT，72.6 秒上限不再适用**（那是 `max_mel_tokens=1815` 对 GPT 的约束）。
- 参考段（`ref_mel` 前缀）会参与每次前向：参考 15s ≈1292 帧，按需可缩到 5–8s 换取上下文与算力余量（参考越短音色锚定越弱，可测）。

## A.3 音色通路：可接线（且角色 LoRA 理论上同步生效）

- 三路音色注入全部来自参考音频（`infer_v2_5.py:626-661`）：`ref_mel`（80 维梅尔前缀）、`style`（campplus fbank→192 维）、`prompt_condition`（参考连续特征过同一个 length_regulator）。**换参考 = 换音色**，与 TTS 完全同一机制。
- 因此本项目既有资产在 VC 路径上**原样可用**：音色库参考、音频工作台处理过的干净参考、角色档案。
- **一个待验证的重要推论**：CFM LoRA 训练器（`webui_app/training/cfm_lora.py`）学的就是这条 s2mel 链的权重（音色），如果它能作用于 VC 路径，则「角色 LoRA 音色直接用于翻唱」不需要任何新训练——即：**你为 TTS 训的角色音色，可能天然就是 SVC 的目标音色**。这需要一次探针验证（见 A.5 实验 4）。
- 音色可切换、发音可懂之外的「音乐性」（气声、颤音的音色细节）不在此轴保证范围。

## A.4 音高通路：不是接线，是训练（三选一）

**权重级实锤**（本会话直接检查 `checkpoints/s2mel.pth`）：
- 284 个参数键，**f0 相关 0 个**；`length_regulator` 下存在 `content_in_proj`（1024→512）与 conv 栈，但**没有 `f0_embedding`/`f0_mask`**；
- `diffusion_transformer.py` 全文无 f0 字样；`flow_matching.py` 的 estimator 调用只有 6 个参数（`x, prompt_x, x_lens, t, style, mu`）——2.5 已把 2.0 时代的 f0 支路整体移除；
- 配置里 `f0_condition: false`（`checkpoints/config.yaml` s2mel 段）是**遗留开关**：即使打开，加载的权重里没有对应参数（随机初始化），必须先训练。

三条路的对比：

| 路 | 做法 | 工程量 | 预期 | 建议 |
|---|---|---|---|---|
| **P1 后处理兜底（零训练）** | 输出音频做 WORLD 分析（`sp/ap`），源歌 F0 轮廓（harvest）映射到输出音区后替换 F0，重合成 | 小：项目已有 harvest/cheaptrick/d4c 全链与「F0 统计/音区」工具，缺的只是「轮廓映射」一小块 | 旋律一定对上；代价是 WORLD 的相位感（整曲重合成有"塑料味"，可只对偏差大的段局部修） | **首选兜底**，与路线无关，RVC 输出同样适用 |
| **P2 最小训练（吃本项目存量）** | 打开 `f0_condition`：新增 `f0_embedding(512×512)+mask` 参数，在 `length_regulator` 调用处传 f0（**代码路径已存在**，只是没权重）；用现有 CFM-LoRA 训练器在歌声数据上微调 | 中：训练前向加 f0 入参 + 特征管线加 F0 提取（RMVPE 代码仓库里就有，`indextts/s2mel/modules/rmvpe.py`）；数据需带 F0 的歌声（公开歌声语料可起步） | 粗粒度旋律跟随可期；音色由参考锚定、内容由 mu 决定，F0 只加"旋律"一轴——**理论上正是 SVC 的标准配方** | 研究线，但复用度最高（8GB 可训，CFM 只有 98M） |
| **P3 DiT 加 F0（上限路线）** | 给 DiT 增加 F0 条件通道并重训 | 大 | 不确定 | 不推荐（上游没有，收益不明） |

## A.5 验证清单（本次未执行；按"先便宜后贵"排序，每项含判定线）

| # | 实验 | 指标 | 判定线 |
|---|---|---|---|
| 1 | **内容/时长直连**：源语音 → A.2 接线（变体 A 与 B 各一次）→ 输出 | 时长误差（mel 帧）、whisper CER、campplus SS（对目标 vs 对源） | 时长帧级对齐；CER<15% 且 SS(目标)>SS(源) 即三轴打通 |
| 2 | **音高跟随（核心判定）**：用 WORLD 给源音频合成一段旋律轮廓（同样的音色与内容、只有 F0 变成"歌"）→ 过同一管线 → harvest 比较 F0 | log-F0 相关系数 r、八度错误率、半音误差 | **r>0.8 且无八度错误** → 隐式音高够用，SVC 仅接线即可；**r<0.6 或频繁八度错误** → 音高必须走 P1/P2。预期结果：介于两者之间（说话段高、旋律段塌） |
| 3 | **实时成本**：1s/2s/4s 分块测 encode / 码本 / CFM 25 步 / BigVGAN 各段耗时与显存 | 每段 RTF、峰值显存 | 总 RTF<0.5 → 准实时（300–600ms）可立项；否则实时只交 RVC 内核。参考数据点：本项目 README FAQ 记录单段 s2mel ≈0.9s；Seed-VC 实时用 10 步 + 更小编码器 |
| 4 | **LoRA 联动**：挂一个 CFM 角色 LoRA → 走接线 → 听/测音色 | campplus SS 变化 | SS 显著上升 → A.3 推论成立，角色音色资产直接复用于翻唱 |

## A.6 对「实时变声」结论的修正

上一版报告说「结构上不可能」——准确的说法应是：**TTS 路径不可能**（GPT 自回归 + 段级流式）。而 VC 路径**绕开了 GPT**：延迟结构变成 `w2v-BERT 编码 + 码本 + CFM(25 步) + BigVGAN`，**全部前馈**，其中 BigVGAN 全卷积可分块、CFM 可分块+重叠（Seed-VC 实时模式即此形态）。所以：

- 「准实时」（约 300–600ms）是**待测量的工程问题**，不是结构性死路——关键未知量是 w2v-BERT 编码成本（0.58B 参数的编码器不轻）与 25 步 CFM 的每块耗时（可用步数蒸馏/减步换延迟）；
- 「真低延迟」（<200ms）仍应交给 RVC ONNX/CUDA Graph 或 LLVC 类专用引擎；
- 无论哪条路，**唱歌的 F0 都需要额外条件**（A.4），"唱"比"说"多一道。

## A.7 结论（回答「能不能基于这套模型打通参数」）

1. **能打通部分**：内容（听/唱什么）、音色（像谁，含角色 LoRA 复用）、时长（帧级精确）——三轴都是接线工程，通路已逐行定位（A.2/A.3），不需要训练、不需要改上游（自建旁路函数即可）。
2. **打不通的部分**：音高（旋律）——权重里不存在这条轴（284 键零 F0 参数）。要么后处理移植（P1，推荐兜底），要么训练开轴（P2，复用现有 CFM-LoRA 与 RMVPE，属研究线）。
3. **工程边界**：单次 8192 帧（≈95s 含参考）→ 整歌分块；VC 模式不受 72.6s 限制；显存与现有引擎一致（4.94GB）。
4. **和 RVC 的关系**：RVC 是「今天就能用」，本路线是「同生态、可与角色 LoRA/情感库/评分体系联动的自持方案」。两者的第一步验证都很便宜（A.5 实验 1/2 各半天量级），**建议先做实验 2 的音高判定**——它一个数就能决定这条路线值不值得走下去。

---

# 附录 B：「怎么做」操作级配方（2026-10-16 补）

**本附录解决的是「怎么落地」而不是「选什么」**：每条链给到「步骤 → 参数 → 产出物 → 检验点」粒度。数字来源标注：〔仓库核实〕= 官方 README/Release/本项目源码实测；〔社区共识〕= 多来源教程一致；〔实验设计〕= 本项目验收线。

## B.1 翻唱端到端配方（离线，六步）

**Step 0 素材与目标准备**
- 目标角色素材（RVC 训练）：10–50 分钟干声，44.1kHz 单声道 WAV〔社区共识〕；说话素材可用（旋律由 F0 条件补），音域不足时用本项目「F0 音域恢复」思路做高音区增广〔实验设计〕；素材治理顺序 = 去 BGM/和声 → 去混响 → 分频段降噪 → 响度中位数锚定（后三步本项目已有）。
- 硬前提：训练/推理素材保留 44.1k（一键三连现在统一重采样到 22050，需 44.1k 分支，见 §5.1）。

**Step 1 分离**：`python-audio-separator`（MIT，自动下载权重，最适服务化）
- 主唱：`BS-Roformer ViperX-1297`（稳）或 `BS-Roformer 124 bands`（2026-07 新，MVSEP 榜首，二手来源）〔社区共识〕
- 主唱/和声：Karaoke Mel-Roformer（Gabox/Aufr33 Small）→ 和声单独保留或轻处理
- 检验点：主唱轨无伴奏残留（听 0–10s 与副歌各一段）、有伴奏轨可作混音基线；两轨**时间轴与源曲完全一致**（分离不改时长，这是后续对齐的前提）。

**Step 2 去混响**：`anvuew dereverb Mel-Band Roformer`（社区首选；注意权重许可未声明）
- 轻混响用 `sdr_18.8050`，重混响用 `sdr_19.1729`；UVR DeEcho 三件套为退路
- 检验点：干声无「水下感/呼吸抽吸」；过度去混响的症状是音色发闷、齿音发脆——**宁可欠一点，残余混响由 Step 6 的统一空间覆盖**。

**Step 3 转换（RVC 推理参数速查）**〔社区共识 + 仓库核实〕

| 参数 | 推荐值 | 作用 / 备注 |
|---|---|---|
| f0 提取 | RMVPE（默认）→ FCPE（大跨度高音/气声更准、更慢） | 2026-07 版新增 FCPE〔仓库核实〕 |
| transpose | 先各自 harvest 统计 F0 中位数再定：同性别 0，男→女 **+12**，女→男 **−12** | 本项目已有 harvest 统计工具，直接算 |
| index_rate | 0.5–0.75（音色优先）；0–0.3（保咬字/实时） | 高值可能吞辅音、带噪 |
| protect | 0.33 默认；吞字 → 升到 0.4–0.5 | 清辅音/呼吸保护，越高音色替换越弱 |
| filter_radius | 3（≥3 抑呼吸毛刺） | 0 关闭 |
| rms_mix_rate | 0.25–0.5（保留源动态，唱歌推荐） | 1 = 全用输出包络 |
| resample_sr | 0（保持模型采样率） | 需要 48k 交给 Step 4 |
| **高音唱不上去/发虚** | 预处理技巧：源整体移调 −2~−4 再转换（转换后 +2~+4 回来） | 〔社区共识〕减轻模型高音外推；或改用 FCPE |

**Step 4 后处理**：齿音/爆冲治理（本项目 de-esser 与浊音撕裂限幅，阈值可直接迁移）→ 频谱画像对齐（本项目 `match_band_profile`，浊音门控，向角色画像 ±4dB）→ 可选 BWE（离线用 AudioSR；追求速度用 AP-BWE/LavaSR）→ 高通 60–80Hz。
**Step 5 混音**：干湿比起步人声比伴奏高约 3dB；母线上限 −1 dBTP；目标响度 **−14 LUFS**（流媒体口径）〔社区共识〕；人声可加侧链压缩让伴奏让位（进阶）。
**Step 6 空间一致性（把「干唱」变成「在歌里唱」）**：把转换后干声送入与原曲相同的混响空间。进阶做法：从「原湿声 − 去混响输出」求残差反卷积估一个 IR〔社区共识，中等置信〕，或直接用同参混响器按听感对齐；这一步是「像成品 vs 像 demo」的分水岭。

## B.2 变声实时配方（一条链，四件事）

1. **模型与运行时**：RVC + ONNX 或 CUDA Graph（官方 2.3.260718 实时 GUI 支持 CUDA Graph，算法延迟最高 4.7× 加速、显存 −20~30%〔仓库核实〕）；chunk 128–256ms；`index_rate 0–0.3`、nprobe=1（索引是延迟大项）；F0 用 RMVPE（仓库里已有 ONNX 代码），追求极限换 FCPE。
2. **输入侧**：轻量降噪（RTX Voice 或自建 <10ms NS）——内容编码器会把噪声当内容〔仓库核实：fork README 明示〕；歌手场景沿用同一链（F0 跨块连续，颤音不断）。
3. **输出路由**：VCClient 式「服务器设备模式」= Output→VoiceMeeter/VB-Cable（给 Discord/OBS），Monitor→耳机直连；三设备同采样率（48k）〔仓库核实〕。
4. **延迟测法（自建 UI 必须内置）**：双通道拍手回环（物理麦 + 虚拟麦同录，量峰值偏移）；注意 GUI 里显示的 ms 只是算法延迟〔仓库核实：VCClient issue #497 GUI 500ms vs 实际 4s〕。预算：捕获 10–30 + chunk 128–256 + 推理 20–80 + SOLA 20–40 + 输出 10–30 ≈ **200–450ms**〔实验设计〕。

## B.3 训练配方（三档，按投入递增）

| 档 | 做法 | 数据 | 超参/时长 | 适用 |
|---|---|---|---|---|
| **L0 零样本（0 训练）** | Seed-VC v1 200M（44.1k，SVC 款）或 HQ-SVC 直接推理 | 参考 10–30s | `--f0-condition True --diffusion-steps 30–50`〔仓库核实〕 | 先验证音色可行性、临时音色 |
| **L1 微调（分钟级）** | Seed-VC fine-tune 到目标说话人 | **≥1 句** | **≥100 步、T4 约 2 分钟**〔仓库核实〕 | 目标素材很少时要相似度跃升 |
| **L2 RVC 全训练（小时级）** | RVC WebUI/CLI 训练 v2 模型 + faiss 索引 | 10–50min 干声，切 3–20s，44.1k mono | epochs 200–500（数据越多越少）；batch 依显存（8GB → 6–8）；F0 用 RMVPE〔社区共识〕 | 长期主力音色；产出 `.pth + .index` 两件套 |
| **L3 生态内进阶（本项目独有）** | 把翻唱候选交给现有评分/择优体系：音色 SS + F0 轮廓 + 伪影三项打分，BoN 选最好；优/劣对导入 DPO | 复用 L2 产物 | 复用 checkpoint 锦标赛与 BoN 框架 | 「以时间换质量」；与 S²Voice 的 SFT→DPO 打法同构（§3.3） |

数据侧口径：RVC 原话「≥10 分钟低噪语音即可训练」〔仓库核实〕；质量 > 数量（一句干净胜过一段嘈杂）〔社区共识〕；**跳过 BGM/混响前置不可省**（否则模型把混响当音色学走）。

## B.4 失败模式 → 对策速查表

| 症状 | 对策 |
|---|---|
| 听得出源歌手音色（tone leakage） | 提高 index_rate；确认 F0 条件开启；补训练数据；或换 Seed-VC v2（ASTRAL 量化，源音色抑制最好〔仓库核实〕） |
| 吞字/咬字不清 | 降 index_rate；升 protect；filter_radius 降到 0–1 |
| 呼吸声/毛刺多 | filter_radius ≥3；输入降噪；本项目呼吸库可反向用于「补正自然气口」 |
| 高音发虚/唱不上去 | 源预移调 −2~−4（转换后移回）；FCPE；训练集补高音区素材 |
| 齿音刺耳/爆冲/撕裂 | 本项目间隙 de-esser + 浊音撕裂限幅（阈值已标定，直接迁移） |
| 输出发闷/缺高频 | 离线 BWE（AudioSR；或 AP-BWE/LavaSR）；确认未在降采样链里丢高频 |
| 混响残留发糊 | 去混响加一档（但警惕水下感）；Step 6 统一空间掩盖残余 |
| 和声串进主唱 | Karaoke 模型分离；转换只走主唱轨 |
| 对拍不准 | 分离/转换都保时长，天然对齐；用互相关校验，**禁止整段 time-stretch**；个别段落手动微移 |
| 音量忽大忽小 | rms_mix_rate 收向 0.25；响度中位数锚定（本项目已有） |
| 音区跳变/音色漂移 | 本项目 BoN 并列组决胜（音区+音色衔接）与频谱画像对齐，直接套用 |
| 8GB 显存互踩 | 分离/AudioSR/引擎/SVC 全部串行排程（复用现有 guard） |

## B.5 验收怎么做

- **客观指标**（全部可挂进现有 CPU 打分管线）：音色 = campplus SS（对目标参考）；旋律 = 源/输出 F0 轮廓相关系数 + 八度错误率 + 半音误差（pyworld/RMVPE）；内容 = 歌词 CER 只作**候选间相对比较**（歌声 ASR 有系统偏差〔仓库核实：reward.py 的 pause/WER 假设说话〕）；伪影 = 撕裂率/爆冲（本项目已有）。
- **建议通过线**〔实验设计〕：SS ≥ 0.80；F0 轮廓相关 ≥ 0.90（说话段）/ 八度错误帧 <5%；无听感级伪影；三人盲听 A/B 偏好不劣于基线。
- **评分器扩展**：`reward.py` 新增唱歌指标组（F0 系），其余复用 —— 这是把「翻唱质量」从主观变成可排名的关键一步。

## B.6 前沿加分项（把 2025–2026 的先进做法落到本配方上）

1. **检索增强**：RVC 的 faiss top-1 检索本质是 kNN-VC 思路，已内建；实时侧用 nprobe=1 控成本。
2. **零样本基座 + LoRA/DPO 二段式**：SVCC 2025 冠军（Vevo1.5 + S₁FT→DPO）的结论可直接迁移——「先微调、再偏好对齐」，且 DPO 的价值在于**减少失败模式而不只是抬平均分**（§3.3）。
3. **RL 后训练**：YingMusic-SVC 的 Flow-GRPO（奖励 = 音色 + 可懂度 + 美学）在「带伴奏污染输入」上收益最明显（§3.3）——列为观察项，等其代码/权重可用。
4. **残差混响卷积**：Step 6 的 IR 估计法（社区技巧，中等置信），比手工加混响更贴原曲空间。
5. **F0 移植兜底**：本项目 WORLD 链（附录 A.4 P1），对所有 SVC 输出都适用，用于修个别走音段。
6. **频谱画像对齐**：本项目独有（市面工具没有），用于「换声后频谱不贴角色」的系统性修正。
7. **源预移调**：高音区的社区标准技巧（B.1 Step 3 表末行）。

> 本附录与 §5.1（模块清单）、附录 A（原生接线）互补：**A 回答「能不能用这套模型打通」，B 回答「用任何内核时，每一步具体怎么做」**。两者合起来构成完整的落地方案。
