# 本次会话开发规划：E1-T 全管线落地 + 推理分块改进（2026-10-17）

> **修订 r1（子代理规划评审后）**：已并入评审 10 条修订——F0 归一化单一职责定稿（r1-1）、
> prompt 帧区段 F0=参考自身 F0（r1-2）、SVC 租约拒绝 cfm 通道 LoRA 挂卸（r1-3）、
> 声码器微调入口骨架（r1-4，WP4b）、technique-balanced 采样与 cycle pitch-shift 增广（r1-5）、
> `infer_v2_5.py:840` 上游 VC 分支不改的显式声明（r1-6）、`tools/e1t_baseline.py` 基线脚本（r1-7）、
> `technique` 参数只追加在 `inference` 签名尾部保位置兼容（r1-8）、GPU 端 TTS A/B 列入用户运行项 +
> 测试计数修正为按实际（r1-9）、merge 白名单走 BaseGuard 参数而非调用侧绕过（r1-10）。
> 另固化评审确认的好性质：DiT 的 class_dropout 会把 `x_in[..., 80:]` 整段置零，
> style_t 落在该区间，**训练期条件 null-dropout 自动覆盖 conditioner 输出**，无需额外代码，测试固化。

**性质：** 一次性会话执行规划，服从 `docs/dev-plan-e1t-2026-10.md`（唯一执行规划）。
**本会话交付的是「代码级全管线」**：模型接线、特征/训练/合并/推理/UI 全链代码 + 测试 + 冒烟脚本。
**不属于本会话的**（需要数据下载与 GPU 训练时长，属用户后续运行项）：
GTSinger 数据下载与 QA、Base capability 实际训练（Step 6）、G1a/G1b 评测报告、
声码器微调（Step 12 的训练部分）、44.1k 分支。所有这些入口都必须在本会话内可用（有 CLI/UI/脚本可跑）。

---

## 0. 范围澄清（用户指令中「自动分块策略」的解释）

用户要求「计划稿内最后一部分关于自动分块策略等部分的改进」。经全文检索，
E1-T 规划本身无「自动分块」章节；该主题位于 `docs/dev-plan-ai-cover-vc-2026-10.md` 的
末尾章节（§11 阶段46：低显存阶段调度与连续性模式，C2 模块级阶段卸载为规划未实现项）
与翻唱侧分块管理（D4）。本规划按以下**超集**理解执行（两部分都做，均为推理基建、可独立回退）：

- **A. 翻唱侧自动分块改进**（`svc_core.py`）：空闲显存感知的动态块长上限、
  切点吸附到 mel 帧栅格、硬切点自适应加大重叠、纯静音块跳过转换（保长）。
- **B. TTS 侧阶段46 C2**（`infer_v2_5.py` 可选钩子 + `engine.py`）：模块级阶段卸载
  （参考特征构建后 w2v-BERT→CPU、GPT 生成完→CPU），由连续性模式联动开关控制，
  默认关，异常自动恢复原状——**只加默认 None 的可选参数，不改默认路径**。

若用户本意不同（例如仅指其中一项），两块互相独立，可单独丢弃，不影响 E1-T 主线。

---

## 1. 硬约束（每一步的验收红线，违反任何一条即回退）

1. **默认 TTS 完全不变**：
   - `checkpoints/config.yaml`、`checkpoints/s2mel.pth`、原 BigVGAN 目录**零字节改动**（hash 对比）。
   - 默认 config（`technique_condition` 缺省=false）构造的模型 state_dict 键集合与改前完全一致（新增键=0）。
   - `BASECFM.inference/forward` 在 `f0=None, technique=None` 时与改前**逐位一致**（CPU 小模型 seeded 张量断言）。
   - `infer_v2_5.infer` 不加参数调用时行为不变（stage_hooks=None 原路径）。
2. **singing 模块三重隔离**：新模块只在 `config_singing.yaml`（`technique_condition: true`）下构造；
   cover 会话内临时换装 singing CFM、结束 finally 恢复引擎原 CFM；identity LoRA 只挂 singing CFM、
   结束 detach。TTS 会话永不加载 `s2mel_singing.pth`。
3. **合并红线**：merge 产物只允许写 `checkpoints/s2mel_singing.pth` 与 `checkpoints/config_singing.yaml`
   这两个**新名字**；`s2mel.pth`/`config.yaml`/BigVGAN 永不覆盖（在合并入口做名字白名单硬校验）。
4. **旧数据兼容**：已有数据集特征缺 `f0/tech` 字段时默认 control/零向量，现有 TTS 音色训练流程不受影响。
5. **可回退性**：所有新功能默认关闭（开关在翻唱页/系统页），关掉即回到当前行为；
   所有上游文件改动仅为「新增可选参数 + getattr 门控」，`git revert` 单提交可整体回退。
6. **开发习惯**：中文注释写「为什么」并引用实锤/红线；测试无 GPU 可跑（mock/纯函数/小张量）；
   产物目录三处登记（清理页模式沿用）；`test_webui_syntax.py` 必须过（tab 真实构建）。

---

## 2. 工作包分解（按依赖排序）

### WP0 · 基线 + TTS 回归护栏（E1-T Step 0）
- 记录 `s2mel.pth`/`config.yaml`/BigVGAN 的 md5 → `tests/baseline_tts_hashes.json`（工具脚本生成，只读记录）。
- 新增 `tests/test_tts_regression.py`：
  a) hash 守卫（文件缺失时 skip，不误报）；b) 默认 config 构造 DiT 无新键；
  c) 默认 config.yaml 仍 `f0_condition: false` 且无 `technique_condition`。
- 产出：护栏测试绿。

### WP1 · 模型接线：TimbreConditioner + CFM/DiT 透传（E1-T Step 4）
- 新文件 `indextts/s2mel/modules/conditioners.py`：`TimbreConditioner`
  （`f0_proj: Linear(1→D)`；`tech_emb: Embedding(7, D)`；
  `style_mlp: Linear(192+2D→D)→SiLU→Linear(D→192)`；`alpha: Parameter(0.5)`）。
  **签名（r1-1 定稿）**：接收**原始逐帧 f0(B,T) Hz（0=清音）**、`technique(B,T)`（0..6）、
  以及逐样本 `(f0_mean, f0_std)` 张量（浊音 log2-F0 统计，训练来自样本 meta、
  推理来自**参考音频** harvest 统计）；归一化在 conditioner 内部完成
  （清音先 mask 再 log，归一化后清音位填 0），collate 不做归一化——单一职责。
  输出 `style_t(B,T,192)`。
- `diffusion_transformer.py`：`DiT.__init__` 读 `getattr(args.DiT,"technique_condition",False)`
  → 仅 true 时构造 conditioner；`DiT.forward(..., f0=None, technique=None, f0_stats=None)`：
  两者皆 None → `style_t = style`（与原逐帧 repeat 完全同一语句路径）；否则 `style_t = conditioner(...)`。
- `flow_matching.py`：`inference/solve_euler/forward` 增加 `f0=None, technique=None, f0_stats=None`；
  **（r1-8）`technique` 等新参数一律追加在签名尾部**（`inference_cfg_rate` 之后），
  不得插在 `f0` 与 `n_timesteps` 之间——`infer_v2_5.py:840` 与 `svc_core.py:267` 都是位置传参；
  CFG null 分支 `stacked_f0/stacked_tech` 拼零（条件在空分支不得生效，null 零输入→零输出自然成立）；
  estimator 调用透传（None 时调用形态与改前一致）。
- `checkpoints/config_singing.yaml`（新）：复制默认 config 的 s2mel 段 + `technique_condition: true`
  + `f0_condition: false`（length_regulator 仍不用）+ `s2mel_checkpoint: s2mel_singing.pth`。
- 新增 `tests/test_e1t_wiring.py`：小 config DiT 上验证——a) 默认路径逐位一致；
  b) 条件生效（输出随 f0/tech 改变）；c) CFG null 零化；d) conditioner 参数量/形状；
  e) 默认 config 无新键；f) class_dropout 触发时 style_t 段被清零（评审确认的自动 null-dropout 性质）。
- 风险与回退：若 `load_checkpoint2(strict=False)` 在默认模型上暴露新键（不应该，因为门控），
  退回「只在 singing config 构造」并加测试断言。

### WP2 · 特征管线：F0 / technique（E1-T Step 2）
- `features.py`：
  - `extract()` 新增可选 `singing: bool = False`（或按数据集标记）：pyworld harvest 5ms（复用
    `svc_f0.harvest_f0` 的整段/分窗策略）→ 线性插值对齐 mel 帧网格（86.13Hz，清音不架桥）→
    存 `f0`（fp32 (Tm,) Hz，0=清音）、`f0_mean/f0_std`（浊音 log2-F0 统计，标量）、
    `tech`（uint8 (Tm,)，默认 0=control）。
  - **FEATURE_VERSION 不动**：新字段进 `SINGING_SCHEMA`，`is_usable` 主 schema 不变；
    `check_feature` 增加 singing 字段校验（存在才查）。
  - technique 标签来源：数据集 meta 可选列 `technique`（逐句）或伴生 `<uid>.tech.csv`
    （GTSinger phoneme 级，时间戳→mel 帧）；无标签全 control。
  - `build_cfm_sample/build_cfm_pair/truncate_*`：透传 f0/tech（截断同步）；
    **（r1-2）配对时 prompt 帧区段的 F0/tech 用 prompt 样本自身的值**
    （训练与推理一致：推理时参考段填参考音频自身 harvest F0，同一归一化统计口径）。
- `cfm_lora.py`：`collate_cfm` 拼 f0/tech/f0_stats（pad 0；归一化统一在 conditioner 内，
  collate 只负责把样本 meta 的 mean/std 带进 batch）；`CfmSamplePool._convert` 已走
  build_cfm_sample 自动透传。
- **（r1-5）增广与采样（config 门控，默认关）**：
  - cycle pitch-shift 增广：`singing_pitch_aug`（半音集合，默认空=关）——对 mel 域不可行，
    在 f0 条件与 audio 重提取代价高，故第一版实现为**条件侧增广**：对 f0 乘 2^(k/12) 并同步
    f0_stats 重建（等价教模型「同一内容跨音区的音色映射」，SPA-SVC 思想的条件侧近似）；
    全音频域 cycle 增广留作用户运行项（脚本位）。
  - technique-balanced sampler：`singing_tech_balance: bool`——按 tech 帧占比加权采样，
    避免 control 淹没 falsetto/mixed。
- 新增 `tools/gtsinger_import.py`：GTSinger → 本项目数据集（音频+文本+technique csv+manifest），
  含许可清单模板与 split 建议（unseen singer / unseen technique / top10% F0）——只做导入工具，不下载数据。
- 测试：`tests/test_e1t_features.py`（合成音频 F0 对齐精度、tech 展开到帧、旧特征缺字段默认值、
  collate 透传、pitch 增广的 f0/stats 同步、balanced 采样分布）。

### WP3 · 训练接线（E1-T Step 6/9 的代码路径）
- `cfm_lora.py`：
  - `load_base_cfm(..., cfg_path=None, pth_name=None)`：支持 `config_singing.yaml` 与
    `s2mel_singing.pth`（默认仍走原路径）；校验 singing config 必须含 technique_condition=true。
  - `CfmTrainOptions` 增 `singing: bool = False`：true 时要求样本带 f0（缺 tech 默认 control），
    `_train_micro_batch` 传 f0/tech/f0_stats 给 `cfm_training_forward`；false 时行为与现在完全一致。
  - **（r1-5 续）辅助损失（可选开关，默认关）**：`tech_ce_weight`（conditioner 输出接 7 类
    判别头的 CE——用 WP5 分类器蒸馏，保条件不塌缩）+ `f0_aux_weight`（mel 域谐波回归，
    第一版先留接口与开关，权重默认 0；identity contrastive/SSIM cycle 明示为用户运行项的
    训练策略实验，不在本会话实现——防单会话改动面失控）。
- `forward.py`：`cfm_training_forward(..., f0=None, technique=None, f0_stats=None)` 透传给 `core.forward`。
- `trainer_base.py`：`_write_adapter` 额外保存 `timbre_conditioner.pt`（存在时）；
  `_load_adapter_weights` 续训回灌；`run.json` 记录 singing 标志。
- 测试：小模型前向——singing=True 时条件进入 estimator（梯度非零）、singing=False 时与旧路径一致。

### WP4 · Capability 合并（E1-T Step 8）
- `merge.py`：`merge_lora_to_checkpoint` 支持 `arch="cfm_singing"`：底座=singing config+pth（或原始
  s2mel.pth + conditioner 新建），合并 LoRA + 写入 conditioner state → 输出**仅允许**
  `checkpoints/s2mel_singing.pth` + `checkpoints/config_singing.yaml`。
  **（r1-10）白名单实现为 BaseGuard 的新显式参数**（`allow_checkpoint_new_names` 白名单集合），
  不在调用侧绕过；测试含「非白名单 checkpoints/ 路径被拒」用例。
- 产出后自动校验：新 pth 用 singing config 加载 missing/unexpected=0；原 `s2mel.pth` hash 不变。

### WP4b · 声码器微调入口骨架（E1-T Step 12 的入口，r1-4）
- `tools/bigvgan_finetune.py`：22k BigVGAN 在歌声数据集上的微调**入口骨架**——
  数据清单读取（复用 WP2 数据集格式）、训练循环占位（含 range-controlled split 提示）、
  独立输出目录约定（`training_runs/bigvgan_singing/`，绝不覆盖 `checkpoints/hf_cache/bigvgan`）；
  `tools/vocoder_extrapolation_test.py`：Perrotin 式 held-out F0 外推测试
  （训练分布外 F0 段的频谱塌陷检测）。两者均为可跑入口，本会话不跑真实训练。

### WP5 · Technique 分类器（E1-T Step 3）
- 新文件 `webui_app/training/technique_cls.py`：轻量模型（mel → 3 层 CNN + 池化 → 7 类），
  `train()`（GTSinger 帧/段级标签）、`predict(audio) -> (帧类别, 置信度)`；懒加载/用完释放。
- 推理端 auto 模式调用；分类器缺失/不可靠时回 control（preserve 模式=源标签直通，manual=用户选）。
- 测试：合成谱差异样本的分类器训练 1 步可收敛 + predict 形状；不可用时回退路径。

### WP6 · 推理 / UI 集成（E1-T Step 13 + BoN Step 14 代码路径）
- `svc_core.py`：
  - `convert_window(..., f0=None, technique=None, f0_stats=None)`：非 None 时传给 `cfm.inference`
    （f0 从 5ms 网格插到该窗 mel 网格；窗内偏移对齐）。**（r1-2）条件张量按
    `[ref.prompt_condition | cond]` 的拼接结构组织：参考帧区段填参考音频自身 harvest F0
    （归一化统计也用参考侧），目标帧区段填八度映射后的源 F0。**
  - 新增 `SingingCfmSession` 上下文管理器：加载独立 singing CFM（98M）→ 临时换入
    `engine.tts.s2mel.models["cfm"]` → 退出 finally 恢复原对象并释放。
    **（r1-3）SVC 租约存在时 `engine.attach_lora/detach_lora(target="cfm")` 必须拒绝**
    （与 `mount_qwen` 同款报错，防推理页挂载挂到换装对象上、防中途摘掉翻唱 identity LoRA）；
    换装/恢复在引擎锁内原子完成。
    **（评审 P2-C4）挂卸归属**：singing 会话内 LoRA 的 detach 在恢复原 CFM **之前**由会话自己完成；
    `cover_pipeline` 级 finally 的 detach 只处理非 singing 路径，杜绝双重 detach/
    detach-after-restore 错序；mock 守恒测试覆盖两条异常路径。
  - BoN 高音奖励（可选、默认关）：仅对「浊音 F0 p75 ≥ 阈值(默认350Hz)」的块生成 N(≤3) 候选，
    奖励 = HNR 代理(HPSS) + SS(campplus，对参考) − 八度错误罚；记录选中候选与奖励明细进报告。
- 新文件 `webui_app/services/svc_singing.py`：HNR 代理（HPSS 谐波/残余 dB）、高音子集统计、
  逐句指标——供报告与 BoN 共用。
- **（r1-7）`tools/e1t_baseline.py`**：对既有翻唱产物目录（02/03/04/05 分阶段文件）生成
  `outputs/e1t_baseline.json`（逐句 HNR 代理/谱平坦度/F0/SS），为 G1a/G1b 提供对比锚点。
- `cover_pipeline.py`：`CoverRequest` 增 `singing_enable: bool = False`（默认关=原链路原样）、
  `technique_mode: "auto"/"manual"/"preserve"`、`technique_manual: int`、`bon_high_pitch: bool = False`；
  singing 开启时：源 F0 先整曲提取（复用移植分支逻辑）→ 八度映射 → 逐块条件；
  F0 移植模式 auto 时若模型条件已启用，WORLD 全曲移植降级为「局部修复/兜底」；
  报告新增 `singing` 段（条件消费统计/HNR/高音子集/BoN 明细）。
- `params_cover.py` + `tabs/cover_tab.py`：新分组「歌声模型（E1-T 实验性）」，默认全关；
  翻唱页安全预设不变。
- 测试：mock 引擎的换装/恢复守恒（异常路径也恢复）、条件透传调用形态、报告字段、BoN 择优纯函数。

### WP7 · 自动分块策略改进（范围澄清之 A+B）
- `svc_core.py`：`cap_chunk_sec` 融合空闲显存——**（评审 P2-F2）保留现有 total 分档表
  （8GB→30s 等）为硬下限，free 只用于在分档内继续收紧**（WDDM 下 free 不可靠，不放大块长）；
  `plan_chunks` 切点吸附 256 样本栅格、谷切点 40ms/硬切 90ms 自适应重叠（重叠=输入重叠，保对拍不变）；
  纯静音块（RMS < -55dB）跳过转换直通（保长、不破坏对拍）。
- `infer_v2_5.py`：`infer(..., stage_hooks=None)` 可选钩子（默认 None 原路径），
  在「参考特征构建后」「GPT 生成后」两个边界调用；`engine.py` 连续性模式联动新开关
  「阶段卸载（C2，实验性）」：默认关；开启时钩子把 w2v-BERT/GPT 搬 CPU，infer 结束/异常恢复。
- 测试：分块改进纯函数单测（栅格吸附、重叠自适应、静音跳过、free-VRAM 只收紧不放宽）；
  stage_hooks=None 等价性（mock tts 验证调用序列不变）。

### WP8 · 文档 / 验收 / 提交
- `docs/dev-log-e1t-impl-2026-10.md`：**真实更新文档**（做了什么/怎么验证/剩余用户运行项/回退方式）。
- `NOTICE` 更新（模型代码修改声明）。
- **（r1-9）用户运行项清单**（写入更新文档）：GTSinger 导入与训练（WP2/WP3 入口）、
  G1a/G1b 评测、`tools/e1t_baseline.py` 生成基线、声码器微调（WP4b 入口）、
  **GPU 端 TTS A/B 音频对比**（同 text/ref/seed 开关前后波形一致性）。
- 子代理验收两轮（找问题→修复→复验）。
- `gh`/git 本地提交（对比 `origin/main` 差异并报告；**按推送策略只 commit 不 push**，
  用户验证后再推）。
- 重写 `README.md`：徽章 + 对上游全部改进总览 + 部署指导 + 全功能实现说明
  （用户明确要求的会话级加项，非 E1-T 母规划范围，已确认）。

---

## 3. 独立先进性调研（开发前派出，结论并入实现）

调研重点（子代理带网执行，只读不改）：
1. F0/technique 条件注入点：style 残差（YingMusic 式）vs FiLM/AdaLN vs cross-attention，
   2025–2026 有无更优且可在 98M DiT 上低风险实现的方案；
2. F0 表示：连续归一化 log-F0 vs 量化 bin（length_regulator 的 512 bin 死代码）对**外推**的影响；
3. CFG 与条件交互：null 分支零化是否为社区共识，有无 CFG-free 的条件外推更稳的做法；
4. 少数据高音外推：pitch augmentation（SPA-SVC）与 technique 条件化的组合是否仍是 SOTA 路线；
5. 22k 声码器高音：不重训前提下的应急手段（谐波感知后处理等）是否有依据。
结论直接决定 WP1 的 conditioner 细节（α、归一化、注入点），不改变总体架构。

## 4. 验收流程

1. 全量测试绿：既有 58 项 + 新增（e1t wiring / features / regression / 分块 / singing 会话）。
2. `tests/test_webui_syntax.py` 覆盖新 tab 控件真实构建。
3. 子代理验收报告：逐条对照本规划与 E1-T 规划 §13 检查清单，找遗漏/风险/优化点 → 修复 → 复验。
4. hash 三件套不变、默认路径逐位一致测试通过、singing 全部开关默认关。

## 5. 明确不做（本会话）

- 不下载 GTSinger、不跑真实训练、不做 G1 评测（入口全部就绪并写入更新文档的用户运行项）。
- **（r1-6）不改 `indextts/infer_v2_5.py:840` 所在的上游 VC 分支**：该分支无 WebUI 调用方
  （翻唱走 `svc_core` 重实现），改它只会扰动上游、没有收益；母规划 §8 该行以本声明结案。
- 不动 `checkpoints/config.yaml`、`s2mel.pth`、BigVGAN。
- 不实现 44.1k/NSF 声码器分支（仅保留 config_singing 的扩展位）。
- identity contrastive / SSIM cycle 损失不实现（训练策略实验，留用户运行项；本会话只到
  tech CE 蒸馏 + f0 aux 的可选开关接口）。
- 不 push（推送策略：用户验证后手动推）。
