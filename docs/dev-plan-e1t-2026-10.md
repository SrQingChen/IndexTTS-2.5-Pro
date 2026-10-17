# E1-T 全量开发规划（执行版 · 无工期 · 按依赖排序）

**日期：** 2026-10-17
**用途：** 本文件是 E1-T（音高 / 技巧条件化歌声转换）的**唯一执行规划**。不排开发时间，只排先后顺序；每一步都有输入、动作、产出、完成判据（Done when）和回退。
**前置：**
- `docs/research-high-pitch-timbre-extrapolation-2026-10.md`（2025–2026 技术调研）
- `docs/handoff-high-pitch-singing-2026-10.md`（高音/戏腔交接）
- `docs/dev-plan-e1t-2026-10.md`（本文件，替换旧版）
**硬约束：** 默认语音 TTS 的权重、配置、vocoder、调用路径不得改变；E1-T 是第二条模型曲线，不是替换 TTS。

---

## 0. 一页总览

### 0.1 问题

- 当前翻唱在 358–442 Hz 已经明显破音；这不是“600 Hz 才出现”的高音问题。
- 角色说话中位 257 Hz；歌曲中位 390.8 Hz ≈ **+7.3 半音**，逐句 358–442 Hz ≈ **+5.7 ~ +9.4 半音**。对角色而言，这已经是外推。
- 当前 SS 0.5269 只说明“像谁”，不说明“唱得稳不稳”；HNR 已证明主要掉话发生在 **CFM 转换阶段**，不是 F0 移植。

### 0.2 解决思路

1. **Base capability 一次性训练**：用公开高质量歌声（GTSinger 等）教模型 `音高/技巧 → 音色`，所有角色共享。
2. **Character identity LoRA 每个角色训练**：用角色说话（可选歌声）教“这个人是谁”，不负责从零学高音。
3. **Optional character singing LoRA**：角色有 10–30min 歌声时，把 base 的高音先验个性化到该角色。
4. **F0 + technique 条件化**：不只给 F0，还要给 mixed/falsetto/breathy/pharyngeal/vibrato/glissando 等技巧状态。
5. **22k 先过 G1，44.1k/NSF 再上 G2**；高音 HNR/CPP 是硬指标，SS 不是唯一指标。

### 0.3 执行顺序（只列依赖，不列时间）

```
Step 0  基线 + TTS 回归护栏
  ↓
Step 1  数据与许可
  ↓
Step 2  特征管线（F0 / technique）
  ↓
Step 4  模型接线（config 门控 + None 原路径）
  ↓
Step 5  10 首歌 smoke test
  ↓
Step 6  Base capability 训练
  ↓
Step 7  Base capability 评测（G1a）
  ↓
Step 8  Capability merge → singing checkpoint + TTS 隔离复检
  ↓
Step 9  角色 identity LoRA
  ↓
Step 10 角色评测（G1b）
  ↓
Step 12 声码器 / 高频（22k → 48k/44.1k）
  ↓
Step 13 推理 / UI 集成
  ↓
Step 14 BoN 高音奖励
  ↓
Step 15 最终闸门（G1 → G2 → G3）
  ↓
Step 16 许可 / NOTICE / 发布检查

Step 3（technique 分类器）与 Step 2/4 可并行，但必须在 Step 6 前可用。
Step 11（角色 singing LoRA）依赖 Step 9 和角色歌声数据，可在 Step 10 后插入。
Step 1 的数据下载与 Step 4 的代码接线可并行；Step 12 的声码器实验可与 Step 6–10 并行，但评测分开归因。
```

---

## 1. 诊断事实（为什么先修转换器）

### 1.1 逐句 HNR 代理

最新 run：`outputs/cover/20261017-154205`（A 档、25 步、cfg 0.7、LoRA 0.9、未移调、F0 auto/full）。
HNR 代理 = HPSS 谐波/残余能量比（dB），不是 Praat HNR，但同管线内可比。

| 时间 | 句中位 F0 | 02 源干声 | 03 转换 | 04 F0 后 | 05 最终 |
|---|---|---|---|---|---|
| 0.0s | 381.8 | 11.2 | 7.2 | 5.1 | 5.7 |
| 3.9s | 438.4 | 8.5 | **1.0** | 2.1 | 2.4 |
| 18.8s | 430.5 | 12.5 | 7.8 | 8.1 | 8.2 |
| 33.3s | 391.9 | 11.5 | 8.1 | 8.5 | 8.5 |
| 41.1s | 390.6 | 11.0 | 6.8 | 6.8 | 6.8 |
| 73.4s | 358.1 | 14.6 | **4.6** | 2.6 | 3.1 |
| 82.3s | 390.9 | 15.1 | **4.0** | 3.6 | 4.0 |
| 93.7s | 378.3 | 11.3 | 7.9 | 9.0 | 10.4 |
| 97.7s | 442.5 | 10.9 | **3.8** | 3.2 | 3.5 |
| 116.3s | 391.5 | 15.0 | 11.6 | 10.4 | 10.7 |
| 127.2s | 390.0 | 13.5 | 9.9 | 8.1 | 8.1 |

### 1.2 结论

1. **主要掉话在 03 转换**，04 F0 移植是次要；后处理 04→05 影响很小。
2. 最差是 438/442 Hz 两句（转换后 1.0/3.8 dB）；模型在 350–450 Hz 已吃力。
3. SS 0.5269 是身份指标，不是质量指标；G1 必须加 HNR/CPP/谱平坦度。
4. 当前“中音”是对歌曲而言；对角色是 +5.7~+9.4 半音。不能把“中低音听起来不错”当作模型舒适区证据。

### 1.3 当前 SS / F0 基线（同一 run）

| 指标 | 整首 | 逐句（11 句，103.2s） |
|---|---|---|
| 03 转换 SS | 0.5673 | 均值 0.469 / 中位 0.459 |
| 04 F0 后 SS | 0.5392 | 均值 0.416 / 中位 0.421 |
| 05 最终 SS | 0.5269 | 均值 0.399 / 中位 0.401；p10–p90 0.346–0.462 |
| F0 50 音分内（报告 20s 探针） | 91.64% | 逐句约 57%（口径不同） |
| F0 相关 r（报告探针） | 0.7018 | 逐句 03 均值 0.351 → 04 均值 0.659 |
| 八度错误（报告探针） | 2.38% | 逐句需单独统计 |

> 报告里的 91.64% / 0.7018 是 20 秒代表段；逐句整首口径低很多。不要混用。

---

## 2. 硬约束：TTS 独立，不影响正常语音合成

### 2.1 三层隔离

| 隔离层 | 做法 | 对 TTS 的影响 |
|---|---|---|
| **权重/配置** | 默认 `checkpoints/s2mel.pth` + `checkpoints/config.yaml` 不动；E1-T 用 `s2mel_singing.pth` + `config_singing.yaml`；新模块只在 singing config 的 `technique_condition: true` 时构造 | 无；默认 TTS 模型结构和参数不变 |
| **运行时** | cover 路径单独加载 singing CFM（98M，fp32 约 0.39GB），不替换引擎 TTS CFM；capability/identity LoRA 只挂 singing CFM；结束 detach/unload；technique 分类器懒加载、用完释放 | 无；TTS 不加载 singing 权重，不挂 adapter |
| **代码路径** | `BASECFM.inference/forward` 新增 `f0=None, technique=None`；两者都为 None 时 `style_t=style`，走原 estimator 调用；默认 `f0_condition=false`，TTS 不传 f0/tech | 无；默认调用序列不变 |

### 2.2 四个最容易误伤 TTS 的点

1. **不要把 `f0_condition: true` 写进默认 `checkpoints/config.yaml`**。默认保持 false；singing 用单独 config。
2. **不要无条件把 `TimbreConditioner`/F0/tech 模块加进 DiT 构造函数**。用 `getattr(args.DiT, "technique_condition", False)` 门控；默认 TTS 的 state_dict 和原来一样。
3. **不要覆盖 `s2mel.pth`、原 BigVGAN、原 config**。capability 训练完 merge 到新文件；vocoder 微调另存；默认 TTS 用原文件。
4. **不要把 singing LoRA 挂到引擎 CFM 上不卸**。它是 cover 会话的 adapter，结束必须 `detach_lora`；现有 cover 路径已这么做，E1-T 沿用并指向独立 CFM。

### 2.3 TTS 回归护栏（Step 0 就要建立，之后每步都跑）

- **文件 hash**：`checkpoints/s2mel.pth`、`checkpoints/config.yaml`、原 BigVGAN 目录；每次模型改动后对比。
- **TTS A/B**：同一 text/ref/seed，开关关闭时输出应与改动前一致（或差异在数值噪声内）；记录 SS/波形差异。
- **运行时检查**：TTS 会话中 `engine.stats.lora_adapters` 为空；未加载 `s2mel_singing.pth`；显存峰值不变。
- **测试**：现有 58 项 + 新增 `tests/test_tts_regression.py`。
- **cover 会话**：singing CFM + identity LoRA 加载；结束释放；TTS 可立即恢复。

### 2.4 上游代码改动边界

- E1-T 会改 `indextts/s2mel/modules/flow_matching.py` / `diffusion_transformer.py` 的**可选参数和门控逻辑**；这不是“上游代码零改动”。
- 若“零改动”是绝对红线：在 `webui_app/services/` 做子类/包装器，默认 TTS 仍走原类；代价是维护两份逻辑。
- 无论哪种，**默认 TTS 的权重、config、vocoder、调用路径都不变**。
- 修改模型后更新 `NOTICE`（原“上游权重未修改”不再成立）。

---

## 3. 训练分工：能力一次，身份每次

### 3.1 三类训练

| 训练 | 数据 | 频率 | 学什么 | 不学什么 |
|---|---|---|---|---|
| **Base capability** | GTSinger 80.59h + M4Singer/Opencpop 等公开歌声 | 一次，全角色共享 | `P(音色 | 身份, F0, technique, 元音)`；高音/假声/混声/咽音的谱包络与声门源 | 不学某个角色身份；不混角色说话 |
| **Character identity LoRA** | 角色说话 10–60s；可选角色歌声 | 每个角色一次 | 角色音色锚点；参考 → style/prompt | 不从零学高音；不混公开歌手身份 |
| **Character singing LoRA（可选）** | 角色歌声 10–30min，覆盖高音/技巧 | 有数据才做 | 把 base 高音先验个性化到该角色 | 不替代 base；数据不足不做 |

### 3.2 直接回答“一次性还是每次角色 LoRA 扩展能力”

- **能力一次训，身份每次训。**
- 只给角色说话时，角色 LoRA 不可能学会高音——说话里没有高音样本；高音行为必须来自 base capability。
- 角色有 10–30min 歌声，才做 character singing LoRA；它是 base capability 之后的第二阶段。
- 最稳组合：**singing-capable base（共享）+ 角色 identity LoRA（每个角色）+ 可选角色 singing LoRA**。

```
公开歌声（GTSinger 等）
        ↓
Base capability（一次）──> singing-capable base checkpoint
                                      │
角色说话 10–60s ──> identity LoRA ────┼──> 推理
角色歌声 10–30min（可选）─> singing LoRA ┘
```

---

## 4. 数据总表与使用规则

### 4.1 公开数据

| 数据 | 规模/内容 | 用途 | 许可/注意 |
|---|---|---|---|
| **GTSinger** | 80.59h、20 专业歌手、9 语言、4 音域、6 technique（mixed/falsetto/breathy/pharyngeal/vibrato/glissando）、16.16h 配对语音、48kHz/24bit、phoneme-level 标注、technique/control 对照 | Base capability 主训练集 | 检索显示 CC BY-NC-SA 4.0；商用需另找授权/自录 |
| M4Singer | ~30h 多歌手 | 扩歌手/音域 | 研究用途，逐项核对 |
| Opencpop | ~5h 中文流行 | 扩中文 | 研究用途 |
| OpenSinger | 多歌手 | 补充 | 核对 |
| SingNet | ~3000h | 大规模预训练/声码器 | 核对 |
| SVCC 2025 | ~70h、7 style | style 转换/评测 | 竞赛条款 |
| KunquDuoVox | 7.68h 昆曲 | 戏腔研究起点 | 研究用途 |
| Yue Opera 2026 基准 | 越剧，F3–F#5 | 戏曲高音参考 | 回源核对 |
| Peking Opera ACV-003 | 京剧，F0/note-slur | 京剧参考 | 回源核对 |

### 4.2 角色数据

| 数据 | 用途 | 注意 |
|---|---|---|
| 角色说话 10–60s | identity LoRA；style/prompt 锚点 | 干净、无混响；不要移调；不混入 base |
| 角色歌声 10–30min | character singing LoRA；个性化高音 | 覆盖目标音域/技巧；按歌曲划分 |
| 角色戏腔 1–5h | 戏腔个性化 | 最难获取；有才做 |

### 4.3 数据使用规则

- **公开高质量歌声 = 能力数据**：只进 base capability；不要混进角色 LoRA。
- **角色说话 = 身份数据**：只进 identity LoRA；不要混进 base。
- **角色歌声 = 个性化数据**：base 之后第二阶段；没有就不做。
- **划分**：按歌手/歌曲划分 train/val/test；不能按帧随机划分。
- **range-controlled split**：每个歌手留出最高 10% F0 作为 high-pitch test；留出至少一种 technique 作为 unseen technique。
- **technique-balanced sampling**：GTSinger technique/control 成对采样；避免 falsetto/mixed 被 control 淹没。
- **pitch augmentation**：训练期 cycle pitch-shift 6–18 半音 + SSIM；推理期不移调参考/源。
- **数据 QA**：F0 分布、technique 平衡、静音/削波/混响、采样率、时长、标签对齐；不合格样本剔除并记录。

---

## 5. 架构与代码设计

### 5.1 现状（代码事实）

| 位置 | 现状 |
|---|---|
| `indextts/s2mel/modules/flow_matching.py:31` | `BASECFM.inference(mu, x_lens, prompt, style, f0, ...)` 有 `f0` 形参 |
| `flow_matching.py:57` | `solve_euler(..., f0, ...)` 仍接收，但 `estimator(...)` 没传 f0 |
| `flow_matching.py:116` | `forward(x1, x_lens, prompt_lens, mu, style)` 无 f0/tech |
| `diffusion_transformer.py:186` | `DiT.forward(x, prompt_x, x_lens, t, style, cond, mask_content=False)` |
| `diffusion_transformer.py:221` | `style[:, None, :].repeat(1, T, 1)` 全局 style 逐帧复制 |
| `diffusion_transformer.py:226` | `cond_x_merge_linear` 输入 80+80+512+192=864 |
| `length_regulator.py:76-134` | 有 `f0_condition` 分支（F0 embedding 加到内容），当前 false/死代码 |
| `checkpoints/config.yaml:73/88` | `f0_condition: false` |
| `checkpoints/config.yaml:107` | `s2mel_checkpoint: s2mel.pth` |
| `indextts/s2mel/modules/commons.py:621` | `load_checkpoint2` 用 `strict=False` |
| `indextts/infer_v2_5.py:840` | `cfm.inference(..., ref_mel, style, None, ...)` 传 None |
| `cover_pipeline.py:376/589` | cover 会话 `attach_lora(cfm)` → `detach_lora(cfm)` |
| `engine.py:770/843` | `attach_lora` / `detach_lora` |
| `cfm_lora.py:520` | `load_base_cfm` 独立加载一份 CFM（98M，0.39GB） |
| `features.py:222` | `FeatureExtractor.extract` 存 mel/mu/style/codes |
| `features.py:389` | `check_feature` 校验 mel/mu/style |
| `features.py:809` | `build_cfm_sample` |
| `cfm_lora.py:265` | `CfmSamplePool` |
| `cfm_lora.py:358` | `collate_cfm` |
| `cfm_lora.py:1034` | `_train_micro_batch` |

### 5.2 目标条件设计

```
F0_emb(t)   = MLP_f0(logF0(t))                  # 256/512 维
Tech_emb(t) = Embedding_tech(tech_id(t))        # 7 类：6 technique + control
style_t     = style + α · MLP([F0_emb(t); Tech_emb(t); style])   # 逐帧，192 维
x_in        = cat([x, prompt_x, cond, style_t]) # 仍是 864，不改 merge 维度
```

- α 初始 0.5，可学习/退火。
- 可选内容残差：`cond = cond_projection(cond) + f0_proj(F0_emb) + tech_proj(Tech_emb)`；第一版先不做，减少变量。
- `style_t` 是逐帧的，替换 `style[:, None, :].repeat(1,T,1)`。
- F0 归一化：speaker-mean normalized log-F0，避免绝对音高泄漏身份。
- technique：GTSinger 6 类 + control；推理支持 auto/manual/preserve。

### 5.3 TimbreConditioner

```python
class TimbreConditioner(nn.Module):
    f0_proj: Linear(1 or D -> D)
    tech_emb: Embedding(7, D)
    style_mlp: Sequential(Linear(192 + 2D, D), SiLU, Linear(D, 192))
    alpha: Parameter(0.5)
```

- 只在 singing config（`technique_condition: true`）时构造；默认 TTS 无此模块。
- 和 LoRA 一起保存/加载；或 merge 进 singing base。
- checkpoint 保存/加载要覆盖该模块，不能只存 PEFT LoRA。

### 5.4 CFM / DiT 接线

- `BASECFM.inference(..., f0=None, technique=None)`：
  - 内部 `solve_euler(..., f0, technique, ...)`；
  - 两者都为 None → 走原 estimator 调用；
  - 否则传给 estimator。
- `solve_euler`：
  - CFG 的 null 分支要同时 zero `f0/technique`（`stacked_f0 = cat([f0, zeros])`），否则 null 仍在用音高条件。
- `BASECFM.forward(..., f0=None, technique=None)`：
  - 训练时传入；None 时走原逻辑。
- `DiT.forward(..., f0_emb=None, tech_emb=None)`：
  - 若为 None：`style_t = style`（严格等价原路径）；
  - 否则：`style_t = style + α·MLP([f0_emb; tech_emb; style])`；
  - 用 `style_t` 替换逐帧 style。
- `length_regulator.f0_condition`：
  - 默认 false；singing config 可选 true；
  - 若启用，先解决 5ms/200Hz → 25/50Hz 网格对齐；第一版可不启用，避免错位。

### 5.5 训练管线接线

| 层 | 改动 |
|---|---|
| `FeatureExtractor.extract` | 存 `f0`（mel 帧率，log-F0）、`f0_voiced`、`tech`（mel 帧率）、`f0_mean/std` |
| `check_feature` | 校验 f0/tech 帧数 = mel_len；tech 0..6 |
| `build_cfm_sample` | 增加 f0/tech |
| `CfmSamplePool._convert` | 透传 f0/tech |
| `collate_cfm` | 拼 f0/tech，padding 0；长度 Tmax |
| `_train_micro_batch` | 传 f0/tech |
| `CFM.forward` / `BASECFM.forward` | 增加 f0/tech 形参 |
| `DiT.forward` | 增加 f0_emb/tech_emb 或 style_t |
| checkpoint | 保存 `TimbreConditioner`；旧数据集缺 f0/tech 时默认 control/zero，不破坏现有训练 |

### 5.6 Checkpoint 组合

```
原始 s2mel.pth ──> 只给 TTS
        │
        └─> singing config + 新模块 ──> Base capability 训练
                                              ↓
                                      merge 到新文件 s2mel_singing.pth
                                              ↓
                              角色 identity LoRA（PEFT）
                                              ↓
                              可选角色 singing LoRA
```

- 不覆盖 `s2mel.pth`；capability merge 到 `s2mel_singing.pth`。
- `config_singing.yaml` 与默认 `config.yaml` 分开；默认 `f0_condition` 保持 false。
- 角色 LoRA 挂在 singing base 上；TTS 不挂。
- 若用“两份逻辑”方案：`webui_app/services/` 子类/包装器，默认 TTS 走原类。

### 5.7 推理路径

```
源干声 → RMVPE/FCPE F0 + technique 分类器
       → w2v-BERT/RVQ 内容
       → CFM(F0_emb, Tech_emb, ref identity)
       → mel → 22k/44.1k 声码器
       → 可选 BoN 高音奖励
```

- 参考音频不移调。
- technique 选择：auto / manual / preserve。
- F0 只在模型条件里用；WORLD 移植退居兜底（局部八度修正）。
- 保留 A/B 安全预设作为回退。

---

## 6. 执行顺序（Step 0 → Step 16）

> 每一步的“Done when”是硬判据；不满足不进下一步。没有工期，只有依赖。

### Step 0：基线 + TTS 回归护栏

**目的：** 在改任何模型前，先把“当前 TTS 输出”和“当前翻唱基线”固定下来。

**动作：**
1. 记录 `checkpoints/s2mel.pth`、`config.yaml`、BigVGAN 的 hash。
2. 跑 TTS A/B：固定 text/ref/seed，保存输出与 SS/波形。
3. 跑当前翻唱 11 句 HNR/CPP/谱平坦度/F0/SS，生成 `outputs/e1t_baseline.json`。
4. 新增 `tests/test_tts_regression.py`：固定输入，断言开关关闭时输出与基线一致/差异在噪声内。
5. 跑现有 58 项测试，记录基线。

**产出：** `outputs/e1t_baseline.json`、TTS 回归测试、hash 清单。

**Done when：** 基线可复现；TTS 回归测试通过；hash 已记录。

**回退：** 无（只读/记录）。

---

### Step 1：数据与许可

**目的：** 把训练数据和许可边界一次性盘清。

**动作：**
1. 下载/核对 GTSinger；确认 CC BY-NC-SA 4.0 的使用边界。
2. 核对 M4Singer/Opencpop/OpenSinger/SingNet/SVCC 2025/KunquDuoVox 许可。
3. 建 manifest：路径、歌手、语言、音域、technique、时长、采样率、文本/音素、F0 统计。
4. 数据 QA：静音、削波、混响、标签对齐、重复、采样率；剔除并记录。
5. 规划 split：unseen singer / unseen technique / top 10% F0。
6. 角色数据：说话 10–60s；可选歌声 10–30min；记录权利来源。

**产出：** `data/manifests/*.jsonl`、许可清单、数据统计。

**Done when：** 每个数据集有许可结论；manifest 可被特征管线读取；split 不泄漏。

**回退：** 许可不清的数据不进训练；商用路线改用自录/授权数据。

---

### Step 2：特征管线（F0 / technique）

**目的：** 让每个训练样本都带 F0 和 technique，且与 mel 帧对齐。

**动作：**
1. `features.py` 增加 F0 提取：harvest 5ms → 插值到 mel 帧（86.13Hz）；存 log-F0、voiced mask、speaker-mean/std。
2. 解析 GTSinger phoneme-level technique → mel 帧；无标签数据默认 control。
3. `check_feature` 增加 f0/tech 形状/范围校验。
4. `build_cfm_sample`/`CfmSamplePool`/`collate_cfm` 透传 f0/tech。
5. 抽取 100 首 smoke 集，检查 F0 分布/technique 平衡/对齐。

**产出：** 带 f0/tech 的特征池；100 首 smoke 数据。

**Done when：** 100 首样本全部通过 `check_feature`；f0/tech 与 mel 帧对齐；旧数据集缺字段时可默认 control/zero，不破坏现有训练。

**回退：** 若 F0 网格错位，先只存帧级 F0，不启用 `length_regulator.f0_condition`。

---

### Step 3：Technique 分类器（可与 Step 2/4 并行）

**目的：** 推理期 auto 选择 technique；评测期量化技巧保真。

**动作：**
1. 用 GTSinger 6 类 + control 训练分类器（mel/SSL 特征；轻量）。
2. 输出混淆矩阵、每类 F1；重点看 falsetto/mixed/pharyngeal。
3. 保存为独立小模型；懒加载、用完释放。
4. 在源干声/转换输出/最终输出上跑，量化“源技巧是否丢失”。

**产出：** technique 分类器 + 基线报告。

**Done when：** 验证集每类可解释；falsetto/mixed 的召回可用于评测；不常驻显存。

**回退：** 分类器不可靠时，推理期改 manual 选择；评测用人工 ABX。

---

### Step 4：模型接线（config 门控 + None 原路径）

**目的：** 把 F0/technique 条件接进 CFM/DiT，同时保证默认 TTS 完全不变。

**动作：**
1. `DiT.__init__` 增加 `technique_condition`（默认 false）、`f0_embed_dim`、`num_techniques`；仅 true 时构造 `TimbreConditioner`。
2. `DiT.forward` 增加 `f0_emb=None, tech_emb=None`；None 时 `style_t=style`。
3. `BASECFM.inference/solve_euler/forward` 增加 `f0=None, technique=None`；None 时走原调用。
4. CFG null 分支 zero f0/tech。
5. `config_singing.yaml` 新建；默认 `config.yaml` 不动；默认 `f0_condition` 保持 false。
6. `FeatureExtractor`/`CfmSamplePool`/`collate_cfm`/`_train_micro_batch` 透传。
7. `TimbreConditioner` 的保存/加载接入 checkpoint；`load_base_cfm` 支持 `cfg_path` 参数。
8. 静态检查/单测：`f0=None,tech=None` 时输出与改前一致。

**产出：** 可训练/可推理的 F0+tech 条件代码；默认 TTS 路径不变。

**Done when：** TTS 回归测试通过；`f0=None,tech=None` 输出一致；singing config 能构造新模块；旧 checkpoint 在默认 config 下加载无新键。

**回退：** 若 `strict=False` 导致默认模型出现新键，改为“只在 singing config 构造新模块”；必要时做子类/包装器。

---

### Step 5：10 首歌 Smoke Test

**目的：** 在真实数据上验证管线、显存、梯度、条件是否真的被消费。

**动作：**
1. 10 首歌（含 falsetto/mixed/pharyngeal）跑特征→batch→CFM forward/backward。
2. 检查 loss 有限、梯度非零、VRAM 在 8GB 内。
3. 消融：`f0=None,tech=None` vs 有条件的 loss/输出差异；确认条件不是“死参数”。
4. 检查 CFG null 分支不泄漏条件。
5. 跑 TTS 回归。

**产出：** smoke 报告；最小可训练配置。

**Done when：** 1 个 batch 能稳定前后向；条件对输出有可测影响；TTS 无回归。

**回退：** 条件无影响 → 检查 f0/tech 是否进入 estimator、维度/网格是否对齐；不要继续加数据。

---

### Step 6：Base capability 训练

**目的：** 用公开歌声教“音高/技巧 → 音色”的先验。

**动作：**
1. 先 5–10h 子集训，确认 loss/VRAM/速度；再扩 GTSinger 中/英 + M4Singer/Opencpop。
2. 22k、10–20s 片段、batch 1–2、grad checkpoint、LoRA rank 8–32；冻结 w2v-BERT/BigVGAN。
3. 损失：flow matching + F0 aux + technique CE + identity contrastive + SSIM cycle（可选）。
4. 训练期 cycle pitch-shift 6–18 半音；technique-balanced sampler。
5. 每轮在 range-controlled val 上测 HNR/F0/SS/technique。
6. 保存 capability checkpoint（LoRA + TimbreConditioner）。

**产出：** capability 训练 run、checkpoint、val 曲线。

**Done when：** loss 收敛；range-controlled val 上 HNR 相对基线改善；条件消融有效；TTS 回归通过。

**回退：** 8GB 不够 → 降片段/rank/层数；CFM 容量不够 → 加 Adapter 层；数据不够 → 先 GTSinger 中/英。

---

### Step 7：Base capability 评测（G1a）

**目的：** 决定是否进入角色阶段。

**动作：**
1. 固定测试集：unseen singer + unseen technique + top 10% F0。
2. 指标：HNR/CPP、谱平坦度、F0 50c、八度错误、SS、technique accuracy、ABX/MOS。
3. 与当前 A 基线对比；与 03/04/05 分阶段对比。
4. 输出 `outputs/e1t_g1a.json` + 报告。

**G1a 建议闸门：**
- 350–600 Hz 子集 HNR 不低于源干声 3 dB 以内（当前最差句掉 6–12 dB）。
- F0 50 音分内 ≥85%；八度错误 <1%。
- SS ≥0.70 或较 A 基线 +0.08。
- technique 分类准确率 ≥70%（falsetto/mixed 单独看）。
- ABX“更像目标”≥60%。

**Done when：** G1a 通过，或明确记录失败原因并回 Step 6。

**回退：** 不通过 → 补数据/technique 条件/22k 声码器；不要直接上 44.1k。

---

### Step 8：Capability Merge → singing checkpoint + TTS 隔离复检

**目的：** 把 capability 固化成共享 base，供所有角色使用；再次确认 TTS 不受影响。

**动作：**
1. 用 `merge.py` 把 capability LoRA merge 到一份 base CFM 拷贝；`TimbreConditioner` state_dict 一并写入。
2. 保存 `s2mel_singing.pth` + `config_singing.yaml`；不覆盖 `s2mel.pth`/`config.yaml`。
3. 用 singing checkpoint 跑 TTS 回归：默认 TTS 仍加载原 `s2mel.pth`。
4. 检查 `load_checkpoint2` 的 missing/unexpected keys；确保 config 与 checkpoint 匹配。
5. 记录 hash：原文件不变，新文件新增。

**产出：** `checkpoints/s2mel_singing.pth`、`checkpoints/config_singing.yaml`。

**Done when：** singing base 可加载；原 TTS hash 不变；TTS A/B 无回归；cover 路径可切到 singing base。

**回退：** merge 失败/键不匹配 → 保留 LoRA + conditioner 分离加载；不改原文件。

---

### Step 9：角色 identity LoRA

**目的：** 用角色说话（可选歌声）学身份，不重学高音。

**动作：**
1. 在 singing base 上训练 identity LoRA；数据 = 角色说话 10–60s（+ 可选歌声）。
2. 冻结 capability；低 LR；加少量 GTSinger singing replay 防遗忘。
3. 参考音频保持原样，不移调；style/prompt 用角色参考。
4. 保存角色 LoRA 到 `training_runs/<role>_singing_identity/`。
5. 跑 TTS 回归；确认角色 LoRA 未挂到 TTS。

**产出：** 角色 identity LoRA + 训练报告。

**Done when：** 角色 LoRA 可挂载/卸载；TTS 不受影响；同一首歌能跑通。

**回退：** 身份不稳 → 增加参考时长/质量；高音退化 → 冻结更多 capability 层/加 replay。

---

### Step 10：角色评测（G1b）

**目的：** 判断“中音/高音是否真的比当前好”，而不是只看 SS。

**动作：**
1. 同一首歌 A/B：当前 A 档 vs E1-T（singing base + identity LoRA）。
2. 逐句 HNR/CPP/谱平坦度/F0/SS/technique。
3. 高音子集单独统计；ABX/MOS。
4. 输出 `outputs/e1t_g1b.json` + 报告。

**G1b 建议闸门：**
- 中音区 SS ≥0.5269 或 +0.03。
- HNR 提升 ≥3 dB（重点 350–450 Hz）。
- 高音子集不劣化；八度错误不增加。
- ABX“更像目标/更自然”≥60%。

**Done when：** G1b 通过，或记录失败原因回 Step 6/9。

**回退：** 不通过 → 先查转换器 HNR，再查身份 LoRA 是否遗忘；不要用后处理掩盖。

---

### Step 11：可选角色 singing LoRA（G3）

**目的：** 角色有 10–30min 歌声时，把高音先验个性化。

**动作：**
1. 只在 Step 9/G1b 通过后做；数据覆盖目标音域/技巧。
2. 在 singing base + identity LoRA 上继续训练；冻结 capability 大部分层。
3. 按歌曲划分 train/val；高音子集单列。
4. 保存到独立目录；可与 identity LoRA 组合/替换测试。
5. 跑 TTS 回归。

**产出：** 角色 singing LoRA + G3 报告。

**Done when：** 角色高音/技巧 ABX 通过；不劣化中低音；TTS 无回归。

**回退：** 数据不足/过拟合 → 不做；回到 base prior。

---

### Step 12：声码器 / 高频

**目的：** 修渲染层，不让 22k/声码器成为高音上限。

**动作：**
1. 22k BigVGAN 在 GTSinger 歌声上微调；保存独立 checkpoint。
2. Perrotin 式 range-controlled 测试：held-out F0 区间不塌回训练中位。
3. 可选 S4 式高频补全（辅助 48k SVC/带宽扩展）。
4. G2：44.1k BigVGAN v2 或 NSF-HiFiGAN；24GB 级硬件。
5. 默认 TTS 仍用原 vocoder；singing 路径才用新 vocoder。

**产出：** 22k/44.1k vocoder checkpoint + 外推测试报告。

**Done when：** 高音 HNR 不降；held-out F0 不塌；TTS 不受影响。

**回退：** 44.1k 资源不够 → 22k + 高频补全；声码器无改善 → 先修 CFM。

---

### Step 13：推理 / UI 集成

**目的：** 让 E1-T 在翻唱页可用，且 TTS 路径不受影响。

**动作：**
1. `svc_core.py` 传 F0/technique；`cover_pipeline.py` 编排；`params_cover.py`/`cover_tab.py` 加 technique 选择。
2. 加载 singing CFM/角色 LoRA；结束 detach/unload；沿用租约/GPT offload。
3. 报告加 HNR/technique/高音子集/F0 口径。
4. 保留 A/B 安全预设；singing 开关关闭时走原链路。
5. 跑 TTS 回归 + 现有测试。

**产出：** 可用的 E1-T 翻唱路径 + UI。

**Done when：** 翻唱页可切换 E1-T；TTS 开关关闭无回归；报告完整。

**回退：** UI/显存问题 → 先 CLI/离线；不影响 TTS。

---

### Step 14：BoN 高音奖励

**目的：** 用推理期择优提升高音稳定性，不改模型。

**动作：**
1. 候选生成 N 个；奖励 = HNR/CPP + 八度错误 + F1–F0 tuning + technique 置信度 + SS。
2. 只在高音子集启用；记录选中候选与奖励。
3. 与不启用 BoN 的 A/B；确认不牺牲中低音。
4. 不改变 TTS 路径。

**产出：** BoN 配置 + 报告。

**Done when：** 高音 HNR/ABX 改善；中低音不劣化；TTS 无回归。

**回退：** 奖励不稳定 → 降低候选数/只用 HNR+八度；不影响主链。

---

### Step 15：最终闸门（G1 → G2 → G3）

**G1（22k）综合：**
- G1a base + G1b character 均通过。
- 高音子集 HNR/F0/technique/ABX 达标。
- TTS 回归通过。

**G2（44.1k/NSF）：**
- 与 Seed-VC/HQ-SVC/YingMusic 同参考 A/B（若允许外部对照）。
- SS ≥0.80 或较 G1 +0.08；高音 MOS 不劣。
- Perrotin 式外推测试通过。
- TTS 仍走原 vocoder（除非单独 A/B 证明无回归）。

**G3（角色歌声/戏腔）：**
- 角色 singing LoRA；角色 SS ≥0.85（有数据时）。
- 戏腔人工 A/B 通过；pharyngeal/falsetto 数据覆盖。
- 许可/权利清晰。

**Done when：** 每级闸门有报告；失败有明确回退；TTS 始终无回归。

---

### Step 16：许可 / NOTICE / 发布检查

**动作：**
1. 更新 `NOTICE`：上游权重/代码修改、衍生权重、第三方数据。
2. 核对 GTSinger CC BY-NC-SA 4.0、M4Singer/Opencpop/SVCC 条款、角色声音/歌曲版权。
3. 确认 bilibili 协议 3.4(c)/商业门槛；商用路线改用授权/自录数据。
4. 衍生权重/代码许可证（GPL-3.0 新增代码、bilibili 权重协议）。
5. 发布前法务复核。

**Done when：** 许可清单完整；NOTICE 更新；商用/非商用边界明确。

---

## 7. 评测协议

### 7.1 指标

| 类别 | 指标 | 说明 |
|---|---|---|
| 身份 | SS（campplus）、SECS、SIM-MOS、ABX | 不同 embedding 不可横向比；SS 不是“百分比” |
| 自然度 | MOS/CMOS、NISQA、UTMOS、SingMOS | 破音主要掉这里 |
| 音准 | F0 RMSE（音分）、FPC、LogF0PCC、50 音分内、八度错误 | 报告探针 ≠ 整首 |
| 高音 | F0>500/600 子集、HNR/CPP、谱平坦度、F1–F0 tuning、双音高 | 必须单独统计 |
| 技巧 | technique 分类准确率/F1、style similarity | falsetto/mixed/pharyngeal 单列 |
| 内容 | CER/WER | 音域适配可能用 WER 换 SIM |
| 分布 | FAD、density/coverage、Audiobox aesthetics | 辅助 |

### 7.2 评测规则

- **range-controlled split**：unseen singer + unseen technique + top 10% F0。
- **高音子集**：F0>500/600 单独报告；不要用整首平均掩盖。
- **ABX**：目标参考 vs 另一人；50% 瞎猜，60–70% 有感知优势。
- **SS 警戒**：高 SS + 低 HNR = “像但破”；不能只看 SS。
- **报告口径**：整首/逐句/探针分开写；不要混用 91.64% 与逐句 57%。

### 7.3 报告 JSON 建议字段

```json
{
  "run": "...",
  "stage": "baseline|g1a|g1b|g2|g3",
  "split": {"unseen_singer": true, "unseen_technique": "pharyngeal", "top_f0_pct": 10},
  "metrics": {"ss": 0.0, "secs": 0.0, "f0_50c": 0.0, "octave_err": 0.0,
              "hnr_db": 0.0, "cpp": 0.0, "tech_acc": 0.0, "cer": 0.0},
  "subsets": {"f0_gt_500": {...}, "falsetto": {...}, "mixed": {...}},
  "tts_regression": {"hash_ok": true, "ab_diff": 0.0, "tests_passed": true},
  "notes": []
}
```

---

## 8. 代码触点总表

| 文件/函数 | 改动 | 风险 |
|---|---|---|
| `flow_matching.py:31/57/116` | `f0/technique` 透传；None 原路径；CFG null zero | 默认路径必须严格等价 |
| `diffusion_transformer.py:186/221` | `style_t` 逐帧；`TimbreConditioner`；config 门控 | 新模块不能无条件构造 |
| `length_regulator.py:76-134` | 可选 `f0_condition`；先不依赖 | 网格对齐 |
| `checkpoints/config.yaml` | 默认不改；新增 `config_singing.yaml` | 不改默认 f0_condition |
| `indextts/infer_v2_5.py:840` | VC 路径传 F0/tech | TTS 路径不传 |
| `webui_app/training/features.py:222/389/809` | 存/校验/透传 f0/tech | 旧数据兼容 |
| `webui_app/training/cfm_lora.py:265/358/1034/520` | pool/collate/train/load；conditioner 保存 | checkpoint 完整性 |
| `webui_app/training/trainer_base.py` | 新模块保存/加载 | 不能只存 LoRA |
| `webui_app/training/merge.py` | capability merge 到新文件 | 不覆盖原权重 |
| `webui_app/services/svc_core.py` | 推理传 F0/tech；独立 singing CFM | 不替换引擎 CFM |
| `webui_app/services/svc_f0.py` | F0/technique 分类器/音域分析 | 懒加载 |
| `webui_app/services/cover_pipeline.py:376/589` | 挂/卸 singing adapter；technique 选择 | 必须 detach |
| `webui_app/params_cover.py` / `tabs/cover_tab.py` | technique 控件/预设 | 默认 A 安全预设不变 |
| `checkpoints/hf_cache/bigvgan` | 22k/44.1k 独立微调 | 默认 TTS 用原 vocoder |

---

## 9. 环境与命令

### 9.1 环境

- 项目根：`E:\AI\indextts2.5\index-tts-2.5.0`
- Python：`.venv\Scripts\python.exe`
- 模型：`checkpoints/`
- 翻唱产物：`outputs/cover/<timestamp>/`
- 日志：`outputs/logs/indextts.log` / `error.log`
- 参考：`voice_bank/audio/达妮娅.wav`
- 训练：`training_runs/<role>_cfm/`、`training_runs/<role>_gpt/`

### 9.2 测试

```bash
.venv\Scripts\python.exe -m pytest tests/test_svc.py tests/test_separation_fetch.py tests/test_webui_syntax.py -q
```

新增：`tests/test_tts_regression.py`

### 9.3 Hash 检查

```python
import hashlib, pathlib
for p in ["checkpoints/s2mel.pth", "checkpoints/config.yaml"]:
    print(p, hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest())
```

### 9.4 训练/评测入口

- 特征抽取：`webui_app/training/features.py`（CLI/UI 入口按现有训练页）
- CFM LoRA：`webui_app/training/cfm_lora.py`
- 评测：`webui_app/training/evaluate.py`
- 合并：`webui_app/training/merge.py`
- 翻唱：`webui_app/services/cover_pipeline.py`

> 具体 CLI 参数以现有训练页/入口为准；新增 `--cfg_path config_singing.yaml` 和 `--singing` 开关。

---

## 10. 风险登记与回退

| 风险 | 触发信号 | 处理 | 回退 |
|---|---|---|---|
| TTS 回归 | TTS A/B 差异/新键/显存变化 | 停用 singing 模块；检查 config 门控 | 子类/包装器；恢复原文件 |
| 8GB 显存 | OOM/共享内存降速 | 22k、10–20s、batch 1、grad checkpoint | 云 24GB |
| GTSinger 许可 | 商用不可用 | 自录/授权数据 | 研究阶段先用 |
| 角色说话太短 | 身份不稳 | 增加参考/质量 | 10–30min 角色歌声 |
| 身份 LoRA 遗忘 | 高音退化 | 冻结 capability、低 LR、replay | 只做 identity，不做 singing |
| CFM 容量不足 | 条件无影响 | 加 Adapter/扩 rank | 22k 小步验证 |
| F0 网格错位 | 条件错位/不收敛 | 统一到 mel 帧 | 先不启用 length_regulator F0 |
| CFG null 泄漏 | 条件在空分支仍生效 | zero f0/tech | 关闭 CFG 对照 |
| 破音是源唱法 | HNR 低但 F0 对 | technique 条件/数据 | 接受上限，不做“完美”承诺 |
| SS 掩盖破音 | SS 高但 HNR 低 | 加 HNR/ABX 闸门 | 不以 SS 单独验收 |
| 44.1k 资源不足 | OOM/慢 | 22k + 高频补全 | G2 延后 |

---

## 11. 许可与商用

- **GTSinger**：检索显示 CC BY-NC-SA 4.0；商用需授权/自录。
- **M4Singer/Opencpop/OpenSinger/SingNet**：研究用途；逐项核对。
- **SVCC 2025**：竞赛条款；商用核对。
- **KunquDuoVox/Yue Opera/ACV-003**：研究用途；戏腔商用需授权。
- **角色说话/歌声/戏腔**：声音权利、游戏/声优/歌曲版权独立。
- **bilibili 协议**：微调允许；保留 LICENSE/NOTICE/DISCLAIMER；3.4(c) 商业限制；商用门槛按中文 2.2（MAU 1 亿或年收入 1 亿人民币）。
- **NOTICE**：修改模型后更新；衍生权重受 bilibili 协议；新增代码 GPL-3.0。
- 本文不是法律意见；发布前法务复核。

---

## 12. 参考链接

**核心方法**
- YingMusic-SVC：https://arxiv.org/abs/2512.04793 ；https://github.com/GiantAILab/YingMusic-SVC
- HQ-SVC：https://arxiv.org/abs/2511.08496 ；https://github.com/ShawnPi233/HQ-SVC
- S²Voice：https://arxiv.org/abs/2601.13629
- SVCC 2025 分析：https://arxiv.org/abs/2509.15629
- S4（SVCC 2025）：https://arxiv.org/abs/2604.05526
- TechSinger：https://arxiv.org/abs/2502.12572 ；https://github.com/gwx314/TechSinger
- FABYOL：https://proceedings.mlr.press/v303/hsu26a.html
- Everyone-Can-Sing：https://arxiv.org/abs/2501.13870
- TCSinger 2：https://arxiv.org/abs/2505.14910
- Vevo2：https://arxiv.org/abs/2508.16332 ；https://github.com/microsoft/Vevo2
- SoulX-Singer：https://arxiv.org/abs/2602.07803
- OneVoice：https://arxiv.org/abs/2601.18094
- SPSinger：https://ieeexplore.ieee.org/abstract/document/10888907
- SPA-SVC：https://www.isca-archive.org/interspeech_2024/bai24_interspeech.html
- SpSiVC：https://www.jstage.jst.go.jp/article/ast/46/1/46_e24.47/_article
- R2-SVC：https://arxiv.org/abs/2510.20677
- DAFMSVC：https://arxiv.org/abs/2508.05978
- PFlow-VC：https://arxiv.org/abs/2502.05471
- Prompt-Singer：https://arxiv.org/abs/2403.11780
- CoMelSinger：https://arxiv.org/abs/2509.19883
- VaSAB：https://arxiv.org/abs/2310.03444
- VibE-SVC2：https://arxiv.org/abs/2606.17126

**声码器/物理**
- DDSP：https://arxiv.org/abs/2001.04643
- SawSing：https://arxiv.org/abs/2208.04756
- BigVGAN：https://arxiv.org/abs/2206.04658
- SnakeGAN：https://arxiv.org/abs/2309.07803
- Perrotin 2021：https://hal.science/hal-03338483
- Drugman 2020：https://arxiv.org/abs/2006.04142
- Sundberg 1975：Formant technique in a professional female singer, Acustica 32(2):89–96
- Titze 2004：F0–F1 interaction in speaking/singing

**数据/工具**
- GTSinger：https://arxiv.org/abs/2409.13832 ；https://huggingface.co/datasets/GTSinger/GTSinger
- SingNet：https://arxiv.org/abs/2505.09325
- KunquSinger：https://hualizhou167.github.io/KunquSinger-demo/
- Seed-VC：https://github.com/Plachtaa/seed-vc
- DiffSinger：https://github.com/openvpi/DiffSinger

---

## 13. 最终检查清单

### 13.1 隔离

- [ ] 默认 `checkpoints/config.yaml` 未改，`f0_condition: false`
- [ ] 默认 `checkpoints/s2mel.pth` hash 未变
- [ ] 原 BigVGAN hash 未变
- [ ] 默认 TTS 不加载 `s2mel_singing.pth`
- [ ] 默认 TTS 不挂 singing LoRA
- [ ] `f0=None, technique=None` 路径与改前一致
- [ ] TTS 回归测试通过
- [ ] cover 结束 `detach_lora` / 释放 singing CFM

### 13.2 数据

- [ ] GTSinger 许可结论
- [ ] manifest/split 无泄漏
- [ ] unseen singer / unseen technique / top 10% F0
- [ ] technique-balanced sampling
- [ ] 角色数据权利记录
- [ ] 训练/推理参考不移调

### 13.3 模型

- [ ] F0/tech 条件真的进入 estimator
- [ ] CFG null 分支 zero 条件
- [ ] `TimbreConditioner` 保存/加载
- [ ] capability merge 到新文件
- [ ] 角色 LoRA 可挂/可卸
- [ ] 22k G1 通过后再上 44.1k

### 13.4 评测

- [ ] 高音子集单独统计
- [ ] HNR/CPP/谱平坦度
- [ ] F0 50c/八度错误
- [ ] technique accuracy
- [ ] ABX/MOS
- [ ] 报告口径（整首/逐句/探针）写清
- [ ] SS 高但 HNR 低时判失败

### 13.5 发布

- [ ] NOTICE 更新
- [ ] 第三方数据/角色/歌曲许可
- [ ] 商用/非商用边界
- [ ] 衍生权重/代码许可
- [ ] 法务复核

---

## 14. 一句话原则

**Base capability 一次训“怎么唱”，角色 identity LoRA 每次训“谁在唱”，角色 singing LoRA 有数据才做“本人怎么唱”；默认 TTS 始终走原权重、原配置、原 vocoder、原 CFM，E1-T 只是 cover 会话里的第二条曲线。**
