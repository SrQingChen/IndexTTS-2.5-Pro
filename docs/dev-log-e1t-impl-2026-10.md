# E1-T 实施更新文档（代码级全管线落地）

**日期：** 2026-10-17
**性质：** 真实更新记录——本会话**实际交付**的代码、测试与验证结果；
以及需要用户在真机/真数据上运行的后续项（未跑，如实列出）。
**服从：** `docs/dev-plan-e1t-2026-10.md`（母规划）+ `docs/session-plan-e1t-impl-2026-10-17.md`（会话规划 r1）。
**先进性调研：** 已并入设计（残差零初始化 / uv 通道 / 软截断 / CFG null=旁路 /
SHC 八度修正 / 音区分数映射 / SYKI 高频移植 / BoN 全局八度罚），来源与证据见调研结论（会话内完成）。

---

## 0. 一句话总结

E1-T（音高/技巧条件化歌声转换）的**代码级全管线**已交付：模型接线、特征管线、
训练链、capability 合并、technique 分类器、推理/UI 集成、BoN 高音择优、
自动分块改进与 TTS 阶段卸载——全部默认关闭、可单提交回退；
**默认 TTS 三件套（s2mel.pth / config.yaml / BigVGAN）字节级未动（hash 实测）**，
`f0=None,technique=None` 原路径逐位一致（冻结参照测试固化）。
真实训练/评测（GTSinger 下载、capability 训练、G1a/G1b）属用户运行项，入口全部就绪。

## 1. 交付清单（按母规划 Step 对照）

| 母规划 Step | 交付物 | 验证 |
|---|---|---|
| Step 0 护栏 | `tests/test_tts_regression.py`（hash 守卫/默认 config 门控/无新键）+ `tests/baseline_tts_hashes.json` + `tools/e1t_baseline.py`（基线 JSON 生成，r1-7） | 4 项测试绿；hash 实测全 OK |
| Step 2 特征 | `features.py`：singing 提取（F0 5ms→mel 网格、清音不架桥）、`f0/f0_mean/f0_std/tech/tech_dom` 字段、`SINGING_SCHEMA`（FEATURE_VERSION 不动，旧缓存兼容）、`technique.json` 伴生标签、`check_feature` 扩展 | `test_e1t_features.py` 12 项绿 |
| Step 3 分类器 | `webui_app/training/technique_cls.py`（3 层 CNN，训练/懒加载推理/均衡权重）；推理 auto 模式回退 control | 训练函数冒烟可用（真实训练属用户项） |
| Step 4 模型接线 | `indextts/s2mel/modules/conditioners.py`（TimbreConditioner：连续归一化 log2-F0 + uv 通道 + 4tanh 软截断 + 零初始化残差 + α=0.5 可学习 + cond_drop 旁路）；`diffusion_transformer.py`（getattr 门控 + None 原路径）；`flow_matching.py`（f0/technique/f0_stats 透传，**新参数只在签名尾部**（r1-8）；CFG null 分支=**旁路 conditioner**（非输入置零））；`checkpoints/config_singing.yaml`（新文件，默认 config.yaml 零改动） | `test_e1t_wiring.py` 14 项绿（含 bit-exact 与冻结参照 solve_euler 对比） |
| Step 5 冒烟 | 条件消费/CFG/丢条件报错均在 wiring 测试固化；真数据冒烟属用户项（入口=训练页） | — |
| Step 6 训练 | `cfm_lora.py`：`load_base_cfm(cfg_path, pth_name)`、`CfmTrainOptions.singing`（含 tech_balance）、`_train_micro_batch/_mean_loss` 条件透传、preflight singing 字段预检、technique 均衡上采样；`trainer_base.py`：`timbre_conditioner.pt` 与 adapter 成对存取（漏存=能力训练白跑的红线）；`forward.py` 透传 | 导入+小模型前向绿；**条件侧移调增广被否决**（调研结论：只改标签不改 mel 会教模型忽略条件），改为导入期音频域增广（`gtsinger_import.py --pitch-aug`） |
| Step 8 合并 | `merge.py` arch=`cfm_singing`（singing 底座/叠加/conditioner 回灌/校验）；`guard.py`：BaseGuard `SINGING_WHITELIST` 显式白名单参数（r1-10，只放行 checkpoints/ 顶层两个歌声新名字，子目录不放行） | 白名单语义 6 项断言实测 |
| Step 9 identity LoRA | 训练入口就绪（singing base 上跑 CfmTrainer，cfg_path 指向 config_singing.yaml） | 用户项 |
| Step 12 声码器 | `tools/bigvgan_finetune.py`（入口骨架：range-controlled split/训练循环/独立输出目录/隔离校验）+ `tools/vocoder_extrapolation_test.py`（Perrotin 式 held-out F0 外推测试）（r1-4） | --smoke 冒烟属用户项（需真数据） |
| Step 13 推理/UI | `svc_core.py`：`convert_window` 条件化（prompt 帧区段=参考自身 F0，r1-2）、`SingingCond`（逐帧分区统计）、`SingingCfmSession`（换装/恢复/LoRA 归属，异常路径恢复）；`engine.py`：SVC 租约拒绝 cfm 通道挂/卸（bypass 豁免，r1-3）+ 卸载后幻影标签清理；`cover_pipeline.py`：singing 编排（SHC 修正→分数音区映射→technique 条件→F0 auto 降级为只测量）+ 报告 singing 段（HNR/高音子集/逐句，§7.2 红线）+ SYKI 高频移植开关；`params_cover.py`/`cover_tab.py`：新分组 7 控件（默认全关/修正默认开）+ 运行前权重缺失友好报错 | `test_e1t_inference.py` 15 项绿；WebUI 真实构建 |
| Step 14 BoN | 高音块（浊音 p75≥350Hz）N≤3 候选，奖励=HNR+50 音分内−八度错误−**全局掉八度罚**+SS，明细进报告 | 好候选/坏候选奖励差实测 >10 分 |
| Step 16 NOTICE | `NOTICE` 追加模型代码修改声明 | 见 §4 |
| 分块改进（用户指令） | `svc_core.py`：切点吸附 mel 栅格、谷切 40ms/硬切 90ms 自适应重叠、纯静音块跳过（保长）、free-VRAM **只收紧不放宽**（WDDM 不可靠，评审 P2-F2） | `test_svc.py` 扩展绿 |
| 阶段46 C2（用户指令） | `infer_v2_5.py`：`stage_hooks` 可选钩子（默认 None=原路径，after_reference/after_gpt 两边界）；`engine.py`：`stage_offload` 开关（w2v-BERT/GPT 按阶段搬 CPU，结束/异常恢复）；系统页新开关 + **顺手修复了 C1 连续性复选框从未接线的缺陷** | 签名 None 默认断言 + WebUI 构建绿 |

**测试总账：122 项全绿**（新增 e1t_wiring 18 / e1t_features 13 / e1t_inference 15 /
tts_regression 5 / e1t_guard_merge 8 / svc 扩展 2，加既有用例合计 122；
验收第一轮后补 8 项白名单与栅格测试）。
WebUI 真实构建成功（含 E1-T 分组 7 控件与系统页 C2 开关）。

## 2. 硬约束核验（母规划 §13.1 对照）

- [x] 默认 `checkpoints/config.yaml` 未改（md5 实测一致），`f0_condition: false` ×2
- [x] 默认 `checkpoints/s2mel.pth` hash 未变（md5 实测一致）
- [x] 原 BigVGAN 两个文件 hash 未变
- [x] 默认 TTS 不加载 `s2mel_singing.pth`（引擎永不读 config_singing.yaml；换装只在 cover 会话内）
- [x] 默认 TTS 不挂 singing LoRA（SingingCfmSession 换装后才可挂，结束按序卸载）
- [x] `f0=None, technique=None` 路径与改前一致（solve_euler 冻结参照 bit-exact + 零初始化残差下条件路径 bit-exact 等价原路径两条测试）
- [x] cover 结束 `detach_lora` / 释放 singing CFM（会话 __exit__ finally；异常路径测试覆盖）
- [x] merge 白名单只开口 `s2mel_singing.pth`/`config_singing.yaml` 两个新名字（Guard 参数，非调用侧绕过）
- [x] 旧数据集缺 f0/tech → 无条件原路径（collate/pair/训练三级降级 + 测试）
- [x] 所有新功能默认关（singing 五控件/BoN/高频移植/阶段卸载全默认 False）

## 3. 用户运行项（本会话未跑，入口就绪）

1. **数据**：下载 GTSinger → `tools/gtsinger_import.py --src … --dataset … --pitch-aug "-12,+12"` → 数据集页体检 → **singing 模式**提取特征。
2. **capability 训练**：训练页 CFM LoRA，`singing=True`（或 CLI `--cfg_path checkpoints/config_singing.yaml`），底座=s2mel.pth（首个 run）。
3. **合并**：`merge`（arch=cfm_singing）→ `checkpoints/s2mel_singing.pth`。
4. **基线/评测**：`tools/e1t_baseline.py` 生成 `outputs/e1t_baseline.json`；G1a/G1b 按母规划 §7。
5. **identity LoRA**：singing base 上跑角色说话 LoRA；翻唱页选它 + 开「歌声模型」。
6. **声码器**：`tools/bigvgan_finetune.py --smoke` → 真训练 → `tools/vocoder_extrapolation_test.py` 对照。
7. **GPU 端 TTS A/B**（r1-9）：同 text/ref/seed，开关全部关闭下合成，与改动前波形对比（应在数值噪声内一致）。
8. 长文本 A/B（阶段46 C3 闸门）：连续性+阶段卸载的切段次数/回退率/主观衔接。

## 4. NOTICE / 合规

- `NOTICE` 已追加：上游模型**代码**修改声明（flow_matching/diffusion_transformer 可选参数 + 新模块 conditioners.py），
  上游**权重**仍未修改；衍生歌声权重受 bilibili 协议约束；GTSinger CC BY-NC-SA 4.0 商用边界不变。
- 不构成法务意见；公开发布前按母规划 §11 复核。

## 5. 回退方式

- 整体：本次改动集中在一个提交，`git revert` 即整体回退（默认行为不变，回退无风险）。
- 单项：每个新功能独立开关（翻唱页 5 个 + 系统页 1 个），关掉即回到旧行为；
  singing 权重文件可直接删除（`checkpoints/s2mel_singing.pth` 不被 TTS 引用）。
- 上游文件改动均为「可选参数默认 None + getattr 门控」，逐位一致测试在 CI 里守住。

## 5b. 可行性/优越性评审后的修复（第二轮，已落实）

独立子代理评审结论：八项调研偏离**全部保留**（其中 CFG null 旁路与 BoN 全局八度罚
被判为对母规划真实缺陷的修正）。评审发现的问题已修复：

1. **P0**：`gtsinger_import.py --pitch-aug` 的移调副本此前未写入 meta.jsonl
   （孤儿文件，增广静默失效）——已补 `DS.Utterance` 入册 + 入册计数日志。
2. force 模式的移调报告不再 `int(round())` 取整，与条件的分数移调同口径。
3. 新增两个 UI 开关（评审 #4/#5）：`音区映射`（auto 分数映射 / off 保原调硬唱）与
   `八度跳变修正`（默认开；戏腔/yodel 类 <0.4s 快速换声区内容可关）。
4. 推理侧条件 z 的 σ 改用**参考侧**旋律幅度重标定（收窄训推分布差，评审建议）。
5. 母规划条目未实现项**显式登记**（避免误以为已覆盖）：
   - Step 6 的 identity contrastive 损失与 SSIM cycle 损失（tech CE / F0 aux 有开关默认 0）；
   - Step 6 的**训练期**随机 pitch-shift 6–18 半音（以导入期固定 ±12 替代，取舍已述）；
   - Step 14 奖励中的 technique 置信度项（分类器真实训练后再接）。

### 验收第一轮（独立子代理）修复清单

结论「有条件通过」后已修复并复测（122 项全绿）：

1. **P0**：singing 主路径浮点移调量的 `:+d` 格式化崩溃（默认参数即可触发）——
   日志/进度/报告三处改自适应格式（int 用 `:+d`、float 用 `:+.2f`）。
2. `tools/e1t_baseline.py` 分段切片未取整——修复后**已在真实产物
   `20261017-154205` 上跑通**，四阶段 F0/HNR/平坦度与母规划 §1.1 诊断数字吻合
   （02 干声中位 390.8Hz），`outputs/e1t_baseline.json` 已生成。
3. `tools/bigvgan_finetune.py`/`vocoder_extrapolation_test.py` 的 BigVGAN 导入
   改为引擎同款 `indextts.s2mel.modules.bigvgan` + `from_pretrained`——
   真实构造冒烟通过（112.2M 参数）。
4. 合并白名单 8 项断言固化进 `tests/test_e1t_guard_merge.py`（含负向用例与
   硬切点栅格吸附测试）。
5. 参考 F0 提取与 `ref_seconds` 截断同口径（>15s 参考不再时间轴错位）。
6. 引擎卸载后 cfm 通道幻影 LoRA 标签清理；`tech_ce/f0_aux` 非 0 时明确
   「预留接口未消费」警告；`cover_tab` 重复定义清理。

### 验收第二轮（复验）结果

第一轮 11 项问题中 10 项确认落地无回归；第二轮发现「去重清理」误删了
`cover_tab._KEY_OUT_TO_LABEL` 的唯一定义（懒冒烟只数 blocks 查不出——
build_app 会把 Tab 渲染异常吞进 `ctx.shared["_render_failures"]`）。
已修复：

1. 补回定义；`cover_pipeline.py` 补 `Tuple` 导入（lint 级）。
2. **新增全 Tab 真实渲染冒烟测试**（`test_webui_all_tabs_render`：断言
   `_render_failures` 为空）——堵死「测试绿但 UI 构建崩」这类盲区，
   该测试在修复前实测会红。

第二轮后总账：**123 项测试全绿**；`tools/e1t_baseline.py` 已在真实产物
20261017-154205 上生成 `outputs/e1t_baseline.json`（四阶段整曲 SS
0.5673/0.5392/0.5269 与母规划 §1.3 精确一致）；BigVGAN 入口真实构造
（112.2M 参数）；TTS 三件套 hash 复核未变。

## 6. 已知限制（如实）

- 22k 声码器的高音天花板仍在（调研 Q5：文献无「不重训的天花板突破」；SYKI 移植只是应急，预期收益有限）。
- technique auto 依赖分类器真实训练；未训练前 auto 自动回退 control（报告可见回退原因）。
- BoN 候选的 SS 用 campplus 逐块嵌入（快但与整曲 SS 口径略有差异），报告已分离记录。
- `plan_chunks` 返回三元组（含每块重叠），旧调用方只有本项目内两处（已同步）。
- 推理端参考帧区段的 technique 恒为 control（训练侧 prompt 可带非 0 标签；
  GTSinger 多数帧本就是 control，影响温和——分类器可用后可对参考跑一次补齐）。
- BoN 奖励中 HNR 权重（×0.05）使 F0 项主导，与母规划「HNR 与其他并列」原意有偏差，
  真实候选择优出现偏差时回调权重。
---

## 7. 追加（47b，2026-10-17 晚）：🎵 歌声训练（E1-T）向导页

应用户要求补齐训练链的 UI 接口——**独立标签页 + 分步引导 + 可复用数据的提示**：

- **新页 `webui_app/tabs/e1t_tab.py`**（TAB_SPECS 挂在「🎓 训练」之后）：
  - **七步状态总览**（`e1t_wizard.status_markdown()`，纯文件系统检测）：
    前置文件 / ①引导复制 / ②歌声数据集 / ③歌声特征条数 / ④capability run /
    ⑤合并成果 / ⑥身份 run，每步 ✅/⬜ + 「下一步做什么」+ **可复用性说明**
    （数据一次准备全角色共享；新角色只跑⑥；说话数据集旧特征直接复用）；
  - ②③④⑤⑥ 各为独立 Accordion 步骤卡片；GTSinger 许可边界
    （CC BY-NC-SA：个人研究可用 / 免费共享需署名同协议 / 商用需授权）
    常驻页首与②步骤内；
  - 后台执行与一键三连**同一套 TrainRunner**（互斥/进度/停止/日志 Timer 轮询）：
    训练与特征提取 `require_engine="unloaded"`（自建模型实例期间引擎不得加载），
    导入/合并走 CPU；合并报告落 `merge_md`；
  - ⑦ 静态使用指引（翻唱页选身份 run + 勾歌声模型 + e1t_baseline 对比口径）。
- **服务层 `webui_app/services/e1t_wizard.py`**：状态检测、run 分类
  （run.json 的 `singing.enabled` / `singing.cfg_path` → capability / 身份 /
  普通 TTS 三类）、六步后台 fn 构造器；训练 fn 先 preflight 再 run，
  capability=`singing=True`（条件透传+均衡采样），身份=`singing=False`
  +歌声底座（conditioner 自动冻结）。
- **GTSinger 导入服务化 `webui_app/services/gts_import.py`**：核心逻辑从
  CLI 脚本搬出并函数化（progress/should_stop、technique.json 幂等合并、
  增广副本入册）；`tools/gtsinger_import.py` 改为薄 CLI 壳，行为不变。
- **参数注册 `webui_app/params_e1t.py`**（9 个旋钮：语言过滤/音高增广/
  条数上限/复制开关/预设/帧数/epoch 等）+ `params.py` 聚合。
- 测试：`tests/test_e1t_wizard.py` 9 项（run 分类三态、引导复制不覆盖防护、
  状态面板七步、增广解析）+ 全 Tab 渲染冒烟自动覆盖新页；**全套 130 项绿**。
---

## 8. 追加（47c）：数据目录 + 镜像拉取 + 一键全链 + 成果自动接线

应用户要求「数据在项目下拉取好、默认目录配置好、配置参数后一键完成所有
训练和成果的自动接线」：

- **数据目录约定**：`datasets/_sources/GTSinger`（gitignore 覆盖；
  `list_datasets` 只认带 info 文件的目录，_sources 不会混进数据集下拉）。
- **镜像拉取 `webui_app/services/gts_fetch.py`**（CLI 壳
  `tools/fetch_gtsinger.py`）：hf-mirror 优先（`HF_ENDPOINT` 沿用项目
  config 的 setdefault 默认，**不涉及 GitHub**）；逐文件 + 6 线程拉取
  （GTSinger 是数万个小文件，逐文件才有真实进度且可中断——断点续传由
  hf_hub 本地缓存承担，重复执行增量补齐）；**语言子集过滤**
  `plan_files()`（纯函数，zh+en 子集约 15–20GB vs 全量 40GB 量级；
  过滤结果为空时宁可回退全量，不给「假完成」）。
- **向导页升级**：新增「🚀 一键全链」与「⓪ 镜像拉取」两张步骤卡；
  ② 的源路径默认值 = 数据目录；七步状态总览增加 ⓪ 数据源行。
- **一键全链 `pipeline_fn`**：拉取（数据缺失且允许时）→ ①引导 → ②导入
  → ③特征 → ④capability → ⑤合并 → ⑥身份（可选）→ ⑦**接线翻唱页**。
  每段失败即停、成果保留（状态总览天然增量，再点一次跳过已完成部分）；
  进度按段映射到全链 0–100%。
- **接线 = 写翻唱页参数记忆**（`wire_cover_state`：singing_enable=True +
  身份 LoRA + preserve/auto/octave_fix 默认），下次进翻唱页直接可用；
  只写 E1-T 键，不动用户其它翻唱参数。
- **顺手修了一个真 bug**：`svc_state._COVER_KEYS` 白名单缺 E1-T 七键——
  翻唱页保存的歌声开关重启后会被静默丢弃（阶段37「白名单静默丢键」同款
  事故），已补齐并用回归测试钉住。
- 测试：`tests/test_e1t_wizard.py` 扩到 17 项（语言过滤三态/端点默认/
  全链顺序-失败即停-接线/白名单钉）；全套 **140 项绿**；WebUI 构建通过。
---

## 9. 追加（47d）：⑥ 角色数据集友好显示 + 默认 run 名自动填充

- **痛点**：⑥ 的数据集下拉原来是一排 `oneclick_20261011-1556`——完全无法区分。
- **实现**：`dataset_char_names()` 经 run.json 的 `dataset` 字段反推「数据集 →
  角色名」（run 名剥 `_gpt/_cfm/_dpo` 后缀 = 一键三连当时填的模型名称）；
  `dataset_display_labels()` 生成显示标签（如 `新锁暝 · 一键三连
  （oneclick_20261011-1556）`；无 run 映射回退统计摘要；GTSinger 标记为
  歌声数据且**不进身份下拉**）；⑥ 下拉改 (标签, 值) 元组 choices，
  角色名命名的排前面。
- **默认 run 名**：选数据集后自动填 `<角色名>_singing_identity`（change 事件
  只在空/仍是自动值时覆盖，手改不动；⑥按钮与一键全链另有兜底）。
- **⑥ 说明明确回答了用户的素材问题**：用的就是一键三连切好、降噪/响度锚定、
  ASR 对齐、特征已提好的那批音频，特征直接复用不必重提。
- 测试：TestCharNameOfRun / TestDatasetLabels 共 7 项（后缀剥离、一键三连
  标签、无 run 回退、统计兜底、下拉排序与过滤、默认名）；全套 **147 项绿**。
