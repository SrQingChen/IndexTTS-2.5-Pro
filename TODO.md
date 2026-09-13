# IndexTTS-2.5 增强工程 · 待办清单

> 这份清单是本工程的**唯一进度真相**。每完成一项就把 `[ ]` 改成 `[x]`，
> 并把「验收证据」一栏填上可复核的数字（测试通过数 / 实测指标），
> 不要只写「已完成」—— 三个月后没人记得当时是怎么验的。
>
> 状态标记：`[x]` 已完成并验收 · `[~]` 进行中 · `[ ]` 未开始
> 最近更新：阶段 2 全部完成 + 工程化发布（环境 / 启动器 / 许可 / GitHub），2026-09-13

---

## 原始需求（用户原话拆解）

| # | 需求 | 状态 |
|---|---|---|
| R1 | 优雅的可视化 WebUI，模块化、UI 清晰符合逻辑 | `[x]` 阶段 1 |
| R2 | **所有功能接口都做好 UI** | `[x]` 阶段 1（官方 `webui.py` 未改动） |
| R3 | LoRA 模型**自训练** | `[x]` 阶段 2（GPT / CFM / DPO 三目标全通，真机验收 33/33） |
| R4 | 模型推理 | `[x]` 阶段 1 |
| R5 | **所有参数的提示信息** | `[x]` 阶段 1（参数手册 Tab + 逐项 tooltip） |
| R6 | 补齐缺失的底模（镜像源优先） | `[x]` 7.89 GB 全部就位 |
| R7 | 追加：LoRA 微调的**泛化能力保护** | `[x]` `guard.py` 九道防线 |
| R8 | 追加：**拼音标注**纠音 | `[x]` 66/66 测试通过 |

---

## 阶段 1 · 可视化 UI（已交付验收）

| ID | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| a1 | 拉取全部模型资源 | `[x]` | 7.89 GB，仅缺示例音频（不影响功能） |
| a2 | 建 `.venv --system-site-packages` 并补依赖 | `[x]` | 复用系统 torch 2.7.1+cu118 |
| a3 | 验证 IndexTTS2 加载 + 推理 | `[x]` | 模式 0 / 2 / 3 全部跑通 |
| a4 | `webui_app/` 模块化骨架 | `[x]` | config / params / theme / widgets / context |
| a5 | 引擎服务：懒加载 / 卸载 / 显存管理 | `[x]` | QwenEmotion 串行策略已实测 |
| a6 | Tab 语音合成：全参数 UI + 逐项提示 | `[x]` | — |
| a7 | L0 参考音频工作台 | `[x]` | 分析 / 智能切片 / 降噪归一已实测 |
| a8 | Tab 批量合成 + Tab 预设管理 | `[x]` | — |
| a9 | Tab 模型资源管理 | `[x]` | 复用 `model_fetcher` + 实时进度 |
| a10 | Tab 系统监控 + Tab 参数手册 | `[x]` | — |
| a11 | **阶段 1 验收** | `[x]` | 46/46 集成测试 + 浏览器端到端 |

**入口**：`webui_pro.py`（官方 `webui.py` 一行未动）

---

## 追加需求 · 拼音标注与泛化保护

| ID | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| x1 | 拼音标注：服务层 + 合成页 UI + 手册 | `[x]` | 66/66 |
| x2 | `guard.py` 泛化保护九道防线 | `[x]` | 192/192（`tools/guard_test.py`） |
| b2r | 底座蒸馏回放集 `replay.py` | `[x]` | 已并入 GPT 探针：145/145 |

**九道防线**：底座只读快照 · 参数可配置校验 · 注入面收敛 · 权重漂移体检 ·
早停 · checkpoint 保险库 · 回放抗遗忘 · adapter 强度旋钮 · 一键回滚回放

---

## 阶段 2 · 训练体系（L0 → L3 四层全做）

### L1 · SFT LoRA

| ID | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| b1 | 数据集构建 + 特征离线预提取 | `[x]` | **153/153**（`tools/features_probe.py`）；卸载后显存回落 0.04 GB |
| b2 | **GPT(T2S)** LoRA 训练器 | `[x]` | **148/148**（`tools/gpt_train_probe.py`）；val 5.7057→1.7233（降 69.8%），底座逐字节未变 |
| b3 | **CFM(S2M)** LoRA 训练器 | `[x]` | **163/163**（`tools/cfm_train_probe.py`）；val 1.3378→1.1643（降 13.0%），底座逐字节未变，续训起点 val 与 checkpoint metric 逐位一致 |
| b3a | 核对 `s2mel.pth` 键名与 CFM 结构 | `[x]` | 已确认：256 个张量全在 `estimator.` 下；`load_state_dict` missing 0 / unexpected 0；98,187,344 参数；唯一 Dropout 是 `estimator.wavenet.drop` p=0.2 |
| b3b | 写 `cfm_lora.py` | `[x]` | 1132 行：Options / 配对策略 / collate / load_base_cfm / CfmTrainer；探针 [2][3][6][7][8] 节逐项验证 |
| b3c | `tools/cfm_train_probe.py` | `[x]` | 163/163：小尺寸 DiT 闭环 + 全部陷阱回归钉（mask_content / rng_guard / dropout_off / 注入面 / '*' 展开 / 续训） |

### L2 · 偏好对齐

| ID | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| b4 | 奖励打分（WER via whisper + SS via campplus） | `[x]` | **49/49**（`tools/reward_probe.py`）：whisper small 转写逐字一致、自相似=1.0、端到端满分/坏样本 reward 1.0 vs -0.01 |
| b5 | DPO 偏好对构造 + 训练器 | `[x]` | **70/70**（`tools/dpo_probe.py`）：★ln2 不变量（step-0 loss=ln2、acc=0）、loss 0.693→0.568、acc→1.0、margin 0.275、续训逐位还原、构造器编排（margin 过滤/追加/覆盖/中断） |

### L3 · 评测与落地

| ID | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| b6 | 自动评测台：A/B 对比 + 指标报告 + 试听 | `[x]` | **28/28**（`tools/eval_probe.py`）：同种子公平对比、胜负表、report.json/md、试听 wav、引擎状态还原 |
| b7 | LoRA 权重合并回 `gpt.pth` / `s2mel.pth` + 推理端加载（含强度旋钮） | `[x]` | **25/25**（`tools/merge_probe.py`）：Conv1D 转置/Linear 双向合并数学逐层对账（含 α/r scaling）、红线拦截、备份、挂载/强度旋钮 |
| b8 | 训练相关 4 个 Tab 接入 UI（含泛化保护面板） | `[x]` | **20/20**（`tools/stage2_ui_probe.py`）+ build_check 11 Tab；`runner.py` 后台执行/互斥/取消 + 数据集/训练/对齐/评测部署四个 Tab |
| b9 | **阶段 2 验收**：完整训练闭环跑通 | `[x]` | **33/33**（`tools/stage2_acceptance.py` 真机 4.0 分钟）：引擎合成 28 句建集 → 特征提取 → GPT 真训（峰值 1.87GB，漂移 0.000423）→ CFM 真训（漂移 0.00319）→ DPO 偏好对真机构造（8 候选 1 对成对）→ A/B 评测 → 强度旋钮 48 层 0.7 → GPT/CFM 合并产物校验 missing/unexpected=0，全程底座复校通过 |

### 杂项

| ID | 内容 | 状态 |
|---|---|---|
| z1 | 清理临时文件（`_*.log`、`tools/_gpt_bisect.py`、`tools/_dit_timing.py`、`tools/_cfm_scan.py` 等）；**保留** `tools/_env.py`（探针公共引导，非临时文件） | `[x]` | 已删 10 个临时文件（7 根目录日志/扫描 + 3 个 tools 调试脚本），仅存 `tools/_env.py` |

---

## 已建成的代码资产

| 文件 | 作用 | 行数 |
|---|---|---|
| `webui_app/training/guard.py` | 九道防线：快照 / 校验 / 扫描注入面 / 漂移 / 早停 / 保险库 / 显存体检 | ~1515 |
| `webui_app/training/dataset.py` | 数据集与 `meta.jsonl`（含长度字段回写） | ~700 |
| `webui_app/training/features.py` | 离线特征提取（campplus / w2v-bert / codec / mel / mu_prompt / mu_target） | ~836 |
| `webui_app/training/forward.py` | **唯一**的训练前向，绕开 3 个官方陷阱 | 435 |
| `webui_app/training/trainer_base.py` | GPT / CFM 共用骨架 + `FeaturePool` + `inject_lora` | ~1286 |
| `webui_app/training/gpt_lora.py` | GPT(T2S) LoRA 训练器 | 758 |
| `webui_app/training/replay.py` | 底座蒸馏回放集（抗灾难性遗忘） | ~548 |
| `webui_app/training/runs.py` | 训练记录 / adapter 同步 / 保险库入口 | ~430 |
| `tools/features_probe.py` | 特征链路回归 | 153 项 |
| `tools/gpt_train_probe.py` | GPT 训练器回归 | 148 项 |
| `tools/guard_test.py` | 泛化保护回归 | 194 项 |
| `webui_app/training/cfm_lora.py` | CFM(S2M) LoRA 训练器 | 1132 |
| `webui_app/training/reward.py` | 奖励打分（whisper WER + campplus SS） | ~400 |
| `webui_app/training/dpo.py` | DPO 偏好对存储/构造器/训练器 | ~800 |
| `webui_app/training/evaluate.py` | A/B 评测台（复用 reward） | ~400 |
| `webui_app/training/merge.py` | LoRA 合并 + 推理端挂载/强度 | ~300 |
| `webui_app/training/runner.py` | 后台执行器（互斥/取消/进度） | ~250 |
| `webui_app/tabs/dataset_tab.py` 等 4 个 | 数据集 / 训练 / 对齐 / 评测部署 Tab | ~1600 |
| `tools/cfm_train_probe.py` | CFM 训练器回归 | 163 项 |
| `tools/reward_probe.py` | 奖励打分回归（含真模型段） | 49 项 |
| `tools/dpo_probe.py` | DPO 回归（含 ★ln2 不变量） | 70 项 |
| `tools/eval_probe.py` | 评测台回归 | 28 项 |
| `tools/merge_probe.py` | 合并/挂载回归（双向数学对账） | 25 项 |
| `tools/stage2_ui_probe.py` | UI/Runner 集成回归 | 20 项 |
| `tools/stage2_acceptance.py` | 阶段 2 真机验收闭环 | 33 项 |

---

## 已钉死的关键事实（改动前务必先看）

**架构常量**

- w2v-BERT / codes / mel 帧率 = 50 Hz / 25 Hz / 86.13 Hz（22050÷256）
- `MEL_PER_CODE = 2 × 1.72 ≈ 3.44`；1.72 是 **50Hz→86.13Hz**，不是 codes→mel
- GPT：24 层 / 1280 维 / 20 头 / 813M / `max_mel_tokens=1815` / codebook 8194
- CFM(DiT)：13 层 / 512 维 / 8 头 / **98,187,344** 参数 / L1Loss / `in_channels=80`
- 参考音频在推理时被截到 **15 s**（`_load_and_cut_audio(..., 15)`）→ 1292 mel 帧上限
- 单条音频硬上限 1815 ÷ 25Hz = **72.6 s**
- 真类在 `indextts/gpt/model_v2.py`；`model_v2_5.py` 是死代码

**六个必须绕开的官方陷阱**（全部实测复现，已在 `forward.py` / `trainer_base.py` 里落地）

1. `UnifiedVoice.forward()` 算 text_emb 时**漏了 `lang_embedding`**（:639 vs :681）
2. `UnifiedVoice.forward()` 在 bf16 下 RuntimeError（:633 `torch.zeros` 未指定 dtype）
3. `BASECFM.forward()` **不能在 `eval()` 下调**：`prompt_lens` 被传进 `mask_content`，
   触发 class_dropout → prompt_x / mu / style **全被乘 0**，val loss 变成与条件无关的常数
4. CFM 必须先 `setup_caches()`，且缓存属性住在 **`DiT.transformer`** 上（DiT 只转发）；
   `layers_emit_skip` / `layers_receive_skip` 也在这里设置，不调就 AttributeError
5. **reentrant 梯度检查点 + 全冻结底座 = 训练静默失效**（梯度恒 0）→ 必须 `use_reentrant=False`
6. `Transformer.forward` 只在 `mask is None` 时读 `causal_mask`，而 DiT 总是传 mask
   → 官方按 8192² 分配的 **67 MB causal_mask 从不被读取**，纯浪费

**四个自己踩出来的 bug（已修，均已加回归钉）**

7. `resume_from` 只恢复优化器状态、不恢复 adapter 权重 → 续训从纯底座重来且**不报错**
8. `evaluate()` 的 val 下标落到训练池上 → 「验证集」实际是训练集前几条，145 项测试全过都没发现
9. CFM 注入面预设失效：DiT 的注意力叫 **`wqkv` / `wo`**，不是 `qkv` / `out_proj`
10. `import_audio(copy=True)` 会把文件**重命名成 `<uid>.wav`** → 按原文件名做的映射永远匹配不上

**环境陷阱**

- Windows WDDM：显存溢出后**不报 OOM**，而是静默降速 20~30 倍（CFM 25 步 2.4s → 66s）
  → 所以 `vram_preflight(raise_on_short=True)` 必须硬拦
- RTX 4060 Laptop 8GB：引擎常驻 4.94~5.70 GB；训练前必须先卸载引擎
- `copy.deepcopy(CFM)` **会失败**（`weight_norm` 的 hook 存了非叶子张量）→ 需要副本时重新构造
- PowerShell 不支持 `&&`，用 `;`


---

## 阶段 3 · 工程化与发布（已完成）

| ID | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| c1 | 修复被 `uv sync` 覆盖的环境，补齐缺失依赖 | `[x]` | uv venv（Python 3.11.13 + torch 2.8.0+cu128）补装 gradio/peft/pypinyin；`build_check` 11 Tab 通过 |
| c2 | `pyproject.toml` 声明真实依赖 + 作者 | `[x]` | 基础依赖加 `peft>=0.14`、`pypinyin>=0.51`；作者加 SrQingChen；`uv lock` 一致（191 包） |
| c3 | 启动器 `start.bat` | `[x]` | 自检 venv/依赖/模型/CUDA → 缺依赖可一键补装 → 启动并开浏览器；实测 7860 监听正常 |
| c4 | 作者标签与许可分层 | `[x]` | `NOTICE`（归属链 + 修改清单 + 上游 §4.1a 强制免责声明）、`LICENSE-ADDITIONS.txt`（GPL-3.0）；上游 `LICENSE`/`LICENSE_ZH.txt` 原样保留 |
| c5 | 清理测试产物并推送 GitHub | `[x]` | 保留 4 份验收报告到 `docs/verification/`；推送 **409 文件 / 39.8 MB**（无模型权重/venv/产物），https://github.com/SrQingChen/IndexTTS-2.5-Pro |
| c6 | 项目专属 README（展示 UI 与功能集成） | `[x]` | 新 `README.md`（11 Tab 截图 + 能力对照表 + 训练体系 + 验证结果 + 分层许可）；上游 README 原样存档到 `docs/README_UPSTREAM.md` |
| c7 | `tools/ui_screenshots.py` 自动截图 | `[x]` | 走 Chrome CDP（无新依赖）自动遍历 11 Tab 截图到 `assets/ui/`；**踩坑：Gradio 5 的 Tab 栏在 DOM 里有两份同名按钮**（1px 隐藏副本 + 32px 真实 tab），按文本找 `button` 必点到死元素且不报错 |

**许可分层说明**（为什么不整体换 MIT/Apache）：

- 上游 `LICENSE` 是 bilibili 自定义模型许可，§3.4(b) 要求保留原始版权声明与协议副本、
  §4.1(a) 要求下游声明免责 —— **不可替换**，替换即违约。
- 因此采用两层：Model 与上游代码 → bilibili 许可（不变）；
  SrQingChen 的**原创新增文件** → GPL-3.0（copyleft + 保留署名），
  正好满足「可自由拉取优化，但基于本项目修改需遵循协议并保留贡献」。
- 两层冲突时以上游协议为准（GPL 不延伸至 Model）。详见 `NOTICE`。

---

## 环境陷阱（新增，务必先看）

- **`uv sync` 会就地覆盖 `.venv`**。本项目原环境是
  `.venv --system-site-packages`（借系统 Python 3.10 的 torch 2.7.1+cu118）
  + venv 内装 gradio/peft/pypinyin。直接跑 `uv sync` 会把这三样一起清掉，
  表现为「启动就报 No module named 'gradio'」。
  修复：`uv sync --extra webui`（peft/pypinyin 已在基础依赖里）。
- **不要用 `uv sync --all-extras` 作为默认路径**：它还会拉
  deepspeed / flash-attn / torch_compile，需要 CUDA 工具链且非必需。
- **`.bat` 必须是纯 ASCII**。cmd.exe 用**当前代码页**逐行解析批处理文件，
  UTF-8 中文会被按 GBK 误解并连带破坏后续命令行（实测把 `echo` 行解析成
  命令、label 被截断）。中文说明写在 README / 本文件里。
- **批处理里 `%s` 会被当成环境变量吃掉**。避免在 `.bat` 内嵌 Python 字符串里
  使用 `%`（改用逗号分隔的 print 参数或 `.format()`）。
