# AI 翻唱 + AI 变声 开发规划（原生 IndexTTS2.5 路线）

**日期：** 2026-10-16
**前置：** `docs/research-ai-cover-voice-changer-2026-10.md`（调研报告，含附录 A 接线级调研 / 附录 B 操作配方）
**本轮新增调研：** 学术补强（F0 移植音质、RVQ 音高丢失、少步 CFM、分块流式、WASAPI I/O、audio-separator API）——结论已并入本文。
**路线裁决：** 应用户要求走**自研原生管线**——用 IndexTTS-2.5 引擎本体 + 已挂载 LoRA 做推理，**不引入 RVC/Seed-VC 外部 SVC 引擎**。分离/去混响引入 `python-audio-separator`（MIT，可选 extra，未装时优雅降级）。

---

## 0. 学术补强结论（本轮，指导实现参数）

1. **RVQ 量化丢失音高是实锤**（arXiv 2604.07467 "Lexical Tone is Hard to Quantize"）：SSL 连续特征携带半音级旋律，过 25Hz ResidualVQ 后音高信息大幅退化 → **旋律必须由外置 F0 管线保证，不能指望内容 token**。
2. **WORLD 改 F0 重合成有可闻劣化**（KaraTuner arXiv 2110.09121 等多篇）：参数声码器音质弱于神经声码器 → **默认"局部修复"模式：只对 |源F0−输出F0| > 40 音分且持续 > 80ms 的浊音段做 WORLD 重合成**，全曲重合成仅作兜底；比值曲线先中值滤波（窗5）再滑动平均，防逐帧抖动粗糙感。
3. **cheaptrick 低音陷阱**：F0 移到低音区必须 `f0_floor≤65` 且 `fft_size=2048`（2^ceil(log2(3·22050/f0_floor))），否则整段误判清音；唱歌上限放宽 `f0_ceil=1100`（C6=1046）。
4. **少步推理**（Seed-VC 官方口径）：8–10 步听感接近 25 步；`cfg→0` 提速约 1.5× 且差异细微 → **实时档 = 10 步 + cfg 0 ≈ 5× 提速**；离线保 25 步 + cfg 0.7。
5. **分块流式模板**（Seed-VC 实测 530ms 的配置）：block 0.18s（对齐整数 mel 帧）+ 交叉淡化 0.04s + **左上下文 2.5s**（SSL 特征需要历史，编码后丢弃边缘帧）+ 右上下文 0.02s + VAD 门控静音段。
6. **BigVGAN 非因果**：逐块声码需前后 pad 0.05–0.1s 再裁掉；**离线翻唱整块过声码器**（拼接在波形域，不在 mel 域）。
7. **sounddevice + WasapiSettings(exclusive=True, auto_convert=True)** + blocksize 256–512 + latency='low'；回环延迟 = 脉冲互相关实测（UI 内置测量，不信算法延迟）。
8. **audio-separator 编程 API**：`Separator(output_dir=…, output_format='WAV')` + `load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')`（人声 SDR 12.98，社区默认）+ `UVR-DeEcho-DeReverb.pth`（去混响）；roformer 显存 2–4GB，CPU 分钟级/首（仅离线可接受）。
9. **F0 器分工**：干净人声 harvest/RMVPE/CREPE 差距 <2 RPA；**分离质量比 F0 器选择更关键**。RMVPE 权重本项目缺失（`assets/rmvpe/rmvpe.pt` 不存在）→ 首版默认 pyworld harvest（零新依赖），RMVPE 作可选增强（HF `lj1995/VoiceConversionWebUI` 的 `rmvpe.pt`，走 model_fetcher 三级镜像下载）。

---

## 1. 目标与范围

### 1.1 AI 翻唱（离线，`🎤 AI翻唱` tab）

```
用户导入音频 + 选择输入状态(原歌曲/分离人声/干净人声)
  → 按状态路由: 人声分离(BS-Roformer) → 去混响(DeEcho-DeReverb) → 预处理(去DC/降噪/归一,不动段内停顿)
  → 【自研核心管线】源16k→w2v-BERT→RVQ码本→length_regulator→CFM(25步,挂CFM-LoRA)→BigVGAN
      + F0 管线(源F0提取→八度映射→输出WORLD局部修复/全曲移植)
  → 后处理(HF治理/频谱画像对齐/气声软化,复用audio_lab,低强度)
  → 可选超分(AudioSR,默认关) → 可选"自动合成"(废弃物BGM+转换人声→混音)
  → 产物落盘 + 质量报告(SS/F0相关/八度错误)
```

### 1.2 AI 变声（准实时，`🎧 AI变声` tab）

```
选择输入设备(WASAPI) → 自动输入调理(DC/单声道/重采样/轻AGC)
  → 流式分块: 左上下文2.5s拼接编码→丢弃边缘帧→CFM(10步,cfg0)→BigVGAN(块pad)→等功率交叉淡化
  → 可选逐块F0跟随(源F0→八度映射→块内WORLD移植)
  → sounddevice 输出到监听设备(耳机)/虚拟声卡; WebUI为控制面板+延迟实测+电平表
多档位: 快速/均衡/质量 (块大小×步数×上下文×F0模式的组合)
+ 离线文件变声模式(sounddevice未装/不需要实时时的兜底,同核心管线)
```

### 1.3 明确不做（本期）

- SVC 训练器 / 44.1k 数据分支 / f0_condition 训练开轴（研究线，留接口不实现）
- RVC/Seed-VC 引擎集成、karaoke 主唱/和声分离（留"高级-可选"入口但默认隐藏）
- 导演台联动（对转换链无意义，不迁移——按用户要求"没用或副作用的就不做"）
- 实时变声的浏览器音频输出（gradio 无低延迟音频流通道，输出走本地声卡——UI 内说明）

---

## 2. 关键技术决策（D1–D14）

**D1 内容通路变体**：默认变体 A（`quantize→decode`，分布内，与 GPT 生成路径同域）；高级参数"保留细节(连续特征直通)"切变体 B（与 prompt_condition 同域同构）。两变体运行时可切。

**D2 F0 管线三模式**（`svc_f0.py`）：
- `off`：不移植（说话变声可用，韵律残留可接受时）
- `auto`（默认）：先测源/输出 F0 轮廓相关系数 r → r≥0.75 走**局部修复**（只修 >40 音分且 >80ms 的浊音段），r<0.75 走**全曲移植**（旋律完全由源决定）
- `force`：无条件全曲移植
- 八度映射：源/目标音区中位数差自动取整到八度（+12/-12/0 半音），允许手动微调（±12 内任意半音）；比值曲线中值滤波(窗5)+滑动平均；WORLD 参数 `f0_floor=55` **必须同传 harvest+cheaptrick+d4c 三处**（只传 harvest 会让 <71Hz 帧的谱包络分析错误）、`f0_ceil=1100`、`frame_period=5`、`cheaptrick(fft_size=2048)`；**全曲移植抄 `restore_f0_range` 的时长保持模式**（synthesize 后 `[:len(y)]`+不足补零——绝不能用 `lengthen_abrupt_releases`，它故意插帧改时长，会毁 BGM 对拍；只借其"多合成 sr//20 再裁齐"与谱自检守卫技巧）。
- **全曲 F0 分窗**：harvest 300s@22k 实测 67.3s（RTF≈0.22），全曲可行但慢 → 按 30s 窗分窗提取、5ms 网格对齐拼接（`len(f0)=floor(N/(sr·0.005))+1`，注意 +1），窗间用重叠消除边界跳变；f0_stats 的 30s 截断是调用方护栏，svc_f0 自己管理分窗。

**D3 LoRA 双通道事实**：VC 路径绕过 GPT → **GPT LoRA（语气）不生效**，仅 **CFM LoRA（音色）生效**（`attach_lora(target="cfm")` 包在 `s2mel.models["cfm"]` 上，对 `cfm.inference` 透明）。UI 双下拉都提供，GPT 通道标注"转换路径不生效（旁路 GPT），仅 TTS 合成使用"；音色锚定主路径 = 参考音频（voice_bank/上传）+ CFM LoRA 增强。

**D4 分块（离线）**：**CFM 完全无状态可重入**（审查核验：`setup_caches(use_kv_cache=False)` 不分配 KV，DiT 纯 SDPA 全注意力，input_pos 每次切片重置，无推进状态）→ 逐块独立调用不需要任何重置；硬上限是 input_pos/freqs_cis 缓冲 16384 帧≈190s，**实际约束是 T² 全注意力显存/算力** → 取**安全块 60s**（含参考 ≤1292 帧）。切块在块内**低能量段（歌唱气口）优先**（目标块界 ±5s 内搜 RMS 谷，找不到硬切），**块间输入重叠 = crossfade 时长（40ms）**：相邻块输入首尾重叠 40ms，输出波形域等功率交叉淡化——不重叠的淡化会每块净丢 40ms（4 分钟歌漂移 ~160ms，对拍废）。参考/style 全块共享。

**D5 流式（变声）**：设备采样率原样入环形缓冲（**重采样发生在窗口拼接之后**——Resample 是无状态线性相位 FIR，块首尾滤波瞬态必须落在被丢弃的上下文里）；每窗 = 左上下文（档位决定 1.0–2.5s）+ 新音频块 + 右 0.02s → 整窗 48k→16k/22k → 过 w2v-BERT/quantize/decode → **只保留新块对应帧**（帧映射 j ≈ 0.5805·k − 0.335，mel 帧 k ↔ w2v 帧 j；左边缘丢弃 ≥3 帧覆盖注意力边界效应+重采样瞬态+decode 的奇数 T off-by-one）→ length_regulator → CFM（prompt=参考，**实时档参考强制 ≤5–8s**，参考每块全量参与注意力是延迟大头）→ BigVGAN（**mel 域 pad 0.06s：用 log-mel 下限值/边缘复制填充，禁止 zeros**——log 域补零=能量异常帧→爆点；pad 后裁掉）→ 输出流交叉淡化：**feed() 返回的新块波形自带与上块的重叠段**（重叠=淡化时长），调用方淡化不丢时长。F0 跟随模式：源 F0（harvest 5ms 网格，环形缓冲连续提取）→ 八度映射（**系数由参考音频与源首块预计算一次，全局恒定**，不逐块估计）→ 块内 WORLD 移植，**以转换输出侧 VUV 为门控**（仅输出浊音帧替换 F0，防 voiced/unvoiced 错位产生垃圾声——项目已有教训）。

**D6 档位（变声）**（帧数按 mel 帧公式 (N−256)//256+1 精确计算，以帧为准）：
| 档 | 块 | 步数 | cfg | 左上下文 | 参考时长 | F0 | 预期延迟 |
|---|---|---|---|---|---|---|---|
| 快速 | 21帧(0.242s) | 6 | 0 | 1.0s | ≤5s | off | ~300–500ms |
| 均衡(默认) | 31帧(0.357s) | 10 | 0 | 2.0s | ≤6s | 平滑 | ~450–700ms |
| 质量 | 63帧(0.729s) | 16 | 0.3 | 2.5s | ≤8s | 跟随移植 | ~800–1200ms |
每块重编码冗余（(左+块)/块 ≈ 3.6–6.5×）与 CFM 对参考全长的每步注意力都已计入预算；延迟数标"待实测"，UI 内置**回环实测**（脉冲互相关）显示真实端到端延迟。

**D7 显存排程**（8GB 全链互斥，复用 audio_sr 模式）：翻唱链 = 需分离时 `eng.unload()→分离→release(镜像 audio_sr.release_all 的显式释放+empty_cache+ipc_collect)→eng.load()→VC`；超分段 `eng.unload()→SR→eng.load()`。**全链 try/finally：记录进入前引擎状态，异常/停跑时恢复**（防"引擎停在卸载态"困惑用户）。变声 = 引擎常驻 + **engine 层活动 guard**：在 `engine.load/unload` 内部拦截（复用 `_busy_runner_engine_req` 插点，engine.py:226-242 模式，批量页/训练 runner/合成页全部被保护，而非只挡 UI 按钮）；变声活动期间同时拒绝 QwenEmotion 挂载（4.94+1.2+峰值≈6.4GB，8GB 卡临界）。

**D8 分离模块**：`audio-separator` 为**可选 extra**（`pip install "audio-separator[gpu]"`，与 audiosr 同样的 available() 探测+降级模式；PyPI 元数据核验其 torch 依赖区间 `>=2.3,<3` 与本项目 torch 2.8 兼容）。权重首次使用自动下载到 `checkpoints/separators/`（model_file_dir 指向项目内；**checkpoints/ 是 artifacts 盘点的明文排除区，故分离权重只纳入模型页 model_fetcher 审计、不进清理页产物盘点**——遵守既有不变量）。人声分离默认 `model_bs_roformer_ep_317_sdr_12.9755.ckpt`；去混响默认 `UVR-DeEcho-DeReverb.pth`（均为 UI 可选）。分离不改时长（与源曲同时间轴，混音天然对齐）。

**D15（审查新增）全局约定**：管线内波形一律 **float32 ±1**（int16 尺度陷阱项目已三次踩坑），仅最终落盘转 PCM_16（soundfile）；quantize 的 squeeze 由 num_quantizers=1 触发（dim0 是 N 非 batch，B>1 批处理禁用）；decode 奇数 T 输入有 +1 帧，时长最终由 ylens 强制；mel 帧数公式 `(N−256)//256+1`、w2v 帧数 `floor((N16k−400)/320)+1`。sounddevice 用法：**默认共享模式 + `WasapiSettings(auto_convert=True)`**（一手核验：exclusive 与 auto_convert 互斥，独占模式采样率不匹配直接开流失败）；"独占模式"作为高阶选项，启用前用 `query_devices` 校验 `default_samplerate` 一致。

**D9 预处理（翻唱专用，不动段内停顿）**：去DC → 边缘静音保守裁剪（阈值 -45dB，仅首尾）→ 可选轻降噪（noisereduce 低强度）→ RMS 峰值保护归一（-3dBFS 余量，**不压缩动态**，保演唱起伏）。拒绝复用 `enhance()` 的段内停顿治理。

**D10 超分**：默认**关**（勾选框不选中），提示"约 17.8× 实时耗时"；启用时复用 `audio_sr.enhance_file`（sr_steps 参数沿用记忆键模式）+ 显存互斥流程。混音场景：人声超分后与 BGM（原采样率）合并，输出统一到 max(44.1k, BGM率)。

**D11 自动合成**：勾选后保留分离产出的**伴奏轨**（"废弃物"BGM）+ 转换人声 → 混音：人声增益（默认 +3dB 相对 BGM，-6~+9 可调）、母线峰值 -1dBTP 保护、可选响度归一到 -14 LUFS（RMS 近似）；输入状态≠原歌曲时，提供可选"伴奏轨上传"（用户自带伴奏）。产出的中间产物（干声/伴奏/成品）都落盘保留。

**D12 后处理链（复用 audio_lab，翻唱标定）**：`suppress_wet_hf`（痰爆冲，max_cut 降档 10dB）→ `limit_voiced_hf`（撕裂限幅 ceiling 12dB 沿用）→ `match_band_profile`（频谱画像对齐，目标=角色参考/数据集画像，max_db 4dB，**可选默认关**，实验性标记）→ `soften_breath_hiss`（气声软化，默认关——歌声气口是表现力）。全部可独立开关，默认仅前两项开。加 70Hz 高通（去隆隆）。

**D13 参数记忆**：新建 `outputs/state/cover_state.json` / `vc_state.json`（复用 synth_state 的原子写/钳制模式，独立文件避免破坏 56 控件硬校验）。

**D14 产物与登记**：新目录 `outputs/cover/<时间戳>/`（源/分离干声/BGM/转换人声/成品/报告 json）、`outputs/vc/`（离线文件变声产物）。artifacts.py 三处登记（新 kind：`cover`、`vc`）。实时变声不留产物（可选"录制"开关落 `outputs/vc/`）。

---

## 3. 模块设计（新增文件）

| 文件 | 行数估 | 职责 |
|---|---|---|
| `webui_app/services/svc_core.py` | ~550 | VC 转换核心：`convert_file()` 离线（分块+交叉淡化+LoRA 挂载钩子）；`StreamConverter` 流式类（D5）；变体 A/B；参考特征构建（复用引擎缓存机制） |
| `webui_app/services/svc_f0.py` | ~450 | F0 提取（harvest 封装/RMVPE 可选）、八度映射、轮廓相关系数、局部修复与全曲移植（WORLD）、实时块级 F0 缓存 |
| `webui_app/services/separation.py` | ~280 | audio-separator 封装、权重下载登记、可用性探测、CPU/GPU 自适应 |
| `webui_app/services/cover_pipeline.py` | ~500 | 链路编排（状态路由/预处理/VC/F0/后处理/超分/混音）、停跑+心跳+进度+日志 sink、质量报告（campplus SS + F0 相关 + 八度错误 + 伪影指标） |
| `webui_app/services/vc_realtime.py` | ~400 | sounddevice 双流、输入调理（AGC）、档位、回环延迟测量、电平表、录制 |
| `webui_app/services/svc_state.py` | ~140 | 两份参数记忆（原子写+钳制） |
| `webui_app/tabs/cover_tab.py` | ~650 | 翻唱台 UI |
| `webui_app/tabs/vc_tab.py` | ~550 | 变声台 UI（实时+文件双模式） |
| `webui_app/params_cover.py` | ~260 | 参数注册（分组：翻唱-输入/转换/F0/后处理/超分/混音；变声-设备/档位/转换） |

改动既有文件：`app.py`（TAB_SPECS 两行）、`artifacts.py`（三处）、`params.py`（注册调用）、`pyproject.toml`（webui extra +sounddevice+pyworld；新增 separation extra）、`tools/model_fetcher.py`（RMVPE/分离权重 fetch_aux 条目）。

### 3.1 svc_core 核心契约（伪代码）

```python
def convert_file(engine, src_path, ref_path, *, lora_cfm=None, lora_scale=1.0,
                 variant="A", steps=25, cfg_rate=0.7, ref_seconds=15.0,
                 chunk_sec=60.0, crossfade_ms=40, f0_plan=None, progress=None,
                 should_stop=None, out_dir=None) -> ConvertResult:
    # 1) 源音频 librosa 载入 → 22k/16k 双轨
    # 2) 参考: engine 内部 mel_fn/campplus/length_regulator 构建 ref_mel/style/prompt_condition(缓存键=ref_path+mtime)
    # 3) 低能量切块(切点=块内 RMS 谷值, 距目标块界 ±5s 内搜索)
    # 4) 每块: 16k→extract_features→get_emb→[A: quantize→decode | B: 直通]
    #    → length_regulator(ylens=该块22k mel帧数) → cat([prompt_condition, cond])
    #    → cfm.inference(..., steps, cfg_rate)[:, :, ref_mel_len:] → bigvgan(整块)
    # 5) 波形域交叉淡化拼接 → (f0_plan 执行移植) → ConvertResult{wav, sr, chunks, timing}
```

流式 `StreamConverter.feed(new_16k_chunk) -> np.ndarray`：维护左上下文音频环形缓冲，内部走同一转换函数，返回新块的转换波形（已含 pad 裁剪）；调用方做输出流交叉淡化。

### 3.2 与引擎的边界

- 只读引擎属性：`extract_features / semantic_model / semantic_mean/std / semantic_codec / s2mel.models / mel_fn / campplus_model / bigvgan / _load_and_cut_audio`。不改 indextts/ 上游一行（项目红线）。
- 显式 `torch.inference_mode()` + fp32（与 infer_v2_5 CFM 段一致）；每块后 `torch.cuda.empty_cache()` 仅在低显存模式。
- LoRA：由调用方（tab 层）经 `engine.attach_lora(dir,"cfm")` 挂载后传入本模块；本模块只读不挂（职责分离，与合成页 `_sync_lora` 一致）。

### 3.3 质量报告（cover_pipeline 产出 `report.json`）

| 指标 | 实现 | 参考线 |
|---|---|---|
| 音色相似 SS | reward.RewardScorer.embed（campplus，CPU） | ≥0.75（角色参考 vs 输出） |
| F0 轮廓相关 r | svc_f0（八度对齐后 log-F0 Pearson，浊音帧） | ≥0.85（auto 模式处理后） |
| 八度错误帧率 | | <5% |
| 高频伪影 | AL.analyze（痰/撕裂指标复用） | 无听感级 |
| 块边界衔接 | 相邻块交叉淡化区谱质心跳变 | 报告供排查 |

---

## 4. UI 设计（两 tab）

### 4.1 🎤 AI翻唱（插在 lab 与 emobank 之间）

1. **输入区**：音频上传 + 输入状态单选（原歌曲/分离人声/干净人声，info 提示各自会跑哪些步骤）+ 伴奏上传（仅状态≠原歌曲且勾选自动合成时出现，gr. visible 联动）
2. **音色区**：参考音频（voice_bank 下拉 + 上传，audio 组件）+ CFM LoRA（training_runs 下拉+checkpoint+强度，GPT LoRA 灰显注明）+ 频谱画像目标（可选：该参考/角色数据集画像）
3. **转换参数区**（make_component，params_cover 注册）：变体A/B、步数 25、cfg 0.7、参考时长、块长、F0 模式(off/auto/force)+八度偏移(auto/手动)+F0 器(harvest/RMVPE)
4. **预处理区**：降噪强度（默认 0 低）、去混响强度档（轻/标准，映射 DeEcho 模型选择）
5. **后处理区**（D12 链，开关组）+ **超分区**（默认关，耗时警示，sr_steps）+ **自动合成区**（勾选框+人声增益+响度归一）
6. **运行区**：开始/停止按钮、进度条、**日志窗**（deque+Timer(1.0)，复用阶段22 模式）、产物文件列表（干声/BGM/成品/报告）
7. **报告区**：markdown 表（指标+通过线比对）+ A/B 试听

### 4.2 🎧 AI变声

1. **模式切换**（实时/文件）；文件模式 = 简化版翻唱台（无分离/混音，F0 默认 off，语音参数组）
2. **实时区**：输入设备下拉（sd.query_devices 过滤输入+WASAPI）+ 输出设备下拉（默认系统输出；info 提示"输出到耳机监听或 VB-Cable 虚拟声卡供其他软件采集"）+ 档位单选（D6 表）+ 开始/停止 + 状态行（实测延迟/当前块推理耗时/丢块计数）
3. **输入调理**：AGC 开关（轻量，目标 RMS -20dBFS，attack 慢）/ 静音直通开关（VAD 静音期 bypass 减伪影，Seed-VC 同款做法）
4. **音色区**：同翻唱（参考+CFM LoRA）
5. **工具**：回环延迟测量按钮（发脉冲→互相关）、电平表（Timer 2s 刷新输入/输出 RMS）、录制开关
6. 依赖缺失（sounddevice 未装）：实时区显示安装指引，文件模式仍可用

---

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| VC 输出旋律塌（RVQ 丢音高） | F0 auto 模式自动降级全曲移植；报告强制暴露 r 值 |
| WORLD 移植"塑料感" | 默认局部修复；全曲移植后做谱自检（RMS±4dB/<1kHz±20pp，复用韵尾延长守卫）；不达标回退原输出并在报告标注 |
| 高音区包络估计劣化 | fft_size 2048 + f0_floor 55；输出 F0>500Hz 段在报告中单列 |
| 8GB 串行慢（引擎往返加载） | 全链进度+预计耗时提示；分离与 VC 之间不重复加载（分离在前、VC 加载一次到底，超分按需再卸载） |
| 实时档延迟不达标 | 档位可调 + 实测工具诚实显示；文档给 VB-Cable/48k 同采样率指引 |
| w2v-BERT 逐块重编码成本未知 | 流式实现先测单块耗时（验收项），不达标时降级说明并把"质量档"为主推 |
| audio-separator 与 torch 版本冲突 | 可选 extra + available() 降级 + 文档注明安装命令（含 --no-deps 备选路径） |
| 块边界音色/谱跳变 | 40ms 等功率淡化 + 低能量切点 + 报告谱质心跳变指标 |
| 长任务阻塞 UI | 独立线程 + LoggedProgress(sink) + 停跑 flag（复用 orchestrator 模式） |

---

## 6. 里程碑与验收

| 步骤 | 内容 | 验收 |
|---|---|---|
| M1 | svc_f0.py + 单测（无 GPU）：八度映射/相关系数/局部修复合成正确性 | pytest 通过；合成音 F0 已知曲线的重建误差 <10 音分 |
| M2 | svc_core.py：离线 convert_file + 引擎接线；GPU 烟测（test 标 gpu marker） | 语音源→角色：时长帧级一致、SS(目标)>SS(源) |
| M3 | F0 管线合流：WORLD 合成"旋律音频"（同音色同内容仅变 F0，附录A.5 实验2 法）→ 过管线 → F0 相关 r | auto 模式 r≥0.85；报告四指标齐 |
| M4 | separation.py + cover_pipeline.py（预处理/后处理/混音/报告/停跑） | 干净人声路径全链跑通（不依赖分离权重）；mock 分离测路由 |
| M5 | cover_tab.py + params_cover.py + state + artifacts 登记 | test_webui_syntax 通过 + UI 构建冒烟 |
| M6 | vc_realtime.py + vc_tab.py | 文件模式可用；实时模式在无 audio 环境下逻辑单测（mock sd） |
| M7 | **自主验收**：用 voice_bank 参考做语音转换 + WORLD 旋律源做翻唱路径，听测+指标（痰音/撕裂/跳变/SS/r） | 无严重伪影；指标达参考线；问题清单与修复 |
| M8 | 子代理完整验收（代码审查+链路推演+UI/参数一致性） | 通过后交付 |

**性能验收线**：离线 VC 单 60s 块 < 90s（RTF<1.5）；流式单块（均衡档）推理 < 250ms（4060/8GB 预期，实测为准）；连续 10 分钟流式无显存增长。

---

## 7. 合规与定位

页面页脚注明"个人研究/本地使用"；不引入 GPL/非商用组件（audio-separator MIT、sounddevice MIT、pyworld MIT；RMVPE 权重 RVC 官方 HF 分发，MIT 家族）。bilibili 协议 3.4(c)：本功能不使用 IndexTTS 合成数据训练其他模型，无触碰。

---

## 8. 实施修正记录（2026-10-16 开发中实证发现，后续维护必读）

规划审查 + 单测 + GPU 验收三轮实证出的问题与修正，均已落入代码：

1. **八度映射符号反了**（单测抓到）：`octave_shift_semis` 源/目标比方向颠倒（110→220 返回 −12）。修正为 `log2(mr/ms)`。
2. **等功率淡化写成平方版**（单测抓到）：cos²/sin² 幅度版中点能量仅 0.5（掉 3dB）。正确为幅度域 cos/sin。
3. **WORLD 移植包络栅格**（对照实验实锤，最重要）：cheaptrick/d4c 用输出侧**原 f0** 的谐波栅格分析、再用移调后 f0 合成 → 输出谱栅格失配深谷纹波（220 分析+440 激励 → harvest 锁 103Hz）。**包络分析必须用合成 f0（synth_f0）**，清音帧沿用原 f0 防零。fft_size 无关（默认参数同样复现）。
4. **非八度移调的固有拍频**：WORLD 参数声码器移调 ×1.5（220→330）会产生差频调制（110Hz 拍），harvest 锁到拍频。auto 模式的八度映射（±12/0）天然规避；手动微调 ±1–2 半音时谱自检兜底回退。已写入参数 pitfall。
5. **48k/44.1k 源直接进 torchaudio.transforms.Resample 会混叠**（GPU 验收 E1 实锤）：默认滤波宽度对 ≥3:1 降采样不足，混叠特征喂码本 → 输出高频嘶声、声纹崩坏（SS≈0、质心 4.7k）。convert_file 与 build_reference 全部改 librosa（soxr HQ）重采样。
6. **流式缓冲坐标错位**（mock 测试抓到）：缓冲裁剪后窗口索引未换算相对坐标 → 同段数据重复消费（12 块输出＞输入总量）。引入 `_buf_offset` 绝对坐标约定。
7. **流式重叠死锁**（同上）：首块 overlap=0 时不保存尾部 → 后续永远 overlap=0。改为无条件保存尾部。
8. **隐式旋律能力实测**（附录 A.5 实验 2 的答案，4060 真机）：RVQ 后隐式旋律相关 **r≈0.81**（介于"够用"与"塌掉"之间偏乐观）——语调/缓变旋律大部分保留，精确旋律仍需 F0 移植；auto 分支设计（r≥0.75 局部修复、失败降级全曲）在真机上正确触发降级。
9. **CFM 无状态确认**（审查+实测）：逐块独立调用 cfm.inference 无需重置任何状态；prompt 每块照带正确。
10. **gradio 5.45 构件事实**：gr.Audio 无 info 参数；gr.Timer.active 赋值不可靠（照抄 synthesize 常开模式）；Dropdown choices 用 (label, value) 对时 value 必须传 value 不是 label。

## 9. 交付验收与规划削减记录（2026-10-16 收尾）

**交付验收（子代理全量审查）追加修复**：①P0 clean 路由未赋源路径（默认参数必炸，已修+回归测试）；②实时 F0 follow/smooth 帧预算死代码（已修：供给不足时由 `_interp_to_grid` 补齐）；③LoggedProgress 误传模块非 logger（进度条死+心跳死，已修 `get_logger("cover")`）；④自动合成对拍破坏——预处理砍头未补回（已修：head_cut 全程记录+mixdown head_pad_samples 补回+回归测试）；另修：分离释放挪进 try/finally 等价位置、LoRA 卸载统一 finally、cover 管线引擎租约（与实时会话互斥）、in_q 背压丢旧块、_rt_start 失败清理、convert 停止标志。

**本轮削减（规划 §4/§3.3 有承诺、未交付，留待后续版本）**：
- F0 器 RMVPE 选项（现为 pyworld harvest 单选；RMVPE 需另行下载权重，接口位已留）
- 去混响"轻/标准"档位 UI（separation.py 已定义两档模型，UI 未暴露，走默认）
- LoRA checkpoint 级选择（当前只挂 best；svc_state 已留 lora_ckpt 键）
- 报告区 A/B 试听组件、块边界谱质心跳变指标、输出 F0>500Hz 段单列
- "连续 10 分钟流式无显存增长"验收线（需声卡环境）

**遗留风险（如实告知，见交付说明）**：分离链路（audio-separator 未安装，降级文案安全）、实时变声真跑（延迟数为纸面估计）、超分真跑（17.8× 耗时）、auto_mix 全曲对拍（head_pad 已修+单测，未真机全曲验证）。

## 10. 模型部署与国内镜像（2026-10-17 补）

**依赖安装（清华源）**：

```bash
uv pip install --python .venv/Scripts/python.exe "audio-separator[gpu]" \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

`audio-separator` 自身只认 GitHub 直连（元数据 `download_checks.json` +
Releases 权重），国内不可达。本项目在 `services/separation.py` 里做了**前置
镜像预取**，推理时缺模型会**自动下载并部署**（无需手动步骤）：

- 镜像链默认 `ghproxy.net → gh-proxy.com → 直连`；可用环境变量覆盖：
  `INDEXTTS_GH_MIRRORS=https://ghproxy.net/,`（末尾空项=再试直连；海外可设 `,` 仅直连）。
- 支持 `.part` + HTTP Range **断点续传**、多镜像自动回退、Content-Length/
  Content-Range **大小校验**、HTML 错误页防污染。
- 权重落在 `checkpoints/separators/`（模型资源区，不进清理页产物盘点）；
  同时预置 `download_checks.json`（audio-separator 的模型清单）。
- 手动预下载：`python tools/fetch_separator_models.py --all`（`--list` 看状态）。
  翻唱页也有「⬇ 预下载/补全分离模型（镜像）」按钮与模型就绪计数。

**已部署模型（2026-10-17）**：BS-Roformer ViperX-1297（639MB）、Mel-Band
Roformer 1143（1008MB）、vocals_mel_band_roformer（913MB）、UVR-DeEcho-DeReverb
（224MB）、UVR-DeEcho-Normal（127MB），合计约 2.3GB。`--list` 全部 ✓；
BS-Roformer 与 UVR-DeEcho-DeReverb 已做加载冒烟（6.0s / 0.1s）。

**部署期踩坑（均已在代码里兜住）**：
1. `Separator.__init__` 0.47 **没有 `output_bitdepth` 参数**（传了 TypeError）——
   WAV 默认 16bit，改用 `output_format="WAV"`。
2. VR 模型加载时 `load_model_data_using_hash` 还要拉
   `vr_model_data.json` / `mdx_model_data.json`（远端名 `model_data_new.json`），
   否则 VR 去混响在加载阶段就网络失败。`ensure_metadata` 现在一并预置三件。
3. `UVR-DeEcho-Normal.pth` 的 Release 资产实名是 `UVR-De-Echo-Normal.pth`
   （带连字符）——URL 直拼 404。`FILE_URL_ALIASES` 做映射。
4. 镜像冷缓存偶发 404（重试即 206）；YAML 只有 944 字节，旧的 1024B
   “就绪阈值”会把它判缺失导致每次推理重下——现按类型阈值（YAML 200B /
   权重 1MiB），且探到的体重<1MiB 视为探测失败。
5. **separate() 返回相对文件名**（2026-10-17 用户「原歌曲」首次翻唱失败）：
   audio-separator 0.47 的 `common_separator.py:530` 用单参数 `os.path.join`
   生成输出名，返回的是相对名；管线用 CWD 调 `os.path.isfile` 判定，导致
   “两轨已写出却报未找到人声”。`_stem_path` 现按 `out_dir` 解析相对路径。
6. **模型名里的 "vocals" 污染 stem 匹配**：输出名为
   `000_(other)_vocals_mel_band_roformer.wav`，旧子串匹配会把伴奏轨当人声。
   现先剥掉模型名再匹配，并对人声候选排除 instrumental/other 类关键词；
   去混响的 `no_reverb` 干声不被 `reverb` 排除规则误杀（排除只用于人声
   分离 both=True）。
7. **失败路径不落 report.json**：`run_cover` 的早退（依赖缺失/分离失败/VC
   失败）直接 return，跳过报告写盘，用户无据可查。现统一改为 `_Fail`，
   经 finally 恢复引擎后仍写完整 report.json；UI 侧也补 warning 日志。
8. **60s 分块在 8GB 卡上 OOM**（2026-10-17 第二次实锤）：第 1 块 81.8s
   成功，第 2 块在 w2v-BERT 的 `relative_position_attn_weights` 申请 1.21GB
   失败（PyTorch 报“已分配 13.79GB”，WDDM 已溢出到共享内存）。修复：
   - 内容编码器按 **24s 窗 + 0.5s 重叠**提取后拼回 50Hz 网格——把
     O(T²) 注意力与总时长解耦（60s 单窗单层就要 ~0.6GB）；
   - 按卡容量限制块长：≤9.5GB → 30s、≤13GB → 45s（`cap_chunk_sec`）；
   - 块间 `gc.collect()+empty_cache()`（缓存块不释放是第二块失败的直接原因）；
   - OOM 自动把块长减半（30→15s，下限 10s）从头重试一次，报告标注
     `retried_smaller_chunk` 与实际 `chunk_sec_used`；
   - **VC 期间把 GPT 移到 CPU**（VC 绕过 GPT；8GB 约让出 1.5GB），
     会话结束/异常在 finally 恢复；启用 accel/torch_compile 时跳过；
   - 租约 **token 化**：`unload(bypass_lease=自己的token)` / `load(bypass_lease=...)`
     只绕过自己，其他会话（实时变声、另一条翻唱）仍拦截——旧版
     `bypass_svc=True` 会连别人的租约一起绕过；
   - VC 会话期间 `engine.infer` / `mount_qwen` 直接拒绝（防止合成页
     在 GPT 位于 CPU 时启动 TTS）。
   - 顺带：分离推理阶段补“分离推理中”进度文案（此前 170s 一直显示
     “加载分离模型”）；失败报告也带阶段耗时。
9. **F0 阶段三处接线/设计错误**（2026-10-17 第三次实锤，用户报“F0 返回
   接线问题”时定位）：
   - **包络策略必须方向相关**（推翻 §8 第 3 条的绝对化结论）：上行移调
     用合成-F0 分析包络；**下行移调必须用输出自身 F0 分析包络**。固定用
     合成-F0 时，-12 移调（372→195）合成后基频被抹掉，中位只降到 305Hz、
     轮廓相关反降到 0.39。现按目标/输出中位方向选择，并用 4s 探针核对，
     偏差 >0.5 八度自动换策略，仍不收敛则回退原输出。
   - **harvest_f0 分窗平均是错的**：旧实现 30s 窗+1s 重叠，把两个窗边缘
     的 F0 做线性平均（边缘常有八度错误），产生 223/264/364 这类中间值，
     看起来像“F0 跳变”。现 ≤12 分钟整段一次 harvest；更长才分窗，且
     丢弃每窗首尾 1s、只取中心帧按绝对帧号拼接，不做任何数值平均。
   - **自动移调改为保调优先**：同音区（目标/源中位差 ≤8 半音）返回 0，
     跨音区才 ±12。旧逻辑对 390Hz 源 vs 257Hz 参考选 -12，既改调又踩
     下行八度的病态路径。手动 ±24 半音仍可覆盖。
   - 报告指标改为**移植后探针**：`f0_r` 取 20s 代表段（目标浊音最多）的
     相关，另报 `f0_r_pre` / `within_50c` / 局部八度错误率；旧 `octave_err`
     用 round(cents/1200) 残差，对局部 ±12 跳变完全失明，已改为对齐后
     |cents|>600 的帧占比。
   - 超分阶段进度：184s 人声 17.8× ≈ 55 分钟，此前心跳一直显示上一条
     “F0 移植”，看起来像卡死；现先报“超分前卸载引擎”，再报“超分中
     （预计 N 分钟）”，完成后 0.92。

**已知环境告警**：`onnxruntime-gpu 1.31`（CUDA 13 编译）与项目 torch
`2.8+cu128`（CUDA 12.8）不匹配，启动时打印 cublasLt/cudart 加载失败。
本项目 UI 暴露的模型（Roformer/VR）走 PyTorch 路径，**不受影响**；若未来
启用 MDX(ONNX) 模型，需将 onnxruntime-gpu 固定到 1.22.x（CUDA 12）或升级
torch 到 cu13 分支。
