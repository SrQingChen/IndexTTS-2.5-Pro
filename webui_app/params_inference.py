"""推理参数登记：音色 / 文本 / 情感 / 采样 / 分句 / 引擎 / 显存 / 日志。"""

from __future__ import annotations

EMO_ZH = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]
EMO_EN = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
EMO_BIAS = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
EMO_LIB = [3, 17, 2, 8, 4, 5, 10, 24]


def build(reg, P):
    _voice(reg, P)
    _text(reg, P)
    _emotion(reg, P)
    _sampling(reg, P)
    _segment(reg, P)
    _engine(reg, P)
    _memory(reg, P)
    _logging(reg, P)


# ---------------------------------------------------------------------------

def _logging(reg, P):
    """调试日志（阶段 3.5）。出问题先看这里说的文件。"""
    reg(P(
        key="log_level", group="logging", label="日志级别",
        kind="dropdown", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        summary="决定记多少。排查问题用 DEBUG，平时 INFO。",
        info="启动参数 --log-level；运行期可在「系统 → 调试日志」里即时切换，无需重启。",
        affects="日志文件体积与信息量",
        detail_md="""
四个级别的取舍：

| 级别 | 会记什么 | 什么时候用 |
|---|---|---|
| `ERROR` | 只记失败 | 只想看坏事 |
| `WARNING` | 加上「可疑但不致命」（状态脱节、无效操作） | 默认观察 |
| `INFO` | **默认**：引擎加载/卸载、LoRA 挂载/卸载、后台任务起止、训练起止、识别与优化的用时与线程数 | 日常 |
| `DEBUG` | 加上每次 UI 回调的入参与耗时、强度旋钮的实际生效值、参数校验细节 | 排查具体问题时**临时**打开 |

DEBUG 会明显增加日志体积（每次点按钮都记一行），排查完记得切回 INFO。
""",
        pitfall_md="""
级别只影响**记录**，不影响任何行为 —— 开 DEBUG 不会让训练变慢，但会多写盘。
`outputs/logs/` 上限约 50 MB（两个文件各 5 MB × 5 份轮转）。
""",
    ))

    reg(P(
        key="log_dir", group="logging", label="日志目录",
        kind="text", default="outputs/logs",
        summary="日志落盘位置。默认 ./outputs/logs。",
        info="启动参数 --log-dir；写入两个文件：indextts.log（全量）+ error.log（只记问题）。",
        affects="排查时去哪里找证据",
        detail_md="""
```
outputs/logs/indextts.log    全量日志，5 MB × 5 份轮转
outputs/logs/error.log      只记 WARNING 及以上，**含完整异常堆栈**
```

**`.log` 与 `log.txt` 不是一回事**：`training_runs/<run>/log.txt` 是训练器
自己的日志（只在那一个训练目录里），`outputs/logs/` 才是全局的。查
「界面点了没反应」「切换模型报错」这类问题要看后者。

`error.log` 单独一份的意义：出了问题不用在几千行里翻，直接开它。

另外，**未捕获异常也会落盘** —— 主线程与子线程的 excepthook 都挂上了。
后台任务线程崩溃以前只会静默消失，现在会留下完整堆栈与线程名。
""",
        pitfall_md="""
Windows 上只要日志文件还被进程打开着，日志目录就删不掉。
脚本里要清理日志目录，先调 `logging_setup.shutdown()` 释放句柄。
""",
    ))

    reg(P(
        key="quiet", group="logging", label="静默模式（只写文件）",
        kind="checkbox", default=False,
        summary="不往控制台打日志，但文件照写。",
        info="启动参数 --quiet。适合把服务挂在后台或重定向输出的场景。",
        affects="控制台整洁程度；不影响文件日志",
        detail_md="""
`--quiet` 只是摘掉控制台处理器，`indextts.log` / `error.log` 照常写 ——
「静默」不等于「不留证据」，事后照样能查。
""",
    ))


# ---------------------------------------------------------------------------

def _voice(reg, P):
    reg(P(
        key="spk_audio_prompt", group="voice", label="音色参考音频", kind="audio",
        summary="决定「谁在说」。零样本音色克隆的唯一输入。",
        info="3~15 秒干净人声；超过 15 秒会被硬截断（取前 15 秒）",
        affects="音色相似度（决定性）",
        detail_md="""官方 `_load_and_cut_audio(spk_audio_prompt, 15)` 会重采样并**硬截断到前 15 秒**，超出部分直接丢弃（不是择优，是取前面）。

这段音频兵分三路，构成音色的全部来源：

| 分支 | 处理 | 去向 | 作用 |
|---|---|---|---|
| ① 16kHz | CAMPPlus → 192维 `style` | `spk_emb_proj(192→1280)` → GPT 条件 token | 全局声纹，影响韵律层「人味」 |
| ② 16kHz | w2v-BERT 第17层隐状态(1024维) | 未指定情感音频时充当情感条件 | 说话人自带的情感基调 |
| ③ 22.05kHz | `mel_fn` → 80维 mel | CFM 的 `prompt` 参数 | **声学音色细节的主要来源** |

所以：**参考音频的质量，几乎直接等于最终音色的质量。**""",
        tuning_md="""- **时长 8~15 秒最佳**。<3s 声纹统计不稳定；官方上限 15s，再长也浪费。
- **只放你要的那一段**：截断是「取前 15 秒」，如果前 5 秒是静音或别人说话，会污染整个声纹。先用「参考音频工作台」裁好。
- **情绪要与目标一致**：分支②决定默认情感基调。用平静参考音频合成激动台词，效果打折。
- **采样率 ≥16kHz**。8kHz 电话录音在 CAMPPlus 和 w2v-BERT 上都会失真。
- **单人、无 BGM、无混响**。多说话人会让 CAMPPlus 输出「平均声纹」，结果四不像。""",
        pitfall_md="""- ⚠️ 上传后**不要移动或删除原文件**：预设存的是路径，文件消失会加载失败。
- ⚠️ 换参考音频时官方会清空 `cache_spk_cond / cache_s2mel_style / cache_mel` 并 `empty_cache()`。**连续用同一条参考音频合成多段文本会显著更快**，频繁切换反而慢。
- ⚠️ 背景音乐会被 CAMPPlus 当成声纹的一部分，务必先降噪。""",
    ))

    reg(P(
        key="emo_audio_prompt", group="voice", label="情感参考音频", kind="audio",
        summary="决定「怎么说」。与音色参考完全独立，可来自不同的人、不同语言。",
        info="仅在情感控制方式=「使用情感参考音频」时生效",
        affects="情感表现力",
        detail_md="""这是 IndexTTS-2.5「音色-情感解耦」的入口。

```
emo_audio → 16kHz → w2v-BERT h17 → Conformer(emo_conditioning_encoder, 4块)
          → PerceiverResampler(num_latents=1) → 1024维
          → emovec_layer(1024→1280) → emo_layer(1280→1280) = emo_vec
```
随后与音色条件**相加**送入 GPT：
```
conds_latent = cat(spk_emb + emo_vec.unsqueeze(1), zeros(B,2,D))
```

**关键**：情感向量只进 GPT(T2S)，**完全不进 CFM(S2M)**。CFM 只接收 `prompt=ref_mel` 和 `style=campplus`（都是音色）。这正是解耦的实现方式 —— 改情感不动音色，改音色不动情感。""",
        tuning_md="""- 同样受 15 秒截断限制。
- 选**情感表达夸张、干净**的片段。情感编码器只输出 1 个 latent（`num_latents=1`），信息高度压缩，平淡的参考音频会让情感特征糊掉。
- 跨语言允许：用中文「哭腔」驱动英文合成，官方评测确认有效。""",
        pitfall_md="""- ⚠️ 一旦指定情感参考音频，音色参考音频的**情感分支就被覆盖**（不再回退到 `spk_audio_prompt`）。
- ⚠️ 情感参考音频的**音色不会泄漏**到输出（它不进 CFM），但强烈口音可能通过韵律间接影响结果。""",
    ))


def _text(reg, P):
    reg(P(
        key="text", group="text", label="目标文本", kind="textarea", default="", lines=6,
        summary="要合成的文本内容。支持发音标注语法。",
        info="支持 <文字|发音> 标注：中文拼音 / 英文CMU音素 / 日语假名",
        affects="内容",
        detail_md="""### 处理流水线（v2.5）
```
原文
 → clean_pattern 字符替换（全角/异形字符归一）
 → text_normalization（若开启）
     · zh / zhen / en → TextNormalizer.normalize()
     · ja / es        → nemo_text_normalize()
     · ar             → 不处理
 → 大小写规整（ja/zh/zhen/en 转小写；es 转大写）
 → apply_pronunciation_annotations()   ← 处理 <文字|发音>
 → ja: JapaneseG2PProcessor.process_ja_text()
 → <|xx|> 标记统一转大写
 → split_text_by_tokens(text, max_text_tokens_per_segment, lang_prefix)
 → tiktoken 编码，每段末尾补 token id=1
```

### 发音标注语法
`<文字|发音>`，由 `PRONUNCIATION_ANNOTATION_PATTERN = <([^|>\\n]+)\\|([^>\\n]+)>` 匹配：

| 文字类型 | 转换结果 | 例子 |
|---|---|---|
| 含中文 | `<\\|SPECIAL_TOKEN_2\\|>发音<\\|SPECIAL_TOKEN_2\\|>` | `<行\\|XING2>` |
| 纯英文 | `<\\|SPECIAL_TOKEN_1\\|>发音<\\|SPECIAL_TOKEN_1\\|>` | `<minute\\|M IH1 . N AH0 T>` |
| 发音是假名 | 直接空格包裹，不加标记 | `<上手\\|じょうず>` |

发音内容**统一转大写**。合法拼音见 `checkpoints/pinyin.vocab`。""",
        tuning_md="""- 多音字务必用标注，别指望模型猜对：`他在银<行|XING2>里<行|HANG2>走了半天`
- 数字、日期建议**自己写成读法**：「2026年」写成「二零二六年」比交给 TN 更可控。
- 长文本不用手动切分，`split_text_by_tokens` 会处理；但**段落之间加标点**能让切分点更自然。""",
        pitfall_md="""- ⚠️ **低显存自动切块**：`low_vram=True`（显存<10GB）且文本 >40 字符时，`infer()` 先用 `split_text_by_punctuation(text, max_chars=40)` 粗切，逐块独立合成再拼接，块间插静音。**后果**：块与块之间韵律不接续。显存 ≥10GB 不触发。
- ⚠️ 标注用了全角 `｜` 而不是半角 `|` 不会被识别，会当普通文本读出来。
- ⚠️ 空文本不报错但返回 None，界面表现为「没有输出」。""",
    ))

    reg(P(
        key="lang", group="text", label="语言", kind="dropdown", default="ZH",
        choices=["ZH", "EN", "JA", "AR", "ES"], version="2.5",
        summary="目标语言。决定语言前缀 token、归一化路径和大小写策略。",
        info="跨语言合成时选「目标语言」，不是参考音频的语言",
        affects="发音正确性、跨语言音色保持",
        detail_md="""三重作用：

1. **语言前缀**：`lang_prefix = f'<|{lang.lower()}|> '`，拼在每段文本前参与编码 —— 官方「边界感知对齐」策略。
2. **归一化路径**：`zh/zhen/en` → TextNormalizer；`ja/es` → nemo_tn；`ar` → 不处理。
3. **大小写**：`ja/zh/zhen/en` 转小写，`es` 转大写。

最后 `lang_to_token(lang)` 转 id，经 `lang_embedding` 加到文本 embedding 上（官方「Token 级拼接」策略）：
```
text_emb = text_embedding(text_input) + text_pos_embedding(pos)
text_emb += lang_embedding(langs[i])
```""",
        tuning_md="""- **跨语言克隆**：参考音频说中文、`lang` 选 EN → 用中文音色说英文。官方评测 zh→ja 的 SS 高达 74.16%。
- 日语建议配合假名标注处理多音词：`<上手|じょうず>` vs `<上手|うわて>`。""",
        pitfall_md="""- ⚠️ 选错语言会走错归一化路径。中文文本用 `lang=EN` 合成，中文数字不会被转换。
- ⚠️ `zhen`（中英混）在 `lang_to_token` 里有 token，但下拉框只暴露 5 种。中英混合直接用 `ZH` 即可。
- ⚠️ 阿拉伯语是 RTL 文字，输入框显示顺序可能看着别扭，但不影响合成。""",
    ))

    reg(P(
        key="text_normalization", group="text", label="文本归一化", kind="checkbox",
        default=True,
        summary="是否自动把数字、符号、日期等转换成可读文本。",
        info="关闭后文本原样送入模型，适合已手工标注读音的场景",
        affects="发音正确性",
        detail_md="""按 `lang` 分流：`zh/zhen/en` → `TextNormalizer.normalize()`（依赖 wetext / WeTextProcessing）；`ja/es` → `nemo_text_normalize()`；`ar` 等**不做任何处理**。

关闭后 `123` 会被直接当成 token 序列编码，模型可能读成「一二三」也可能读错。""",
        tuning_md="""- **默认保持开启**。
- 只有当已把所有数字/符号手工写成读法，或发现 TN 把专有名词改坏了，才关闭。""",
        pitfall_md="""- ⚠️ TN 会改写文本。`verbose=True` 会打印 `text after normalization: ...`，建议开 verbose 确认改写结果。
- ⚠️ Windows/macOS 用 `wetext`，Linux 用 `WeTextProcessing`（pyproject 的 platform marker），两者规则略有差异。""",
    ))


def _emotion(reg, P):
    reg(P(
        key="emo_control_method", group="emotion", label="情感控制方式", kind="dropdown",
        default=0, choices=[0, 1, 2, 3],
        summary="四种互斥的情感来源，决定哪些控件生效。",
        info="0=跟随音色音频 1=情感参考音频 2=8维向量 3=文本描述(需QwenEmotion)",
        affects="情感表现力",
        detail_md="""| 模式 | 名称 | 情感来源 | 额外模型 |
|---|---|---|---|
| 0 | 与音色参考音频相同 | 音色音频自身的 w2v-BERT 情感特征 | 否 |
| 1 | 使用情感参考音频 | 独立上传音频 → Conformer+Perceiver | 否 |
| 2 | 使用情感向量控制 | 8维滑块 → `emo_matrix`/`spk_matrix` 查表加权 | 否 |
| 3 | 使用情感描述文本 | 文本 → QwenEmotion(Qwen3-0.6B) → 8维向量 | **是** |

模式 2 的向量合成路径：
```python
random_index = [find_most_similar_cosine(style, tmp) for tmp in spk_matrix]
emo_matrix   = [tmp[i].unsqueeze(0) for i, tmp in zip(random_index, self.emo_matrix)]
emovec_mat   = sum(weight_vector[i] * emo_matrix[i])
emovec       = emovec_mat + (1 - sum(weight_vector)) * emovec   # 剩余权重回落本人情感
```
`emo_num = [3, 17, 2, 8, 4, 5, 10, 24]` —— 8 种情感的候选库大小，合计 **73 条情感原型**。""",
        tuning_md="""- **模式 0 是最稳的默认值**，音色还原度最高。
- 模式 1 适合「用 A 的音色说 B 的情绪」，如配音场景。
- 模式 2 适合精确、可复现的情感配比（可存进预设反复用）。
- 模式 3 最省事但最不可控，官方标注为**实验功能**。""",
        pitfall_md="""- ⚠️ 模式 2/3 会**强制清空外部情感音频**：`if use_emo_text or emo_vector is not None: emo_audio_prompt = None`。即模式 2、3 无法叠加情感参考音频。
- ⚠️ 模式 3 需要 QwenEmotion。低显存默认不常驻，本 UI 首次使用时按需加载、用完释放。""",
    ))

    reg(P(
        key="emo_alpha", group="emotion", label="情感权重", kind="slider", default=0.65,
        minimum=0.0, maximum=1.0, step=0.01,
        summary="情感混合强度。**在不同模式下语义完全不同**。",
        info="模式0下无效；模式1是插值系数；模式2/3是向量缩放系数",
        affects="情感强度、音色相似度",
        detail_md="""### 模式 1（情感参考音频）：线性插值系数
`merge_emovec()` 实现：
```python
emo_vec  = get_emovec(emo_speech_conditioning_latent)   # 情感参考音频的
base_vec = get_emovec(speech_conditioning_latent)       # 音色参考音频自己的
out = base_vec + alpha * (emo_vec - base_vec)
```
- `0.0` → 完全用**音色音频自身**情感（等价模式 0）
- `1.0` → 完全用**情感参考音频**情感
- `0.65`（默认）→ 65% 偏向情感参考音频

### 模式 2/3（向量/文本）：向量缩放系数
```python
emo_vector_scale = clamp(emo_alpha, 0, 1)
emo_vector = [round(x * emo_vector_scale, 4) for x in emo_vector]
```
等比缩放 8 维向量幅值，再走 `emovec_mat + (1-sum)*emovec`。alpha 越小 → 越回落说话人自身情感。

### 模式 0：无效
代码里 `if emo_audio_prompt is None: emo_alpha = 1.0` 会**强制覆盖**，且 `out = base_vec + 1.0*(base_vec-base_vec) = base_vec`。""",
        tuning_md="""- 模式 1：**0.6~0.8** 最自然。拉到 1.0 情感很足但可能盖过说话人的语气习惯。
- 模式 3：官方明确建议 **0.6 或更低**，因为 QwenEmotion 输出的向量幅值本身偏大。
- 模式 2：先固定 alpha=1.0 调向量配比，调好后再用 alpha 整体收放。""",
        pitfall_md="""- ⚠️ **模式 0 下拖这个滑块没有任何效果**，不是 bug。
- ⚠️ 模式 1 下 alpha=0 让情感参考音频完全失效，但缓存已建立则不额外耗时。""",
    ))

    for i, (zh, en, bias, lib) in enumerate(zip(EMO_ZH, EMO_EN, EMO_BIAS, EMO_LIB)):
        reg(P(
            key=f"emo_vec_{i}", group="emotion", label=f"{zh} ({en})", kind="slider",
            default=0.0, minimum=0.0, maximum=1.0, step=0.05,
            summary=f"8维情感向量第 {i+1} 维：{zh}。内置偏置系数 {bias}，候选库 {lib} 条原型。",
            info=f"内部偏置 ×{bias}；8维总和 >0.8 会被静默等比压缩",
            affects="情感表现力",
            detail_md=f"""维度顺序固定为 `[高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静]`，对应 `emo_num = [3, 17, 2, 8, 4, 5, 10, 24]` 的候选库切分。本维度（**{zh}**）候选库有 **{lib}** 条情感原型。

`normalize_emo_vec(vec, apply_bias=True)` 两步处理：
1. **偏置**：每维乘固定系数，弱化容易产生怪异结果的情感
   ```python
   emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
   ```
   「惊喜」(×0.6875) 和「平静」(×0.5625) 被压得最狠。
2. **限幅**：8维总和 >0.8 时整体等比缩放到总和 = 0.8。""",
            tuning_md="""- 单次只给 **1~2 个维度**非零值，效果最干净。8维全拉满会被限幅压平，反而得到模糊的「什么都有点」。
- 想要「平静」，正确做法是**全部归零**（回落到说话人自身基调），而不是拉高第8维 —— 它有 ×0.5625 的重偏置。
- 保存成预设后可精确复现同一套配比。""",
            pitfall_md="""- ⚠️ 总和 0.8 的限幅是**静默**的，官方界面不提示。本 UI 会实时显示归一化后的实际值。
- ⚠️ 「低落」和「悲伤」是两个独立维度，但 QwenEmotion 分不出来（见 `melancholic_words` workaround）。手动控制时请自己区分。""",
        ))

    reg(P(
        key="use_random", group="emotion", label="情感随机采样", kind="checkbox",
        default=False,
        summary="从情感候选库随机取原型，而非取与当前声纹最匹配的。",
        info="开启会降低音色还原度（原因见详解）",
        affects="情感多样性 ↑ / 音色相似度 ↓",
        detail_md="""```python
if use_random:
    random_index = [random.randint(0, x - 1) for x in self.emo_num]
else:
    random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
```

默认(False)：用 **CAMPPlus 声纹 `style`** 在 `spk_matrix` 每组里做余弦检索，选出与该说话人最匹配的原型索引，再用**同一索引**去 `emo_matrix` 取情感原型。

开启(True)：完全随机选索引。

**为什么降低音色还原度**：`feat1.pt`(spk_matrix) 和 `feat2.pt`(emo_matrix) 是**成对**的 73 条原型 —— 第 i 条 spk 原型和第 i 条 emo 原型来自同一个说话样本。余弦检索保证「情感原型」与「当前声纹」在同一说话人分布上；随机取打破了这个配对，引入不属于该说话人的情感原型，导致情感与音色轻微错位。""",
        tuning_md="""- 想要**同一套参数生成多个不同演绎版本**时开启，然后挑最好的。
- 追求最高音色还原度时**务必关闭**。""",
        pitfall_md="""- ⚠️ 开启后结果不可复现（无固定 seed）。本 UI 提供 seed 控件解决。
- ⚠️ 仅在情感控制方式 = 2 或 3 时生效。""",
    ))

    reg(P(
        key="emo_text", group="emotion", label="情感描述文本", kind="text", default="",
        experimental=True,
        summary="用自然语言描述想要的情绪，由 QwenEmotion 转成 8维向量。",
        info="留空则自动用目标文本作为情感描述",
        affects="情感表现力",
        detail_md="""`QwenEmotion` 是 **Qwen3-0.6B 的情感分类微调版**（`checkpoints/qwen0.6bemo4-merge/`，1.14GB）。

```python
messages = [{"role":"system","content":"文本情感分类"},
            {"role":"user","content": emo_text or text}]
# chat_template(enable_thinking=False) → generate(max_new_tokens=32768)
# 找 token 151668 (</think>) 之后的内容 → json.loads
```
输出 8 键字典，clamp 到 `[0.0, 1.2]`，全零时回落 `calm=1.0`。

**允许内容与情感分离**：`text` 是念的内容，`emo_text` 是情绪描述。
例：`text="快躲起来！"` + `emo_text="你吓死我了！你是鬼吗？"`。
留空时 `infer_generator` 执行 `emo_text = text`，即**用目标文本自身推断情感**。""",
        tuning_md="""- 官方例子：`委屈巴巴`、`危险在悄悄逼近` —— **写氛围/情境比写情绪名词更有效**。
- 配合 `emo_alpha ≈ 0.6` 或更低（官方建议）。
- 想强制「低落」而非「悲伤」，文本里需出现 `低落 / melancholy / melancholic / depression / depressed / gloomy` 之一（触发 `melancholic_words` 的向量交换 workaround）。""",
        pitfall_md="""- ⚠️ **官方标注为实验功能，结果尚不稳定。**
- ⚠️ QwenEmotion 分不清「悲伤」和「低落」，代码用关键词硬交换打补丁。
- ⚠️ `max_new_tokens=32768` 是官方硬编码。情感 JSON 很短，但模型跑飞会一直生成，**无超时保护**。本 UI 加了长度上限。
- ⚠️ 官方用 `device_map="auto"` + `torch_dtype="float16"`：显存不足时 accelerate 会**静默**把层 offload 到 CPU，慢到不可用且不报错。本 UI 改为显式指定 device + 显存余量预检。""",
    ))


def _sampling(reg, P):
    reg(P(
        key="do_sample", group="sampling", label="do_sample", kind="checkbox",
        default=True,
        summary="是否启用随机采样。关闭则退化为贪心/纯 beam search。",
        info="关闭后结果完全可复现，但会明显变平淡、机械",
        affects="多样性 ↑ / 稳定性",
        detail_md="""传给 HF `generate()`。与 `num_beams` 组合出四种模式：

| do_sample | num_beams | 实际算法 |
|---|---|---|
| True | 1 | 纯采样（temperature/top_p/top_k 生效） |
| True | >1 | **Beam Sample**（官方默认：True + 3） |
| False | 1 | 贪心解码 |
| False | >1 | Beam Search |

TTS 的语义 token 需要一定随机性来产生自然韵律，**贪心解码会让语调变平**。""",
        tuning_md="""- 保持 True。
- 只有做 A/B 对照实验、需要严格可复现时才关闭。""",
        pitfall_md="""- ⚠️ 关闭后 `temperature / top_p / top_k` **全部失效**（HF 会忽略并打 warning）。""",
    ))

    reg(P(
        key="temperature", group="sampling", label="temperature", kind="slider",
        default=0.8, minimum=0.1, maximum=2.0, step=0.05,
        summary="采样温度。越高越随机，越低越保守。",
        info="官方默认 0.8；>1.2 容易读错字，<0.5 语调发平",
        affects="多样性 / 发音稳定性",
        detail_md="""在 logits 上除以 temperature 后再 softmax。

对 TTS 的特殊影响：语义 token 序列**同时承载内容和韵律**。温度过高时模型可能选到低概率的错误音素（表现为读错字、吞字）；温度过低时韵律收敛到最「安全」的路径，听起来像念稿。""",
        tuning_md="""- **0.7~0.9** 是安全区，官方默认 0.8。
- 出现读错字/吞字 → 降到 0.6。
- 觉得太机械、想要更生动的演绎 → 升到 1.0，同时把 `top_p` 降到 0.7 兜底。
- 与 `top_k` 配合：`top_k` 先粗筛，`temperature` 再调整分布形状。""",
        pitfall_md="""- ⚠️ 官方 UI 步长是 0.1，颗粒太粗。本 UI 改为 0.05。
- ⚠️ `do_sample=False` 时此参数无效。""",
    ))

    reg(P(
        key="top_p", group="sampling", label="top_p", kind="slider",
        default=0.8, minimum=0.0, maximum=1.0, step=0.01,
        summary="核采样：只从累积概率达到 p 的最小 token 集合里采样。",
        info="官方默认 0.8；越低越稳但越单调",
        affects="多样性 / 稳定性",
        detail_md="""Nucleus sampling。把 token 按概率降序累加，取累积概率首次 ≥ top_p 的集合。`top_p=1.0` 等价于不做核采样截断（仍受 top_k 约束）。""",
        tuning_md="""- 官方默认 0.8，配合 top_k=30 已经比较保守。
- 长文本合成出现前后风格漂移 → 降到 0.7。
- 想要更丰富的情感起伏 → 升到 0.9，同时把 temperature 降到 0.7 平衡。""",
        pitfall_md="""- ⚠️ `top_p` 和 `top_k` 是**串联**生效的（先 top_k 截断，再 top_p 截断），两个都调低会导致候选极少、输出趋同。""",
    ))

    reg(P(
        key="top_k", group="sampling", label="top_k", kind="slider",
        default=30, minimum=0, maximum=100, step=1,
        summary="只从概率最高的 k 个 token 里采样。**设为 0 表示禁用**。",
        info="0=禁用；官方默认 30",
        affects="多样性 / 稳定性",
        detail_md="""界面值经 `int(top_k) if int(top_k) > 0 else None` 转换后传给 `generate()`，所以 **0 → None → 不做 top-k 截断**（不是「只取第 0 个」）。

词表大小是 `number_mel_codes = 8194`（含 start=8192 / stop=8193）。top_k=30 意味着每步只考虑 8194 个语义 token 里的前 30 个，截断相当激进。""",
        tuning_md="""- 保持 30。
- 出现「卡顿感」「节奏死板」→ 提到 50。
- 出现明显读错 → 降到 20。""",
        pitfall_md="""- ⚠️ 别把 0 理解成「不采样」。0 是**禁用**该过滤器。""",
    ))

    reg(P(
        key="num_beams", group="sampling", label="num_beams", kind="slider",
        default=3, minimum=1, maximum=10, step=1,
        summary="束搜索宽度。官方默认 3（配合 do_sample=True 即 Beam Sample）。",
        info="越大越稳越慢，KV cache 显存线性增长",
        affects="质量 ↑ / 速度 ↓↓ / 显存 ↑",
        detail_md="""`num_beams=3` + `do_sample=True` → HF 的 **Beam Sample**：维护 3 条候选序列，每步在束内做采样。

对 TTS 的收益：抑制单步采样失误导致的整段崩坏（比如某个音节读错后后面全部跑偏）。代价是每步算 3 倍前向。""",
        tuning_md="""- **8GB 显存建议 1~3**。3 是官方默认，质量/速度平衡点。
- 追求速度 → 设 1（纯采样，快约 2~3 倍）。
- 长句频繁崩坏 → 提到 5，但耗时明显增加。""",
        pitfall_md="""- ⚠️ **显存杀手**。KV cache = `num_beams × 序列长度 × 层数 × 2 × head_dim`。24层/1280维/序列2400/beam=3 时约 1.3GB(fp32)。beam=10 直接翻 3 倍多，8GB 卡会 OOM。
- ⚠️ `num_beams>1` 时 `min_tokens_to_keep` 被设为 2（见 typical_sampling 分支）。
- ⚠️ 启用 `--accel` 时代码走 `num_return_sequences == 1` 的加速分支，beam 行为与标准 HF 路径不同。""",
    ))

    reg(P(
        key="repetition_penalty", group="sampling", label="repetition_penalty",
        kind="number", default=10.0, minimum=0.1, maximum=20.0, step=0.1,
        summary="重复惩罚。**官方默认 10.0，远高于常规 LLM 的 1.0~1.3**。",
        info="10.0=官方默认，强力抑制语义token重复；调低韵律更自由但可能结巴",
        affects="发音稳定性 / 韵律自然度",
        detail_md="""走标准 HF `RepetitionPenaltyLogitsProcessor`：已出现过的 token，logit 为正则**除以** penalty，为负则**乘以** penalty。

`penalty=10.0` 极激进 —— 任何已生成的语义 token 再被选中的概率被压到接近零。

**为什么 TTS 要这么高**：语义 token 序列里同一个 token 天然会连续出现（长元音、静音 token 52、持续辅音）。不强力惩罚，自回归模型容易陷入「重复同一 token 直到 max_mel_tokens」的死循环，表现为拖长音或卡住不停。官方用高惩罚换取生成必定终止。

代码里还有第二重保险：`remove_long_silence(codes, silent_token=52, max_consecutive=30)` 把连续超过 30 个的静音 token 压缩到最多 10 个。""",
        tuning_md="""- **保持 10.0**，这是官方调出来的稳定值。
- 觉得语音「太跳跃」「每个音短促、缺乏连贯」→ 试 5.0~8.0。
- 出现拖长音/重复 → 提到 12.0~15.0。
- 调低时**务必同时观察是否触发 max_mel_tokens 截断**。""",
        pitfall_md="""- ⚠️ 不要按 LLM 直觉设成 1.0~1.3 —— 在 TTS 上几乎必然导致重复死循环。
- ⚠️ 这个参数和 `max_mel_tokens` 是一对：惩罚太低 → 生成不收敛 → 撞上 max_mel_tokens 被硬截断 → 音频末尾被切掉。""",
    ))

    reg(P(
        key="length_penalty", group="sampling", label="length_penalty",
        kind="number", default=0.0, minimum=-2.0, maximum=2.0, step=0.1,
        summary="beam search 的长度归一化指数。仅 num_beams>1 时生效。",
        info="0=不归一化；>0 偏好更长序列，<0 偏好更短",
        affects="生成长度",
        detail_md="""HF 长度归一化：`score / (length ** length_penalty)`。

- `0.0`（默认）→ 不做长度归一化，直接用累积分数
- `>0` → 除以长度的幂，长序列惩罚被摊薄 → **偏好更长输出**
- `<0` → **偏好更短输出**""",
        tuning_md="""- 默认 0.0 即可。
- 音频总是被过早截断（句子没说完就停）→ 试 0.5~1.0。
- 音频末尾总拖一段无意义的音 → 试 -0.5。""",
        pitfall_md="""- ⚠️ `num_beams=1` 时**完全无效**。
- ⚠️ 效果比 `repetition_penalty` 弱得多，别指望用它解决重复问题。""",
    ))

    reg(P(
        key="max_mel_tokens", group="sampling", label="max_mel_tokens", kind="slider",
        default=1500, minimum=50, maximum=1815, step=10,
        summary="单段最多生成的语义 token 数。**过小会截断音频**。",
        info="语义token 为 25Hz → 1500≈60秒；上限 1815 来自 config.yaml",
        affects="完整性 / 速度 / 显存",
        detail_md="""`max_length = trunc_index + max_generate_length`，即 条件token数 + 文本token数 + max_mel_tokens。

**换算**：语义 codec 的 `downsample_scale=2` 把 w2v-BERT 的 50Hz 压到 **25Hz**，所以 1 个语义 token ≈ 40ms 音频：

| max_mel_tokens | 约等于音频时长 |
|---|---|
| 500 | 20 秒 |
| 1000 | 40 秒 |
| **1500（默认）** | **60 秒** |
| 1815（上限） | 72.6 秒 |

上限 1815 来自 `config.yaml` 的 `gpt.max_mel_tokens`，也是 `mel_pos_embedding` 的训练长度，**超过会位置编码越界**。

超出时 `codes[:, -1] != stop_mel_token`，官方 warn：
> generation stopped due to exceeding max_mel_tokens. Consider reducing max_text_tokens_per_segment or increasing max_mel_tokens.""",
        tuning_md="""- 单段文本 ≤60 秒 → 保持 1500。
- 用了很大的 `max_text_tokens_per_segment`（长分句）→ 相应提高，最多 1815。
- 想省显存/加速且文本本身很短 → 可降到 800。""",
        pitfall_md="""- ⚠️ **不要超过 1815**，位置编码会越界。
- ⚠️ 这个值是**每个分句**的上限，不是整段文本的。分句多时总 token 数会远超它。
- ⚠️ 撞到上限时音频是**硬截断**的，末尾会听到突兀切断，不会淡出。""",
    ))


def _segment(reg, P):
    reg(P(
        key="max_text_tokens_per_segment", group="segment",
        label="分句最大 Token 数", kind="slider", default=120,
        minimum=20, maximum=600, step=2,
        summary="每个合成段最多容纳多少文本 token。决定分句粒度。",
        info="建议 80~200；值越大分句越长，越小分句越碎",
        affects="质量 / 速度 / 连贯性",
        detail_md="""v2.5 的分句由 `split_text_by_tokens()` 完成：
```python
capacity = gpt.text_pos_embedding.emb.num_embeddings   # = max_text_tokens + 2 = 602
budget   = min(max_tokens, capacity - 2) - token_len(lang_prefix)
```
切分策略（三级降级）：
1. 整段 token 数 ≤ budget → **不切**
2. 保护 `<|SPECIAL_TOKEN_x|>...<|SPECIAL_TOKEN_x|>` 发音标注块不被切开（`SPLIT_PROTECTED_PATTERN`），其余按 `，。！？、；：,.!?;:\\n` 切
3. 单个标点片段仍超 budget → **按字符硬切**

每段独立走一次 GPT + CFM + BigVGAN，段间插入 `interval_silence` 毫秒静音。""",
        tuning_md="""- **80~200 是官方推荐区间**，默认 120。
- 值大(200+)：分句少、韵律更连贯，但单段更易撞 `max_mel_tokens`，KV cache 更大、显存更高。
- 值小(60-)：分句碎，每段重新起调，听感像「一句一顿」，但单段失败影响范围小。
- **8GB 显存建议 100~140**，配合 num_beams ≤3。""",
        pitfall_md="""- ⚠️ v2.5 **没有** `tokenizer.split_segments()`（那是 v2 的 BPE 路径）。v2.5 用 tiktoken，界面上的「预览分句结果」在 v2.5 下只显示总 token 数。
- ⚠️ 上限 600 来自 `config.yaml` 的 `gpt.max_text_tokens`。设 600 时加上语言前缀和 2 个特殊 token 正好触到 602 的位置编码容量。
- ⚠️ 发音标注块受保护不会被切开，所以一个超长标注可能让实际段长超过 budget。""",
    ))

    reg(P(
        key="duration_factor", group="segment", label="时长系数", kind="slider",
        default=1.0, minimum=0.5, maximum=2.0, step=0.01,
        summary="语速控制。**真·时长控制，不是后期变速，所以不变调**。",
        info="<1.0 加快，>1.0 放慢；官方推荐 0.8~1.2",
        affects="语速 / 音质",
        detail_md="""IndexTTS-2 论文的核心贡献之一。实现在 `infer_generator`：
```python
target_lengths = torch.LongTensor([int(S_infer.shape[1] * 1.72 * duration_factor)])
cond = s2mel.models['length_regulator'](S_infer, ylens=target_lengths, n_quantizers=3)
```

`S_infer` 是解码后的语义特征(25Hz)。`1.72` 是语义帧率到 mel 帧率的换算比（mel 为 22050/256 ≈ 86.1Hz，86.1/50 ≈ 1.72）。

**关键**：它改变的是 `InterpolateRegulator` 把语义特征**插值拉伸到多少 mel 帧**，然后 CFM 在这个新长度上生成 mel，BigVGAN 再转波形。整条链路是「重新生成」而不是「拉伸已有音频」，所以**音高不变、共振峰不变、音质不损失**。

这与传统变速（time_stretch / WSOLA / PSOLA）有本质区别 —— 后者会引入金属感或音高偏移。""",
        tuning_md="""- **0.9~1.1** 几乎无损，安全区。
- 0.8 明显加快，用于短视频配音。
- 1.2~1.3 放慢，用于有声书、教学。
- 极值(0.5 或 2.0)会出现音节粘连或过度拉长，官方给了范围但不推荐。""",
        pitfall_md="""- ⚠️ 时长变化**同步改变 mel 帧数**，进而改变 CFM 计算量。`duration_factor=2.0` 时 CFM 耗时约翻倍。
- ⚠️ 与低显存自动切块叠加时，每个 40 字块都独立应用该系数。
- ⚠️ 不影响 GPT 生成的语义 token 数量（GPT 不知道这个参数），所以**韵律节奏本身不变**，变的是每个音的持续时间。""",
    ))

    reg(P(
        key="interval_silence", group="segment", label="段间静音", kind="slider",
        default=200, minimum=0, maximum=1000, step=10, unit="ms",
        summary="分句之间插入的静音时长（毫秒）。",
        info="默认 200ms；设为 0 则各段无缝拼接",
        affects="节奏自然度",
        detail_md="""`insert_interval_silence()` 在相邻 wav 段之间插入 `int(22050 * interval_silence / 1000)` 个零样本。

低显存自动切块路径用同一段逻辑（`torch.zeros(1, sr*ms/1000, dtype=int16)`）。""",
        tuning_md="""- 200ms 接近自然句间停顿，保持默认。
- 分句很碎（max_text_tokens_per_segment 小）时降到 80~120ms，否则太拖。
- 想做「无缝长句」效果设 0，但可能听到段与段的接缝。""",
        pitfall_md="""- ⚠️ 插入的是**纯数字零**，不是环境底噪。如果参考音频有底噪，段间会形成「噪-静-噪」的呼吸感，反而暴露拼接痕迹。此时给参考音频降噪比调这个参数更有效。""",
    ))

    reg(P(
        key="seed", group="segment", label="随机种子", kind="number", default=-1,
        minimum=-1, maximum=2**31 - 1, step=1, precision=0,
        summary="固定随机种子以获得可复现结果。-1 表示每次随机。",
        info="-1=随机；填固定值可精确复现同一结果（含情感随机采样）",
        affects="可复现性",
        detail_md="""官方 webui.py **没有暴露 seed**，所以同样的参数每次结果都不同（`do_sample=True` 本身就有随机性，`use_random=True` 还会随机取情感原型）。

本 UI 在推理前调用 `torch.manual_seed(seed)` / `random.seed(seed)`，让结果可复现 —— 这对**调参对比**和**训练数据采集**都是必需的。

`-1` 表示用 `random.randint` 生成一个新种子并显示出来，方便你「抽到满意的结果后把种子记下来」。""",
        tuning_md="""- 调参时**务必固定 seed**，否则你分不清是参数变化还是随机性导致的差异。
- 挑音色时反过来：固定参数、遍历多个 seed，选最好的那个。""",
        pitfall_md="""- ⚠️ 固定 seed 只在**同一硬件、同一精度、同一 num_beams** 下可复现。换 GPU 或切换 bf16 会因浮点运算顺序不同而产生差异。
- ⚠️ 启用 `--accel` 加速引擎后走自定义 CUDA Graph 路径，可复现性需另行验证。""",
    ))


def _engine(reg, P):
    reg(P(
        key="use_bf16", group="engine", label="BF16 半精度", kind="checkbox",
        default=False, version="2.5",
        summary="v2.5 用 BF16 推理。显存减半、速度提升，质量损失极小。",
        info="需 GPU 支持 BF16（RTX 30 系及以上）",
        affects="显存 ↓↓ / 速度 ↑ / 质量 ≈无损",
        detail_md="""```python
use_bf16 = HALF_PRECISION and torch.cuda.is_bf16_supported()
self.dtype = torch.bfloat16 if use_bf16 else None
# GPT: self.gpt.eval().bfloat16()
# 推理: torch.amp.autocast(device_type, enabled=dtype is not None, dtype=dtype)
```

BF16 vs FP16：BF16 保留 FP32 的 8 位指数（动态范围相同），只有 7 位尾数。所以**不会溢出**，无需 loss scaling，对 TTS 这种激活值范围大的场景更安全。FP16 的 5 位指数在 attention logits 上容易上溢。

官方 README：「使用 FP16/BF16 推理非常有益，推理更快且显存占用更低，质量损失极小。」官方实测 BF16 下 RTF 低至 0.20。""",
        tuning_md="""- **RTX 4060 支持 BF16，强烈建议开启。**
- 低显存(<10GB)时本 UI 自动开启。""",
        pitfall_md="""- ⚠️ 硬件不支持 BF16 时回退全精度并打印提示，不报错。
- ⚠️ CPU 模式下强制忽略（`use_bf16 = False if device == "cpu"`）。
- ⚠️ v2（非 2.5）用的是 `use_fp16`，不是 bf16。""",
    ))

    reg(P(
        key="use_cuda_kernel", group="engine", label="BigVGAN CUDA Kernel",
        kind="checkbox", default=False,
        summary="BigVGAN 融合激活的自定义 CUDA kernel。",
        info="首次使用需 JIT 编译，失败自动回退到 torch 实现",
        affects="速度 ↑（声码器阶段，占比 <10%）",
        detail_md="""```python
if self.use_cuda_kernel:
    from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d
    # 预加载 anti_alias_activation_cuda
```
加载失败时打印 `>> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.` 并自动置 False。

只加速 BigVGAN 这一段。从官方打印的 `bigvgan_time` 看，它在总耗时里通常占比不到 10%。""",
        tuning_md="""- 收益有限(<10% 耗时占比)，且需要 CUDA 编译环境。
- **Windows + 无 nvcc 环境下建议关闭**，避免首次启动卡在编译上。""",
        pitfall_md="""- ⚠️ 需要本机有匹配 CUDA 版本的 nvcc。cu118 的 torch 配 CUDA 12.x toolkit 会编译失败。
- ⚠️ 编译产物会缓存，首次慢、后续快。
- ⚠️ 非 CUDA 设备(cpu/mps/xpu)会被强制关闭。""",
    ))

    reg(P(
        key="use_deepspeed", group="engine", label="DeepSpeed 推理加速",
        kind="checkbox", default=False,
        summary="用 DeepSpeed 的 kernel injection 加速 GPT2InferenceModel。",
        info="效果高度依赖硬件，可能加速也可能变慢，建议实测对比",
        affects="速度 ?（不确定）",
        detail_md="""```python
self.ds_engine = deepspeed.init_inference(
    model=self.inference_model, mp_size=1,
    replace_with_kernel_inject=True,
    dtype=torch.float16 if half else torch.float32)
self.inference_model = self.ds_engine.module.eval()
```
加载失败(ImportError / OSError / CalledProcessError)会自动回退普通推理。""",
        tuning_md="""- 官方原话：「DeepSpeed *可能*在部分系统上加速推理，但也可能变慢，效果取决于具体硬件、驱动及操作系统。建议分别开启和关闭测试。」
- **单卡消费级 GPU 上通常收益不明显**，`mp_size=1` 没有并行收益，只剩 kernel fusion。""",
        pitfall_md="""- ⚠️ **Windows 上 DeepSpeed 安装困难**（官方 README 明确提示）。
- ⚠️ `dtype=torch.float16 if half` —— v2.5 用 BF16 时这里传的 `half=use_bf16`，但 DeepSpeed 内部按 fp16 处理，可能精度不匹配。
- ⚠️ 与 `--accel` 互斥（两者都替换 inference_model）。""",
    ))

    reg(P(
        key="use_accel", group="engine", label="GPT2 加速引擎", kind="checkbox",
        default=False,
        summary="自研 GPT2 推理引擎（flash-attn + CUDA Graph + 分页 KV cache）。",
        info="需 flash-attn；仅 num_return_sequences=1 时启用",
        affects="速度 ↑↑（T2S 阶段）",
        detail_md="""`post_init_gpt2_config` 里构建：
```python
accel_gpt = GPT2AccelModel(gpt_config)
accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)
self.accel_engine = AccelInferenceEngine(
    model=accel_gpt, lm_head=nn.Sequential(final_norm, mel_head),
    num_layers=24, num_heads=20, head_dim=64,
    block_size=256, num_blocks=16,      # 16*256 = 4096 tokens 容量
    use_cuda_graph=True)
```
启用后走 `accel_engine.generate(...)` 而不是 HF 的 `generate()`。

**这是收益最大的加速选项** —— T2S 自回归是整个流程的耗时大头。""",
        tuning_md="""- 能装上 flash-attn 就开，T2S 阶段提速明显。
- **但注意它绕过了 HF 的 LogitsProcessor**：只传 temperature，`top_p / top_k / repetition_penalty / num_beams / length_penalty` **在加速引擎路径下不生效**（见 `accel_engine.generate()` 的参数列表）。""",
        pitfall_md="""- ⚠️ **flash-attn 在 Windows 上极难安装**，需要预编译 wheel。pyproject 锁的是 `flash-attn==2.8.3.post1` + `triton-windows`。
- ⚠️ 需要 torch 2.8 + cu128 才能匹配官方预编译 wheel。当前环境是 torch 2.7.1+cu118，**大概率装不上**。
- ⚠️ KV cache 容量硬编码 4096 token，超长文本会溢出。
- ⚠️ 开启后采样参数失效，**无法用 top_p/repetition_penalty 调音**。建议调参阶段关闭，定稿后再开。""",
    ))

    reg(P(
        key="use_torch_compile", group="engine", label="torch.compile (s2mel)",
        kind="checkbox", default=False,
        summary="对 CFM 做 torch.compile 图优化。",
        info="需 triton；首次推理有额外编译耗时，之后 CFM 阶段提速",
        affects="速度 ↑（S2M 阶段）/ 首次启动 ↓↓",
        detail_md="""```python
if self.use_torch_compile:
    self.s2mel.enable_torch_compile()   # → self.models['cfm'].enable_torch_compile()
```
只编译 CFM（DiT 估计器），不影响 GPT 和 BigVGAN。

CFM 推理跑 25 步 Euler（`diffusion_steps = 25` 在 `infer_generator` 里硬编码），循环调用同一个 estimator，**非常适合 compile** —— 编译一次，25 步全受益。""",
        tuning_md="""- 批量合成大量文本时开启很划算（编译成本被摊薄）。
- 只合成一两句就关掉，编译耗时可能比省下的还多。
- **Windows 需要 `triton-windows`**（pyproject 已配 win32 marker）。""",
        pitfall_md="""- ⚠️ CFM 输入长度是**动态的**（`cat_condition.size(1)` 随分句变化）。每个新长度都可能触发重编译，分句长度差异大时会反复编译。
- ⚠️ 首次编译可能耗时 1~5 分钟，界面上看起来像卡死。
- ⚠️ 与 `setup_caches(max_batch_size=1, max_seq_length=8192)` 的固定 cache 交互需验证，compile 后可能报 dynamic shape 错误。""",
    ))


def _memory(reg, P):
    reg(P(
        key="use_qwen_emo", group="memory", label="QwenEmotion 常驻",
        kind="checkbox", default=False,
        summary="是否把 QwenEmotion(Qwen3-0.6B, 1.14GB) 常驻显存。",
        info="关闭=按需加载用完释放（低显存推荐）；开启=常驻，切换零延迟",
        affects="显存 ↑1.2GB / 情感文本控制响应速度",
        detail_md="""官方行为（`IndexTTS2.__init__`）：
```python
if use_qwen_emo:
    self.qwen_emo = QwenEmotion(os.path.join(model_dir, cfg.qwen_emo_path))
else:
    self.qwen_emo = None
    print(">> QwenEmotion not loaded (use_qwen_emo=False)")
```
`infer(use_emo_text=True)` 时若 `self.qwen_emo is None` 直接抛 RuntimeError。

**本 UI 的改进**：`self.qwen_emo` 只是普通属性，可在运行时动态挂卸，无需重启、无需改官方代码：
```python
tts.qwen_emo = QwenEmotion(path)     # 挂载
tts.qwen_emo = None                  # 卸载
torch.cuda.empty_cache()             # 归还显存
```

显存预算（8GB 卡，BF16）：

| 组件 | 占用 |
|---|---|
| GPT (777M, bf16) | ~1.55 GB |
| w2v-BERT-2.0 (580M) | ~1.2 GB |
| semantic codec | ~0.3 GB |
| s2mel/CFM (100M) | ~0.2 GB |
| BigVGAN | ~0.2 GB |
| CAMPPlus | ~0.03 GB |
| **常驻小计** | **~3.5 GB** |
| QwenEmotion (fp16) | +1.2 GB |
| KV cache + 激活 + CUDA 上下文 | ~1.5-2.5 GB |""",
        tuning_md="""- **8GB 显存：关闭常驻**，用按需加载。首次用情感文本控制有几秒加载延迟，之后本次会话内可复用。
- ≥12GB 显存：可以常驻，切换零延迟。""",
        pitfall_md="""- ⚠️ 官方用 `device_map="auto"`：显存不足时 accelerate 会**静默**把层 offload 到 CPU，速度掉几十倍且**不报错**。本 UI 改为显式指定 device + 余量预检。
- ⚠️ 官方硬编码 `torch_dtype="float16"`，新版 transformers 已改名 `dtype`，会打 deprecation warning。本 UI 做兼容。
- ⚠️ 卸载后 `empty_cache()` 只把缓存归还 CUDA，PyTorch 的显存碎片不会完全消除。频繁挂卸可能逐渐变慢。""",
    ))

    reg(P(
        key="low_vram_auto_split", group="memory", label="低显存自动分块",
        kind="checkbox", default=True, readonly=True,
        summary="显存 <10GB 时自动把长文本按标点切成 ≤40 字的块逐块合成。",
        info="由 IndexTTS2 内部根据显存自动决定，不可手动关闭",
        affects="显存 ↓ / 韵律连贯性 ↓",
        detail_md="""```python
# IndexTTS2.__init__
if total_vram_gb < 10.0:
    self.low_vram = True
    print(f">> Low-VRAM mode enabled ({total_vram_gb:.1f} GB < 10 GB), ...")

# IndexTTS2.infer
if self.low_vram and not stream_return and len(text) > 40:
    segments = self.split_text_by_punctuation(text, max_chars=40)
    for seg_text in segments:
        ...  # 每段独立完整推理
```

`split_text_by_punctuation` 按 `，。！？、；：,.!?;:\\n` 切，贪心累加到 ≤40 字符；无标点且超长的片段**保持原样不切**（避免切碎词）。

注意这是**独立于** `max_text_tokens_per_segment` 的**第二层**切分：低显存时先按 40 字符粗切，每块内部再按 token 数细分。""",
        tuning_md="""- 8GB 卡上无法关闭（除非改源码）。
- **缓解韵律断裂的办法**：
  1. 把 `interval_silence` 降到 80~120ms，减弱块间「换气感」
  2. 自己按语义手动分段，分批合成后用音频工具拼接
  3. 用「批量合成」Tab 按段落拆任务，每段单独调参""",
        pitfall_md="""- ⚠️ 每块会重新建立参考音频缓存吗？**不会** —— `cache_spk_cond` 跨块复用，只有首块慢。但每块都要重跑 GPT+CFM+BigVGAN。
- ⚠️ 40 字符是**硬编码**的，不受任何参数影响。
- ⚠️ 块间韵律不接续是最主要的听感损失，长段落尤其明显。""",
    ))
