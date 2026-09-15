"""训练相关参数的注册表（阶段 2 + 一键三连）。

`params.register_training_params()` 按约定导入本模块的 `build(reg, P)`。
在此之前这个模块不存在，那次导入被 `except ImportError: pass` 静默吞掉，
于是「训练 · 数据集 / LoRA / 优化器 / DPO / 评测」几个分组在手册里一直是
空的 —— 控件有 tooltip，但手册查不到。

这里登记的是**训练体系的全部旋钮**（与 guard.LoRAConfig、各组件的
Options dataclass 一一对应），加上「一键三连」自动操作的那几个参数。
描述以源码为准：每一条的默认值、边界、行为都能在对应模块里找到出处，
不留「写了但不生效」的摆设项。

分组顺序即手册里的展示顺序（params.group_order 按注册顺序推导）：
一键三连 → 数据集 → LoRA → 优化器 → DPO → 评测指标。
"""

from __future__ import annotations

from typing import Any, Callable


def build(reg: Callable[[Any], Any], P: Any) -> None:
    # =====================================================================
    # 一键三连
    # =====================================================================
    reg(P(
        key="oc_input_path", group="oneclick", label="服务器路径（长音频 / 目录）",
        kind="text", default="",
        summary="长音频最实际的入口：填一个音频文件或一个目录的路径。",
        info="填目录会递归找出目录下所有支持的音频（wav/flac/mp3/ogg/m4a/opus）。"
             "填文件就是单个长音频。也可改用上面的上传控件。",
        affects="整个流水线的输入范围",
        detail_md="""
「一键三连」有两种喂法，可以同时给：

1. **上传控件** —— 拖若干文件进去，适合一堆短音频。
2. **服务器路径** —— 一个文件（长音频，比如一整期播客）或一个目录。

为什么长音频建议走路径：两小时的 wav 通过浏览器上传要先生成临时副本再拷进
数据集，既慢又占双份磁盘；直接给路径就是本机拷贝。
""",
        tuning_md="""
- 短音频（3~15 秒一条）直接给目录最省事。
- 长音频给单个文件，流水线会按声学打分切成 8~15 秒的片段。
- 同一个文件重复给只会导入一次（按源文件绝对路径去重）。
""",
        pitfall_md="""
路径里的文件必须是**本地可读的音频**。给一个文本文件会被跳过并计入
「非音频」；给一个不存在的路径会列入 missing 并在报告开头提示。
""",
    ))

    reg(P(
        key="oc_slice_over_sec", group="oneclick", label="超过多少秒就切片",
        kind="slider", default=20.0, minimum=5.0, maximum=20.0, step=1.0,
        unit="秒",
        summary="判定「长音频」的阈值，默认 20 秒 = 训练可接受的最长样本。",
        info=f"超过它的音频会被自动切成 8~15 秒的片段；默认 20 秒与 "
             f"dataset.MAX_TRAIN_SEC 一致。",
        affects="样本数量与单条时长分布",
        detail_md="""
这个值与数据集体检是同一条线：`dataset.MAX_TRAIN_SEC = 20.0`。超过 20 秒的
样本体检直接判 `too_long`，**永远不参与训练**。所以不切片的话，一段 10 分钟
的录音就是一条永远用不上的废样本。

切片复用「参考音频工作台」的 `find_segments`：按语音占比、信噪比、削波、
响度打分，并优先让切分点落在停顿处，比按固定长度硬切好得多。
""",
        tuning_md="""
- 录音本身干净、句子间隔清楚 → 保持默认 20 即可。
- 想切得更细（每片更短、样本更多）→ 调小到 12~15。
- 注意：**片段太短会丢失韵律信息**，1~3 秒的片段既学不到语气，又容易被
  体检判 `too_short`（下限 1 秒）。
""",
        pitfall_md="""
切片后每条片段**不带文本**（各片说的不是同一句话），必须靠语音识别补 ——
这正是流水线紧接着要做的第 3 步。如果关掉了「自动转写」，切出来的片段会
全是 `no_text`，下一步的筛选会把它们全部丢掉。
""",
    ))

    reg(P(
        key="oc_slice_target_sec", group="oneclick", label="每片目标时长",
        kind="slider", default=12.0, minimum=2.0, maximum=20.0, step=0.5,
        unit="秒",
        summary="切片的目标长度。8~15 秒是零样本 TTS 的甜点区。",
        info="默认 12 秒：够长能学到韵律，又不至于让训练显存吃紧。",
        affects="样本数量 / 单条信息量 / 显存",
        detail_md="""
参考音频的推荐规格是 3~15 秒（`config.REF_AUDIO_IDEAL`），训练样本同理：

| 时长 | 后果 |
|---|---|
| < 3 秒 | 音色信息不足，容易学过拟合到某一个音上 |
| 8~15 秒 | 足够体现音色与韵律，显存也友好 |
| > 20 秒 | 体检判 `too_long`，不参与训练 |

默认 12 秒落在甜点区中间。
""",
        pitfall_md="""
上限被 `MAX_TRAIN_SEC`（20 秒）卡死 —— 设成 20 以上会被参数自检直接拦下，
因为「切了也白切」。
""",
    ))

    reg(P(
        key="oc_enhance", group="oneclick", label="自动优化音频",
        kind="checkbox", default=True,
        summary="降噪 / 归一 / 掐静音 / 重采样，把参考音频处理干净。",
        info="关掉则只做体检、不改动音频。这是投入产出比最高的一步。",
        affects="音色相似度与训练稳定性",
        detail_md="""
IndexTTS-2.5 是零样本 TTS，音色几乎完全由参考音频决定（CAMPPlus 声纹 +
w2v-BERT 情感特征 + ref_mel 声学模板三路注入）。所以「把参考音频处理好」
往往比多训几轮更有效、更零风险。

优化按固定顺序执行（`audio_lab.enhance`）：去直流 → 掐首尾静音 → 降噪 →
响度归一 → 重采样到 22050 → 截断到阈值上限。
""",
        tuning_md="""
- 原始录音有空调声/底噪 → 保持开启，降噪强度 0.6 起步。
- 录音本来就很干净（录音棚干声）→ 可以只留归一与掐静音，把降噪关掉，
  避免连气声、齿音一起抹掉。
- 每条样本处理完会重新体检，报告里给出「优化前 → 优化后」的平均体检分。
""",
        pitfall_md="""
降噪强度调太高（>0.85）会让人声发闷、丢掉高频细节 —— 音色听起来「糊」，
而体检分反而更高（噪声少了）。分数不是唯一标准，**要听**。
""",
    ))

    reg(P(
        key="oc_asr", group="oneclick", label="自动转写为训练文本",
        kind="checkbox", default=True,
        summary="用 whisper 逐条转写，长音频则是「逐片转写」。",
        info="关掉后必须到「数据集」页手动补文本，否则筛选阶段一条都留不下。",
        affects="能否进入训练（没有文本就没有 text_tokens / mu）",
        detail_md="""
**这一步就是整个流程的「对齐」。** 工程里没有强制对齐器（没有 whisperX /
MFA / CTC 对齐，上游只留了一段无人调用的死代码），所以这里用的是
「先按声学质量切片，再对每片独立转写」：

```
长音频 → 切片（10 片）→ 每片单独转写 → 一片音频 ↔ 一段文本
```

文本与音频按构造一一对应。相比「整段转写再按字数硬切」，这种做法不会错位 ——
而错位是长音频做训练数据最常见的翻车方式（文本与音频说的事对不上，
模型学到的就是噪声）。
""",
        tuning_md="""
- whisper-small 在中文上已够用（约 0.55 GB，每条 0.3~1 秒）。
- 更准可以上 medium，但更慢更吃显存，且要能下载到权重。
- 只想快速验证流程 → 用 tiny/base 先跑通，再换 small 重跑。
""",
        pitfall_md="""
- 首次使用需要联网下载 whisper 权重到 `checkpoints/hf_cache/whisper/`。
- 转写结果是**原始识别结果**，多音字、专有名词、标点都可能不准。
  人工校对过的文本更值钱 —— 已有文本的样本不会被覆盖，
  想改就到「数据集」页改。
- 识别为空（纯音乐、噪声、静音）的样本会被记 `note` 并在筛选阶段丢掉。
""",
    ))

    reg(P(
        key="oc_whisper_size", group="oneclick", label="识别模型（whisper）",
        kind="dropdown", default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        summary="转写用的 whisper 尺寸，直接影响文本准确率与耗时。",
        info="默认 medium。tiny 0.10 GB / base 0.20 / small 0.55 / "
             "medium 1.60 / large-v3 3.20 / turbo 1.70（fp16 显存）",
        affects="文本正确率 → 训练质量；以及识别耗时",
        detail_md="""
转写准确率决定训练文本的质量，而训练文本错字多会直接教坏模型 ——
它会照着错误文本去学发音。这是整条流水线里最容易「静默降质」的一环。

显存占用见 `reward.WHISPER_SIZES`。scorer 是独立于推理引擎构造的，
只加载 whisper + campplus（约 0.6 GB），所以识别阶段**不需要**加载大模型。
""",
        tuning_md="""
- **默认 medium**：它的产出直接成为训练文本，准确率决定模型学什么。
  实测同一批合成语音，`small` 把「转眼间」写成「专业间」、把「西莲」写成「西蓮」，
  换成 medium 后逐字正确（WER 从 0.033 降到 0.0）。
- 显存紧张、只想先跑通流程 → 用 small（0.55 GB）。
- 有专业术语/方言/多人对话 → large-v3（3.2 GB），但识别会明显变慢。
- turbo：速度接近 large、质量接近 medium，是个折中。

**打分用的是另一个旋钮**（`oc_score_whisper_size`），见那一条的说明。
""",
        pitfall_md="""
whisper ≥ medium 会触发一条 warn（显存提醒）。8 GB 卡上与其它阶段错峰跑没问题，
但如果同时挂着引擎就会紧张 —— 流水线本身是分阶段串行的，不会撞。
""",
    ))

    reg(P(
        key="oc_score_whisper_size", group="oneclick",
        label="打分模型（whisper，择优用）",
        kind="dropdown", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        summary="择优时给候选打 reward 用的模型，**故意比识别小一档**。",
        info="默认 small。改大它不会让识别更准（那是上一个旋钮的事），"
             "只会让打分更慢、更吃显存。",
        affects="择优打分的速度与显存占用；对名次影响很小",
        detail_md="""
**为什么和识别分开**：两者的需求并不对等。

| | 识别（`oc_whisper_size`） | 打分（这一条） |
|---|---|---|
| 产出用途 | **直接成为训练文本**，决定模型学什么 | 只在**候选之间**做相对比较 |
| 准确率要求 | 高 —— 错字会教坏模型 | 够用即可 —— 转写噪声是所有候选的共同项 |
| 运行时显存 | 引擎**已卸载**，1.6 GB 放得下 | 引擎**必然驻留**，与它抢显存 |

把打分也设成 medium 时，8 GB 卡上会出现
`引擎 4.94 + medium 1.6 + campplus ≈ 7.85 / 8.19 GB`，
只剩几百 MB —— 实测后果是 Windows 把扩散采样挤到共享内存，
**s2mel 从 0.9 秒变成 25 秒**（WDDM 静默降速，不报 OOM），
整轮验收从 6.9 分钟涨到 13.9 分钟。所以这里默认 small。

若显存充裕（≥ 12 GB）想统一成 medium，改这一条即可；
流水线在开跑前也会做显存余量体检，不够会直接给出提示。
""",
        pitfall_md="""
改大它**不会**提高识别准确率 —— 那只由 `oc_whisper_size` 决定。
这两条容易看串，label 里已经分别写明「识别」与「打分」。
""",
    ))

    reg(P(
        key="oc_min_score", group="oneclick", label="音频体检分下限",
        kind="slider", default=45.0, minimum=0.0, maximum=100.0, step=1.0,
        summary="低于这个分数的样本直接丢弃。0 = 不按分数筛。",
        info="体检分综合了时长、信噪比、采样率、削波、静音占比、直流偏移。",
        affects="训练集规模与干净程度",
        detail_md="""
体检打分规则在 `audio_lab._score`（相对 `config.REF_AUDIO_IDEAL`）：

| 项 | 扣分 |
|---|---|
| 时长 < 3s / < 8s / > 15s | −35 / −15 / −5 |
| 信噪比 < 15 / < 25 dB | −25 / −10 |
| 采样率 < 16k / < 22.05k | −25 / −6 |
| 削波比例 > 0.1% | −20 |
| 响度太轻（< −35 dBFS）或太响（> −6） | −12 |
| 静音占比 > 30% | −15 |
| 开头静音 > 1 秒 | −10 |
| 直流偏移 > 0.01 | −4 |

分档：≥88 优秀 · ≥72 良好 · ≥55 可用 · ≥38 勉强 · 其余不建议使用。
""",
        tuning_md="""
- 数据充足（几小时）→ 可以放宽到 30~45，多留样本。
- 数据紧张（只有十几分钟）→ 宁可降到 0，先把量凑够；低分样本的危害
  不如「样本不足」大（后者直接训不了，下限 20 条）。
- 手机录音的典型分在 50~75；录音棚干声常在 85 以上。
""",
        pitfall_md="""
门槛设太高会把数据筛干净，然后流水线在第 4 阶段停下并告诉你「只剩 N 条」——
这是**有意设计**的早停：与其拿 5 条样本去训，不如让用户先补数据。
""",
    ))

    reg(P(
        key="oc_max_text_repeats", group="oneclick", label="同一句话最多留几条",
        kind="number", default=3, precision=0, minimum=0,
        summary="同一段文本重复出现的次数上限，防止一句话主导整个训练集。",
        info="0 = 不限。按「去空白 + 转小写」后的文本比对。",
        affects="训练集的多样性",
        detail_md="""
一批短音频里常有反复重录的同一句话（或长音频里反复出现的口头禅）。
这些重复样本会让损失被同一句话主导，模型把资源花在拟合那一句上。

超过上限的会被丢弃并计入报告的「同文本重复」。
""",
        tuning_md="""
- 内容多样的播客/有声书：可以设大一些（5~10）。
- 短句练习、重复朗读：设 2~3。
- 确认数据本身很干净、不需要去重 → 设 0 关闭。
""",
        pitfall_md="""
去重只看**文本是否完全一致**。语速不同、断句不同的同一句话不会被判为重复
（这通常是好事，因为那正是我们想学的韵律变化）。
""",
    ))

    reg(P(
        key="oc_top_k", group="oneclick", label="每个目标保留并参评的档位数",
        kind="slider", default=3, minimum=1, maximum=8, step=1,
        summary="保险库只留 val loss 最好的 top-K 档，它们全部进入真机打分。",
        info="这就是「三个或一个」里的「三个」：K 个候选 → 打分 → 激活最优的 1 个。",
        affects="参评候选数、择优耗时、最终交付的确定性",
        detail_md="""
**「三个或一个」是怎么来的**：训练器每有一次「val loss 改善」就把权重存进
checkpoint 保险库（`guard.CheckpointVault`），保险库按 val loss 只保留 top-K，
被挤掉的当场删除。K 由这个参数决定。

然后第 7 阶段把这些档位**逐个挂到引擎上真机合成**，用 reward（WER +
声纹相似）打分排序，最后把评分最高的那一个激活到该 run 的 `adapter/`。

所以交付的是 **1 个最优模型**，另外 K-1 个备选仍留在保险库里，随时可切换 ——
既不是「只训一个没得比」，也不是「给三个让用户自己挑」。
""",
        tuning_md="""
- 想省时间 → 设 1（只参评最好的一档，直接看它到底行不行）。
- 默认 3：能在「val loss 最好」「实际听感最好」之间发现分歧。
- val loss 与听感经常不一致（val 只能说明拟合得好不好，说明不了像不像本人），
  所以 2~3 是值得的。
""",
        pitfall_md="""
**评估间隔会跟着这个值自动推导**（见「训练 · 优化器」里的
`eval_every` 说明）：小数据集上预设的 50/100/200 整个 run 都触发不到，
只会存下 1 个档位 —— 那 K 就成了摆设。流水线按总步数反推评估频率，
保证能攒出约 K 个可比较的档位。
""",
    ))

    reg(P(
        key="oc_rank_eval", group="oneclick", label="真机打分择优",
        kind="checkbox", default=True,
        summary="把候选逐个挂载合成、whisper 打分，按 reward 排序。",
        info="关掉则只按 val loss 排序 —— 省一次引擎加载 + 一批合成，但不准。",
        affects="最终选出的模型是否真的「像」",
        detail_md="""
**val loss 排不出「像不像本人」。** 它衡量的是模型在训练分布上的预测损失，
可以很低，而听感却很平（例如语气对了但音色飘）。

打分用的是评测台同一套指标（`training/reward.py`）：

```
reward = 0.6 × (1 − WER) + 0.4 × 声纹相似度
```

WER 由 whisper 转写合成音频与原文比对；声纹相似度用 CAMPPlus 对比合成音频
与数据集里的参考音频。合成用**固定种子**，所以候选之间的差异只来自模型本身。
""",
        tuning_md="""
- 时间允许 → 保持开启，这是唯一能给出「听感证据」的一步。
- 只想验证流程能跑通 → 关掉，能省几分钟。
""",
        pitfall_md="""
这一步需要**加载引擎**（要合成）。8 GB 卡上它与训练不能共存，所以流水线会
自动在训练后重新加载。报告里的「评分条数」若为 0，说明该候选一次都没评成
（例如引擎中途被卸载），此时流水线会退回按 val loss 排序并在报告里标明
「已降级」。
""",
    ))

    reg(P(
        key="oc_eval_samples", group="oneclick", label="每个候选评几条",
        kind="slider", default=5, minimum=3, maximum=20, step=1,
        summary="择优时每个候选在验证数据上评几条。",
        info="打分本身带转写噪声，评太少名次会不稳；5 条是性价比折中。",
        affects="排名的稳定性与择优耗时",
        detail_md="""
reward 的两个分量都有噪声：whisper 转写偶尔换个字，WER 就跳几个点；
声纹相似度对背景噪声也敏感。样本数少时，一两条的偶然波动就能颠倒名次。

评 N 条意味着 N 次合成 × 候选数。默认 5 条在「名次够稳」与「不太慢」之间折中。
""",
        tuning_md="""
- 候选只有 1 个 → 这一步没意义，设最小值即可。
- 候选多、且想认真比较 → 提到 10~20。
- 数据里句子长度差异很大时，建议调高，让平均更稳。
""",
    ))

    # =====================================================================
    # 数据集
    # =====================================================================
    reg(P(
        key="ds_max_train_sec", group="dataset", label="样本时长上限",
        kind="number", default=20.0, readonly=True, unit="秒",
        summary="超过它体检判 too_long，不参与训练（硬上限 72 秒）。",
        info="源码常量 dataset.MAX_TRAIN_SEC = 20.0，只读。",
        affects="可用样本范围；一键三连的默认切片阈值就取这个值",
        detail_md="""
两个不同的时限，别混：

| 常量 | 值 | 含义 |
|---|---|---|
| `MAX_TRAIN_SEC` | 20 秒 | 训练样本的**建议上限**，超过判 `too_long` |
| `HARD_MAX_SEC` | 72 秒 | 物理硬上限（1815 语义 token ÷ 25 Hz ≈ 72.6 秒） |

72 秒是 GPT 的 `max_mel_tokens=1815` 推出来的：1815 ÷ 25 Hz = 72.6 秒，
再多就超出位置编码范围。
""",
        pitfall_md="""
「参考音频推理时取前 15 秒」是**推理**侧的规则（官方 `_load_and_cut_audio`），
与这里的训练样本上限是两件事。
""",
    ))

    reg(P(
        key="split_val_ratio", group="dataset", label="验证集比例",
        kind="slider", default=0.1, minimum=0.02, maximum=0.4, step=0.01,
        summary="划分多少样本用于验证（早停与保险库都盯 val loss）。",
        info="划分只在 status=ready 的样本里做，同 seed 结果可复现。",
        affects="早停是否有效、val 曲线的可信度",
        detail_md="""
val 集有两个用途：**早停**判据，以及保险库的排序指标（`mode="min"`，
即 val loss 越小越好）。

划分用固定随机种子打乱后切分，所以同样的数据集与种子会得到同样的划分 ——
这是「val 曲线可比」的前提。
""",
        tuning_md="""
- 样本多（几百条以上）→ 5%~10% 足够。
- 样本很少（20~40 条）→ 至少留 4~8 条，否则 val 曲线抖得比训练改善还大。
- 一键三连默认 0.1。
""",
        pitfall_md="""
验证集为空时早停会被强制关闭（日志里有 warn），保险库也就不再有可信的
排序依据。样本数接近 `batch_size × grad_accum` 时尤其要注意
（`LoRAConfig.validate` 会对此发出 warn）。
""",
    ))

    reg(P(
        key="split_seed", group="dataset", label="划分随机种子",
        kind="number", default=42, precision=0,
        summary="固定它，划分才可复现、val 曲线才可比。",
        info="同一个种子 + 同一份 ready 样本 = 同一个划分。",
        affects="划分结果的可复现性",
        detail_md="""
换种子 = 换一套验证集 = val loss 曲线整体平移，两次实验的数字就没法直接比了。

一键三连把训练种子与划分种子统一用同一个值，便于复现整条流水线。
""",
        pitfall_md="""
**改了数据集就要接受 val 不可比**：新增/删除样本后即使种子不变，
划分内容也会变。想比较两次训练，请保证数据集与种子都不变。
""",
    ))

    reg(P(
        key="ds_min_sec", group="dataset", label="样本时长下限",
        kind="number", default=1.0, readonly=True, unit="秒",
        summary="短于它的样本判 too_short。",
        info="源码常量 dataset.MIN_SEC = 1.0，只读。",
        affects="极短音频是否进入训练",
        detail_md="""
1 秒是人耳能判断「这是谁」的粗略下限，也是特征提取有意义的下限。
切片时 `slice_min_sec` 默认给到 4 秒，留了更大余量。
""",
    ))

    reg(P(
        key="ds_min_snr_db", group="dataset", label="信噪比下限",
        kind="number", default=12.0, readonly=True, unit="dB",
        summary="低于它判 low_snr，不参与训练。",
        info="源码常量 dataset.MIN_SNR_DB = 12.0，只读。",
        affects="含噪样本是否进入训练",
        detail_md="""
信噪比由 `audio_lab.analyze` 用「有声帧平均能量 − 无声帧平均能量」估计
（有声帧定义为峰值 35 dB 以内的帧）。

12 dB 是个宽松的门槛 —— 参考音频的**理想**值是 25 dB 以上
（`config.REF_AUDIO_IDEAL.ideal_snr_db`）。训练样本比参考音频宽松，
因为量比质更难凑。
""",
        pitfall_md="""
背景音乐、多人说话、键盘声都会拉低这个值。这类样本即使过了 12 dB 门槛，
训出来的音色也会带上噪声的「味道」—— 体检分与这条硬门槛都不能替代试听。
""",
    ))

    # =====================================================================
    # LoRA
    # =====================================================================
    reg(P(
        key="lora_rank", group="lora", label="LoRA rank（容量）",
        kind="slider", default=8, minimum=1, maximum=256, step=1,
        summary="旁路矩阵的秩，决定「能学多少」。最影响过拟合的超参。",
        info="出厂预设：保守 4 / 均衡 8 / 激进 32。8 GB 卡建议 4~16。",
        affects="可训练参数量、过拟合风险、显存",
        detail_md="""
LoRA 给每个目标层加一条低秩旁路 `ΔW = B·A`，rank 就是 A/B 的中间维度。

| rank | 可训练参数（GPT attn 面） | 适用 |
|---|---|---|
| 4 | 约 0.18 M | 数据少（< 5 分钟），最抗遗忘 |
| 8 | 约 0.37 M | 默认，多数场景够用 |
| 32 | 约 1.5 M | 数据充足且已用均衡档试过 |

rank 太大而数据太少时，模型会**记住训练样本**而不是学到音色 ——
表现是「训练集里的句子完美，换个说法就崩」。
""",
        pitfall_md="""
`LoRAConfig.validate` 在 rank ≥ 32 时发 warn，rank ≥ 16 且 dropout = 0 时
再发一条 —— 小数据集上高 rank + 无正则是最容易翻车的组合。
""",
    ))

    reg(P(
        key="lora_alpha", group="lora", label="LoRA alpha（旁路增益）",
        kind="number", default=16, minimum=1,
        summary="实际缩放系数 = alpha/rank（rsLoRA 下是 alpha/√rank）。",
        info="出厂预设：保守 8（配 rank 4）/ 均衡 16（配 rank 8）/ 激进 64（配 rank 32）。",
        affects="LoRA 对输出的影响力",
        detail_md="""
旁路输出要乘一个缩放系数才加到原权重上：

```
标准 LoRA：   scaling = alpha / rank
rsLoRA：      scaling = alpha / √rank
```

预设都把 alpha 设成 `2 × rank`（scaling = 2），激进档是 `2 × rank` 配更大 rank。
增益越大，LoRA 对输出的「话语权」越强，底座被带偏得越快。
""",
        pitfall_md="""
`LoRAConfig.validate` 在 `alpha/rank > 4` 时发 warn。推理端还有强度旋钮
（`lora_adapter_scale`）可以在不重训的前提下调整实际影响 —— 训练时不必
为了「稳」而牺牲容量，推理时调回来即可。
""",
    ))

    reg(P(
        key="lora_target_preset", group="lora", label="注入面",
        kind="dropdown", default="attn", choices=["attn", "attn_mlp", "all_linear"],
        summary="把 LoRA 注入到哪些层。注入越多，通用能力损失越大。",
        info="attn = 注意力投影（GPT: c_attn/c_proj；CFM: wqkv/wo）；"
             "attn_mlp = 再加大 MLP；all_linear = 全部被扫描到的线性层。",
        affects="可训练参数量、音色/语气改变幅度、遗忘风险",
        detail_md="""
| 档位 | GPT 注入 | CFM 注入 | 出厂预设 |
|---|---|---|---|
| `attn` | `attn/c_attn`, `attn/c_proj` | `attention/wqkv`, `attention/wo` | 保守 / 均衡 |
| `attn_mlp` | 再加 `mlp/c_fc`, `mlp/c_proj` | 再加 MLP | 激进 |
| `all_linear` | 全部被扫描到的线性层 | 除去死模块的全部 | — |

注入面是**从真实模型扫描出来的**，不是硬编码：`guard.scan_targets` 先扫出
可注入层，`resolve_target_patterns` 再解析。CFM 的 `all_linear` 会把 `"*"`
展开一次，因为把 `"*"` 直接交给 PEFT 会在 `BASECFM.criterion`（一个 L1Loss）
上崩掉。
""",
        pitfall_md="""
只注入注意力投影通常就足够改变音色与语气。`all_linear` 会显著提高遗忘风险
（`LoRAConfig.validate` 对它专门发 warn）。

**GPT 与 CFM 的层名完全不同**：CFM 的注意力叫 `wqkv` / `wo`，不是 `qkv` /
`out_proj`。用 GPT 的注入面去训 CFM 会注到空气上 —— 一键三连走
`cfm_lora.default_config` 就是为了避开这个坑。
""",
    ))

    reg(P(
        key="lora_use_rslora", group="lora", label="rsLoRA（rank 稳定化）",
        kind="checkbox", default=False,
        summary="把缩放从 alpha/rank 换成 alpha/√rank。",
        info="保守档默认开启。rank 越大，二者差距越明显。",
        affects="高 rank 下的训练稳定性",
        detail_md="""
rank-stabilized LoRA 的动机：标准 LoRA 的 scaling 随 rank 增大而线性衰减，
于是「调大 rank」同时悄悄削弱了旁路的影响。开 rsLoRA 后 scaling 只按 √rank
衰减，高 rank 下的有效学习率不会被动变小。

保守档（rank 4）也开它，是为了让 `alpha/√rank` 得到更保守的增益。
""",
        pitfall_md="""
开关会改变 scaling，所以**换了它就不能直接比较 val loss**，也不能沿用
之前的强度旋钮值（推理端的 `set_adapter_scale` 按 nominal scaling 折算）。
""",
    ))

    reg(P(
        key="lora_keep_checkpoints", group="lora", label="保险库保留档位数",
        kind="slider", default=3, minimum=1, maximum=8, step=1,
        summary="每个 run 在保险库里保留几个最好的档位（按 val loss）。",
        info="只保留 top-K，被挤掉的当场删除；best 永不被剪掉。",
        affects="磁盘占用与「备选档位」的多少",
        detail_md="""
保险库（`guard.CheckpointVault`）每次「val loss 改善」都会存一份权重，
然后按 metric 排序只留 top-K。

- metric = **验证集 loss**，`mode="min"`（越小越好）—— 在 `runs.vault()` 里
  硬编码，因为训练全程盯的就是 val loss。
- **best 永不被剪掉**：即使它不在 top-K 里（正常不会发生）也会被保留。
- 写入是原子的（先写临时目录再 `os.replace`），中途断电不会留下半个权重。
- 保险库**永不写 `checkpoints/`**（底座目录），这是红线。

一键三连的 `top_k` 直接映射到这个值 —— 「三个候选」就是这么来的。
""",
        pitfall_md="""
「final」（训练结束那一刻的权重）是用**最后一次**评估值存的，不是最好的，
所以它可能一存进去就被剪掉。要「最好」请用 `best`。
""",
    ))

    reg(P(
        key="lora_adapter_scale", group="lora", label="推理期强度旋钮",
        kind="slider", default=1.0, minimum=0.0, maximum=1.5, step=0.05,
        summary="推理时把 LoRA 的影响按比例缩放，0 = 纯底座。",
        info="0.6~0.8 是「像」与「稳」的常见折中；不用重训。",
        affects="音色相似度 vs 发音稳定性",
        detail_md="""
`set_adapter_scale` 按 nominal scaling 折算后写入 `layer.scaling[adapter]`，
所以反复调用不会累积（不是每次乘一遍）。

这是九道防线里的第 8 道：**不用重训**就能在「更像本人」和「更稳」之间找平衡。
训练完发现有点飘 → 调到 0.7 试听；发现不够像 → 调到 1.2。

在「🎙 合成」页的「LoRA 音色模型」区，或「🏁 评测/部署」页都能调。
""",
        pitfall_md="""
调高到 1.5 以上（面板上限）会开始出现发音漂移、偶发读错字 —— 那是 LoRA
的话语权强过底座了。
""",
    ))

    # =====================================================================
    # 优化器与调度
    # =====================================================================
    reg(P(
        key="train_lr", group="optim", label="学习率",
        kind="number", default=1e-4,
        summary="LoRA 常用 1e-5 ~ 5e-4，比全参微调高得多。",
        info="出厂预设：保守 5e-5 / 均衡 1e-4 / 激进 2e-4。",
        affects="收敛速度与稳定性",
        detail_md="""
只训练 LoRA 旁路（可训练参数约万分之几），所以学习率可以比全参微调高 1~2 个
数量级。调度是带 warmup 的余弦退火，下限 = `lr × min_lr_ratio`。
""",
        pitfall_md="""
`LoRAConfig.validate` 要求 `0 < lr < 0.1`。损失发散（loss 变 NaN 或突然飙升）
时先降学习率，而不是加秩。
""",
    ))

    reg(P(
        key="train_epochs", group="optim", label="训练轮数",
        kind="number", default=4, precision=0, minimum=1,
        summary="过一遍数据集的次数。与 max_steps 至少要有一个为正。",
        info="出厂预设：保守 2 / 均衡 4 / 激进 8。小数据集上早停通常先触发。",
        affects="训练时长与过拟合程度",
        detail_md="""
总步数 = 每轮步数 × epochs，而每轮步数 = 样本数 ÷ (batch_size × grad_accum)。
轮数本身不是关键 —— 早停开着的时候它只是个上限。
""",
        pitfall_md="""
`LoRAConfig.validate` 在总步数 > 20000 时发 warn：8 GB 单卡上会跑很久，
且早停大概率先触发，不如直接调低轮数。
""",
    ))

    reg(P(
        key="train_eval_every", group="optim", label="评估间隔（多少个 step）",
        kind="number", default=100, precision=0,
        summary="每多少个优化步评估一次验证集。0 = 只靠轮末兜底评估。",
        info="小数据集上预设的 50/100/200 可能整个 run 都触发不到，"
             "一键三连会按总步数自动反推。",
        affects="保险库能攒出几个档位、早停的灵敏度",
        detail_md="""
评估是训练里最贵的操作之一（要跑完整个验证集）。所以预设给的是 50/100/200 ——
那是为大数据集定的。

**小数据集上的坑**：20 条样本、global_batch=4 → 每轮 5 个 step，
`eval_every=100` 永远触发不到，保险库最终只有「轮末兜底评估」存下的 1 个档位。
于是「留 K 个档位再择优」就无档可选。

一键三连按总步数反推：`eval_every = max(1, total_steps ÷ (K+1))`，
保证能攒出约 K 个可比较的档位。
""",
        pitfall_md="""
轮末会强制兜底评估一次。如果最后一次 step 恰好撞上 `eval_every`，
轮末这次会被**跳过**（避免同一 step 重复评估 —— 两次 val 完全相同会被
EarlyStopper 误判成「没改善」，白白吞掉一次耐心）。
""",
    ))

    reg(P(
        key="train_val_patience", group="optim", label="早停耐心",
        kind="number", default=3, precision=0, minimum=0,
        summary="val loss 连续几次没改善就停。0 = 关闭早停。",
        info="出厂预设：保守 2 / 均衡 3 / 激进 0（关闭）。",
        affects="训练时长与过拟合程度",
        detail_md="""
早停是九道防线里的第 5 道：盯住 val 曲线，连续 N 次评估不改善就停止 ——
**后面升上去的部分全是过拟合**。

验证集为空时早停会被强制关闭（日志里会写明），因为无从判断改善。
""",
        pitfall_md="""
关闭早停（0）会一路跑到 epochs 结束。激进档就是这么设计的，但它只在数据充足
（≥ 30 分钟）时才建议使用。
""",
    ))

    reg(P(
        key="train_base_dropout", group="optim", label="底座 dropout",
        kind="slider", default=0.0, minimum=0.0, maximum=0.5, step=0.05,
        summary="冻结底座的 dropout 概率，默认全关。",
        info="0 = 关闭（底座原本的 dropout 只会让 val 抖动）；"
             "CFM 传 -1 表示保留官方的 wavenet p=0.2。",
        affects="val 曲线的稳定性",
        detail_md="""
训练时底座是冻结的（`freeze_base` 会校验），但它内部仍有 Dropout 层。
这些随机性会给 val loss 叠加噪声 —— **噪声比训练带来的改善还大**，
于是早停会在错误的信号上决策。

所以默认全关，让 val 是确定性的。
""",
        pitfall_md="""
DPO 训练时会临时关掉全部 Dropout 包住整批前向：policy 与 ref 必须看到同一个
「dropout 世界」，否则 `(policy − ref)` 里会混进两次不同采样的随机差，
margin 直接抖成噪声。
""",
    ))

    reg(P(
        key="train_grad_checkpointing", group="optim", label="梯度检查点",
        kind="checkbox", default=True,
        summary="用算力换显存：不存中间激活，反向时重算。",
        info="慢约 30%，但显存占用明显下降。CFM 上是空操作（会被自动关掉）。",
        affects="显存占用与训练速度",
        detail_md="""
8 GB 卡训 813 M 底座的 LoRA 时，激活值比权重更吃显存，所以默认开启。

**关键陷阱**：reentrant 版梯度检查点 + 全冻结底座 = **训练静默失效**
（梯度恒为 0，不报错）。必须用 `use_reentrant=False` ——
这一点在 `training/forward.py` 里已经落地。
""",
        pitfall_md="""
CFM 的 `grad_checkpointing` 是**空操作**（preflight 会发 warn 说明），
`cfm_lora.default_config` 干脆把它设成 False，免得用户以为它在起作用。
""",
    ))

    reg(P(
        key="train_bf16", group="optim", label="BF16 训练",
        kind="checkbox", default=True,
        summary="半精度训练，省显存。CFM 上建议关闭。",
        info="CFM 的 L1 损失量级与 bf16 尾数不匹配，会被自动关掉。",
        affects="显存占用与数值稳定性",
        detail_md="""
GPT 用 bf16 没问题：底座前向在 bf16 下有专门的 dtype 处理
（`training/forward.py` 绕开了官方 `torch.zeros` 未指定 dtype 的崩溃）。

CFM 不一样：它的 L1 速度场损失在 bf16 的尾数精度下会失真，
`cfm_lora.default_config` 因此强制 `bf16=False`。
""",
        pitfall_md="""
LoRA 参数本身会被提为 fp32（`promote_adapter_fp32`），所以半精度影响的只是
前向，不改变适配器的精度。
""",
    ))

    reg(P(
        key="train_seed", group="optim", label="训练随机种子",
        kind="number", default=42, precision=0,
        summary="固定它，val 曲线与保险库结果才可复现。",
        info="一键三连把训练种子与划分种子统一用这一个值。",
        affects="可复现性",
        detail_md="""
种子影响：权重初始化、数据打乱顺序、回放样本抽样、以及 val 评估用的固定噪声。

**val 必须是确定性的**：固定噪声 + 固定配对，否则 val 曲线的抖动比训练带来的
改善还大，早停会在噪声上决策。
""",
    ))

    reg(P(
        key="train_replay_ratio", group="lora", label="回放比例（抗遗忘）",
        kind="slider", default=0.3, minimum=0.0, maximum=0.95, step=0.05,
        summary="每个 batch 里混入多少底座蒸馏样本，钉住通用能力。",
        info="出厂预设：保守 0.5 / 均衡 0.3 / 激进 0（无回放）。",
        affects="灾难性遗忘的程度 —— 这是防线 7",
        detail_md="""
只拿目标角色的数据训练，模型会逐步丢掉通用发音能力（灾难性遗忘）。
回放的做法是混入一批「底座自己说的样本」（`replay.py` 构建的蒸馏集），
让模型在每个 batch 里都复习一遍原来的能力。

数据越少，比例应该越高 —— 保守档给到 0.5。
""",
        pitfall_md="""
`LoRAConfig.validate` 在 `replay_ratio = 0` 时发 warn，并点名它是**灾难性遗忘
最主要的成因**。激进档就是这么设的，只在数据充足时使用。
""",
    ))

    # =====================================================================
    # DPO
    # =====================================================================
    reg(P(
        key="dpo_beta", group="dpo", label="DPO beta（贴近参考策略的程度）",
        kind="number", default=0.1,
        summary="控制策略偏离纯底座的惩罚强度。越大越保守。",
        info="DPO 参考策略 = 关掉 LoRA 的底座本身（adapters_off）。",
        affects="对齐强度 vs 稳定性",
        detail_md="""
DPO 的损失里有一个 KL 项把策略拉向参考策略，beta 就是它的系数。
本项目里参考策略实现为 `adapters_off(model)` —— 关掉 LoRA 就正好是纯底座。

beta 越大越不敢偏离底座（更稳、改变更小）；越小越激进。
""",
        pitfall_md="""
DPO 一次训练要吃 **4 次前向**（policy×2 带梯度 + ref×2 不带），显存约为 SFT 的
1.5 倍（preflight 会按这个系数估）。
""",
    ))

    reg(P(
        key="dpo_sft_weight", group="dpo", label="chosen NLL 锚权重",
        kind="number", default=0.1,
        summary="给「好的那个」加一个绝对似然锚，防止它塌掉。",
        info="DPO 只看相对差，chosen 的绝对似然可以塌到没人说得出话。",
        affects="DPO 后模型是否还能正常说话",
        detail_md="""
DPO 的损失只关心 `(chosen − rejected)` 的**相对**差。这意味着 chosen 的绝对
似然可以一路下滑 —— 只要它比 rejected 高一点就行，最后模型可能说得含混不清。

所以额外加一项 chosen 的负对数似然（按长度归一），把「说得好」拉回
「至少说得出」。权重就是它。

注意 `length_normalize=True` 时 logprob 已经是**每 token 平均**了，
锚只能再对 batch 取均值 —— 再除一次整批 token 数会把锚压小约三个数量级，
让这个权重形同虚设。
""",
    ))

    reg(P(
        key="dpo_min_margin", group="dpo", label="偏好对最小 margin",
        kind="slider", default=0.05, minimum=0.0, maximum=0.3, step=0.01,
        summary="候选之间的 reward 差小于它就丢弃这一对。",
        info="转写噪声就能造成的差距不值得学。默认 0.05。",
        affects="偏好对数量与信噪比",
        detail_md="""
「对齐」页会用当前模型对同一句话合成多个候选、逐个打分，取最好与最差成对。
但如果两者分数只差 0.001，那大概率是 whisper 转写的随机波动，不是真的优劣。

margin 门槛就是过滤这种噪声对 —— **学噪声不如不学**。
""",
        pitfall_md="""
构造出来的对经常几乎全被 margin 刷掉。这通常说明候选之间没真差别：
把合成温度或温度扰动调高，或换更难的文本。
""",
    ))

    reg(P(
        key="dpo_length_normalize", group="dpo", label="按长度归一 logprob",
        kind="checkbox", default=True,
        summary="把序列 logprob 除以有效 token 数，避免长句天然吃亏。",
        info="与 SFT 同量纲；chosen NLL 锚的正负号与量级都依赖它。",
        affects="DPO 损失的尺度",
        detail_md="""
不归一的话，长句的 logprob 绝对值天然更大，DPO 会比较「谁更长」而不是
「谁更好」。默认开启，与 SFT 的交叉熵同量纲。

这个开关会改变 chosen 锚的计算方式（见 `dpo_sft_weight` 的说明）——
改它必须连带确认锚的实现，否则权重会静默失效。
""",
    ))

    # =====================================================================
    # 评测指标
    # =====================================================================
    reg(P(
        key="reward_wer_weight", group="reward", label="WER 权重",
        kind="slider", default=0.6, minimum=0.0, maximum=1.0, step=0.05,
        summary="reward 里「读得对」占的比重。",
        info="reward = w1×(1−WER) + w2×声纹相似度，默认 0.6 / 0.4。",
        affects="择优时偏向「准确」还是偏向「像」",
        detail_md="""
```
reward = (w1 × (1 − WER) + w2 × 相似度) / (w1 + w2)
```

- WER：whisper 转写合成音频，与原文做字符级编辑距离（中文按字）。
- 声纹相似度：CAMPPlus 分别嵌入合成音频与参考音频，取余弦相似度。

两个分量都有意义：WER 抓「读错字、漏字」，相似度抓「像不像本人」。
""",
        tuning_md="""
- 更在意**别读错**（有声书、播报）→ 提高 WER 权重到 0.7~0.8。
- 更在意**音色像**（角色配音）→ 提高相似度权重。
- 注意 WER 有转写噪声下限：合成完全正确时 WER 也可能不是 0。
""",
    ))

    reg(P(
        key="reward_beam_size", group="reward", label="whisper beam 宽度",
        kind="number", default=5, precision=0, minimum=1,
        summary="转写的束搜索宽度。温度固定为 0，所以结果是确定的。",
        info="beam 越大越准、越慢。打分与识别共用这个设置。",
        affects="WER 的稳定性与打分耗时",
        detail_md="""
用 `temperature=0` + beam search，保证同一段音频每次转写出同样的文本 ——
打分结果必须可复现，否则「择优」会变成掷骰子。

识别阶段与打分阶段用的是同一套参数，所以「输入文本」与「评价标准」是一致的。
""",
    ))

    reg(P(
        key="reward_sim_weight", group="reward", label="声纹相似度权重",
        kind="slider", default=0.4, minimum=0.0, maximum=1.0, step=0.05,
        summary="reward 里「像不像本人」占的比重。",
        info="CAMPPlus 嵌入余弦相似度；同一条音频自比 = 1.0。",
        affects="择优时对音色的敏感度",
        detail_md="""
CAMPPlus（`campplus_cn_common.bin`，嵌入维度 192）是独立于推理引擎构造的，
所以打分不需要加载大模型，只占约 27 MB 显存。

它是说话人验证模型，对**音色**敏感、对内容不敏感 —— 正好与 WER 互补。
""",
        pitfall_md="""
如果参考音频本身含噪或含背景音乐，相似度会被系统性拉低（合成音频是干净的，
两边分布不一致）。这也是「参考音频工作台」值得先做一遍的原因。
""",
    ))
