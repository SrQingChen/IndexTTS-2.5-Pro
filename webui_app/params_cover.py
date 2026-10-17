"""翻唱 / 变声参数登记（params 注册表分文件惯例：params_<域>.py）。"""

from __future__ import annotations


def build(reg, P):
    _input(reg, P)
    _vc(reg, P)
    _f0(reg, P)
    _key(reg, P)
    _singing(reg, P)
    _post(reg, P)
    _mix(reg, P)
    _realtime(reg, P)


def _singing(reg, P):
    reg(P(
        key="cover_singing_enable", group="cover_sing", label="歌声模型（E1-T 实验性）",
        kind="checkbox", default=False,
        summary="换装歌声 CFM（s2mel_singing.pth）并把音高/技巧条件喂进模型。",
        info="需要先完成 capability 训练与合并；关闭时走原链路（默认）",
        affects="高音区音色（HNR）、转换耗时",
        detail_md="""
**E1-T（docs/dev-plan-e1t-2026-10.md）**：把「音高/技巧 → 音色」的映射
学进 CFM（TimbreConditioner 逐帧残差），高音不再靠 WORLD 后处理拉上去。

前置（一次）：
1. `tools/gtsinger_import.py` 导入公开歌声数据集；
2. singing 模式提取特征 → CfmTrainer(singing=True) 训练 capability；
3. `merge`（arch=cfm_singing）产出 `checkpoints/s2mel_singing.pth`；
4. 本开关即可启用（会话内换装、结束自动恢复，TTS 不受影响）。

关闭本开关时翻唱链路与旧版逐字节一致（回归测试固化）。
""",
        pitfall_md="未训练 capability 就开启会在加载处直接报错（这是故意的——拿语音底座跑条件等于白跑）。",
    ))
    reg(P(
        key="cover_technique_mode", group="cover_sing", label="技巧条件来源",
        kind="dropdown",
        choices=["preserve 保持（control 直通）", "auto 分类器自动", "manual 手动指定"],
        default="preserve 保持（control 直通）",
        summary="technique 条件的来源；auto 需要 checkpoints/technique_cls.pt。",
        info="preserve=全 control（保守）；auto=分类器逐帧；manual=整曲一个技巧",
    ))
    reg(P(
        key="cover_technique_manual", group="cover_sing", label="手动技巧 id",
        kind="slider", minimum=0, maximum=6, step=1, default=0,
        summary="0=control 1=mixed 2=falsetto 3=breathy 4=pharyngeal 5=vibrato 6=glissando。",
        info="只在技巧来源=manual 时生效",
    ))
    reg(P(
        key="cover_bon_high", group="cover_sing", label="高音块 BoN 择优",
        kind="checkbox", default=False,
        summary="对高音块（浊音 p75≥350Hz）生成 3 候选按 HNR/音准/SS 择优。",
        info="只作用于高音块，中低音不受影响；转换耗时约 ×3（仅高音块）",
    ))
    reg(P(
        key="cover_hf_transplant", group="cover_sing", label="源高频移植（SYKI）",
        kind="checkbox", default=False,
        summary="转换结果低通 + 源干声 >9kHz 直接加回（零训练应急）。",
        info="22k 引擎下收益带宽有限；依赖源干声质量",
    ))
    reg(P(
        key="cover_register_mode", group="cover_sing", label="音区映射",
        kind="dropdown",
        choices=["auto 分数音区映射（推荐）", "off 保原调硬唱"],
        default="auto 分数音区映射（推荐）",
        summary="auto=把源 F0 连续拉进参考音区（保旋律，Everyone-Can-Sing 单八度原则）。",
        info="off=不移调，条件按源音高原样喂（唱得高但音色外推压力最大）",
    ))
    reg(P(
        key="cover_octave_fix", group="cover_sing", label="八度跳变修正（SHC）",
        kind="checkbox", default=True,
        summary="修正源 F0 的八度估计错误（短促回返跳变）。",
        info="颤音安全；但 <0.4s 的真实快速换声区（yodel/戏腔闪跳）会被抹平——做这类内容时关闭",
    ))


def _input(reg, P):
    reg(P(
        key="cover_input_state", group="cover_in", label="输入音频状态",
        kind="dropdown",
        choices=["原歌曲（含伴奏）", "分离人声（可能有混响/残留）", "干净人声（干声）"],
        default="干净人声（干声）",
        summary="决定预处理管线路由：原歌曲跑分离+去混响，分离人声跑去混响，干净人声直通。",
        info="原歌曲=人声分离→去混响；分离人声=仅去混响；干净人声=直通转换",
        affects="链路时长、是否产出伴奏轨（自动合成用）",
        detail_md="""
**路由表**：原歌曲 → BS-Roformer 分离（人声+伴奏，不改时长保对拍）→
UVR-DeEcho-DeReverb 去混响 → 预处理；分离人声 → 仅去混响 → 预处理；
干净人声 → 预处理直通。

**坑**：选「原歌曲」需要安装 audio-separator（pip install
"audio-separator[gpu]"）并首跑下载权重（约 1GB）；没有该依赖时请选
「干净人声」并用外部工具预分离。
""",
        pitfall_md="选错状态会浪费一条分离链或残留伴奏污染内容特征（模型会把伴奏当内容学进转换）。",
    ))
    reg(P(
        key="cover_dereverb", group="cover_in", label="去混响/去回声",
        kind="checkbox", default=True,
        summary="对分离/输入人声做干声化（UVR-DeEcho-DeReverb）。",
        info="混响残留会被模型当音色学走；宁可欠一点，残余由混音空间感覆盖",
    ))
    reg(P(
        key="cover_denoise", group="cover_in", label="预处理降噪强度",
        kind="slider", minimum=0.0, maximum=1.0, step=0.05, default=0.0,
        summary="noisereduce 轻降噪（0=关）。翻唱建议 0~0.3。",
        info="0=关闭；过高会伤气声与高频细节",
        affects="干声干净度 vs 高频细节",
    ))


def _vc(reg, P):
    reg(P(
        key="cover_variant", group="cover_vc", label="内容通路变体",
        kind="dropdown", choices=["A 量化重构（默认）", "B 连续特征直通（实验，歌声易嘶哑）"],
        default="A 量化重构（默认）",
        summary="A=语义码本 quantize→decode（与 TTS 生成路径同分布，稳）；"
                "B=连续特征直通（与参考条件槽同构，细节更多但分布外）。",
        info="A 稳（默认）；B 保留更多源细节，属实验特性",
        experimental=False,
        detail_md="""
变体 A 把源特征过 25Hz ResidualVQ 码本再重构——与 GPT 生成 token 的
目标域一致，是最稳的口径。变体 B 直接用 w2v-BERT 连续特征——与
prompt_condition 槽（参考音频同样直通）完全同构，理论上细节更丰富，
但对 CFM 属分布外输入，可能出伪影。听感无明确优势时保持 A。
""",
    ))
    reg(P(
        key="cover_steps", group="cover_vc", label="CFM 步数",
        kind="slider", minimum=5, maximum=50, step=1, default=25,
        summary="Flow Matching 求解步数（官方 25；8-10 步听感接近，离线不必省）。",
        info="离线建议 25；追求上限可 30-50（Seed-VC 官方口径）",
        affects="质量/耗时",
    ))
    reg(P(
        key="cover_cfg", group="cover_vc", label="CFG 引导强度",
        kind="slider", minimum=0.0, maximum=1.0, step=0.05, default=0.7,
        summary="classifier-free guidance（官方 0.7；0 可省一半计算，差异细微）。",
        info="0.7 默认；音色不够像可升到 0.9",
    ))
    reg(P(
        key="cover_ref_seconds", group="cover_vc", label="参考音频时长",
        kind="slider", minimum=5.0, maximum=15.0, step=1.0, default=15.0,
        summary="音色参考截取长度（引擎惯例 15s；实时档建议 5-8s 换算力）。",
        info="参考越长音色锚定越稳、上下文占用越大",
    ))
    reg(P(
        key="cover_chunk_sec", group="cover_vc", label="分块长度",
        kind="slider", minimum=10.0, maximum=60.0, step=5.0, default=30.0,
        summary="长曲分块转换（w2v-BERT/CFM 的 O(T²) 注意力约束）。"
                "8GB 卡自动压到 ≤30s；OOM 时还会自动减半重试一次。",
        info="30s 默认（8GB 安全）；块间优先在歌唱气口切、40ms 输入重叠",
    ))


def _f0(reg, P):
    reg(P(
        key="cover_f0_mode", group="cover_f0", label="F0 旋律移植",
        kind="dropdown", choices=["auto 自动（推荐）", "force 全曲移植", "off 关闭"],
        default="auto 自动（推荐）",
        summary="语义码本量化后音高信息大量丢失——旋律由 WORLD F0 移植外置保证。",
        info="auto=轮廓相关 r≥0.75 局部修复，否则全曲移植；说话变声可 off",
        affects="旋律准度（翻唱核心）",
        detail_md="""
**为什么需要**：内容 token（25Hz ResidualVQ）在量化时丢弃大部分音高
信息（arXiv 2604.07467），直接转换的旋律不可靠。

**三模式**：auto（默认）先测源/输出 F0 轮廓相关系数 r——r≥0.75 只修
偏差 >40 音分且 >80ms 的浊音段（局部 WORLD 重合成，谱自检不过自动回退）；
r<0.75 全曲移植（旋律完全由源决定）。force 无条件全曲移植（源演唱表现
优先）。off 不移植（说话变声韵律残留够用时）。

**八度映射**：源/参考音区中位数差自动取整到 ±12 半音（男↔女自动翻八度），
可在页面手动覆盖。
""",
        pitfall_md="全曲移植用的是 WORLD 参数声码器（包络保留、激励替换），高音区（>500Hz）"
                   "包络估计退化属已知限制——报告里会给出移植是否通过谱自检。",
    ))
    reg(P(
        key="cover_transpose_auto", group="cover_f0", label="八度映射自动",
        kind="checkbox", default=True,
        summary="按源/参考音区中位数差自动决定移调（同性别 0 / 跨性别 ±12）。",
        info="关闭后用手动半音",
    ))
    reg(P(
        key="cover_transpose_semis", group="cover_f0", label="手动移调（半音）",
        kind="slider", minimum=-24, maximum=24, step=1, default=0,
        summary="八度映射自动关闭时的手动微调（±24 内）。",
        info="男→女 +12、女→男 −12；细调 ±2",
    ))


def _key(reg, P):
    """音域适配（B 方案）：默认关闭，不是纯增益。"""
    reg(P(
        key="cover_key_mode", group="cover_key", label="音域适配",
        kind="dropdown",
        choices=["off 关闭（默认）", "auto 自动降调", "manual 手动降调"],
        default="off 关闭（默认）",
        summary="把源人声整体降调，让模型在舒适音域内转换；只降低高音外推压力，"
                "不改善音色，且可能改变调性或引入二次移调伪影。",
        info="默认关；auto 按源 F0 p99 超过上限自动降；整首降调输出音质最好",
        affects="高音破音压力、调性、可能的移调伪影",
        detail_md="""
**为什么需要**：IndexTTS 的 CFM/BigVGAN 是语音训练，源歌高音超出舒适区时
嘶哑/破音会加重。把源人声整体降 K 半音再转换，可以让模型在更稳的音域工作。

**两种输出**：
- `整首降调（推荐音质）`：人声与伴奏一起降 K，不升回；避免二次移调，
  但成品不是原调（属于翻唱改编）。
- `仅人声升回原调`：伴奏保持原调，人声升回 K；会引入相位声码器伪影，
  建议 K≤4，超过 4 半音会明显失真。

**实测提醒**：预降 5 半音再升回，campplus SS 0.639→0.664→0.637，净收益≈0；
它只解决音域，不解决音色。**默认关闭，按歌试验。**
""",
        pitfall_md="不是纯增益：整首降调改变调性；仅人声升回有二次移调伪影。",
    ))
    reg(P(
        key="cover_key_semitones", group="cover_key", label="手动降调（半音）",
        kind="slider", minimum=-12, maximum=0, step=1, default=0,
        summary="manual 模式生效；负值=降调。",
        info="建议 -2~-7；先听高音段再定",
    ))
    reg(P(
        key="cover_key_output", group="cover_key", label="输出调性",
        kind="dropdown",
        choices=["整首降调（推荐音质）", "仅人声升回原调（有移调伪影）"],
        default="整首降调（推荐音质）",
        summary="整首降调避免二次移调；仅人声升回保留原调但有伪影。",
    ))
    reg(P(
        key="cover_key_ceiling", group="cover_key", label="模型舒适音高上限 (Hz)",
        kind="slider", minimum=450, maximum=800, step=10, default=600,
        summary="auto 模式：源 F0 p99 超过此值时自动降调。",
        info="8GB/22k 保守值 600；戏腔可降到 550 更稳",
    ))


def _post(reg, P):
    reg(P(
        key="cover_hf_deess", group="cover_post", label="间隙痰爆冲抑制",
        kind="checkbox", default=True,
        summary="audio_lab 间隙 de-esser（翻唱标定 max_cut 10dB，比语音档温和）。",
        info="转换输出的齿音/爆冲伪影治理，默认开",
    ))
    reg(P(
        key="cover_hf_limit", group="cover_post", label="浊音 HF 撕裂限幅",
        kind="checkbox", default=True,
        summary="浊音段 5.5-11kHz 超语音带限幅（阈值与合成链一致）。",
        info="防撕裂感，默认开",
    ))
    reg(P(
        key="cover_spectral_match", group="cover_post", label="频谱画像对齐",
        kind="checkbox", default=False, experimental=True,
        summary="向角色参考的 6 频段画像对齐（±4dB，本project独有）。",
        info="换声后频谱不贴角色时开（实验）",
    ))
    reg(P(
        key="cover_breath_soften", group="cover_post", label="气声软化",
        kind="checkbox", default=False,
        summary="8-11kHz 嘶声压低。歌声气口是表现力——默认关。",
        info="转换气声刺耳才开",
    ))
    reg(P(
        key="cover_highpass", group="cover_post", label="高通滤波 (Hz)",
        kind="slider", minimum=0, maximum=150, step=10, default=70,
        summary="去除低频隆隆（0=关）。",
    ))
    reg(P(
        key="cover_sr_enable", group="cover_post", label="超分（AudioSR → 48k）",
        kind="checkbox", default=False,
        summary="人声高频重建。默认关：约 17.8× 实时耗时（4 分钟人声 ≈ 70 分钟）"
                "且翻唱人声不总适合超分。",
        info="默认关；耗时极高，仅成品需要 48k 时开",
        pitfall_md="超分需临时卸载推理引擎（显存互斥），完成后自动恢复。",
    ))
    reg(P(
        key="cover_sr_steps", group="cover_post", label="超分步数",
        kind="slider", minimum=10, maximum=100, step=5, default=35,
        summary="AudioSR ddim_steps（官方 35）。",
    ))


def _mix(reg, P):
    reg(P(
        key="cover_auto_mix", group="cover_mix", label="自动合成（BGM 回混）",
        kind="checkbox", default=False,
        summary="把预处理分离出的「废弃物」伴奏轨与转换人声混回成品"
                "（人声与伴奏同源同时轴，天然对拍；分离不改时长）。",
        info="原歌曲=用分离出的伴奏；分离人声/干净人声=可上传伴奏",
    ))
    reg(P(
        key="cover_vocal_gain", group="cover_mix", label="人声增益 (dB)",
        kind="slider", minimum=-6.0, maximum=9.0, step=0.5, default=3.0,
        summary="混音时人声相对伴奏的增益（起步人声高约 3dB）。",
    ))
    reg(P(
        key="cover_loudnorm", group="cover_mix", label="母线响度归一",
        kind="checkbox", default=False,
        summary="成品 RMS 归一到 −14dB（流媒体口径的近似；不做真 LUFS）。",
        info="发行向开；试听对比关",
    ))


def _realtime(reg, P):
    reg(P(
        key="vc_preset", group="vc_realtime", label="档位",
        kind="dropdown",
        choices=["快速（低延迟）", "均衡（推荐）", "质量（更像）"],
        default="均衡（推荐）",
        summary="块大小×CFM步数×上下文×F0 的组合档：快速≈0.24s块/6步，"
                "均衡≈0.36s块/10步，质量≈0.73s块/16步+F0跟随。",
        info="延迟换音色相似度；真实延迟以页面回环实测为准",
        affects="端到端延迟、音色相似度、算力",
        detail_md="""
| 档 | 块 | 步数 | 左上下文 | F0 |
|---|---|---|---|---|
| 快速 | 21 帧(0.24s) | 6 | 1.0s | 关 |
| 均衡 | 31 帧(0.36s) | 10 | 2.0s | 平滑 |
| 质量 | 63 帧(0.73s) | 16 | 2.5s | 跟随移植 |

每块都对参考全长做每步注意力——参考时长被档位强制限制（≤5/6/8s）。
0.58B 参数的 w2v-BERT 每块带上下文重编码是固定开销，延迟低于
300ms 不现实（研究级 20ms 方案需要专用流式模型）。
""",
        pitfall_md="延迟显示的是算法值；真实端到端用页面「回环实测」（GUI 数字偏小是行业通病，VCClient issue #497）。",
    ))
    reg(P(
        key="vc_agc", group="vc_realtime", label="输入自动增益",
        kind="checkbox", default=True,
        summary="轻量 AGC：过低的输入 RMS 缓慢抬升到目标电平（防模型吃不到信号）。",
        info="麦克风电平不稳时开",
    ))
    reg(P(
        key="vc_silence_bypass", group="vc_realtime", label="静音直通",
        kind="checkbox", default=True,
        summary="VAD 静音期直接零输出（Seed-VC 同款做法），减少拼接毛刺与算力浪费。",
        info="不说话时不过模型，默认开",
    ))
    reg(P(
        key="vc_record", group="vc_realtime", label="录制输出",
        kind="checkbox", default=False,
        summary="把变声输出落盘到 outputs/vc/（会与源产生每块 40ms 淡化区，供回听）。",
    ))
