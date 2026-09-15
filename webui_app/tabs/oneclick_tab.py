"""Tab：一键三连（长音频 / 批量短音频 → 训练 → 择优）。

用户只需要把音频丢进来，点一个按钮。剩下七道工序在 training/oneclick.py
里全自动跑：采集切片 → 音频优化 → 识别对齐 → 筛选划分 → 自动调参 →
LoRA 训练 → 真机择优。

这一页只做三件事：
    · 收输入（上传文件，或填服务器上一个长音频 / 目录的路径）
    · 给一个大按钮 + 一个停止按钮
    · **在按钮下方把「本次会自动操作的全部参数」摆出来** —— 计划阶段显示
      将要用的取值，运行中显示实时阶段与日志，结束后显示完整的裁决报告
      （候选配置逐条通过/被拒的原因、训练指标、评分排名、推荐模型）。
"""

from __future__ import annotations

from typing import Any, Dict, List

import gradio as gr

from webui_app import logging_setup as LOG
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.services import audio_lab as AL
from webui_app.training import dataset as DS
from webui_app.training import oneclick as OC
from webui_app.training import parallel as PL
from webui_app.training import reward as RW
from webui_app.training.runner import get_runner

# 高级选项控件的顺序，必须与 OPT_KEYS 一一对应（_opts_from 靠位置装配）。
# model_name 虽然在主区域（不藏在高级选项里），但同样算一个取值来源。
OPT_KEYS = [
    "model_name", "lang", "slice_target_sec", "slice_min_sec",
    "slice_over_sec", "slice_max_pieces", "enhance", "denoise",
    "denoise_strength", "normalize", "trim_silence", "asr", "whisper_size",
    "score_whisper_size",
    "min_score", "max_text_repeats", "val_ratio", "arch_list", "preset_mode",
    "top_k", "rank_eval", "eval_samples", "cpu_workers", "seed",
]

LANG_CHOICES = [("中文 (ZH)", "ZH"), ("英语 (EN)", "EN"), ("日语 (JA)", "JA"),
                ("阿拉伯语 (AR)", "AR"), ("西班牙语 (ES)", "ES")]

PRESET_CHOICES = [
    ("自动（按数据量挑，推荐）", "auto"),
    ("🟢 保守（数据 < 5 分钟）", "conservative"),
    ("⚖️ 均衡（默认档）", "balanced"),
    ("🔴 激进（数据 ≥ 30 分钟）", "aggressive"),
]


def _opts_from(vals: Dict[str, Any]) -> OC.OneClickOptions:
    kw = dict(vals)
    archs = kw.pop("arch_list", None) or ["gpt"]
    kw["arches"] = ",".join(archs)
    return OC.OneClickOptions.from_dict(kw)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "开" if v else "关"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def plan_markdown(opt: OC.OneClickOptions) -> str:
    """把「本次会自动操作的全部参数」按阶段列清楚。

    这是按钮下方那块面板的内容 —— 用户在开跑前就能看清流水线会拿哪些
    参数、按什么规则去动它，而不是点完按钮等一个黑箱。
    """
    notices = opt.validate()
    errs = [n.message for n in notices if n.level == "error"]
    warns = [n.message for n in notices if n.level == "warn"]

    L: List[str] = []
    L.append("### 本次自动流程与参数")
    L.append("")
    if errs:
        L.append(T.err("<br>".join(errs)))
        L.append("")

    # ---- 身份与并行：最常被问到的两件事，放在最前面 ----
    names = opt.plan_run_names()
    if names:
        pairs = " · ".join(f"{a} → <code>{n}</code>" for a, n in names.items())
        L.append(T.hint(f"训练记录名（模型名称）：{pairs}"
                        + ("" if (opt.model_name or "").strip()
                           else "　留空则自动生成，填了便于区分管理")))
    w = PL.default_workers(99, opt.cpu_workers)
    L.append(T.hint(
        f"CPU 并行：<b>{w} 线程</b>"
        + ("（自动）" if not int(opt.cpu_workers or 0) else "")
        + "，只作用于音频体检与增强；GPU 阶段串行，结果与串行逐字节一致。"
        + ("" if AL.noisereduce_available()
           else "　⚠️ <b>未安装 noisereduce，降噪会被跳过</b>")))
    L.append("")

    rows: List[str] = []
    rows.append(
        "| 阶段 | 自动做什么 | 用到的参数 |")
    rows.append("|---|---|---|")
    rows.append(
        "| S1 采集与切片 | 导入音频（复制副本进数据集）；时长 > "
        f"{_fmt(opt.slice_over_sec)}s 的按声学打分切片 | "
        f"切片目标 {_fmt(opt.slice_target_sec)}s · 最短 "
        f"{_fmt(opt.slice_min_sec)}s · 最多 {opt.slice_max_pieces} 片 |")
    rows.append(
        "| S2 音频优化 | 去直流 / 掐静音 / 降噪 / 归一 / 重采样 / 截断 | "
        f"总开关 {_fmt(opt.enhance)} · 降噪 {_fmt(opt.denoise)}"
        f"（强度 {_fmt(opt.denoise_strength)}）· 归一 {_fmt(opt.normalize)}"
        f" · 掐静音 {_fmt(opt.trim_silence)} |")
    rows.append(
        "| S3 识别与对齐 | 逐条 whisper 转写；**长音频先切片再逐片转写**，"
        "于是文本与音频按「一片一段」配对 | "
        f"总开关 {_fmt(opt.asr)} · 识别用 whisper-{opt.whisper_size}"
        f"（产出训练文本）· 语言 {opt.lang} |")
    rows.append(
        "| S4 筛选与划分 | 体检复算 → 丢掉不合格 → 同文本去重 → "
        f"划 train/val | 体检分下限 {_fmt(opt.min_score)} · 同文本最多 "
        f"{opt.max_text_repeats} 条 · val {_fmt(opt.val_ratio)} · "
        f"样本下限 {OC.MIN_SAMPLES} 条 |")
    rows.append(
        "| S5 自动调参 | 按数据量排候选预设，**逐个真跑 preflight**，"
        "取第一个显存与数据量都通过的 | "
        f"目标 {', '.join(opt.arch_list())} · "
        f"模式 {opt.preset_mode} · 每个目标留 {opt.top_k} 个档位 |")
    rows.append(
        "| S6 LoRA 训练 | 卸载引擎 → 逐目标真训练；底座快照 / 漂移体检 / "
        "早停 / 保险库九道防线全程生效 | "
        f"rank·alpha·lr·epochs 来自上一步选定的预设；种子 {opt.seed} |")
    rows.append(
        "| S7 择优与交付 | 逐个档位挂载 → 真机合成 → reward 打分 → 排序 → "
        "激活最优 | "
        f"评分开关 {_fmt(opt.rank_eval)} · 每候选评 "
        f"{opt.eval_samples or '全部'} 条 · 指标 whisper-"
        f"{opt.score_whisper_size}（只做相对比较，故意比识别小一档）"
        " + 声纹相似 |")
    L.extend(rows)

    L.append("")
    L.append("**参数是怎么选出来的**")
    L.append("")
    L.append(
        "- 预设不是查表拍定的，而是**按数据量排序后逐个真跑预检**："
        f"当前规则为 <5 分钟 → {' → '.join(OC.preset_ladder(3.0))}；"
        f"5~30 分钟 → {' → '.join(OC.preset_ladder(10.0))}；"
        f"≥30 分钟 → {' → '.join(OC.preset_ladder(40.0))}。"
        "第一个通过（配置自检 + 样本量 + 空闲显存都够）的被采用，"
        "被拒的候选连原因一起列在结果里。")
    L.append(
        f"- 「三个或一个」：每个目标在保险库里留 **top-{opt.top_k}** 个档位"
        "（按 val loss 排），这最多 3 个档位全部进入 S7 真机打分，"
        "最后**激活评分最高的那一个** —— 所以交付的是 1 个最优模型，"
        "备选的另外几个仍留在保险库里可随时切换。")
    if len(opt.arch_list()) > 1:
        L.append(
            f"- 训练目标有 {len(opt.arch_list())} 个（{', '.join(opt.arch_list())}）："
            "GPT 管「怎么说」的语气韵律，CFM 管「像谁」的音色音质，"
            "两个都练才既像又自然；每个目标各留 top-K 档位一起参评。")
    if not opt.asr:
        L.append("- ⚠️ 语音识别已关闭：数据集里的文本会是空的，"
                  "S4 会把所有样本筛掉。请到「数据集」页补文本后再跑。")

    if warns:
        L.append("")
        L.append(T.warn("<br>".join(warns)))

    L.append("")
    L.append(f"<sub>预计耗时：优化与识别按样本数线性增长，训练约占一半以上；"
             f"8 GB 卡上 30 条样本跑到 GPT+CFM 双目标的量级约十分钟起。"
             f"全流程可随时「⏹ 停止」，已完成的阶段不会白跑"
             f"（样本、特征、adapter 都留在磁盘上）。</sub>")
    return "\n".join(L)


def render(ctx: AppContext):
    eng = ctx.engine
    sb = ctx.component("statusbar")
    runner = get_runner()

    gr.HTML(T.section(
        "一键三连", "🚀",
        "丢一段<b>长音频</b>或<b>一堆短音频</b>进来，点一个按钮：自动切片、"
        "降噪归一、语音识别与对齐、按数据量挑参数、训练 LoRA、"
        "真机打分择优 —— 最后交付评分最高的模型，并直接挂到推理页可用。"))

    with gr.Row(equal_height=False):
        # ============================ 左：输入 + 按钮 + 参数面板 ============================
        with gr.Column(scale=3, min_width=520):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("喂什么给它", "🎧",
                                  "两种都行，也可以同时给：上传若干文件，"
                                  "或填服务器上的路径。长音频建议走"
                                  "<b>服务器路径</b>（一个两小时的 wav 上传很痛苦）。"))
                files_in = gr.File(
                    label="上传音频（可多选；短音频直接放，长音频也行）",
                    file_count="multiple", file_types=["audio"],
                    type="filepath")
                extra_path_tb = gr.Textbox(
                    label="或：服务器上的一个音频文件 / 一个目录",
                    value="",
                    placeholder=r"例：E:\recordings\ep01.wav   或   E:\recordings\ep01",
                    info="填目录会递归找出目录下所有支持的音频；"
                         "长音频（超过切片阈值）会被自动切段")
                lang_dd = gr.Dropdown(choices=LANG_CHOICES, value="ZH",
                                      label="语种",
                                      info="训练语种，也决定语音识别的转写语言")
                model_name_tb = gr.Textbox(
                    label="模型名称（训练记录名，便于区分管理）",
                    value="", placeholder="例：小明_播客_2026",
                    info="留空则自动生成 oneclick_<时间戳>。"
                         "两个目标时自动加后缀（_gpt / _cfm）以免互相覆盖；"
                         "重名会在开工前拦下，不会让你白等一轮训练")

            with gr.Row():
                run_btn = gr.Button("🚀 一键三连（切片→识别→调参→训练→择优）",
                                    variant="primary", scale=4)
                stop_btn = gr.Button("⏹ 停止", size="sm", scale=1)

            # 改了右侧高级选项后点它刷新下方面板（同一份参数，不会两处不一致）
            with gr.Row():
                plan_btn = gr.Button("📋 刷新参数预览", size="sm", scale=0)
                gr.HTML(T.hint("改了右边「高级选项」之后点一下，"
                               "下方面板会按新参数重算自动流程与取值。"))

            # ------- 按钮下方：自动流程与参数面板（计划 / 进行中 / 结果）-------
            with gr.Column(elem_classes=["ix-section"]):
                progress_html = gr.HTML("")
                panel_md = gr.Markdown("")
                log_ta = gr.Textbox(label="实时日志", lines=10, max_lines=18,
                                    interactive=False)

        # ============================ 右：高级选项 ============================
        with gr.Column(scale=2, min_width=380):
            with gr.Accordion("⚙️ 高级选项（默认值已适配多数场景）", open=False):
                gr.Markdown(
                    "改完点下方「📋 刷新参数预览」即可在左侧面板看到"
                    "新的自动流程与取值。")

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("切片（只对长音频生效）", "✂️", ""))
                    slice_over_sl = gr.Slider(
                        5.0, float(DS.MAX_TRAIN_SEC),
                        value=float(DS.MAX_TRAIN_SEC), step=1.0,
                        label="超过多少秒就切片",
                        info=f"默认 {DS.MAX_TRAIN_SEC:g}s = 训练可接受的最长样本，"
                             "更长的样本体检直接判 too_long、永远不参与训练")
                    slice_target_sl = gr.Slider(
                        2.0, 20.0, value=12.0, step=0.5, label="每片目标时长（秒）",
                        info="8~15 秒是零样本 TTS 的甜点区：够长能学到韵律，"
                             "又不会让显存吃紧")
                    slice_min_sl = gr.Slider(
                        1.0, 10.0, value=4.0, step=0.5, label="最短片段（秒）",
                        info="短于它的片段直接丢掉（不足以体现音色）")
                    slice_pieces_nb = gr.Number(
                        120, label="单条长音频最多切几片", precision=0,
                        info="默认 120：半小时素材按 10~12 秒一片大约 100 来条，"
                             "正好够训 LoRA（下限 20 条）。设太小会白白浪费素材")

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("音频优化", "🧹",
                                      "零样本 TTS 的音色几乎完全由参考音频决定，"
                                      "所以「把音频弄干净」往往比多训几轮更有效。"))
                    enhance_cb = gr.Checkbox(True, label="启用优化（关掉则只体检不改音频）")
                    denoise_cb = gr.Checkbox(True, label="降噪")
                    denoise_sl = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                           label="降噪强度",
                                           info="调太高会连同气声、齿音一起抹掉，"
                                                "音色会变闷")
                    norm_cb = gr.Checkbox(True, label="响度归一")
                    trim_cb = gr.Checkbox(True, label="掐掉首尾静音")

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("语音识别", "📝",
                                      "长音频在切片之后**逐片转写**，于是文本与"
                                      "音频按「一片一段」配对 —— 这就是本流程的"
                                      "对齐环节（工程里没有强制对齐器）。"))
                    asr_cb = gr.Checkbox(True, label="自动转写为训练文本")
                    whisper_dd = gr.Dropdown(
                        choices=list(RW.WHISPER_SIZES.keys()),
                        value=RW.DEFAULT_WHISPER, label="识别模型（转写成训练文本）",
                        info="默认 medium：它的产出**直接成为训练文本**，"
                             "准确率决定模型学什么，值得用大一点的")
                    score_whisper_dd = gr.Dropdown(
                        choices=list(RW.WHISPER_SIZES.keys()),
                        value="small", label="打分模型（择优时评 reward）",
                        info="默认 small：打分只在候选之间做**相对比较**，"
                             "而它运行时引擎还占着显存 —— 用 medium 会挤爆 8 GB 卡，"
                             "触发 WMDD 静默降速（实测把扩散采样从 0.9s 拖到 25s）".replace("WMDD", "WDDM"))

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("筛选", "🧽", ""))
                    min_score_sl = gr.Slider(
                        0.0, 100.0, value=45.0, step=1.0,
                        label="音频体检分下限（0 = 不按分筛）",
                        info="体检分综合了时长、信噪比、采样率、削波、静音占比、"
                             "直流偏移。低于下限的样本留着只会教坏模型")
                    repeats_nb = gr.Number(
                        3, label="同一句话最多留几条", precision=0,
                        info="防止一段反复重录的话主导整个训练集；0 = 不限")
                    val_ratio_sl = gr.Slider(0.05, 0.3, value=0.1, step=0.01,
                                             label="验证集比例")

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("调参与训练", "🎛",
                                      "预设档位决定 rank / alpha / lr / epochs / "
                                      "回放比例 —— 它比任何单个超参都更能决定"
                                      "「学得像」还是「把底座带坏」。"))
                    arch_cg = gr.CheckboxGroup(
                        choices=[("gpt —— 语气、节奏、停顿", "gpt"),
                                 ("cfm —— 音色、音质、频谱细节", "cfm")],
                        value=["gpt", "cfm"], label="训练目标（可多选）",
                        info="两个都练才既像又自然：GPT 管「怎么说」，"
                             "CFM 管「像谁」")
                    preset_dd = gr.Dropdown(choices=PRESET_CHOICES, value="auto",
                                            label="预设档位",
                                            info="自动 = 按数据量排候选并逐个预检")
                    topk_sl = gr.Slider(1, 5, value=3, step=1,
                                        label="每个目标保留并参评的档位数",
                                        info="保险库只留 val loss 最好的 top-K 个，"
                                             "它们全部进真机打分，最后激活最优的那一个")
                    seed_nb = gr.Number(42, label="随机种子", precision=0,
                                        info="固定种子才能复现划分与采样")

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("性能", "⚡",
                                      "这里只并行**纯 CPU** 的音频处理（体检与增强）："
                                      "每条音频各读各写、互不干扰，所以加速与结果无关 —— "
                                      "输出和串行逐字节一致。GPU 阶段（特征提取、"
                                      "择优合成）必须串行，不会被动到。"))
                    cpu_workers_sl = gr.Slider(
                        0, 4, value=0, step=1,
                        label="CPU 并行线程数（0 = 自动，1 = 关掉）",
                        info="自动 = min(4, CPU 核数÷4)，本机约 4 线程。"
                             "上限 4 是刻意的：训练与推理也要 CPU，抢太狠整体更慢")

                with gr.Column(elem_classes=["ix-section"]):
                    gr.HTML(T.section("择优", "🏆",
                                      "val loss 只说明「拟合得好不好」，"
                                      "说明不了「像不像本人」—— 后者只能"
                                      "合成出来再打分。"))
                    rank_cb = gr.Checkbox(
                        True, label="真机打分择优（关掉则只按 val loss 排序）",
                        info="要合成 + whisper 打分，每个候选几十秒；"
                             "关掉可省一次引擎加载")
                    eval_n_sl = gr.Slider(
                        3, 20, value=5, step=1, label="每个候选评几条",
                        info="打分本身有转写噪声，评太少名次会不稳")

            with gr.Accordion("ℹ️ 它到底做了什么（与手动流程的对应）", open=False):
                gr.Markdown(
                    "这一页等价于把下面这些手动步骤按顺序自动跑一遍，"
                    "用的是同一套后端函数，没有另起一套逻辑：\n\n"
                    "| 手动流程 | 一键三连调用的东西 |\n|---|---|\n"
                    "| 数据集页：导入音频 | `dataset.import_audio` |\n"
                    "| 数据集页：切片（此前无入口） | `dataset.split_long` |\n"
                    "| 音频工作台：降噪归一 | `audio_lab.enhance` |\n"
                    "| 数据集页：补文本 | `reward.RewardScorer.transcribe` |\n"
                    "| 数据集页：体检 / 划分 | `dataset.refresh_all` / "
                    "`make_split` |\n"
                    "| 数据集页：特征提取 | `features.extract_dataset` |\n"
                    "| 训练页：选预设 → 预检 → 训练 | `guard.CONFIG_PRESETS` / "
                    "`*.preflight()` / `*.run()` |\n"
                    "| 评测页：A/B 打分 | `evaluate.run_eval`（单选手模式）|\n"
                    "| 评测页：激活档位 | `runs.activate` |\n\n"
                    "**为什么必须先卸载引擎再训练？** 8 GB 卡上引擎常驻 "
                    "4.9~5.7 GB，与训练器放不下两份。Windows WDDM 下显存"
                    "溢出不报 OOM，而是静默降速 20~30 倍，所以流水线在训练"
                    "前后主动加载/卸载，并在训练期间禁止任何人加载引擎。\n\n"
                    "**样本太少怎么办？** 低于 "
                    f"{OC.MIN_SAMPLES} 条会在筛选阶段就停下并告诉你原因，"
                    "不会白跑一遍训练。")

    # =====================================================================
    # 回调
    # =====================================================================
    opt_controls = [model_name_tb, lang_dd, slice_target_sl, slice_min_sl,
                    slice_over_sl, slice_pieces_nb, enhance_cb, denoise_cb,
                    denoise_sl, norm_cb, trim_cb, asr_cb, whisper_dd,
                    score_whisper_dd,
                    min_score_sl, repeats_nb, val_ratio_sl, arch_cg,
                    preset_dd, topk_sl, rank_cb, eval_n_sl, cpu_workers_sl,
                    seed_nb]

    assert len(opt_controls) == len(OPT_KEYS), \
        f"控件数 {len(opt_controls)} 与 OPT_KEYS {len(OPT_KEYS)} 不一致"

    def _opts(*vals) -> OC.OneClickOptions:
        return _opts_from(dict(zip(OPT_KEYS, vals)))

    @LOG.ui_guard("oneclick.on_plan")
    def on_plan(*vals):
        return plan_markdown(_opts(*vals))

    plan_btn.click(on_plan, inputs=opt_controls, outputs=[panel_md])

    @LOG.ui_guard("oneclick.on_run")
    def on_run(files, extra_path, *vals):
        opt = _opts(*vals)
        errs = [n.message for n in opt.validate() if n.level == "error"]
        if errs:
            return T.err("<br>".join(errs))
        got = OC.collect_inputs(files or [], extra_path or "")
        if not got["files"]:
            return T.err(
                "还没有输入：请上传音频文件，或在「服务器路径」里填一个"
                "音频文件 / 目录的路径。<br>"
                + (f"（未找到：{got['missing'][:3]}）" if got["missing"] else ""))
        n = len(got["files"])

        def fn(progress, should_stop):
            return OC.run_oneclick(eng, opt, paths=got["files"],
                                   extra_path="", progress=progress,
                                   should_stop=should_stop, tracker=runner)

        # require_engine="none"：本流水线横跨「引擎加载」（特征提取/择优）
        # 与「引擎卸载」（训练）两种状态，只有 "none" 不会被自己的要求挡死。
        # 阶段边界上流水线会用 runner.set_engine_req 把真实需求同步回去，
        # 所以训练期间的互斥保护依然生效。
        r = runner.submit("oneclick", f"一键三连 · {n} 个文件", fn,
                          require_engine="none", engine=eng)
        if not r["ok"]:
            return T.err(r["message"])
        gr.Info("一键三连已在后台启动")
        return T.tip(
            f"🚀 已启动，处理 {n} 个输入。下方面板会实时更新阶段进度；"
            "整条流水线要跑一段时间（识别与训练是大头），"
            "期间可以切到别的页面，回来后仍在这里看结果。")

    run_btn.click(on_run, inputs=[files_in, extra_path_tb] + opt_controls,
                  outputs=[panel_md])

    @LOG.ui_guard("oneclick.on_stop")
    def on_stop():
        r = runner.cancel()
        return T.tip(r["message"]) if r.get("ok") else T.warn(r["message"])

    stop_btn.click(on_stop, inputs=[], outputs=[panel_md])

    # ---------- 轮询：进度条 + 面板 + 日志 ----------
    poll_cache: Dict[str, Any] = {"snap": None, "rendered": None}

    def on_poll():
        snap = runner.snapshot()
        if snap == poll_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update(), gr.update()
        poll_cache["snap"] = snap

        done = snap["ok"] is not None and not snap["running"]
        bar = ""
        if snap["running"] or done:
            pct = int(snap["progress"] * 100)
            icon = "🎉" if (done and snap["ok"]) else ("🔴" if done else "▶️")
            bar = (f'<div style="margin:4px 0">{icon} <b>{snap["label"]}</b> · '
                   f'{pct}% · {snap["message"]}'
                   + (" · <b>已请求停止</b>" if snap["stop_requested"] else "")
                   + f' · {snap["seconds"]}s'
                   + (f'<br><span style="color:#c00">{snap["error"]}</span>'
                      if (done and not snap["ok"] and snap["error"]) else "")
                   + '<div style="background:var(--border-color-primary);'
                     'border-radius:6px;height:8px;margin-top:4px">'
                     f'<div style="width:{pct}%;height:8px;border-radius:6px;'
                     'background:var(--color-accent)"></div></div></div>')

        # 面板：运行中显示阶段进度，结束后显示完整裁决报告
        panel = gr.update()
        if snap["running"]:
            res = snap.get("result")
            rep = res.get("report") if isinstance(res, dict) else None
            if rep:
                panel = gr.update(value=_live_markdown(rep))
        elif done and snap["kind"] == "oneclick":
            res = snap.get("result") or {}
            md = res.get("markdown") if isinstance(res, dict) else None
            if md and poll_cache["rendered"] != md:
                poll_cache["rendered"] = md
                panel = gr.update(value=md)
            elif not md and snap.get("error"):
                panel = gr.update(value=T.err(snap["error"]))

        return (gr.update(value=bar), panel,
                gr.update(value=runner.log_text()))

    timer = gr.Timer(value=2.5, active=True)
    timer.tick(on_poll, inputs=[],
               outputs=[progress_html, panel_md, log_ta])

    return {
        "page_load": (_page_load_plan, [panel_md]),
        "components": {"run_btn": run_btn, "panel_md": panel_md},
    }


def _page_load_plan() -> str:
    """页面首次加载时的面板内容（用默认选项，不依赖控件取值）。"""
    return plan_markdown(OC.OneClickOptions())


def _live_markdown(rep: Dict[str, Any]) -> str:
    """运行中的面板：阶段表 + 已完成阶段的关键数字。"""
    icon = {"done": "✅", "running": "⏳", "skipped": "⏭",
            "failed": "🔴", "pending": "⏸"}
    L = ["### 正在跑：阶段进度", ""]
    L.append("| # | 阶段 | 状态 | 耗时 | 结果 |")
    L.append("|---|---|---|---|---|")
    for i, s in enumerate(rep.get("stages") or [], 1):
        L.append(f"| {i} | {s.get('title', '')} | "
                 f"{icon.get(s.get('status'), s.get('status'))} "
                 f"{s.get('status', '')} | "
                 f"{(str(s.get('seconds')) + 's') if s.get('seconds') else '—'} | "
                 f"{s.get('detail') or '—'} |")
    cands = rep.get("candidates") or []
    if cands:
        L.append("")
        L.append("**自动调参裁决（进行中）**")
        L.append("")
        L.append("| 目标 | 预设 | 步数 | 预估显存 | 裁决 |")
        L.append("|---|---|---|---|---|")
        for c in cands:
            v = "✅ 采用" if c.get("chosen") else f"❌ {c.get('reject', '')}"
            L.append(f"| {c.get('arch', '')} | {c.get('preset', '')} | "
                     f"{c.get('total_steps', '—')} | "
                     f"{c.get('est_vram_gb', '—')} GB | {v} |")
    return "\n".join(L)
