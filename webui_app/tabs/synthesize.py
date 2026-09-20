"""Tab：语音合成（主页面）。

布局：
    左栏 = 音色来源 → 文本与语言 → 情感控制 → 生成与输出
    右栏 = GPT 采样参数 → 分句与时长 → 参数详解抽屉 → 示例

所有控件都由 params.REGISTRY 生成，提示文案与「参数手册」页共用同一份定义。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import params as P
from webui_app import theme as T
from webui_app import widgets as W
from webui_app.config import EMO_BIAS, EMO_SUM_LIMIT, EMO_VECTOR_LABELS
from webui_app.context import AppContext
from webui_app.services import audio_lab as AL
from webui_app.services import inference as INF
from webui_app.services import pronunciation as PR
from webui_app.services import synth_state as SS
from webui_app import logging_setup as LOG
from webui_app.services import voice_bank
from webui_app.services.engine import EngineError
from webui_app.training import guard as GD
from webui_app.training import merge as MG
from webui_app.training import runs as RN

# 配置档直接复用官方预设系统（与「💾 预设」页 / 官方 webui.py 完全互通）
try:
    from indextts.utils.presets import (delete_preset, list_presets, load_preset,
                                        safe_preset_name, save_preset)
except Exception:      # pragma: no cover - 官方模块缺失时的降级
    list_presets = lambda: []           # noqa: E731
    load_preset = lambda n: None        # noqa: E731
    save_preset = None                  # type: ignore[assignment]
    delete_preset = lambda n: False     # noqa: E731
    safe_preset_name = lambda n: n      # noqa: E731

# ---------------------------------------------------------------------------
# LoRA 挂载区的辅助函数（模块级：不依赖 ctx，便于单独测试）
# ---------------------------------------------------------------------------

LORA_NONE = "（不使用 LoRA）"


def _lora_run_choices() -> List[Any]:
    """带 adapter 的训练记录 → [(显示文本, run 名)]。

    只列出 `has_adapter` 的 run —— 训练失败的记录挂在引擎上只会报错，
    不该出现在推理页里让人误选。
    """
    out: List[Any] = [(LORA_NONE, "")]
    try:
        for r in RN.list_runs():
            if not getattr(r, "has_adapter", False):
                continue
            bv = (f"val {r.best_val:.4f}"
                  if getattr(r, "best_val", None) is not None else "val —")
            out.append((f"{r.name} · {r.arch} · {bv}", r.name))
    except Exception:
        pass
    return out


def _lora_ckpt_choices(run: str) -> List[str]:
    """某个 run 可选的档位。用 list_checkpoints（不创建目录）。

    `best` 指的是 `<run>/adapter` —— 训练器每次改善都会把最好的一档同步
    过去，所以它就是「该 run 表现最好的权重」；保险库里的具名档位也一并
    列出（一键三连正是靠它们做择优的）。
    """
    if not run:
        return ["best"]
    out = ["best"]
    try:
        out.extend(c.name for c in RN.list_checkpoints(run))
    except Exception:
        pass
    try:
        if os.path.isdir(os.path.join(RN.run_dir(run), "final")):
            out.append("final")
    except Exception:
        pass
    seen: List[str] = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def lora_action(run: str, mounted_runs: set) -> str:
    """「下拉框选了什么」×「引擎上实际挂着什么」→ 该做什么。

    纯函数（便于回归测试）。三种结果：

        "mount"    选择与现状不一致 → 需要挂载
        "unmount"  选择了「不使用 LoRA」但引擎上还挂着 → 卸掉
        "none"     已经一致 → 什么都不做（避免每次生成都重读 adapter 文件）

    之所以要显式判定「实际挂着什么」而不是只比一个「上次挂过什么」的记录：
    那个记录在引擎被卸载/重载（可能发生在别的页面）之后会过期，
    只信它就会出现「以为还挂着、其实早没了」——结果是**静默用底座合成**，
    听到的声音不对却没有任何报错。
    """
    run = (run or "").strip()
    mounted = set(mounted_runs or ())
    if not run:
        return "unmount" if mounted else "none"
    return "none" if run in mounted else "mount"


def _lora_state_html(eng) -> str:
    """当前引擎上挂着什么、强度多少。

    读的是**引擎报告的真实模块状态**（`engine.lora_status()`），不是标签
    列表 —— 标签和实际包装状态可能脱节，那时面板会明说，而不是让用户
    对着「显示纯底座但声音还是那个人」猜。
    """
    try:
        st = eng.lora_status()
    except Exception as e:
        LOG.get_logger("ui.lora").warning("读取 LoRA 状态失败：%s", e)
        return T.warn(f"读取 LoRA 状态失败：{type(e).__name__}: {e}")

    if not st:
        return T.hint("引擎未加载。挂载 LoRA 时会自动加载；"
                      "当前显示不了实际挂载状态。")

    mounted = [x for x in st if x.get("wrapped")]
    if not mounted:
        return T.hint("当前引擎上是 <b>纯底座</b>（未挂载 LoRA）。"
                      "选一个训练记录后点「🧬 挂载到引擎」。")

    rows, bad = [], []
    for x in mounted:
        tgt = x["target"]
        mean = None
        try:
            mod = (getattr(eng.tts, "gpt", None) if tgt == "gpt"
                   else GD.lora_target_module(eng, tgt))
            if mod is not None:
                mean = float(GD.get_adapter_scale(mod).get("_mean", 1.0))
        except Exception:
            pass
        label = "、".join(x.get("tags") or []) or "(无标签)"
        rows.append(f"<code>{tgt}</code> · {label}"
                    + (f" · 强度 {mean:.2f}" if mean is not None else ""))
        if not x.get("consistent"):
            bad.append(tgt)

    html = T.tip("🧬 <b>已挂载</b>：" + " ｜ ".join(rows)
                 + "<br><sub>强度 0 = 纯底座，1 = 完整 LoRA，"
                   "0.6~0.8 是「像」与「稳」的常见折中。</sub>")
    if bad:
        # 状态脱节的可见化：这类不一致正是「切 LoRA 偶现持续报错」的温床
        html += T.warn("⚠️ " + "、".join(bad) + " 的标签与实际包装状态不一致。"
                       "建议点「🧵 卸载 LoRA」清干净后重挂；"
                       "已写入日志（系统页可看，error.log 里有堆栈）。")
    return html

EXAMPLE_TEXTS = [
    ("中文 · 日常", "大家好，欢迎使用 IndexTTS 二点五，这是一段用于测试的中文语音。", "ZH"),
    ("中文 · 多音字标注", "他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。", "ZH"),
    ("中文 · 情感", "快躲起来！是他要来了！他要来抓我们了！", "ZH"),
    ("英文 · 音素标注", "He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.", "EN"),
    ("英文 · 日常", "IndexTTS can clone a voice from just a few seconds of reference audio.", "EN"),
    ("日语 · 假名标注", "彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。", "JA"),
]


def _prof_choices() -> List[str]:
    """配置档下拉的选项（官方预设目录）。"""
    try:
        return [""] + list_presets()
    except Exception:
        return [""]


# ---------------------------------------------------------------------------
# 纯函数：情感向量归一化（与官方 normalize_emo_vec 完全一致，但不需要引擎）
# ---------------------------------------------------------------------------

def normalize_vec(vec: List[float], apply_bias: bool = True) -> List[float]:
    v = [float(x or 0.0) for x in vec]
    if len(v) < 8:
        v += [0.0] * (8 - len(v))
    v = v[:8]
    if apply_bias:
        v = [a * b for a, b in zip(v, EMO_BIAS)]
    s = sum(v)
    if s > EMO_SUM_LIMIT:
        k = EMO_SUM_LIMIT / s
        v = [x * k for x in v]
    return v


def vec_meter_html(vec: List[float]) -> str:
    """8 维情感的实时可视化：原始值 vs 归一化后的实际生效值。"""
    eff = normalize_vec(vec)
    raw_sum = sum(float(x or 0.0) for x in vec)
    clipped = raw_sum > EMO_SUM_LIMIT

    bars = []
    for i, (lab, e, r) in enumerate(zip(EMO_VECTOR_LABELS, eff, vec)):
        pct = min(100.0, e * 100 / 0.8)          # 以 0.8 满量程显示
        bias = EMO_BIAS[i]
        bars.append(
            f'<div style="display:flex;align-items:center;gap:7px;margin:2px 0">'
            f'<span style="width:34px;font-size:11.5px;text-align:right;'
            f'opacity:.75">{lab}</span>'
            f'<div style="flex:1;height:9px;border-radius:5px;overflow:hidden;'
            f'background:var(--input-background-fill);'
            f'border:1px solid var(--input-border-color)">'
            f'<div style="height:100%;width:{pct:.1f}%;border-radius:5px;'
            f'background:linear-gradient(90deg,#5B6EE1,#8B9BF0)"></div></div>'
            f'<span class="ix-mono" style="width:96px;font-size:11px">'
            f'{float(r or 0):.2f} → <b>{e:.3f}</b> <span style="opacity:.5">×{bias}</span>'
            f'</span></div>'
        )

    warn = ""
    if clipped:
        warn = (
            f'<div class="ix-warn" style="margin-top:6px">'
            f'⚠️ 8 维原始总和 {raw_sum:.2f} 超过上限 {EMO_SUM_LIMIT}，'
            f'已被<b>静默等比压缩</b>到 {sum(eff):.2f}。'
            f'滑块上的值不是最终生效值，右侧箭头后才是。</div>'
        )
    total = sum(eff)
    head = (
        f'<div style="font-size:12.5px;margin-bottom:5px">'
        f'实际生效总和 <b>{total:.3f}</b> / {EMO_SUM_LIMIT} '
        f'<span style="opacity:.6">（左侧=滑块值，右侧=经偏置与限幅后的生效值）</span></div>'
    )
    return head + "".join(bars) + warn


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(ctx: AppContext):
    cfg = ctx.cfg
    eng = ctx.engine
    sb = ctx.component("statusbar")     # 顶栏状态条，由 app.py 预先注册

    # ------------------------------------------------------------------
    # 参数记忆：把上次的值直接作为控件初始值。
    # 恢复走「渲染时初始值」而不是页面加载事件回填 —— render() 每个进程
    # 只跑一次，首屏 payload 就是正确的，无闪烁、无事件竞态。
    # ------------------------------------------------------------------
    _st_payload = SS.load_state()
    _remember = SS.remember_enabled(_st_payload)
    ST: Dict[str, Any] = _st_payload.get("values", {}) if _remember else {}
    _st_prompt = _st_payload.get("prompt_audio") if _remember else None
    _st_emo_audio = _st_payload.get("emo_audio") if _remember else None

    def _v(key: str):
        """带记忆的初始值；没有记忆时回退注册表默认值。"""
        return ST[key] if key in ST else "__default__"

    # 选择类的值要做「仍在合法集合内」校验，否则下拉框会显示非法值
    _bank_names = voice_bank.names()
    _st_voice = ST.get("voice_name") if ST.get("voice_name") in _bank_names else ""
    _st_lang = ST.get("lang") if ST.get("lang") in cfg.languages else "__default__"
    _lora_values = [v for _l, v in _lora_run_choices()]
    _st_lora_run = ST.get("lora_run") if ST.get("lora_run") in _lora_values else ""
    _st_lora_ckpt = ("best" if not _st_lora_run else
                     (ST.get("lora_ckpt")
                      if ST.get("lora_ckpt") in _lora_ckpt_choices(_st_lora_run)
                      else "best"))
    _st_emo_label = (ST.get("emo_mode_label")
                     if ST.get("emo_mode_label") in W.EMO_MODE_LABELS
                     else "__default__")
    # 恢复的模式决定各情感分组的初始可见性（与 on_emo_mode 的联动一致）
    _st_mode = (W.emo_mode_index(_st_emo_label)
                if _st_emo_label != "__default__" else 0)

    if _st_payload and _remember:
        _mem_note = (f"已恢复上次参数（{len(ST)} 项 · 保存于 "
                     f"{SS.saved_at_text(_st_payload)}）"
                     + ("" if _st_payload.get("prompt_audio")
                        or not ST.get("voice_name")
                        else "；上次的参考音频文件已失效，请重新选择"))
        _mem_init = (f'<span class="ix-chip"><span class="ix-dot ok"></span>'
                     f'参数记忆 · {_mem_note}</span>')
    elif _st_payload:
        _mem_init = ('<span class="ix-chip"><span class="ix-dot warn"></span>'
                     '参数记忆 · 已关闭（重启后不恢复）</span>')
    else:
        _mem_init = ('<span class="ix-chip"><span class="ix-dot idle"></span>'
                     '参数记忆 · 尚无记录</span>')

    # 共享快照在渲染期就用「恢复值」播种：否则重启后用户什么都没动就点
    # 「存为配置档」时，快照还是空的，会把默认值当成当前参数存进档里。
    ctx.shared["syn_live_values"] = dict(ST)
    ctx.shared["syn_live_values"].update({
        "prompt_audio": _st_prompt, "emo_audio": _st_emo_audio,
        "_remember": _remember,
    })
    ctx.shared["_remember_flag"] = _remember

    with gr.Row(equal_height=False):
        # =================================================================
        # 左栏
        # =================================================================
        with gr.Column(scale=3, min_width=460):

            # ---------- 音色来源 ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("音色来源", "🎙",
                                  "决定「谁在说」。这段音频会被拆成三路：CAMPPlus 声纹、"
                                  "w2v-BERT 情感特征、ref_mel 声学模板 —— 音色的全部来源。"))
                with gr.Row():
                    voice_dd = gr.Dropdown(
                        choices=[""] + _bank_names, value=_st_voice,
                        label="从音色库载入", scale=3,
                        info="在「参考音频工作台」体检并增强后入库的素材",
                        allow_custom_value=False,
                    )
                    voice_reload_btn = gr.Button("↻", scale=0, variant="secondary",
                                                 size="sm")
                    voice_refresh_btn = gr.Button("刷新列表", scale=1, size="sm")
                prompt_audio = W.make_component("spk_audio_prompt", label="",
                                                value=_st_prompt)
                with gr.Row():
                    voice_detail_btn = gr.Button("查看该音色体检报告", size="sm", scale=1)
                    to_lab_btn = gr.Button("→ 送去工作台优化", size="sm", scale=1)
                voice_info = gr.HTML("")

            # ---------- LoRA 音色模型（自训练产物） ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section(
                    "LoRA 音色模型", "🧬",
                    "挂载自训练出来的 adapter。下拉框直接读 "
                    "<code>training_runs/</code> 里的训练记录 —— "
                    "「🚀 一键三连」或「🎓 训练」跑完的模型会自动出现在这里。"
                    "挂上之后照常点「生成」即可，参数不必改。"))
                with gr.Row():
                    lora_run_dd = gr.Dropdown(
                        choices=_lora_run_choices(), value=_st_lora_run,
                        label="训练记录（run）", scale=3,
                        allow_custom_value=False,
                        info="只列出已经产出 adapter 的记录")
                    lora_reload_btn = gr.Button("↻", scale=0,
                                                variant="secondary", size="sm")
                with gr.Row():
                    lora_ckpt_dd = gr.Dropdown(
                        choices=_lora_ckpt_choices(_st_lora_run),
                        value=_st_lora_ckpt, label="档位", scale=1,
                        info="best = 该 run 表现最好的一档；"
                             "具名档位来自 checkpoints 保险库")
                    lora_scale_sl = gr.Slider(
                        0.0, 1.5, value=float(ST.get("lora_scale", 1.0)),
                        step=0.05, label="强度", scale=2,
                        info="推理期实时生效，不用重训")
                with gr.Row():
                    lora_mount_btn = gr.Button("🧬 挂载到引擎", size="sm", scale=1)
                    lora_unmount_btn = gr.Button("卸载 LoRA", size="sm", scale=1)
                lora_state_html = gr.HTML(_lora_state_html(eng))

            # ---------- 输出后处理（提亮 / 空气感） ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section(
                    "输出后处理", "✨",
                    "IndexTTS2 的输出固定 <b>22050 Hz</b>（11 kHz 上限）—— 这是模型"
                    "设计，不是处理链弄丢的。这里做的是<b>听感补偿</b>：把已有的"
                    "清晰度抬出来、用高频自身的谐波补一点空气感，<b>不伪造细节</b>。<br>"
                    "默认只做高通与峰值对齐；想更亮就调 presence，想要空气感再加 exciter。"))
                polish_cb = gr.Checkbox(
                    bool(ST.get("polish_on", True)), label="启用输出后处理",
                    info="关掉则输出与模型原始结果完全一致（最保真）")
                with gr.Row():
                    polish_presence = gr.Slider(
                        0.0, 6.0, float(ST.get("polish_presence", 2.5)),
                        step=0.5, label="提亮 presence (dB)",
                        info="4.5 kHz 以上平滑抬升。2~4 dB 明显更「亮」更清楚，"
                             "超过 5 容易齿音发刺")
                    polish_exciter = gr.Slider(
                        0.0, 0.3, float(ST.get("polish_exciter", 0.08)),
                        step=0.01, label="空气感激励",
                        info="从 5 kHz 以上生成谐波、只叠加 9 kHz 以上的部分。"
                             "0.05~0.15 是安全区；再高会明显失真")
                polish_md = gr.Markdown(
                    T.hint("生成后这里会显示谱质心前后对比 —— 「亮了没有」有客观数字。"))

            # ---------- 文本 ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("文本与语言", "📝",
                                  "支持 <文字|发音> 标注：中文拼音 / 英文 CMU 音素 / 日语假名。"
                                  "官方会截断参考音频到前 15 秒，但文本长度只受分句参数约束。"))
                text_in = W.make_component("text", value="", placeholder="请输入要合成的文本…")
                with gr.Row():
                    if cfg.is_v25:
                        lang_dd = W.make_component("lang", scale=1, value=_st_lang)
                    else:
                        lang_dd = gr.State(value=None)
                    dur_sl = W.make_component("duration_factor", scale=2,
                                              value=_v("duration_factor"))
                    seed_n = W.make_component("seed", scale=1, value=_v("seed"))
                with gr.Row():
                    tn_cb = W.make_component("text_normalization", scale=1,
                                             value=_v("text_normalization"))
                    token_stat = gr.HTML("")

                # ---------- 读音纠正 ----------
                # 读错字是 TTS 最常被抱怨的问题，而它**不需要重训模型**：
                # 官方内置 <字|拼音> 标注，实测 1728 条拼音全部可用。
                # 所以把它放在文本区而不是埋进手册里。
                with gr.Accordion("🔤 读音纠正（拼音 / 音素 / 假名标注）", open=False):
                    gr.HTML(T.hint(
                        "某个字读错时，<b>不用重训模型、也不用改采样参数</b>："
                        "在文本里写 <code>&lt;字|拼音&gt;</code> 就能强制指定读法。"
                        "这是官方内置能力，已实测 1728 条拼音全部可用。"))
                    pr_issue = gr.HTML("")
                    with gr.Row():
                        pr_word = gr.Textbox(
                            label="要纠正的字 / 词", scale=3,
                            placeholder="例如：行（只输入一个字可看到全部多音）",
                        )
                        pr_query_btn = gr.Button("查读音候选", size="sm", scale=1)
                    pr_table = gr.Dataframe(
                        headers=PR.CAND_HEADERS, datatype=["str"] * 6,
                        wrap=False, elem_classes=["ix-table"],
                        label="候选读音（点击一行自动填入下方）",
                    )
                    pr_note = gr.Markdown("")
                    with gr.Row():
                        pr_pron = gr.Textbox(label="指定发音", scale=2,
                                             placeholder="HANG2")
                        pr_which = gr.Number(label="第几处（从 1 起）", value=1,
                                             precision=0, minimum=1, scale=1)
                        pr_apply_btn = gr.Button("➕ 插入标注到文本",
                                                 variant="primary", size="sm", scale=2)
                    with gr.Row():
                        pr_preview_btn = gr.Button("🔍 预览模型看到的分词",
                                                   size="sm", scale=1)
                        pr_strip_btn = gr.Button("🧹 去掉全部标注", size="sm", scale=1)
                    pr_out = gr.Markdown("")
                    with gr.Accordion("📖 标注语法与易踩点", open=False):
                        gr.Markdown(PR.SYNTAX_DOC)
                pr_rows = gr.State([])

                with gr.Accordion("分句预览（v2.5 按 token 预算切分）", open=False) as seg_acc:
                    seg_table = gr.Dataframe(
                        headers=["#", "分句内容", "字符数", "Token数"],
                        datatype=["number", "str", "number", "number"],
                        wrap=True, elem_classes=["ix-table"],
                    )
                    seg_note = gr.HTML("")
                seg_sl = W.make_component("max_text_tokens_per_segment",
                                          value=_v("max_text_tokens_per_segment"))
                sil_sl = W.make_component("interval_silence",
                                          value=_v("interval_silence"))

            # ---------- 情感控制 ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("情感控制", "🎭",
                                  "情感只注入 GPT(T2S)，完全不进 CFM(S2M) —— "
                                  "这就是「音色-情感解耦」：改情感不动音色。"))
                emo_mode = W.make_component("emo_control_method",
                                            value=_st_emo_label
                                            if _st_emo_label != "__default__"
                                            else W.EMO_MODE_LABELS[0])
                _MODE_HINTS_BOOT = [
                    T.tip("<b>模式 0</b>：情感跟随音色参考音频自身。音色还原度最高，最稳的默认值。"
                          "此模式下「情感权重」滑块<b>无效</b>（代码里 emo_alpha 被强制为 1.0）。"),
                    T.hint("<b>模式 1</b>：音色与情感分别指定。「情感权重」是<b>线性插值系数</b>："
                           "<code>out = base + alpha*(emo - base)</code>，0=完全用音色音频自身情感，"
                           "1=完全用情感参考音频情感。推荐 0.6~0.8。"),
                    T.hint("<b>模式 2</b>：手动指定 8 维向量。「情感权重」此时是<b>向量缩放系数</b>，"
                           "等比缩放 8 个值。此模式<b>不能</b>叠加情感参考音频（代码会强制清空它）。"),
                    T.warn("<b>模式 3 · 实验功能</b>：用自然语言描述情绪，由 QwenEmotion(Qwen3-0.6B) "
                           "转成 8 维向量。官方建议情感权重设 <b>0.6 或更低</b>。"
                           "本 UI 采用<b>串行执行</b>：挂载 → 算向量 → 立即卸载 → 再走模式 2 的路径，"
                           "避免与主推理抢显存（实测峰值 6.26GB / 8GB）。"),
                ]
                emo_hint = gr.HTML(_MODE_HINTS_BOOT[_st_mode])

                with gr.Group(visible=_st_mode == 1) as g_emo_audio:
                    emo_audio = W.make_component("emo_audio_prompt",
                                                 value=_st_emo_audio)

                with gr.Group(visible=_st_mode == 2) as g_emo_vec:
                    gr.HTML('<div class="ix-hint">顺序固定为 '
                            '[喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静]。'
                            '每维有内置偏置系数，且 8 维总和超过 0.8 会被静默压缩。</div>')
                    with gr.Row(elem_classes=["ix-emo-grid"]):
                        with gr.Column():
                            v0 = W.make_component("emo_vec_0", value=_v("emo_vec_0"))
                            v1 = W.make_component("emo_vec_1", value=_v("emo_vec_1"))
                            v2 = W.make_component("emo_vec_2", value=_v("emo_vec_2"))
                            v3 = W.make_component("emo_vec_3", value=_v("emo_vec_3"))
                        with gr.Column():
                            v4 = W.make_component("emo_vec_4", value=_v("emo_vec_4"))
                            v5 = W.make_component("emo_vec_5", value=_v("emo_vec_5"))
                            v6 = W.make_component("emo_vec_6", value=_v("emo_vec_6"))
                            v7 = W.make_component("emo_vec_7", value=_v("emo_vec_7"))
                    vec_meter = gr.HTML(
                        vec_meter_html([float(ST.get(f"emo_vec_{i}", 0.0) or 0.0)
                                        for i in range(8)]))
                    with gr.Row():
                        vec_zero_btn = gr.Button("全部归零", size="sm", scale=1)
                        vec_rand_btn = gr.Button("随机一组", size="sm", scale=1)

                with gr.Group(visible=_st_mode == 3) as g_emo_text:
                    emo_text = W.make_component(
                        "emo_text", value=_v("emo_text"),
                        placeholder="例如：委屈巴巴 / 危险在悄悄逼近")
                    with gr.Row():
                        emo_probe_btn = gr.Button("预览情感向量（不合成）", size="sm", scale=2)
                        emo_cache_btn = gr.Button("清空向量缓存", size="sm", scale=1)
                    emo_probe_out = gr.HTML("")

                with gr.Row(visible=_st_mode in (1, 2, 3)) as g_emo_alpha:
                    emo_alpha = W.make_component("emo_alpha", scale=3,
                                                 value=_v("emo_alpha"))
                with gr.Row(visible=_st_mode in (2, 3)) as g_emo_rand:
                    emo_rand = W.make_component("use_random", scale=1,
                                                value=_v("use_random"))

            # ---------- 生成 ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("生成", "▶", ""))
                with gr.Row():
                    gen_btn = gr.Button("🎧  生成语音", variant="primary", scale=3, size="lg")
                    stop_note = gr.HTML("")
                out_audio = gr.Audio(label="生成结果", type="filepath",
                                     elem_id="ix-output-audio")
                out_info = gr.HTML("")

        # =================================================================
        # 右栏
        # =================================================================
        with gr.Column(scale=2, min_width=380):
            with gr.Accordion("⚙️ GPT 采样参数（T2S 自回归）", open=False):
                gr.HTML(T.hint(
                    "这一组控制 <b>语义 token</b> 的自回归采样。语义 token 同时承载"
                    "<b>内容</b>和<b>韵律</b>，所以这里调的是「读得对不对、节奏自不自然」，"
                    "不是音色。"))
                with gr.Row():
                    do_sample = W.make_component("do_sample", scale=1,
                                                 value=_v("do_sample"))
                    temperature = W.make_component("temperature", scale=2,
                                                   value=_v("temperature"))
                with gr.Row():
                    top_p = W.make_component("top_p", scale=1, value=_v("top_p"))
                    top_k = W.make_component("top_k", scale=1, value=_v("top_k"))
                num_beams = W.make_component("num_beams", value=_v("num_beams"))
                with gr.Row():
                    rep_pen = W.make_component("repetition_penalty", scale=1,
                                               value=_v("repetition_penalty"))
                    len_pen = W.make_component("length_penalty", scale=1,
                                               value=_v("length_penalty"))
                max_mel = W.make_component("max_mel_tokens",
                                           value=_v("max_mel_tokens"))
                gr.HTML(T.warn(
                    "<b>repetition_penalty=10.0 不是笔误。</b>常规 LLM 用 1.0~1.3，"
                    "但 TTS 的语义 token 天然会连续重复（长元音、静音 token 52）。"
                    "调低到 1.x 几乎必然导致「重复同一个 token 直到撞上 "
                    "max_mel_tokens」的死循环，表现为拖长音或音频被硬截断。"))
                with gr.Row():
                    preset_fast = gr.Button("⚡ 快速档", size="sm", scale=1)
                    preset_bal = gr.Button("⚖️ 均衡档（官方默认）", size="sm", scale=1)
                    preset_hq = gr.Button("💎 质量档", size="sm", scale=1)

            with gr.Accordion("🧠 引擎与显存", open=False):
                engine_state = gr.HTML("")
                with gr.Row():
                    load_btn = gr.Button("加载模型", variant="primary", scale=1)
                    unload_btn = gr.Button("卸载模型", scale=1)
                with gr.Row():
                    clear_cache_btn = gr.Button("清空参考音频缓存", size="sm", scale=1)
                    clear_emo_btn = gr.Button("清空情感向量缓存", size="sm", scale=1)
                qwen_dev = gr.Radio(
                    choices=["auto", "cuda", "cpu"], value="auto",
                    label="QwenEmotion 运行设备",
                    info="auto=显存够就上GPU，不够自动回退CPU（零显存风险）",
                )
                gr.HTML(T.hint(
                    "官方用 <code>device_map=\"auto\"</code>：显存不足时 accelerate 会"
                    "<b>静默</b>把层丢到 CPU，慢几十倍且不报错。这里改成"
                    "「要么全 GPU、要么全 CPU」，行为可预测。"))

            W.group_help("sampling", cfg.is_v25, "采样参数详解")
            W.group_help("emotion", cfg.is_v25, "情感控制详解")
            W.group_help("segment", cfg.is_v25, "分句与时长详解")
            W.group_help("voice", cfg.is_v25, "参考音频详解")

            with gr.Accordion("💡 示例文本", open=False):
                ex_dd = gr.Dropdown(
                    choices=[f"{t}｜{lang}" for t, _x, lang in EXAMPLE_TEXTS],
                    value=None, label="点击载入示例",
                )
                gr.HTML(T.hint(
                    "官方 <code>examples/</code> 下的示例音频可用「音色库」页一键导入。"
                    "注意 <code>voice_01.wav</code> 实测只有 <b>2.44 秒</b>，"
                    "低于 3 秒的最低推荐值 —— 用它做参考音频，音色稳定性会打折。"))

            # ---------- 参数记忆与配置档 ----------
            with gr.Accordion("💾 参数记忆与配置档", open=True):
                gr.HTML(T.hint(
                    "本页参数改动<b>自动记住</b>，重启后自动恢复（含上传的参考音频副本）。"
                    "合成文本不记忆 —— 那是内容不是配置。"))
                mem_status = gr.HTML(_mem_init)
                with gr.Row():
                    remember_cb = gr.Checkbox(
                        _remember, label="记住参数改动（重启后恢复）", scale=2)
                    forget_btn = gr.Button("🧹 清除记忆并恢复默认", size="sm",
                                           scale=1)
                gr.HTML(T.hint(
                    "<b>配置档</b> = 命名的参数快照，存于 "
                    "<code>outputs/presets/&lt;名称&gt;/</code>，"
                    "与「💾 预设」页及官方 <code>webui.py</code> 完全互通。"))
                with gr.Row():
                    prof_dd = gr.Dropdown(
                        choices=_prof_choices(), value="", label="配置档",
                        scale=3, allow_custom_value=False,
                        elem_id="ix-profile-dd")
                    prof_reload_btn = gr.Button("↻", scale=0, size="sm",
                                                variant="secondary")
                prof_name = gr.Textbox(
                    label="存为配置档（名称）", scale=2,
                    placeholder="例如：主播A-温暖-慢速")
                with gr.Row():
                    prof_save_btn = gr.Button("💾 存为配置档", size="sm", scale=1,
                                              variant="primary")
                    prof_apply_btn = gr.Button("⬆️ 应用配置档", size="sm", scale=1)
                    prof_del_btn = gr.Button("🗑 删除", size="sm", scale=0,
                                             variant="stop")
                prof_out = gr.HTML("")
                prof_armed = gr.State("")     # 删除两步确认的布防状态

    # =====================================================================
    # 回调
    # =====================================================================

    all_inputs = [
        prompt_audio, text_in, lang_dd, seed_n, dur_sl, tn_cb,
        seg_sl, sil_sl, emo_mode, emo_audio, emo_alpha,
        v0, v1, v2, v3, v4, v5, v6, v7, emo_text, emo_rand,
        do_sample, top_p, top_k, temperature, num_beams,
        rep_pen, len_pen, max_mel,
        # 末尾这几个不是合成参数：3 个「挂哪个 LoRA」+ 3 个「输出后处理」。
        # 它们必须排在 _collect 的 keys 之后，由 on_generate 单独解包（见下）。
        lora_run_dd, lora_ckpt_dd, lora_scale_sl,
        polish_cb, polish_presence, polish_exciter,
    ]

    def _collect(*vals) -> Dict[str, Any]:
        keys = [
            "spk_audio_prompt", "text", "lang", "seed", "duration_factor",
            "text_normalization", "max_text_tokens_per_segment", "interval_silence",
            "emo_control_method", "emo_audio_prompt", "emo_alpha",
            "emo_vec_0", "emo_vec_1", "emo_vec_2", "emo_vec_3",
            "emo_vec_4", "emo_vec_5", "emo_vec_6", "emo_vec_7",
            "emo_text", "use_random",
            "do_sample", "top_p", "top_k", "temperature", "num_beams",
            "repetition_penalty", "length_penalty", "max_mel_tokens",
        ]
        return dict(zip(keys, vals))

    def _mounted_runs() -> set:
        """引擎上**实际**挂着的 adapter 对应的 run 名集合。

        只看真实包装状态（`lora_status`），不看 `lora_want` —— 后者只是
        「上次挂过什么」，引擎在别的页面被卸载/重载之后就过期了。
        """
        try:
            return {t.split(":", 1)[1]
                    for x in eng.lora_status() if x.get("wrapped")
                    for t in (x.get("tags") or [])}
        except Exception:
            return set()

    def _sync_lora(run: str, ckpt: str, scale: float):
        """保证「下拉框选的」与「引擎上挂的」一致。返回 (提示, 是否成功)。

        只在选择变化时才真的重挂：每次生成都重挂要重读一遍 adapter 文件，
        没必要。换 run / 换档位 / 改强度都会触发。

        **挂载失败时返回 ok=False**，让上层中止这次生成。原因：
        用户明确选了某个音色，系统却做不到 —— 这时用底座静默出一版音频
        比直接报错更糟（听起来"像"，但其实是错的模型）。
        """
        act = lora_action(run, _mounted_runs())
        if act == "unmount":
            # 选「不使用 LoRA」= 要纯底座。挂着的就卸掉，与下拉框语义一致。
            try:
                for tag in list(eng.stats.lora_adapters):
                    eng.detach_lora(target=str(tag).split(":", 1)[0])
                ctx.shared["lora_want"] = None
                return "<br>🧬 已按选择卸下 LoRA，本次用<b>纯底座</b>合成", True
            except Exception as e:
                return (f"<br>⚠️ 卸载 LoRA 失败：{type(e).__name__}: {e}", False)
        if act == "none":
            return "", True

        want = (run, ckpt or "best", round(float(scale), 2))
        try:
            if not eng.loaded:
                eng.load()
            tag = MG.mount_run(eng, run, checkpoint=want[1], scale=want[2])
        except Exception as e:
            return (f"LoRA 挂载失败：{type(e).__name__}: {e}"
                    "（已中止本次合成，避免用错模型出声）", False)
        ctx.shared["lora_want"] = want
        return f"<br>🧬 已自动挂载 <code>{tag}</code>（强度 {want[2]}）", True

    @LOG.ui_guard("synthesize.on_generate", slow_sec=1.0)
    def on_generate(*vals, progress=gr.Progress(track_tqdm=False)):
        (*core, lora_run, lora_ckpt, lora_scale,
         pol_on, pol_presence, pol_exciter) = vals
        raw = _collect(*core)
        raw["emo_control_method"] = W.emo_mode_index(raw["emo_control_method"])
        # 记下本次参数快照，供「预设管理」页的「保存当前参数」使用
        ctx.shared["last_gen_values"] = dict(raw)
        req = INF.GenRequest.from_ui(raw, cfg)

        if not eng.loaded:
            gr.Warning("模型尚未加载，正在自动加载…（首次约 20~30 秒）")
            try:
                progress(0.02, desc="正在加载模型…")
                eng.load()
            except EngineError as e:
                gr.Error(str(e))
                return (gr.update(),
                        T.err(f"<b>加载失败</b>：{e}"),
                        ctx.status_html(),
                        _lora_state_html(eng), gr.update())

        # 选的 LoRA 与挂的不一致就先挂上，再合成。挂不上就**中止** ——
        # 用户指定了音色却用底座出声，听起来"像"但其实是错模型，比报错更糟。
        lora_note, lora_ok = _sync_lora(lora_run, lora_ckpt, lora_scale)
        if not lora_ok:
            gr.Error(lora_note.replace("<br>", " "))
            return (gr.update(),
                    T.err(f"<b>未合成</b>：{lora_note}"),
                    ctx.status_html(), _lora_state_html(eng), gr.update())

        try:
            progress(0.05, desc="准备中…")
            res = INF.generate(eng, req, progress=progress)
        except EngineError as e:
            gr.Error(str(e))
            return (gr.update(), T.err(f"<b>合成失败</b>：{e}"),
                    ctx.status_html(), _lora_state_html(eng), gr.update())
        except Exception as e:
            gr.Error(f"{type(e).__name__}: {e}")
            return (gr.update(),
                    T.err(f"<b>合成失败</b>：{type(e).__name__}: {e}"),
                    ctx.status_html(), _lora_state_html(eng), gr.update())

        kw = res["kwargs"]
        rtf = res.get("rtf")
        rows = [
            ("输出文件", f'`{os.path.basename(res["path"])}`'),
            ("音频时长", f'{res["audio_duration"]:.2f} 秒'),
            ("耗时", f'{res["seconds"]:.2f} 秒'),
            ("RTF", f'{rtf:.3f}' if rtf else "-"),
            ("随机种子", f'`{res["seed"]}`（填回种子框可精确复现）'),
            ("情感模式", W.EMO_MODE_LABELS_SHORT[int(raw["emo_control_method"] or 0)]),
        ]
        if kw.get("emo_vector") is not None:
            vec = kw["emo_vector"]
            rows.append(("生效情感向量",
                         "`[" + ", ".join(f"{x:.3f}" for x in vec) + "]`"))
        if kw.get("emo_audio_prompt"):
            rows.append(("情感参考", f'`{os.path.basename(kw["emo_audio_prompt"])}`'))

        # 输出后处理：**另存**一个文件，原始输出保留，方便 A/B 对比
        pol_note = ""
        pol_report = gr.update()
        out_audio_value = res["path"]
        if pol_on:
            pol_path = os.path.join(
                os.path.dirname(res["path"]),
                os.path.splitext(os.path.basename(res["path"]))[0]
                + "_polish.wav")
            pr = AL.polish(res["path"], pol_path,
                           presence_db=float(pol_presence or 0.0),
                           exciter=float(pol_exciter or 0.0))
            if pr.ok:
                out_audio_value = pol_path
                rows = [(k, (f"`{os.path.basename(pol_path)}`"
                             if k == "输出文件" else v))
                        for k, v in rows]
                arrow = ("提亮了" if pr.centroid_after > pr.centroid_before + 30
                         else "基本持平" if abs(pr.centroid_after
                                             - pr.centroid_before) <= 30
                         else "变暗了")
                pol_note = T.tip(
                    f"✨ 已后处理：谱质心 <b>{pr.centroid_before:.0f} → "
                    f"{pr.centroid_after:.0f} Hz</b>（{arrow}）")
                pol_report = gr.update(value=AL.polish_markdown(pr))
            else:
                pol_note = T.warn(f"后处理失败，已用原始输出：{pr.error}")
        else:
            pol_report = gr.update(value=T.hint(
                "未启用输出后处理 —— 输出与模型原始结果一致（最保真）。"))

        info = "\n".join([
            '<div class="ix-tip">✅ <b>合成完成</b></div>',
            "| 项 | 值 |", "|---|---|",
        ] + [f"| {a} | {b} |" for a, b in rows]) + lora_note + pol_note

        if getattr(eng.tts, "low_vram", False) and len(req.text or "") > 40:
            info += T.warn(
                "本次触发了<b>低显存自动分块</b>：文本被按标点粗切成 ≤40 字的块，"
                "逐块独立合成后拼接。块与块之间韵律不接续是正常现象，不是 bug。"
                "缓解办法见「参数手册 → 显存策略 → 低显存自动分块」。")

        return (out_audio_value, info, ctx.status_html(),
                _lora_state_html(eng), pol_report)

    gen_btn.click(
        on_generate, inputs=all_inputs,
        outputs=[out_audio, out_info, sb, lora_state_html, polish_md],
    )

    # ---------- LoRA 选择与挂载 ----------

    @LOG.ui_guard("synthesize.on_lora_run")
    def on_lora_run(run):
        """换 run 时刷新档位列表。"""
        cks = _lora_ckpt_choices(run)
        return gr.update(choices=cks, value=cks[0])

    lora_run_dd.change(on_lora_run, inputs=[lora_run_dd],
                       outputs=[lora_ckpt_dd])

    @LOG.ui_guard("synthesize.on_lora_mount")
    def on_lora_mount(run, ckpt, scale):
        if not run:
            return T.hint("先在上面的下拉框里选一个训练记录。")
        if not eng.loaded:
            gr.Info("引擎未加载，正在自动加载…")
            try:
                eng.load()
            except Exception as e:
                return T.err(f"引擎加载失败：{type(e).__name__}: {e}")
        try:
            tag = MG.mount_run(eng, run, checkpoint=(ckpt or "best"),
                               scale=float(scale))
        except FileNotFoundError as e:
            return T.err(f"找不到 adapter：{e}")
        except Exception as e:
            return T.err(f"挂载失败：{type(e).__name__}: {e}")
        ctx.shared["lora_want"] = (run, ckpt or "best", round(float(scale), 2))
        gr.Info(f"已挂载 {tag}")
        return _lora_state_html(eng)

    lora_mount_btn.click(on_lora_mount,
                         inputs=[lora_run_dd, lora_ckpt_dd, lora_scale_sl],
                         outputs=[lora_state_html])

    @LOG.ui_guard("synthesize.on_lora_unmount")
    def on_lora_unmount():
        tags = list(getattr(eng.stats, "lora_adapters", []) or [])
        if not tags:
            return _lora_state_html(eng)
        for tag in tags:
            try:
                eng.detach_lora(target=str(tag).split(":", 1)[0])
            except Exception as e:
                return T.err(f"卸载 {tag} 失败：{type(e).__name__}: {e}")
        ctx.shared["lora_want"] = None
        gr.Info("已卸载 LoRA，回到纯底座")
        return _lora_state_html(eng)

    lora_unmount_btn.click(on_lora_unmount, inputs=[],
                           outputs=[lora_state_html])

    @LOG.ui_guard("synthesize.on_lora_refresh")
    def on_lora_refresh():
        return (gr.update(choices=_lora_run_choices()),
                _lora_state_html(eng))

    lora_reload_btn.click(on_lora_refresh, inputs=[],
                          outputs=[lora_run_dd, lora_state_html])

    @LOG.ui_guard("synthesize.on_lora_scale")
    def on_lora_scale(scale, run, ckpt):
        """拖动强度旋钮即时生效（已挂载时不用重新挂）。"""
        if not run:
            return gr.update()
        tags = list(getattr(eng.stats, "lora_adapters", []) or [])
        if not tags:
            return gr.update()
        try:
            for tag in tags:
                MG.set_scale(eng, float(scale),
                             target=str(tag).split(":", 1)[0])
        except Exception:
            return gr.update()
        ctx.shared["lora_want"] = (run, ckpt or "best", round(float(scale), 2))
        return _lora_state_html(eng)

    lora_scale_sl.release(on_lora_scale,
                          inputs=[lora_scale_sl, lora_run_dd, lora_ckpt_dd],
                          outputs=[lora_state_html])

    # ---------- 情感模式联动 ----------
    MODE_HINTS = [
        T.tip("<b>模式 0</b>：情感跟随音色参考音频自身。音色还原度最高，最稳的默认值。"
              "此模式下「情感权重」滑块<b>无效</b>（代码里 emo_alpha 被强制为 1.0）。"),
        T.hint("<b>模式 1</b>：音色与情感分别指定。「情感权重」是<b>线性插值系数</b>："
               "<code>out = base + alpha*(emo - base)</code>，0=完全用音色音频自身情感，"
               "1=完全用情感参考音频情感。推荐 0.6~0.8。"),
        T.hint("<b>模式 2</b>：手动指定 8 维向量。「情感权重」此时是<b>向量缩放系数</b>，"
               "等比缩放 8 个值。此模式<b>不能</b>叠加情感参考音频（代码会强制清空它）。"),
        T.warn("<b>模式 3 · 实验功能</b>：用自然语言描述情绪，由 QwenEmotion(Qwen3-0.6B) "
               "转成 8 维向量。官方建议情感权重设 <b>0.6 或更低</b>。"
               "本 UI 采用<b>串行执行</b>：挂载 → 算向量 → 立即卸载 → 再走模式 2 的路径，"
               "避免与主推理抢显存（实测峰值 6.26GB / 8GB）。"),
    ]

    def on_emo_mode(mode):
        m = W.emo_mode_index(mode)
        return (
            gr.update(visible=(m == 1)),      # g_emo_audio
            gr.update(visible=(m == 2)),      # g_emo_vec
            gr.update(visible=(m == 3)),      # g_emo_text
            gr.update(visible=(m in (1, 2, 3))),   # g_emo_alpha
            gr.update(visible=(m in (2, 3))),      # g_emo_rand
            MODE_HINTS[m],
        )

    emo_mode.change(
        on_emo_mode, inputs=[emo_mode],
        outputs=[g_emo_audio, g_emo_vec, g_emo_text, g_emo_alpha, g_emo_rand, emo_hint],
    )

    # ---------- 情感向量实时表 ----------
    vec_components = [v0, v1, v2, v3, v4, v5, v6, v7]

    def on_vec_change(*vals):
        return vec_meter_html(list(vals))

    for c in vec_components:
        c.change(on_vec_change, inputs=vec_components, outputs=[vec_meter])

    def on_vec_zero():
        patch = {f"emo_vec_{i}": 0.0 for i in range(8)}
        return ([gr.update(value=0.0) for _ in range(8)]
                + [vec_meter_html([0.0] * 8), _persist_patch(patch)])

    vec_zero_btn.click(on_vec_zero, inputs=[],
                       outputs=vec_components + [vec_meter, mem_status])

    def on_vec_rand():
        import random
        # 随机 1~2 个维度非零，这是实测最有效的用法（8 维全开会触发限幅压平）
        vec = [0.0] * 8
        for i in random.sample(range(8), k=random.choice([1, 2])):
            vec[i] = round(random.uniform(0.3, 0.9), 2)
        patch = {f"emo_vec_{i}": vec[i] for i in range(8)}
        return ([gr.update(value=x) for x in vec]
                + [vec_meter_html(vec), _persist_patch(patch)])

    vec_rand_btn.click(on_vec_rand, inputs=[],
                       outputs=vec_components + [vec_meter, mem_status])

    # ---------- 模式 3 预览 ----------
    def on_emo_probe(emo_text_val, text_val, dev):
        txt = (emo_text_val or "").strip() or (text_val or "").strip()
        if not txt:
            return T.err("情感描述文本和目标文本都为空，无法推断。")
        if not eng.qwen_available():
            return T.err(
                "QwenEmotion 权重未下载。请到「模型资源」页勾选下载（约 1.14 GB）。")
        eng.qwen_device_pref = dev or "auto"
        try:
            t0 = time.perf_counter()
            vec = eng.text_to_emo_vector(txt)
            dt = time.perf_counter() - t0
        except EngineError as e:
            return T.err(f"推断失败：{e}")
        except Exception as e:
            return T.err(f"推断失败：{type(e).__name__}: {e}")

        rows = "".join(
            f'<tr><td>{lab}</td><td><b>{v:.2f}</b></td>'
            f'<td><div style="height:8px;border-radius:4px;background:#5B6EE1;'
            f'width:{min(100, v/1.2*100):.0f}%"></div></td></tr>'
            for lab, v in zip(EMO_VECTOR_LABELS, vec)
        )
        top = max(range(8), key=lambda i: vec[i])
        return (
            f'<div class="ix-tip">✅ 主导情感：<b>{EMO_VECTOR_LABELS[top]}</b>'
            f'（{vec[top]:.2f}）· 耗时 {dt:.1f}s</div>'
            f'<table class="ix-table" style="width:100%;font-size:12.5px">{rows}</table>'
            + T.hint("这是 QwenEmotion 的<b>原始输出</b>（clamp 到 0~1.2）。"
                     "官方模式 3 会直接用它，<b>不再套 normalize_emo_vec 的偏置</b>。"
                     "点「填入向量框」可切到模式 2 手动微调。")
        )

    emo_probe_btn.click(
        on_emo_probe, inputs=[emo_text, text_in, qwen_dev], outputs=[emo_probe_out],
    )

    def on_clear_emo_cache():
        n = eng.clear_emo_cache()
        gr.Info(f"已清空 {n} 条缓存的情感向量")
        return gr.update()

    emo_cache_btn.click(on_clear_emo_cache, inputs=[], outputs=[emo_probe_out])

    # ---------- 文本 / 分句预览 ----------
    def on_text_change(text, seg_tokens, lang):
        if not text or not text.strip():
            return (gr.update(value=[]),
                    gr.update(value=""),
                    '<span class="ix-chip">Token 0</span>')
        n_tok = eng.count_tokens(text) if eng.loaded else 0
        chip = (f'<span class="ix-chip">字符 <b>{len(text)}</b></span>'
                f'<span class="ix-chip">Token <b>{n_tok}</b></span>')
        if n_tok:
            est = n_tok / 12.0
            chip += f'<span class="ix-chip">预计音频 <b>~{est:.0f}s</b></span>'

        if not eng.loaded:
            return (gr.update(value=[]),
                    T.hint("模型加载后才能预览真实分句结果（需要 tiktoken 编码器）。"
                           "当前只显示字符与 Token 统计。"),
                    chip)

        rows = eng.preview_segments(text, seg_tokens, lang or "ZH")
        data = [[r["index"], r["text"], r["chars"], r["tokens"]] for r in rows]

        note = ""
        if len(rows) > 1:
            note = T.hint(
                f"将分成 <b>{len(rows)}</b> 段独立合成，段间插入 "
                f"静音。段数越多，总耗时越长，且段与段之间起调不连续。")
        if getattr(eng.tts, "low_vram", False) and len(text) > 40:
            note += T.warn(
                "低显存模式：这段文本还会被<b>再切一次</b>（按标点，≤40 字/块），"
                "所以下面的分句数<b>不是</b>最终的合成次数，实际会更多。")
        return gr.update(value=data), note, chip

    for comp in (text_in, seg_sl, lang_dd if cfg.is_v25 else text_in):
        comp.change(on_text_change, inputs=[text_in, seg_sl, lang_dd],
                    outputs=[seg_table, seg_note, token_stat])

    # ---------- 读音纠正 ----------
    def on_pr_check(text, lang):
        return PR.validate_html(text, lang or "ZH")

    text_in.change(on_pr_check, inputs=[text_in, lang_dd], outputs=[pr_issue])
    if cfg.is_v25:
        lang_dd.change(on_pr_check, inputs=[text_in, lang_dd], outputs=[pr_issue])

    def on_pr_query(word):
        w = (word or "").strip()
        if not w:
            gr.Info("请先输入要查询的字或词")
            return gr.update(), gr.update(), []
        rows = PR.candidate_rows(w)
        if not rows:
            return gr.update(value=[]), PR.candidates_note(w), []
        return gr.update(value=rows), PR.candidates_note(w), rows

    pr_query_btn.click(on_pr_query, inputs=[pr_word],
                       outputs=[pr_table, pr_note, pr_rows])
    pr_word.submit(on_pr_query, inputs=[pr_word],
                   outputs=[pr_table, pr_note, pr_rows])

    def on_pr_pick(evt: gr.SelectData, rows):
        """点表格一行 → 把「字」与「读音」填进输入框。

        不直接用 evt.value：Dataframe 的 select 事件只给**单个单元格**的值，
        而我们需要同一行的两列，所以从 pr_rows 里按下标取。
        """
        rows = rows or []
        try:
            r = evt.index[0]
        except Exception:
            return gr.update(), gr.update()
        if r is None or not (0 <= r < len(rows)):
            return gr.update(), gr.update()
        ch, py = rows[r][0], rows[r][1]
        if rows[r][3] == "❌":
            gr.Warning(f"{py} 不在 pinyin.vocab 里，标了大概率无效")
        else:
            gr.Info(f"已选中「{ch}」→ {py}")
        return gr.update(value=ch), gr.update(value=py)

    pr_table.select(on_pr_pick, inputs=[pr_rows], outputs=[pr_word, pr_pron])

    def on_pr_apply(text, word, pron, which, lang):
        w = (word or "").strip()
        p = (pron or "").strip()
        if not w or not p:
            gr.Warning("「字」和「发音」都要填（可先点表格选一个）")
            return gr.update(), gr.update()
        try:
            n = max(0, int(which or 1) - 1)
        except Exception:
            n = 0
        new, ok = PR.apply_annotation(text or "", w, p, which=n)
        if not ok:
            gr.Warning(f"文本里找不到未标注的「{w}」，或发音为空")
            return gr.update(), gr.update()
        if new == text:
            gr.Info(f"「{w}」已经标过 {p.upper()} 了，不重复插入")
        else:
            gr.Info(f"已插入 <{w}|{p.upper()}>")
        return gr.update(value=new), PR.validate_html(new, lang or "ZH")

    pr_apply_btn.click(on_pr_apply,
                       inputs=[text_in, pr_word, pr_pron, pr_which, lang_dd],
                       outputs=[text_in, pr_issue])

    def on_pr_preview(text, lang, tn):
        if not (text or "").strip():
            return "_文本为空，无法预览。_"
        # 首次调用要初始化 TextNormalizer + tiktoken，几秒级，不占显存
        gr.Info("正在跑官方文本处理链路…")
        return PR.preview_markdown(text, lang or "ZH", normalize=bool(tn))

    pr_preview_btn.click(on_pr_preview, inputs=[text_in, lang_dd, tn_cb],
                         outputs=[pr_out])

    def on_pr_strip(text):
        n = len(PR.parse(text or ""))
        if not n:
            gr.Info("文本里没有标注")
            return gr.update()
        gr.Info(f"已去掉 {n} 条标注（保留原字）")
        return gr.update(value=PR.strip_annotations(text or ""))

    pr_strip_btn.click(on_pr_strip, inputs=[text_in], outputs=[text_in])

    # ---------- 音色库联动 ----------
    def on_voice_select(name):
        if not name:
            return gr.update(), "", gr.update()
        e = voice_bank.get(name)
        if e is None:
            return gr.update(), T.err(f"音色 `{name}` 不存在"), gr.update()
        badge = {"优秀": "🟢", "良好": "🟢", "可用": "🟡",
                 "勉强": "🟠", "不建议使用": "🔴"}.get(e.grade, "·")
        html = (
            f'<div class="ix-hint">{badge} <b>{e.name}</b> · 评分 {e.score}/100（{e.grade}）'
            f' · {e.duration:.2f}s · {e.sample_rate}Hz · SNR {e.snr_db:.1f}dB'
            + (f'<br>备注：{e.note}' if e.note else "")
            + (f'<br>标签：{", ".join(e.tags)}' if e.tags else "")
            + "</div>")
        if e.report and e.report.get("issues"):
            html += T.warn("体检遗留问题：" + "；".join(e.report["issues"][:2]))
        # 选择音色会程序化设置参考音频（不触发 change），记忆在这里补
        mem = _persist_patch({"voice_name": name, "prompt_audio": e.audio_path})
        return gr.update(value=e.audio_path), html, mem

    voice_dd.change(on_voice_select, inputs=[voice_dd],
                    outputs=[prompt_audio, voice_info, mem_status])

    def on_voice_reload():
        return gr.update(choices=[""] + voice_bank.names())

    voice_reload_btn.click(on_voice_reload, inputs=[], outputs=[voice_dd])
    voice_refresh_btn.click(on_voice_reload, inputs=[], outputs=[voice_dd])

    def on_voice_detail(name):
        if not name:
            gr.Info("请先在列表里选择一个音色")
            return gr.update()
        return voice_bank.detail_markdown(name)

    voice_detail_btn.click(on_voice_detail, inputs=[voice_dd], outputs=[voice_info])

    def on_to_lab():
        gr.Info("已切换到「参考音频工作台」，请把当前音频上传到那里做体检与增强。")
        return gr.update()

    to_lab_btn.click(on_to_lab, inputs=[], outputs=[voice_info])

    # ---------- 采样档位预设 ----------
    SAMPLING_PRESETS = {
        "fast": dict(do_sample=True, temperature=0.7, top_p=0.7, top_k=20,
                     num_beams=1, repetition_penalty=10.0, length_penalty=0.0,
                     max_mel_tokens=1200),
        "balanced": dict(do_sample=True, temperature=0.8, top_p=0.8, top_k=30,
                         num_beams=3, repetition_penalty=10.0, length_penalty=0.0,
                         max_mel_tokens=1500),
        "quality": dict(do_sample=True, temperature=0.75, top_p=0.75, top_k=40,
                        num_beams=5, repetition_penalty=10.0, length_penalty=0.0,
                        max_mel_tokens=1815),
    }
    sampling_components = [do_sample, temperature, top_p, top_k, num_beams,
                           rep_pen, len_pen, max_mel]

    def apply_preset(which):
        d = SAMPLING_PRESETS[which]
        ups = [gr.update(value=d[k]) for k in
               ("do_sample", "temperature", "top_p", "top_k", "num_beams",
                "repetition_penalty", "length_penalty", "max_mel_tokens")]
        note = {
            "fast": "⚡ 快速档：num_beams=1（纯采样），比均衡档快约 2~3 倍。"
                    "适合快速试听、批量草稿。质量略降但通常可接受。",
            "balanced": "⚖️ 均衡档 = 官方默认值。质量/速度的平衡点，推荐日常使用。",
            "quality": "💎 质量档：num_beams=5 + max_mel_tokens=1815（上限）。"
                       "更稳、更少崩坏，但耗时明显增加，且 KV cache 显存翻近两倍 —— "
                       "8GB 卡上请留意 OOM。",
        }[which]
        gr.Info(note[:60])
        # 快速档按钮程序化设置 8 个采样控件（不触发 change），记忆在这里补
        return ups + [_persist_patch(dict(d))]

    preset_fast.click(lambda: apply_preset("fast"), inputs=[],
                      outputs=sampling_components + [mem_status])
    preset_bal.click(lambda: apply_preset("balanced"), inputs=[],
                     outputs=sampling_components + [mem_status])
    preset_hq.click(lambda: apply_preset("quality"), inputs=[],
                    outputs=sampling_components + [mem_status])

    # ---------- 引擎控制 ----------
    def engine_html():
        s = eng.stats
        if not s.loaded:
            return T.warn("模型<b>未加载</b>。首次生成时会自动加载（约 20~30 秒），"
                          "也可以点下面的按钮提前加载。")
        return (
            f'<div class="ix-tip">🟢 <b>已加载</b> · 耗时 {s.load_seconds:.1f}s · '
            f'显存 {s.vram_alloc_gb:.2f} GB（峰值 {s.vram_peak_gb:.2f} GB）· '
            f'已合成 {s.infer_count} 次</div>'
            + ("".join(T.hint(n) for n in s.notes) if s.notes else "")
        )

    @LOG.ui_guard("synthesize.on_load", slow_sec=5.0)
    def on_load():
        try:
            eng.load()
            gr.Info(f"模型加载完成，耗时 {eng.stats.load_seconds:.1f}s")
        except EngineError as e:
            gr.Error(str(e))
        except Exception as e:
            gr.Error(f"{type(e).__name__}: {e}")
        return engine_html(), ctx.status_html(), _lora_state_html(eng)

    @LOG.ui_guard("synthesize.on_unload")
    def on_unload():
        try:
            eng.unload()
        except EngineError as e:
            gr.Error(str(e))
        else:
            gr.Info("模型已卸载，显存已归还")
        # 卸载会清空 stats.lora_adapters（引擎上的 LoRA 随之消失），
        # 所以「想挂的那个」的标记也要一起失效，否则下次生成会以为还挂着。
        ctx.shared["lora_want"] = None
        return engine_html(), ctx.status_html(), _lora_state_html(eng)

    load_btn.click(on_load, inputs=[],
                   outputs=[engine_state, sb, lora_state_html])
    unload_btn.click(on_unload, inputs=[],
                     outputs=[engine_state, sb, lora_state_html])

    def on_clear_ref_cache():
        if not eng.loaded:
            gr.Warning("模型未加载，无缓存可清")
            return engine_html()
        eng.clear_reference_cache()
        gr.Info("参考音频缓存已清空。下次合成会重新计算声纹（首句会变慢）")
        return engine_html()

    clear_cache_btn.click(on_clear_ref_cache, inputs=[], outputs=[engine_state])

    def on_clear_emo_cache2():
        n = eng.clear_emo_cache()
        gr.Info(f"已清空 {n} 条情感向量缓存")
        return gr.update()

    clear_emo_btn.click(on_clear_emo_cache2, inputs=[], outputs=[engine_state])

    def on_qwen_dev(dev):
        eng.qwen_device_pref = dev
        return gr.update()

    qwen_dev.change(on_qwen_dev, inputs=[qwen_dev], outputs=[emo_probe_out])

    # ---------- 示例 ----------
    def on_example(choice):
        if not choice:
            return gr.update(), gr.update(), gr.update()
        idx = [f"{t}｜{lang}" for t, _x, lang in EXAMPLE_TEXTS].index(choice)
        _t, text, lang = EXAMPLE_TEXTS[idx]
        # 示例会改 lang，而 gr.update 不触发 change 事件，记忆要在这里补
        mem = _persist_patch({"lang": lang}) if cfg.is_v25 else gr.update()
        return (gr.update(value=text),
                gr.update(value=lang) if cfg.is_v25 else gr.update(), mem)

    ex_dd.change(on_example, inputs=[ex_dd],
                 outputs=([text_in, lang_dd, mem_status] if cfg.is_v25
                          else [text_in, text_in, mem_status]))

    # =====================================================================
    # 参数记忆：任何改动自动落盘（重启后由 render 开头恢复）
    # =====================================================================
    # remember_cb 也在这份列表里：拨动开关本身就是一次「改动」。
    remember_comps = [
        voice_dd, prompt_audio, lang_dd, dur_sl, seed_n, tn_cb,
        seg_sl, sil_sl, emo_mode, emo_audio, emo_alpha,
        v0, v1, v2, v3, v4, v5, v6, v7, emo_text, emo_rand,
        do_sample, temperature, top_p, top_k, num_beams,
        rep_pen, len_pen, max_mel,
        lora_run_dd, lora_ckpt_dd, lora_scale_sl,
        polish_cb, polish_presence, polish_exciter, remember_cb,
    ]

    def _live_snapshot(*vals) -> Dict[str, Any]:
        """remember_comps 的当前值 → 语义键字典（含音频路径与记忆开关）。"""
        (voice_name, pa, lang, dur, seed, tn, seg, sil, emo_lab, ea, alpha,
         vv0, vv1, vv2, vv3, vv4, vv5, vv6, vv7, etxt, erand,
         dsamp, temp, topp, topk, beams, rpen, lpen, mmel,
         lrun, lckpt, lscale, pon, ppre, pexc, remember_on) = vals
        return {
            "voice_name": voice_name or "",
            "prompt_audio": pa, "emo_audio": ea,
            "lang": lang, "duration_factor": dur, "seed": seed,
            "text_normalization": tn,
            "max_text_tokens_per_segment": seg, "interval_silence": sil,
            "emo_mode_label": emo_lab,
            "emo_mode_index": W.emo_mode_index(emo_lab),
            "emo_alpha": alpha,
            "emo_vec_0": vv0, "emo_vec_1": vv1, "emo_vec_2": vv2,
            "emo_vec_3": vv3, "emo_vec_4": vv4, "emo_vec_5": vv5,
            "emo_vec_6": vv6, "emo_vec_7": vv7,
            "emo_text": etxt, "use_random": erand,
            "do_sample": dsamp, "temperature": temp, "top_p": topp,
            "top_k": topk, "num_beams": beams, "repetition_penalty": rpen,
            "length_penalty": lpen, "max_mel_tokens": mmel,
            "lora_run": lrun or "", "lora_ckpt": lckpt or "best",
            "lora_scale": lscale, "polish_on": pon,
            "polish_presence": ppre, "polish_exciter": pexc,
            "_remember": bool(remember_on),
        }

    # 「清除记忆」重置 34 个控件会触发一轮 change 回声；两秒内不落盘，
    # 否则刚清掉的记忆立刻被（默认值）写回，界面上看起来像「清除不掉」。
    _suppress_save_until = {"t": 0.0}

    def _persist(live: Dict[str, Any]) -> Any:
        """写共享快照 + 按开关落盘。返回记忆状态行的 gr.update。"""
        ctx.shared["syn_live_values"] = dict(live)
        remember = bool(live.get("_remember", True))
        prev_flag = bool(ctx.shared.get("_remember_flag", True))
        ctx.shared["_remember_flag"] = remember

        if not remember:
            # 关掉开关这个动作本身要落盘一次（否则重启后开关自己变回开），
            # 之后的参数改动不再写。
            if prev_flag:
                SS.save_state(
                    {k: v for k, v in live.items()
                     if k not in ("prompt_audio", "emo_audio")},
                    prompt_audio=live.get("prompt_audio"),
                    emo_audio=live.get("emo_audio"))
            return gr.update(
                value='<span class="ix-chip"><span class="ix-dot warn"></span>'
                      '参数记忆 · 已关闭（重启后不恢复）</span>')

        if time.time() < _suppress_save_until["t"]:
            return gr.update()          # 清除记忆后的重置回声，保持「已清除」显示

        ok = SS.save_state(
            {k: v for k, v in live.items()
             if k not in ("prompt_audio", "emo_audio")},
            prompt_audio=live.get("prompt_audio"),
            emo_audio=live.get("emo_audio"))
        if ok:
            return gr.update(
                value='<span class="ix-chip"><span class="ix-dot ok"></span>'
                      f'参数记忆 · 已记住（{time.strftime("%H:%M:%S")}）</span>')
        return gr.update(
            value='<span class="ix-chip"><span class="ix-dot err"></span>'
                  '参数记忆 · 写入失败（目录只读？）</span>')

    def _persist_patch(patch: Dict[str, Any]) -> Any:
        """程序化设置控件的回调用这个补一次落盘。

        gr.update 设值**不会**触发 change 事件（音色库选择 / 示例 / 快速档 /
        配置档应用都是程序化设值），所以它们的手动补一条；先与最近快照合并，
        保证不把其它字段的记忆冲掉。
        """
        live = dict(ctx.shared.get("syn_live_values") or {})
        live.update(patch)
        live.setdefault("_remember", _remember)
        return _persist(live)

    @LOG.ui_guard("synthesize.on_param_change")
    def on_param_change(*vals):
        return _persist(_live_snapshot(*vals))

    # 单条依赖挂 35 个触发器：任何一个控件变化都整表落盘。
    # show_progress="hidden"：这是高频事件，不能每次拖完滑杆都闪加载动画。
    gr.on(triggers=[c.change for c in remember_comps],
          fn=on_param_change, inputs=remember_comps, outputs=[mem_status],
          show_progress="hidden")

    # ---------- 清除记忆并恢复默认 ----------
    reset_targets = [
        prompt_audio, emo_mode, emo_audio, lang_dd, dur_sl,
        emo_alpha, emo_rand, emo_text,
        v0, v1, v2, v3, v4, v5, v6, v7,
        do_sample, temperature, top_p, top_k, num_beams,
        rep_pen, len_pen, max_mel, seg_sl,
        sil_sl, seed_n, tn_cb, voice_dd,
        lora_run_dd, lora_ckpt_dd, lora_scale_sl,
        polish_cb, polish_presence, polish_exciter,
    ]

    @LOG.ui_guard("synthesize.on_forget")
    def on_forget():
        SS.forget()
        ctx.shared["syn_live_values"] = {}
        ctx.shared["_remember_flag"] = True
        _suppress_save_until["t"] = time.time() + 2.0
        gr.Info("已清除参数记忆，界面已恢复默认值")
        ups = [
            gr.update(value=None),                        # 参考音频
            gr.update(value=W.EMO_MODE_LABELS[0]),        # 情感模式
            gr.update(value=None),                        # 情感参考音频
            gr.update(value=P.get("lang").default if cfg.is_v25 else None),
            gr.update(value=float(P.get("duration_factor").default)),
            gr.update(value=float(P.get("emo_alpha").default)),
            gr.update(value=bool(P.get("use_random").default)),
            gr.update(value=P.get("emo_text").default or ""),
            *[gr.update(value=float(P.get(f"emo_vec_{i}").default))
              for i in range(8)],
            gr.update(value=bool(P.get("do_sample").default)),
            gr.update(value=float(P.get("temperature").default)),
            gr.update(value=float(P.get("top_p").default)),
            gr.update(value=int(P.get("top_k").default)),
            gr.update(value=int(P.get("num_beams").default)),
            gr.update(value=float(P.get("repetition_penalty").default)),
            gr.update(value=float(P.get("length_penalty").default)),
            gr.update(value=int(P.get("max_mel_tokens").default)),
            gr.update(value=int(P.get("max_text_tokens_per_segment").default)),
            gr.update(value=int(P.get("interval_silence").default)),
            gr.update(value=P.get("seed").default),
            gr.update(value=bool(P.get("text_normalization").default)),
            gr.update(value=""),                          # 音色库下拉
            gr.update(value=""),                          # LoRA run
            gr.update(value="best"),                      # LoRA 档位
            gr.update(value=1.0),                         # LoRA 强度
            gr.update(value=True),                        # 后处理开关
            gr.update(value=2.5),                         # presence
            gr.update(value=0.08),                        # exciter
        ]
        assert len(ups) == len(reset_targets)
        mem = gr.update(
            value='<span class="ix-chip"><span class="ix-dot idle"></span>'
                  '参数记忆 · 已清除，回到默认</span>')
        return [mem] + ups

    forget_btn.click(on_forget, inputs=[],
                     outputs=[mem_status] + reset_targets)

    # ---------- 配置档（官方预设系统，与「💾 预设」页互通） ----------
    def on_prof_reload():
        return gr.update(choices=_prof_choices())

    prof_reload_btn.click(on_prof_reload, inputs=[], outputs=[prof_dd])

    @LOG.ui_guard("synthesize.on_prof_save")
    def on_prof_save(name):
        """把当前参数存为命名配置档（官方 preset 格式，音频复制入档）。"""
        name = (name or "").strip()
        if not name:
            gr.Warning("请先填写配置档名称")
            return gr.update(), T.err("名称不能为空。")
        if save_preset is None:
            return gr.update(), T.err("官方 presets 模块不可用，无法保存。")
        live = dict(ctx.shared.get("syn_live_values") or {})
        data = SS.live_to_preset_data(live)
        try:
            save_preset(name, data,
                        prompt_audio=live.get("prompt_audio"),
                        emo_audio=live.get("emo_audio"))
        except Exception as e:
            return gr.update(), T.err(f"保存失败：{type(e).__name__}: {e}")
        final = safe_preset_name(name)
        gr.Info(f"已保存配置档「{final}」")
        msg = T.tip(f"✅ 配置档 <b>{final}</b> 已保存"
                    + ("（名称中的非法字符已清洗）" if final != name else "")
                    + "，在「💾 预设」页能看到同一份。")
        return gr.update(choices=_prof_choices(), value=final), msg

    prof_save_btn.click(on_prof_save, inputs=[prof_name],
                        outputs=[prof_dd, prof_out])

    # 应用目标与「预设页 → 应用到合成页」的 25 个严格同序（见 render 末尾登记）
    prof_apply_targets = [
        prompt_audio, emo_mode, emo_audio, lang_dd, dur_sl,
        emo_alpha, emo_rand, emo_text,
        v0, v1, v2, v3, v4, v5, v6, v7,
        do_sample, temperature, top_p, top_k, num_beams,
        rep_pen, len_pen, max_mel, seg_sl,
    ]

    @LOG.ui_guard("synthesize.on_prof_apply")
    def on_prof_apply(name):
        if not name:
            gr.Warning("请先选择一个配置档")
            return [gr.update()] * 25 + [gr.update()]
        data = load_preset(name)
        if data is None:
            gr.Error(f"配置档 {name} 不存在")
            return [gr.update()] * 25 + [gr.update()]

        adv = data.get("advanced_params") or {}

        def val(k, dflt):
            v = adv.get(k, data.get(k))
            return dflt if v is None else v

        mode = max(0, min(3, int(data.get("emo_control_method", 0) or 0)))
        vec = list(data.get("emo_vector") or [0.0] * 8)
        vec = (vec + [0.0] * 8)[:8]
        pa = data.get("prompt_audio")
        if pa and not os.path.isfile(pa):
            gr.Warning("配置档里的音色参考音频已丢失，跳过该项")
            pa = None
        ea = data.get("emo_audio")
        if ea and not os.path.isfile(ea):
            gr.Warning("配置档里的情感参考音频已丢失，跳过该项")
            ea = None

        patch = {
            "prompt_audio": pa, "emo_audio": ea,
            "emo_mode_label": W.EMO_MODE_LABELS[mode],
            "emo_mode_index": mode,
            "emo_alpha": float(val("emo_alpha", 0.65)),
            "use_random": bool(val("use_random", False)),
            "emo_text": data.get("emo_text", "") or "",
            "lang": val("lang", "ZH"),
            "duration_factor": float(val("duration_factor", 1.0)),
            "do_sample": bool(val("do_sample", True)),
            "temperature": float(val("temperature", 0.8)),
            "top_p": float(val("top_p", 0.8)),
            "top_k": int(val("top_k", 30)),
            "num_beams": int(val("num_beams", 3)),
            "repetition_penalty": float(val("repetition_penalty", 10.0)),
            "length_penalty": float(val("length_penalty", 0.0)),
            "max_mel_tokens": int(val("max_mel_tokens", 1500)),
            "max_text_tokens_per_segment":
                int(val("max_text_tokens_per_segment", 120)),
        }
        for i in range(8):
            patch[f"emo_vec_{i}"] = float(vec[i])

        ups = [
            gr.update(value=pa),
            gr.update(value=W.EMO_MODE_LABELS[mode]),
            gr.update(value=ea),
            gr.update(value=val("lang", "ZH") if cfg.is_v25 else None),
            gr.update(value=float(val("duration_factor", 1.0))),
            gr.update(value=float(val("emo_alpha", 0.65))),
            gr.update(value=bool(val("use_random", False))),
            gr.update(value=data.get("emo_text", "") or ""),
        ] + [gr.update(value=float(x)) for x in vec] + [
            gr.update(value=bool(val("do_sample", True))),
            gr.update(value=float(val("temperature", 0.8))),
            gr.update(value=float(val("top_p", 0.8))),
            gr.update(value=int(val("top_k", 30))),
            gr.update(value=int(val("num_beams", 3))),
            gr.update(value=float(val("repetition_penalty", 10.0))),
            gr.update(value=float(val("length_penalty", 0.0))),
            gr.update(value=int(val("max_mel_tokens", 1500))),
            gr.update(value=int(val("max_text_tokens_per_segment", 120))),
        ]
        gr.Info(f"已应用配置档「{name}」")
        return ups + [_persist_patch(patch)]

    prof_apply_btn.click(on_prof_apply, inputs=[prof_dd],
                         outputs=prof_apply_targets + [mem_status])

    def on_prof_delete(name, armed):
        """两步确认：第一次点击只布防，第二次才真删。"""
        if not name:
            gr.Warning("请先选择配置档")
            return "", gr.update(), gr.update()
        if armed != name:
            gr.Warning(f"再点一次「🗑 删除」确认删除「{name}」")
            return (name,
                    T.warn(f"将删除配置档 <b>{name}</b> —— 再点一次确认。"),
                    gr.update())
        if delete_preset(name):
            gr.Info(f"已删除配置档「{name}」")
            msg = T.tip(f"✅ 已删除 <b>{name}</b>")
        else:
            msg = T.err(f"删除失败：{name} 不存在")
        return "", msg, gr.update(choices=_prof_choices(), value="")

    prof_del_btn.click(on_prof_delete, inputs=[prof_dd, prof_armed],
                       outputs=[prof_armed, prof_out, prof_dd])

    # ---------- 页面加载 ----------
    # 不在这里绑定 demo.load（Tab 拿不到 Blocks 对象），
    # 而是把回调与输出列表交给 app.py 统一绑定。
    def on_page_load():
        return (engine_html(), ctx.status_html(),
                gr.update(choices=[""] + voice_bank.names()),
                gr.update(choices=_lora_run_choices()),
                _lora_state_html(eng),
                gr.update(choices=_prof_choices()))

    # 登记「预设可写入的控件」有序列表，供「预设管理」Tab 的
    # 「应用到合成页」按钮使用。顺序必须与 presets.on_apply 返回的
    # 25 个 gr.update 严格一致：
    #   0-3   prompt_audio / emo_mode / emo_audio / lang
    #   4-7   duration_factor / emo_alpha / use_random / emo_text
    #   8-15  emo_vec_0..7
    #   16-24 do_sample / temperature / top_p / top_k / num_beams /
    #         repetition_penalty / length_penalty / max_mel_tokens / seg_tokens
    ctx.shared["synthesize_apply_targets"] = [
        prompt_audio, emo_mode, emo_audio, lang_dd,
        dur_sl, emo_alpha, emo_rand, emo_text,
        v0, v1, v2, v3, v4, v5, v6, v7,
        do_sample, temperature, top_p, top_k, num_beams,
        rep_pen, len_pen, max_mel, seg_sl,
    ]

    # 合成页把最近一次生成的参数快照存起来，供预设页「保存当前参数」使用。
    # Gradio 服务端无法主动读控件值，所以在 on_generate 里顺带写入。
    return {
        "page_load": (on_page_load,
                      [engine_state, sb, voice_dd, lora_run_dd, lora_state_html,
                       prof_dd]),
        "components": {
            "prompt_audio": prompt_audio,
            "text": text_in,
            "lang": lang_dd,
            "emo_mode": emo_mode,
            "emo_audio": emo_audio,
            "out_audio": out_audio,
            "engine_state": engine_state,
            "lora_run": lora_run_dd,
            "lora_state": lora_state_html,
            "mem_status": mem_status,
        },
    }
