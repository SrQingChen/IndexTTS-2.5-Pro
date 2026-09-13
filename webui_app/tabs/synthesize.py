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
from webui_app.services import inference as INF
from webui_app.services import pronunciation as PR
from webui_app.services import voice_bank
from webui_app.services.engine import EngineError

EXAMPLE_TEXTS = [
    ("中文 · 日常", "大家好，欢迎使用 IndexTTS 二点五，这是一段用于测试的中文语音。", "ZH"),
    ("中文 · 多音字标注", "他在银<行|XING2>里<行|HANG2>走了半天，发现这笔业务办不<行|HANG2>。", "ZH"),
    ("中文 · 情感", "快躲起来！是他要来了！他要来抓我们了！", "ZH"),
    ("英文 · 音素标注", "He had a <minute|M IH1 . N AH0 T> to examine the <minute|M AY0 . N UW1 T> details.", "EN"),
    ("英文 · 日常", "IndexTTS can clone a voice from just a few seconds of reference audio.", "EN"),
    ("日语 · 假名标注", "彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた。", "JA"),
]


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
                        choices=[""] + voice_bank.names(), value="",
                        label="从音色库载入", scale=3,
                        info="在「参考音频工作台」体检并增强后入库的素材",
                        allow_custom_value=False,
                    )
                    voice_reload_btn = gr.Button("↻", scale=0, variant="secondary",
                                                 size="sm")
                    voice_refresh_btn = gr.Button("刷新列表", scale=1, size="sm")
                prompt_audio = W.make_component("spk_audio_prompt", label="")
                with gr.Row():
                    voice_detail_btn = gr.Button("查看该音色体检报告", size="sm", scale=1)
                    to_lab_btn = gr.Button("→ 送去工作台优化", size="sm", scale=1)
                voice_info = gr.HTML("")

            # ---------- 文本 ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("文本与语言", "📝",
                                  "支持 <文字|发音> 标注：中文拼音 / 英文 CMU 音素 / 日语假名。"
                                  "官方会截断参考音频到前 15 秒，但文本长度只受分句参数约束。"))
                text_in = W.make_component("text", value="", placeholder="请输入要合成的文本…")
                with gr.Row():
                    if cfg.is_v25:
                        lang_dd = W.make_component("lang", scale=1)
                    else:
                        lang_dd = gr.State(value=None)
                    dur_sl = W.make_component("duration_factor", scale=2)
                    seed_n = W.make_component("seed", scale=1)
                with gr.Row():
                    tn_cb = W.make_component("text_normalization", scale=1)
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
                seg_sl = W.make_component("max_text_tokens_per_segment")
                sil_sl = W.make_component("interval_silence")

            # ---------- 情感控制 ----------
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("情感控制", "🎭",
                                  "情感只注入 GPT(T2S)，完全不进 CFM(S2M) —— "
                                  "这就是「音色-情感解耦」：改情感不动音色。"))
                emo_mode = W.make_component("emo_control_method",
                                            value=W.EMO_MODE_LABELS[0])
                emo_hint = gr.HTML("")

                with gr.Group(visible=False) as g_emo_audio:
                    emo_audio = W.make_component("emo_audio_prompt")

                with gr.Group(visible=False) as g_emo_vec:
                    gr.HTML('<div class="ix-hint">顺序固定为 '
                            '[喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静]。'
                            '每维有内置偏置系数，且 8 维总和超过 0.8 会被静默压缩。</div>')
                    with gr.Row(elem_classes=["ix-emo-grid"]):
                        with gr.Column():
                            v0 = W.make_component("emo_vec_0")
                            v1 = W.make_component("emo_vec_1")
                            v2 = W.make_component("emo_vec_2")
                            v3 = W.make_component("emo_vec_3")
                        with gr.Column():
                            v4 = W.make_component("emo_vec_4")
                            v5 = W.make_component("emo_vec_5")
                            v6 = W.make_component("emo_vec_6")
                            v7 = W.make_component("emo_vec_7")
                    vec_meter = gr.HTML(vec_meter_html([0.0] * 8))
                    with gr.Row():
                        vec_zero_btn = gr.Button("全部归零", size="sm", scale=1)
                        vec_rand_btn = gr.Button("随机一组", size="sm", scale=1)

                with gr.Group(visible=False) as g_emo_text:
                    emo_text = W.make_component(
                        "emo_text", placeholder="例如：委屈巴巴 / 危险在悄悄逼近")
                    with gr.Row():
                        emo_probe_btn = gr.Button("预览情感向量（不合成）", size="sm", scale=2)
                        emo_cache_btn = gr.Button("清空向量缓存", size="sm", scale=1)
                    emo_probe_out = gr.HTML("")

                with gr.Row(visible=False) as g_emo_alpha:
                    emo_alpha = W.make_component("emo_alpha", scale=3)
                with gr.Row(visible=False) as g_emo_rand:
                    emo_rand = W.make_component("use_random", scale=1)

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
                    do_sample = W.make_component("do_sample", scale=1)
                    temperature = W.make_component("temperature", scale=2)
                with gr.Row():
                    top_p = W.make_component("top_p", scale=1)
                    top_k = W.make_component("top_k", scale=1)
                num_beams = W.make_component("num_beams")
                with gr.Row():
                    rep_pen = W.make_component("repetition_penalty", scale=1)
                    len_pen = W.make_component("length_penalty", scale=1)
                max_mel = W.make_component("max_mel_tokens")
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

    # =====================================================================
    # 回调
    # =====================================================================

    all_inputs = [
        prompt_audio, text_in, lang_dd, seed_n, dur_sl, tn_cb,
        seg_sl, sil_sl, emo_mode, emo_audio, emo_alpha,
        v0, v1, v2, v3, v4, v5, v6, v7, emo_text, emo_rand,
        do_sample, top_p, top_k, temperature, num_beams,
        rep_pen, len_pen, max_mel,
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

    def on_generate(*vals, progress=gr.Progress(track_tqdm=False)):
        raw = _collect(*vals)
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
                        ctx.status_html())

        try:
            progress(0.05, desc="准备中…")
            res = INF.generate(eng, req, progress=progress)
        except EngineError as e:
            gr.Error(str(e))
            return gr.update(), T.err(f"<b>合成失败</b>：{e}"), ctx.status_html()
        except Exception as e:
            gr.Error(f"{type(e).__name__}: {e}")
            return (gr.update(),
                    T.err(f"<b>合成失败</b>：{type(e).__name__}: {e}"),
                    ctx.status_html())

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

        info = "\n".join([
            '<div class="ix-tip">✅ <b>合成完成</b></div>',
            "| 项 | 值 |", "|---|---|",
        ] + [f"| {a} | {b} |" for a, b in rows])

        if getattr(eng.tts, "low_vram", False) and len(req.text or "") > 40:
            info += T.warn(
                "本次触发了<b>低显存自动分块</b>：文本被按标点粗切成 ≤40 字的块，"
                "逐块独立合成后拼接。块与块之间韵律不接续是正常现象，不是 bug。"
                "缓解办法见「参数手册 → 显存策略 → 低显存自动分块」。")

        return res["path"], info, ctx.status_html()

    gen_btn.click(
        on_generate, inputs=all_inputs,
        outputs=[out_audio, out_info, sb],
    )

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
        return [gr.update(value=0.0) for _ in range(8)] + [vec_meter_html([0.0] * 8)]

    vec_zero_btn.click(on_vec_zero, inputs=[], outputs=vec_components + [vec_meter])

    def on_vec_rand():
        import random
        # 随机 1~2 个维度非零，这是实测最有效的用法（8 维全开会触发限幅压平）
        vec = [0.0] * 8
        for i in random.sample(range(8), k=random.choice([1, 2])):
            vec[i] = round(random.uniform(0.3, 0.9), 2)
        return [gr.update(value=x) for x in vec] + [vec_meter_html(vec)]

    vec_rand_btn.click(on_vec_rand, inputs=[], outputs=vec_components + [vec_meter])

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
            return gr.update(), ""
        e = voice_bank.get(name)
        if e is None:
            return gr.update(), T.err(f"音色 `{name}` 不存在")
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
        return gr.update(value=e.audio_path), html

    voice_dd.change(on_voice_select, inputs=[voice_dd],
                    outputs=[prompt_audio, voice_info])

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
        return ups

    preset_fast.click(lambda: apply_preset("fast"), inputs=[], outputs=sampling_components)
    preset_bal.click(lambda: apply_preset("balanced"), inputs=[], outputs=sampling_components)
    preset_hq.click(lambda: apply_preset("quality"), inputs=[], outputs=sampling_components)

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

    def on_load():
        try:
            eng.load()
            gr.Info(f"模型加载完成，耗时 {eng.stats.load_seconds:.1f}s")
        except EngineError as e:
            gr.Error(str(e))
        except Exception as e:
            gr.Error(f"{type(e).__name__}: {e}")
        return engine_html(), ctx.status_html()

    def on_unload():
        eng.unload()
        gr.Info("模型已卸载，显存已归还")
        return engine_html(), ctx.status_html()

    load_btn.click(on_load, inputs=[], outputs=[engine_state, sb])
    unload_btn.click(on_unload, inputs=[], outputs=[engine_state, sb])

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
            return gr.update(), gr.update()
        idx = [f"{t}｜{lang}" for t, _x, lang in EXAMPLE_TEXTS].index(choice)
        _t, text, lang = EXAMPLE_TEXTS[idx]
        return gr.update(value=text), gr.update(value=lang) if cfg.is_v25 else gr.update()

    ex_dd.change(on_example, inputs=[ex_dd],
                 outputs=[text_in, lang_dd] if cfg.is_v25 else [text_in, text_in])

    # ---------- 页面加载 ----------
    # 不在这里绑定 demo.load（Tab 拿不到 Blocks 对象），
    # 而是把回调与输出列表交给 app.py 统一绑定。
    def on_page_load():
        return (engine_html(), ctx.status_html(),
                gr.update(choices=[""] + voice_bank.names()))

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
        "page_load": (on_page_load, [engine_state, sb, voice_dd]),
        "components": {
            "prompt_audio": prompt_audio,
            "text": text_in,
            "lang": lang_dd,
            "emo_mode": emo_mode,
            "emo_audio": emo_audio,
            "out_audio": out_audio,
            "engine_state": engine_state,
        },
    }
