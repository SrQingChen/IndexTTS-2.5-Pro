"""Tab：预设管理。

预设 = 一整套参数快照（情感设置 + 采样参数 + 参考音频）。
底层复用官方 indextts/utils/presets.py（存在 outputs/presets/<name>/），
所以与官方 webui.py 创建的预设**完全互通**。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import theme as T
from webui_app import widgets as W
from webui_app.context import AppContext
from webui_app.services import inference as INF
from webui_app.services import voice_bank
from webui_app.services.engine import EngineError

try:
    from indextts.utils.presets import (delete_preset, list_presets, load_preset,
                                        preset_exists, save_preset)
except Exception:      # pragma: no cover - 官方模块缺失时的降级
    list_presets = lambda: []          # noqa: E731
    load_preset = lambda n: None       # noqa: E731
    save_preset = None
    delete_preset = lambda n: False    # noqa: E731
    preset_exists = lambda n: False    # noqa: E731


def _choices() -> List[str]:
    return [""] + list_presets()


# 「应用到合成页」要写入的控件数量，必须与 on_apply 返回的 gr.update 数量一致。
# 具体顺序见 synthesize.py 末尾登记的 ctx.shared["synthesize_apply_targets"]。
APPLY_COUNT = 25


def render(ctx: AppContext):
    cfg = ctx.cfg
    eng = ctx.engine
    sb = ctx.component("statusbar")

    gr.HTML(T.section(
        "预设管理", "💾",
        "预设保存的是一整套参数快照（情感设置 + 采样参数 + 参考音频），"
        "存在 <code>outputs/presets/&lt;名称&gt;/</code>，与官方 webui.py 完全互通。"))

    # ---------- 与合成页的控件联动 ----------
    # 预设要读写合成页控件，而 Gradio 服务端无法主动读取控件的值，所以：
    #   · 读：合成页在 on_generate 里把参数快照写进 ctx.shared["last_gen_values"]
    #   · 写：合成页在渲染末尾把 25 个可写入控件按固定顺序登记到 ctx.shared
    # app.py 保证合成页先于本页渲染，因此这里能直接取到真实控件。
    apply_targets: List[Any] = list(ctx.shared.get("synthesize_apply_targets") or [])
    syn_ready = len(apply_targets) == APPLY_COUNT
    if not syn_ready:
        # 退化成等量隐藏占位，保证回调返回值数量永远匹配，不抛 Gradio 结构错误
        apply_targets = [gr.State(None) for _ in range(APPLY_COUNT)]

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("新建 / 覆盖", "➕", ""))
                new_name = gr.Textbox(label="预设名称", placeholder="例如 主播A-温暖-慢速")
                with gr.Row():
                    save_current_btn = gr.Button(
                        "💾 保存合成页当前参数", variant="primary", scale=2)
                    save_manual_btn = gr.Button("用下面的表单保存", scale=1)
                save_out = gr.HTML("")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("手动填写参数", "✍️",
                                  "不想去合成页调参时，可以直接在这里填。"))
                m_emo_mode = gr.Dropdown(choices=W.EMO_MODE_LABELS,
                                         value=W.EMO_MODE_LABELS[0],
                                         label="情感控制方式")
                with gr.Row():
                    m_alpha = gr.Slider(0.0, 1.0, 0.65, step=0.01,
                                        label="情感权重", scale=1)
                    m_rand = gr.Checkbox(False, label="情感随机采样", scale=1)
                m_emo_text = gr.Textbox(label="情感描述文本", value="")
                gr.Markdown("**8 维情感向量**")
                with gr.Row():
                    m_v = [gr.Slider(0.0, 1.0, 0.0, step=0.05, label=lab)
                           for lab in ("喜", "怒", "哀", "惧")]
                with gr.Row():
                    m_v += [gr.Slider(0.0, 1.0, 0.0, step=0.05, label=lab)
                            for lab in ("厌恶", "低落", "惊喜", "平静")]
                with gr.Accordion("采样与分句参数", open=False):
                    with gr.Row():
                        m_sample = gr.Checkbox(True, label="do_sample", scale=1)
                        m_temp = gr.Slider(0.1, 2.0, 0.8, step=0.05,
                                           label="temperature", scale=2)
                    with gr.Row():
                        m_topp = gr.Slider(0.0, 1.0, 0.8, step=0.01, label="top_p")
                        m_topk = gr.Slider(0, 100, 30, step=1, label="top_k")
                    with gr.Row():
                        m_beams = gr.Slider(1, 10, 3, step=1, label="num_beams")
                        m_reppen = gr.Number(10.0, label="repetition_penalty")
                    with gr.Row():
                        m_lenpen = gr.Number(0.0, label="length_penalty")
                        m_maxmel = gr.Slider(50, 1815, 1500, step=10,
                                             label="max_mel_tokens")
                    m_segtok = gr.Slider(20, 600, 120, step=2,
                                         label="分句最大 Token 数")
                    m_dur = gr.Slider(0.5, 2.0, 1.0, step=0.01, label="时长系数")
                with gr.Row():
                    m_prompt = gr.Audio(label="音色参考音频（可选）", type="filepath",
                                        sources=["upload"],
                                        elem_classes=["ix-audio-compact"])
                    m_emo_audio = gr.Audio(label="情感参考音频（可选）", type="filepath",
                                           sources=["upload"],
                                           elem_classes=["ix-audio-compact"])
                m_from_bank = gr.Dropdown(choices=[""] + voice_bank.names(), value="",
                                          label="或从音色库选一个作为音色参考",
                                          allow_custom_value=False)

        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("已有预设", "📂", ""))
                with gr.Row():
                    preset_dd = gr.Dropdown(choices=_choices(), value="",
                                            label="选择预设", scale=3,
                                            allow_custom_value=False)
                    reload_btn = gr.Button("↻", scale=0, size="sm")
                with gr.Row():
                    apply_btn = gr.Button(
                        "⬆️ 应用到合成页", variant="primary", scale=2,
                        interactive=syn_ready)
                    delete_btn = gr.Button("🗑 删除", variant="stop", scale=1)
                    export_btn = gr.Button("导出 JSON", scale=1)
                if not syn_ready:
                    gr.HTML(T.err(
                        "合成页未就绪（未注册 25 个可写入控件），"
                        "「应用到合成页」已禁用。请检查 app.py 的 Tab 渲染顺序。"))
                detail_md = gr.Markdown("_选择一个预设查看详情_")
                export_file = gr.File(visible=False)

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("批量操作", "🧰", ""))
                with gr.Row():
                    list_all_btn = gr.Button("列出全部预设", size="sm", scale=1)
                    dup_btn = gr.Button("复制为新预设", size="sm", scale=1)
                dup_name = gr.Textbox(label="新名称（用于复制）", placeholder="xxx-副本")
                batch_out = gr.Markdown("")

    gr.HTML(T.section("预设里存了什么", "ℹ️", ""))
    gr.HTML(T.hint(
        "预设目录 <code>outputs/presets/&lt;名称&gt;/</code> 下会有："
        "<code>preset.json</code>（全部参数）、<code>prompt.wav</code>（音色参考音频的副本）、"
        "<code>emo_ref.wav</code>（情感参考音频的副本，若有）。<br>"
        "音频是<b>复制进去</b>的，所以删掉原始上传文件也不会影响预设。"))
    gr.HTML(T.warn(
        "官方 <code>save_preset</code> 用 <code>safe_preset_name()</code> 清洗名称，"
        "<b>按这三步顺序执行</b>：<br>"
        "① 把 <code>\\ / : * ? \" &lt; &gt; |</code> 这些非法字符换成 <code>_</code><br>"
        "② 把连续空白换成 <code>_</code><br>"
        "③ <b>剥掉首尾的 <code>.</code> 和 <code>_</code></b>，全剥完则退回 <code>untitled</code><br>"
        "所以「主播 A」→「主播_A」；「_测试_」→「测试」；"
        "「a/b:c*d?」→「a_b_c_d」（末尾的 _ 被第③步剥掉了）。"
        "界面显示的一律是清洗后的名字。"))

    # =====================================================================
    # 回调
    # =====================================================================

    def collect_from_syn() -> Dict[str, Any]:
        """合成页最近一次生成时的参数快照；尚未生成过则为空字典。"""
        return dict(ctx.shared.get("last_gen_values", {}))

    def build_data(emo_mode, alpha, rand, emo_text, vec, sample, temp, topp, topk,
                   beams, reppen, lenpen, maxmel, segtok, dur) -> Dict[str, Any]:
        return {
            "emo_control_method": W.emo_mode_index(emo_mode),
            "emo_alpha": float(alpha or 0.65),
            "emo_vector": [float(v or 0.0) for v in vec],
            "emo_text": emo_text or "",
            "use_random": bool(rand),
            "max_text_tokens_per_segment": int(segtok or 120),
            "duration_factor": float(dur or 1.0),
            "do_sample": bool(sample),
            "top_p": float(topp or 0.8),
            "top_k": int(topk or 30),
            "temperature": float(temp or 0.8),
            "num_beams": int(beams or 3),
            "repetition_penalty": float(reppen or 10.0),
            "length_penalty": float(lenpen or 0.0),
            "max_mel_tokens": int(maxmel or 1500),
        }

    def do_save(name, data, prompt_audio=None, emo_audio=None):
        name = (name or "").strip()
        if not name:
            return T.err("预设名称不能为空。")
        if save_preset is None:
            return T.err("官方 presets 模块不可用，无法保存。")
        existed = preset_exists(name)
        try:
            save_preset(name, data, prompt_audio=prompt_audio, emo_audio=emo_audio)
        except Exception as e:
            return T.err(f"保存失败：{type(e).__name__}: {e}")
        from indextts.utils.presets import safe_preset_name
        final = safe_preset_name(name)
        msg = (f'<div class="ix-tip">✅ 已{"覆盖" if existed else "保存"}预设 '
               f'<b>{final}</b></div>')
        if final != name:
            msg += T.hint(f"名称被清洗为 <code>{final}</code>（去掉了非法字符/空格）。")
        return msg

    manual_inputs = [m_emo_mode, m_alpha, m_rand, m_emo_text,
                     m_v[0], m_v[1], m_v[2], m_v[3],
                     m_v[4], m_v[5], m_v[6], m_v[7],
                     m_sample, m_temp, m_topp, m_topk, m_beams,
                     m_reppen, m_lenpen, m_maxmel, m_segtok, m_dur]

    def on_save_manual(name, *vals):
        vec = list(vals[4:12])
        data = build_data(vals[0], vals[1], vals[2], vals[3], vec,
                          vals[12], vals[13], vals[14], vals[15], vals[16],
                          vals[17], vals[18], vals[19], vals[20], vals[21])
        return do_save(name, data), gr.update(choices=_choices())

    save_manual_btn.click(
        on_save_manual, inputs=[new_name] + manual_inputs,
        outputs=[save_out, preset_dd])

    def on_save_current(name, prompt_audio, emo_audio, *vals):
        """保存合成页当前参数。合成页控件的值由 apply 流程回传，
        这里用手动表单的值兜底，保证按钮始终有确定行为。"""
        vec = list(vals[4:12])
        data = build_data(vals[0], vals[1], vals[2], vals[3], vec,
                          vals[12], vals[13], vals[14], vals[15], vals[16],
                          vals[17], vals[18], vals[19], vals[20], vals[21])
        cached = collect_from_syn()
        if cached:
            # 合成页有实时值，用它覆盖手动表单
            merged = INF.request_from_preset(cached)
            data.update({k: v for k, v in merged.items() if k in data})
        return do_save(name, data, prompt_audio, emo_audio), gr.update(choices=_choices())

    save_current_btn.click(
        on_save_current,
        inputs=[new_name, m_prompt, m_emo_audio] + manual_inputs,
        outputs=[save_out, preset_dd])

    def on_bank_pick(name):
        e = voice_bank.get(name) if name else None
        return gr.update(value=e.audio_path if e else None)

    m_from_bank.change(on_bank_pick, inputs=[m_from_bank], outputs=[m_prompt])

    # ---------- 详情 ----------
    def render_detail(name: str) -> str:
        if not name:
            return "_选择一个预设查看详情_"
        data = load_preset(name)
        if data is None:
            return T.err(f"预设 <code>{name}</code> 不存在或已损坏。")

        mode = int(data.get("emo_control_method", 0) or 0)
        mode_label = (W.EMO_MODE_LABELS[mode]
                      if 0 <= mode < len(W.EMO_MODE_LABELS) else f"未知({mode})")
        vec = list(data.get("emo_vector") or [0.0] * 8)
        vec = (vec + [0.0] * 8)[:8]
        eff = None
        try:
            from webui_app.tabs.synthesize import normalize_vec
            eff = normalize_vec(vec) if mode == 2 else None
        except Exception:
            eff = None

        adv_keys = ["do_sample", "temperature", "top_p", "top_k", "num_beams",
                    "repetition_penalty", "length_penalty", "max_mel_tokens",
                    "max_text_tokens_per_segment", "duration_factor"]
        L = [
            f"### 预设 `{name}`",
            "",
            "| 项 | 值 |", "|---|---|",
            f"| 情感控制方式 | {mode_label} |",
            f"| 情感权重 | {data.get('emo_alpha', '-')} |",
            f"| 情感随机采样 | {'开' if data.get('use_random') else '关'} |",
            f"| 情感描述文本 | `{data.get('emo_text') or '-'}` |",
            f"| 语言 | {data.get('lang', '-')} |",
            f"| 种子 | {data.get('seed', '-')} |",
            f"| 音色音频 | `{os.path.basename(data.get('prompt_audio') or '-')}` |",
            f"| 情感音频 | `{os.path.basename(data.get('emo_audio') or '-')}` |",
            "",
            "**8 维情感向量**（原始值）",
            "",
            "`[" + ", ".join(f"{float(x):.2f}" for x in vec) + "]`",
            "",
        ]
        if eff is not None:
            L += ["**归一化后实际生效值**（经偏置与 0.8 限幅）", "",
                  "`[" + ", ".join(f"{x:.3f}" for x in eff) + "]`", ""]
        if sum(float(x or 0) for x in vec) > 0.8:
            L += [T.warn("原始总和超过 0.8，已被静默压缩 —— 生效值不等于滑块值。"), ""]

        # 兼容官方 webui.py 的 advanced_params 嵌套结构
        adv = data.get("advanced_params") or {}
        L += ["**采样与分句参数**", "", "| 参数 | 值 |", "|---|---|"]
        for k in adv_keys:
            v = adv.get(k, data.get(k))
            if v is not None:
                L.append(f"| `{k}` | {v} |")
        L.append("")

        pa = data.get("prompt_audio")
        if pa and not os.path.isfile(pa):
            L += [T.err(f"音色参考音频文件丢失：<code>{pa}</code>")]
        elif pa:
            L += [T.tip("音色参考音频完好，应用后可直接合成。")]

        ver = data.get("version", "?")
        L += ["", f"<sub>预设格式版本 v{ver}</sub>"]
        return "\n".join(L)

    def on_select(name):
        return render_detail(name), gr.update(interactive=bool(name))

    preset_dd.change(on_select, inputs=[preset_dd],
                     outputs=[detail_md, delete_btn])
    reload_btn.click(lambda: gr.update(choices=_choices()),
                     inputs=[], outputs=[preset_dd])

    # ---------- 应用到合成页 ----------
    def on_apply(name):
        """把预设写回合成页的 25 个控件。

        返回值数量必须恒为 APPLY_COUNT：失败分支也要返回等量的空 update，
        否则 Gradio 会抛「outputs 与返回值数量不匹配」。
        """
        if not name:
            gr.Warning("请先选择一个预设")
            return [gr.update()] * APPLY_COUNT
        data = load_preset(name)
        if data is None:
            gr.Error(f"预设 {name} 不存在")
            return [gr.update()] * APPLY_COUNT

        # 兼容官方 webui.py 的 advanced_params 嵌套结构
        adv = data.get("advanced_params") or {}

        def val(k, dflt):
            v = adv.get(k, data.get(k))
            return dflt if v is None else v

        mode = int(data.get("emo_control_method", 0) or 0)
        mode = max(0, min(len(W.EMO_MODE_LABELS) - 1, mode))
        vec = list(data.get("emo_vector") or [0.0] * 8)
        vec = (vec + [0.0] * 8)[:8]

        # 预设里的音频是复制进预设目录的，但仍可能被用户手动删掉
        pa = data.get("prompt_audio")
        if pa and not os.path.isfile(pa):
            gr.Warning("音色参考音频文件丢失，已跳过该项")
            pa = None
        ea = data.get("emo_audio")
        if ea and not os.path.isfile(ea):
            gr.Warning("情感参考音频文件丢失，已跳过该项")
            ea = None

        updates = [
            gr.update(value=pa),
            gr.update(value=W.EMO_MODE_LABELS[mode]),
            gr.update(value=ea),
            gr.update(value=val("lang", "ZH")),
            gr.update(value=float(val("duration_factor", 1.0))),
            gr.update(value=float(val("emo_alpha", 0.65))),
            gr.update(value=bool(val("use_random", False))),
            gr.update(value=data.get("emo_text", "") or ""),
        ]
        updates += [gr.update(value=float(x)) for x in vec]
        updates += [
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
        if len(updates) != APPLY_COUNT:      # 防御性断言，改顺序时能立刻发现
            raise RuntimeError(
                f"on_apply 返回 {len(updates)} 个值，与 APPLY_COUNT={APPLY_COUNT} 不一致")
        gr.Info(f"已应用预设「{name}」到合成页")
        return updates

    apply_btn.click(on_apply, inputs=[preset_dd], outputs=apply_targets)

    def on_delete(name):
        if not name:
            gr.Warning("请先选择预设")
            return gr.update(), gr.update(), "_选择一个预设查看详情_"
        ok = delete_preset(name)
        if ok:
            gr.Info(f"已删除预设「{name}」")
        else:
            gr.Warning("预设不存在")
        return (gr.update(choices=_choices(), value=""),
                gr.update(interactive=False),
                "_选择一个预设查看详情_")

    delete_btn.click(on_delete, inputs=[preset_dd],
                     outputs=[preset_dd, delete_btn, detail_md])

    def on_export(name):
        if not name:
            gr.Warning("请先选择预设")
            return gr.update(visible=False), ""
        data = load_preset(name)
        if data is None:
            gr.Error("预设不存在")
            return gr.update(visible=False), ""
        p = os.path.join(cfg.output_dir, f"preset_{name}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        import json
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        gr.Info(f"已导出到 {p}")
        return gr.update(value=p, visible=True), render_detail(name)

    export_btn.click(on_export, inputs=[preset_dd],
                     outputs=[export_file, detail_md])

    def on_list_all():
        names = list_presets()
        if not names:
            return "_预设库为空。_"
        L = ["| 预设 | 情感模式 | 情感权重 | 音色音频 | 存在 |", "|---|---|---|---|---|"]
        for n in names:
            d = load_preset(n) or {}
            mode = int(d.get("emo_control_method", 0) or 0)
            short = W.EMO_MODE_LABELS_SHORT[mode] if 0 <= mode < 4 else "?"
            pa = d.get("prompt_audio")
            ok = "✅" if (not pa or os.path.isfile(pa)) else "❌ 丢失"
            L.append(f"| `{n}` | {short} | {d.get('emo_alpha', '-')} | "
                     f"`{os.path.basename(pa) if pa else '-'}` | {ok} |")
        return "\n".join(L)

    list_all_btn.click(on_list_all, inputs=[], outputs=[batch_out])

    def on_duplicate(name, newname):
        if not name:
            return T.err("请先选择要复制的预设。")
        newname = (newname or "").strip()
        if not newname:
            return T.err("请填写新名称。")
        data = load_preset(name)
        if data is None:
            return T.err("源预设不存在。")
        data.pop("version", None)
        try:
            save_preset(newname, data,
                        prompt_audio=data.get("prompt_audio"),
                        emo_audio=data.get("emo_audio"))
        except Exception as e:
            return T.err(f"复制失败：{type(e).__name__}: {e}")
        return (T.tip(f"✅ 已复制为 <b>{newname}</b>"),
                gr.update(choices=_choices()))

    dup_btn.click(on_duplicate, inputs=[preset_dd, dup_name],
                  outputs=[batch_out, preset_dd])

    return {
        "page_load": (lambda: gr.update(choices=_choices()), [preset_dd]),
        "components": {"preset_dd": preset_dd},
    }
