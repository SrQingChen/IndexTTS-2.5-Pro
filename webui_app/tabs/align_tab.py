"""Tab：偏好对齐 · DPO 偏好对构造（阶段 2 · L2）。

用**当前挂着的策略**（引擎 + 可选 adapter）对数据集逐条合成多个候选，
`reward.py` 打分后取最优/最差成对 —— margin 不足的对直接丢弃（学噪声
不如不学）。产物写进目标数据集的 `pairs.jsonl`，训练页选 DPO 目标时用它。

前置条件：引擎已加载（要合成）。评测页的「挂载」可以先把某个 adapter
挂上去 —— 蒸出来的偏好对就是对齐那个 adapter 的。
"""

from __future__ import annotations

from typing import Any, Dict, List

import gradio as gr

from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.training import dpo as DP
from webui_app.training import dataset as DS
from webui_app.training import reward as RW
from webui_app.training.runner import get_runner


def render(ctx: AppContext):
    eng = ctx.engine
    sb = ctx.component("statusbar")
    runner = get_runner()

    gr.HTML(T.section(
        "偏好对构造", "⚖️",
        "同文本多候选合成 → 奖励打分（WER + 声纹相似）→ 最优/最差成对。"
        "需要引擎已加载；引擎上挂着 adapter 时，构造的就是对齐它的偏好对。"))

    with gr.Row(equal_height=False):
        # ============================== 左：构造参数 ==============================
        with gr.Column(scale=1, min_width=380):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("输入与去向", "🎯", ""))
                src_dd = gr.Dropdown(choices=DS.list_datasets(),
                                     label="源数据集（提供文本与参考音色）")
                out_tb = gr.Textbox(label="输出数据集名（不存在会创建）",
                                    value="dpo_pairs",
                                    info="偏好对与两侧特征都写进这个数据集")
                n_cand_sl = gr.Slider(2, 6, value=2, step=1,
                                      label="每条合成几个候选",
                                      info="2 = 只取最优与最差；更多候选"
                                           "margin 更可靠，耗时线性增加")
                margin_sl = gr.Slider(0.0, 0.3, value=0.05, step=0.01,
                                      label="最小 margin",
                                      info="reward 差小于它的对被丢弃 —— "
                                           "转写噪声就能造成的差距不值得学")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("合成多样性", "🎲",
                                  "候选之间的差异来自温度扰动与不同种子。"
                                  "温度太低时两次合成几乎一样，全部对会被 "
                                  "margin 刷掉。"))
                temp_sl = gr.Slider(0.1, 1.5, value=0.9, step=0.05,
                                    label="基准温度")
                jitter_sl = gr.Slider(0.0, 0.5, value=0.15, step=0.05,
                                      label="温度扰动（第 2+ 个候选）")
                max_pairs_tb = gr.Number(0, label="最多构造几对（0=不限）",
                                         precision=0)
                keep_cb = gr.Checkbox(False, label="保留中间候选音频（复核用）")
                overwrite_cb = gr.Checkbox(False, label="覆盖旧偏好对"
                                                        "（默认追加）")

            build_btn = gr.Button("🚀 开始构造（后台）", variant="primary")
            stop_btn = gr.Button("⏹ 停止", size="sm")
            build_out = gr.HTML("")
            progress_html = gr.HTML("")
            log_ta = gr.Textbox(label="日志", lines=8, max_lines=14,
                                interactive=False)

        # ============================== 右：浏览 ==============================
        with gr.Column(scale=1, min_width=380):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("偏好对浏览", "👁", ""))
                pairs_dd = gr.Dropdown(choices=DS.list_datasets(),
                                       label="数据集")
                pairs_btn = gr.Button("↻ 查看", variant="secondary")
                pairs_md = gr.Markdown("_选择包含偏好对的数据集。_")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("划分", "✂️",
                                  "按「对」划分（一对的两半永远同侧）。"
                                  "新增的对会自动计入训练集，重新划分可让它们"
                                  "参与验证。"))
                p_val_ratio = gr.Slider(0.02, 0.4, value=0.1, step=0.01,
                                        label="验证比例")
                p_seed = gr.Number(42, label="种子", precision=0)
                p_split_btn = gr.Button("✂️ 重新划分")
                p_split_out = gr.HTML("")

            with gr.Accordion("ℹ️ DPO 怎么用这些对", open=False):
                gr.Markdown(
                    "1. 构造完成后到 **「训练」** 页：目标选 **DPO**、数据集选"
                    "刚才的输出数据集。\n"
                    "2. DPO 只吃偏好对（无回放）；`beta` 控制贴近 SFT 的程度，"
                    "默认 0.1。\n"
                    "3. 训练完到 **「评测与部署」** 页做 A/B：adapter 挂载 vs "
                    "纯底座，看 WER / 声纹相似有没有真的变好。\n\n"
                    "**margin 都贴着下限？** 说明候选之间没真差别 —— "
                    "提高温度或扰动，或换更难的文本。")

    # =====================================================================
    # 回调
    # =====================================================================

    def on_build(src, out_name, n_cand, margin, temp, jitter, max_pairs,
                 keep, overwrite):
        if not src or not DS.exists(src):
            return T.err(f"源数据集 `{src or '(未选)'}` 不存在。")
        if not (out_name or "").strip():
            return T.err("输出数据集名不能为空。")
        opts = DP.PairBuildOptions(
            out_dataset=out_name.strip(), n_candidates=int(n_cand),
            min_margin=float(margin), temperature=float(temp),
            temperature_jitter=float(jitter),
            max_pairs=int(max_pairs or 0), keep_audio=bool(keep),
            overwrite=bool(overwrite))
        errs = [x.message for x in opts.validate() if x.level == "error"]
        if errs:
            return T.err("<br>".join(errs))

        def fn(progress, should_stop):
            return DP.build_pairs(eng, src, opts, progress=progress,
                                  should_stop=should_stop)

        r = runner.submit("pairs", f"偏好对构造 · {src} → {opts.out_dataset}",
                          fn, require_engine="loaded", engine=eng)
        if not r["ok"]:
            return T.err(r["message"])
        gr.Info("偏好对构造已在后台启动")
        return T.tip("🚀 已启动。合成 + whisper 打分较慢（每候选几秒），"
                     "完成后再到右侧浏览成对结果。")

    build_btn.click(
        on_build,
        inputs=[src_dd, out_tb, n_cand_sl, margin_sl, temp_sl, jitter_sl,
                max_pairs_tb, keep_cb, overwrite_cb],
        outputs=[build_out])
    stop_btn.click(lambda: runner.cancel(), inputs=[], outputs=[build_out])

    def on_pairs(name):
        if not name or not DS.exists(name):
            return "_选择一个数据集。_"
        return DP.pairs_markdown(name)

    pairs_btn.click(on_pairs, inputs=[pairs_dd], outputs=[pairs_md])

    def on_split(name, ratio, seed):
        if not name or not DS.exists(name):
            return T.err("先选数据集。")
        if not DP.load_pairs(name):
            return T.err(f"`{name}` 里没有偏好对。")
        sp = DP.split_pairs(name, val_ratio=float(ratio), seed=int(seed))
        return T.tip(f"✅ 训练 {len(sp['train'])} 对 / 验证 {len(sp['val'])} 对")

    p_split_btn.click(on_split, inputs=[pairs_dd, p_val_ratio, p_seed],
                      outputs=[p_split_out])

    # ---------- 轮询 ----------
    poll_cache: Dict[str, Any] = {"snap": None}

    def on_poll():
        snap = runner.snapshot()
        if snap == poll_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update(), gr.update()
        poll_cache["snap"] = snap
        done = snap["ok"] is not None and not snap["running"]
        if snap["running"] or done:
            pct = int(snap["progress"] * 100)
            icon = "🎉" if (done and snap["ok"]) else ("🔴" if done else "▶️")
            extra = ""
            if done and snap["kind"] == "pairs":
                r = snap.get("result") or {}
                extra = (f"<br>成对 {r.get('kept', 0)} · 丢弃 "
                         f"{r.get('dropped', 0)} · 合成 {r.get('synthesized', 0)}")
            bar = (f'<div style="margin:4px 0">{icon} <b>{snap["label"]}</b> · '
                   f'{pct}% · {snap["message"]}'
                   + (" · <b>已请求停止</b>" if snap["stop_requested"] else "")
                   + f' · {snap["seconds"]}s{extra}'
                   + f'<div style="background:var(--border-color-primary);'
                     f'border-radius:6px;height:8px;margin-top:4px">'
                     f'<div style="width:{pct}%;height:8px;border-radius:6px;'
                     f'background:var(--color-accent)"></div></div></div>')
            upd = (gr.update(choices=DS.list_datasets()), ) if done else ()
            outs = [gr.update(value=bar), gr.update(value=runner.log_text())]
            return (outs[0], outs[1],
                    *(upd if upd else (gr.update(),)))
        return gr.update(), gr.update(), gr.update()

    timer = gr.Timer(value=2.5, active=True)
    timer.tick(on_poll, inputs=[],
               outputs=[progress_html, log_ta, pairs_dd])

    def on_page_load():
        return (gr.update(choices=DS.list_datasets()),
                gr.update(choices=DS.list_datasets()),
                "")

    return {
        "page_load": (on_page_load, [src_dd, pairs_dd, pairs_md]),
        "components": {"pairs_dd": pairs_dd},
    }
