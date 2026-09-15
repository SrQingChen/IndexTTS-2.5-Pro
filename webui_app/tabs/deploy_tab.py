"""Tab：评测与部署（阶段 2 · L3 + 出口）。

三件事，一条链：
    · **A/B 评测**：两个配置同文本同种子各合成一遍，WER / 声纹相似 /
      reward 胜负表 + 逐条试听 —— 「像不像」到这里才算数；
    · **挂载**：把某个 run 的 adapter 挂上引擎，强度旋钮实时调
      （0=纯底座，0.6~0.8=常见折中），试听满意了再进下一步；
    · **合并**：把 adapter 的 ΔW 烘进一份独立的 gpt.pth / s2mel.pth，
      可脱离训练目录使用。合并不可逆 —— 先评测，再合并。

最下面是**泛化保护面板**：底座快照状态、完整性校验、权重漂移说明。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import gradio as gr

from webui_app import fsutil
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.training import dataset as DS
from webui_app.training import evaluate as EV
from webui_app.training import guard as GD
from webui_app.training import merge as MG
from webui_app.training import reward as RW
from webui_app.training import runs as RN
from webui_app.training.runner import get_runner


def _runs_choices() -> List[str]:
    return [r.name for r in RN.list_runs()]


def _ckpt_choices(run: str) -> List[str]:
    return ["best", "final"] + _vault_ckpts(run)


def _vault_ckpts(run: str) -> List[str]:
    if not run:
        return []
    try:
        return [c.name for c in RN.list_checkpoints(run)]
    except Exception:
        return []


def render(ctx: AppContext):
    eng = ctx.engine
    sb = ctx.component("statusbar")
    runner = get_runner()

    gr.HTML(T.section(
        "评测与部署", "🏁",
        "A/B 对比 → 挂载试听（强度旋钮）→ 合并成独立权重。"
        "评测期间引擎上挂的 adapter 会被临时管理，结束后还原。"))

    with gr.Tabs():
        # ==============================================================
        # A/B 评测
        # ==============================================================
        with gr.Tab("⚖️ A/B 评测"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=360):
                    gr.HTML(T.section("选手 A", "🅰", ""))
                    a_name = gr.Textbox("adapter", label="名字（用于文件与表格）")
                    a_run = gr.Dropdown(choices=_runs_choices(), label="run（空=纯底座）")
                    a_ckpt = gr.Dropdown(choices=["best", "final"],
                                         value="best", label="checkpoint")
                    a_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                        label="adapter 强度")

                    gr.HTML(T.section("选手 B", "🅱", ""))
                    b_name = gr.Textbox("base", label="名字")
                    b_run = gr.Dropdown(choices=_runs_choices(), label="run（空=纯底座）")
                    b_ckpt = gr.Dropdown(choices=["best", "final"],
                                         value="best", label="checkpoint")
                    b_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                        label="adapter 强度")
                    b_enabled = gr.Checkbox(True, label="启用 B（不勾=单选手体检）")

                with gr.Column(scale=1, min_width=360):
                    gr.HTML(T.section("评测设置", "🧪", ""))
                    e_ds = gr.Dropdown(choices=DS.list_datasets(), label="数据集")
                    e_n = gr.Slider(0, 50, value=10, step=1,
                                    label="评测条数（0=全部）")
                    e_whisper = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "turbo"],
                        value="small", label="whisper 档位",
                        info="small ≈ 0.55GB 显存，与引擎可共存；"
                             "medium 起要先掂量显存")
                    e_seed = gr.Number(42, label="种子", precision=0)
                    eval_btn = gr.Button("🚀 开始评测（后台）", variant="primary")
                    eval_stop = gr.Button("⏹ 停止", size="sm")
                    eval_out = gr.HTML("")
                    eval_progress = gr.HTML("")
                    eval_log = gr.Textbox(label="日志", lines=6, max_lines=10,
                                          interactive=False)

            report_md = gr.Markdown("")
            report_dir_html = gr.HTML("")

        # ==============================================================
        # 挂载与合并
        # ==============================================================
        with gr.Tab("🔌 挂载 / 合并"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=360):
                    gr.HTML(T.section("挂载（可逆，实时可调）", "🔌", ""))
                    m_run = gr.Dropdown(choices=_runs_choices(), label="run")
                    m_ckpt = gr.Dropdown(choices=["best", "final"],
                                         value="best", label="checkpoint")
                    m_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                        label="强度（0=纯底座，1=完整 LoRA）")
                    with gr.Row():
                        mount_btn = gr.Button("🔌 挂载", variant="primary")
                        unmount_btn = gr.Button("卸载")
                        rescale_btn = gr.Button("↻ 只调强度")
                    mount_out = gr.HTML("")
                    mounted_md = gr.Markdown("")

                with gr.Column(scale=1, min_width=360):
                    gr.HTML(T.section("合并（不可逆，先评测）", "🧱", ""))
                    g_run = gr.Dropdown(choices=_runs_choices(), label="run")
                    g_ckpt = gr.Dropdown(choices=["best", "final"],
                                         value="best", label="checkpoint")
                    g_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                        label="合并强度")
                    g_out = gr.Textbox(
                        label="输出 pth（必须在 checkpoints/ 之外）",
                        placeholder="outputs/merged/gpt_merged.pth",
                        value="")
                    merge_btn = gr.Button("🧱 合并（后台）", variant="primary")
                    merge_out = gr.HTML("")
                    merge_md = gr.Markdown("")

        # ==============================================================
        # 泛化保护
        # ==============================================================
        with gr.Tab("🛡 泛化保护"):
            gr.HTML(T.section(
                "底座只读保护", "🛡",
                "九道防线里的地基：训练/合并前后校验底座文件未被动过。"
                "快照不存在时会先建（首次约几秒，哈希 3.2 GB）。"))
            with gr.Row():
                guard_btn = gr.Button("🔍 校验底座完整性", variant="secondary")
                rebuild_btn = gr.Button("♻️ 重建快照", size="sm")
            guard_out = gr.HTML("")
            gr.Markdown(
                "**漂移等级怎么看**（训练报告里每份都有）\n\n"
                "| 全局漂移 | 等级 | 建议 |\n|---|---|---|\n"
                "| < 0.02 | 🟢 保守 | 通用能力基本无损；相似度不够就加 rank |\n"
                "| 0.02~0.05 | 🟢 正常 | 典型成功区间 |\n"
                "| 0.05~0.12 | 🟡 偏激进 | 建议把强度旋钮降到 0.6~0.8 试听 |\n"
                "| 0.12~0.25 | 🟠 过拟合风险 | 回滚到更早的 checkpoint |\n"
                "| > 0.25 | 🔴 严重漂移 | 强烈建议回滚重训 |\n\n"
                "**回滚**：训练页 → 训练记录 → 用保险库里的历史档位续训或激活；"
                "推理端把强度旋钮调 0 等于回到纯底座。")

    # =====================================================================
    # 回调
    # =====================================================================

    # ---------- run 选择联动 checkpoint ----------
    def _on_run(run, ckpt_dd):
        return gr.update(choices=_ckpt_choices(run))

    a_run.change(lambda r: gr.update(choices=_ckpt_choices(r)),
                 inputs=[a_run], outputs=[a_ckpt])
    b_run.change(lambda r: gr.update(choices=_ckpt_choices(r)),
                 inputs=[b_run], outputs=[b_ckpt])
    m_run.change(lambda r: gr.update(choices=_ckpt_choices(r)),
                 inputs=[m_run], outputs=[m_ckpt])
    g_run.change(lambda r: gr.update(choices=_ckpt_choices(r)),
                 inputs=[g_run], outputs=[g_ckpt])

    # ---------- A/B 评测 ----------
    def on_eval(an, ar, ac, asc, bn, br, bc, bsc, ben,
                ds, n, whisper, seed):
        if not ds or not DS.exists(ds):
            return T.err(f"数据集 `{ds or '(未选)'}` 不存在。")
        a = EV.Contender(name=(an or "adapter").strip() or "adapter",
                         run=(ar or "").strip(),
                         checkpoint=ac or "best",
                         adapter_scale=float(asc))
        b = None
        if ben:
            b = EV.Contender(name=(bn or "base").strip() or "base",
                             run=(br or "").strip(),
                             checkpoint=bc or "best",
                             adapter_scale=float(bsc))
            if a.name == b.name:
                return T.err("两个选手名字不能相同。")

        def fn(progress, should_stop):
            opts = EV.EvalOptions(dataset=ds, n_samples=int(n),
                                  seed=int(seed), whisper_size=whisper)
            return EV.run_eval(eng, a, b, opts, progress=progress,
                               should_stop=should_stop)

        r = runner.submit("eval", f"A/B 评测 · {a.name}"
                          + (f" vs {b.name}" if b else ""),
                          fn, require_engine="loaded", engine=eng)
        if not r["ok"]:
            return T.err(r["message"])
        gr.Info("评测已在后台启动")
        return T.tip("🚀 评测已启动。每条要合成 2 次并过 whisper，"
                     "耐心等进度条。")

    eval_btn.click(on_eval,
                   inputs=[a_name, a_run, a_ckpt, a_scale,
                           b_name, b_run, b_ckpt, b_scale, b_enabled,
                           e_ds, e_n, e_whisper, e_seed],
                   outputs=[eval_out])
    eval_stop.click(lambda: runner.cancel(), inputs=[], outputs=[eval_out])

    eval_cache: Dict[str, Any] = {"snap": None}

    def on_eval_poll():
        snap = runner.snapshot()
        if snap == eval_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update(), gr.update(), gr.update()
        eval_cache["snap"] = snap
        done = snap["ok"] is not None and not snap["running"]
        if not (snap["running"] or done):
            return gr.update(), gr.update(), gr.update(), gr.update()
        pct = int(snap["progress"] * 100)
        icon = "🎉" if (done and snap["ok"]) else ("🔴" if done else "▶️")
        bar = (f'<div style="margin:4px 0">{icon} <b>{snap["label"]}</b> · '
               f'{pct}% · {snap["message"]} · {snap["seconds"]}s'
               + f'<div style="background:var(--border-color-primary);'
                 f'border-radius:6px;height:8px;margin-top:4px">'
                 f'<div style="width:{pct}%;height:8px;border-radius:6px;'
                 f'background:var(--color-accent)"></div></div></div>')
        rep_up = gr.update()
        dir_up = gr.update()
        if done and snap["kind"] == "eval":
            r = snap.get("result") or {}
            if r.get("ok") and r.get("out_dir"):
                from webui_app.training import evaluate as EVx
                rep_up = gr.update(value=EVx.eval_report_markdown(
                    EV.Contender(**(r.get("a") or {"name": "a"})),
                    (EV.Contender(**r["b"]) if r.get("b") else None),
                    r.get("rows") or [], r["out_dir"],
                    # run_eval 现在把完整 options 回传进 result；
                    # 老结果的快照没有这个键，退回按 dataset 建（其余字段
                    # 会是默认值，总比空串好）。
                    EV.EvalOptions(**(r.get("options")
                                      or {"dataset": str(r.get("dataset", ""))}))))
                dir_html = f"试听目录：<code>{r['out_dir']}</code>"
                if fsutil.open_in_explorer(r["out_dir"]):
                    dir_html += "（已在文件管理器打开）"
                dir_up = gr.update(value=T.hint(dir_html))
            elif r.get("errors"):
                rep_up = gr.update(value="🔴 " + "; ".join(r["errors"][:3]))
        return (gr.update(value=bar), gr.update(value=runner.log_text()),
                rep_up, dir_up)

    eval_timer = gr.Timer(value=2.5, active=True)
    eval_timer.tick(on_eval_poll, inputs=[],
                    outputs=[eval_progress, eval_log, report_md,
                             report_dir_html])

    # ---------- 挂载 ----------
    def on_mount(run, ckpt, scale):
        if not (run or "").strip():
            return T.err("选一个 run（不挂就点「卸载」）。"), gr.update()
        try:
            tag = MG.mount_run(eng, run.strip(), checkpoint=ckpt or "best",
                               scale=float(scale))
        except Exception as e:
            return T.err(f"挂载失败：{e}"), gr.update()
        return (T.tip(f"✅ 已挂载 <code>{tag}</code>（强度 {float(scale):g}）。"
                      "到「合成」页试听，回这里调强度。"),
                gr.update(value=_mounted_markdown(eng)))

    mount_btn.click(on_mount, inputs=[m_run, m_ckpt, m_scale],
                    outputs=[mount_out, mounted_md])

    def on_unmount():
        tags = list(getattr(eng.stats, "lora_adapters", []) or [])
        for t in tags:
            try:
                eng.detach_lora(target=t.split(":", 1)[0])
            except Exception:
                pass
        return (T.hint("已卸载全部 adapter（回到纯底座）。"),
                gr.update(value=_mounted_markdown(eng)))

    unmount_btn.click(on_unmount, inputs=[], outputs=[mount_out, mounted_md])

    def on_rescale(scale):
        n = 0
        for t in list(getattr(eng.stats, "lora_adapters", []) or []):
            n += MG.set_scale(eng, float(scale), t.split(":", 1)[0])
        if not n:
            return T.warn("当前没有挂载任何 adapter。")
        return T.tip(f"强度已调到 {float(scale):g}（{n} 层）。")

    rescale_btn.click(on_rescale, inputs=[m_scale], outputs=[mount_out])

    def _mounted_markdown(engine) -> str:
        tags = list(getattr(engine.stats, "lora_adapters", []) or [])
        if not tags:
            return "当前：**纯底座**（未挂载 adapter）"
        s = GD.get_adapter_scale(getattr(engine.tts, "gpt", None)) \
            if getattr(engine.tts, "gpt", None) is not None else {}
        lines = ["当前挂载：", ""]
        for t in tags:
            tgt = t.split(":", 1)[0]
            mod = (getattr(engine.tts, "gpt", None) if tgt == "gpt"
                   else GD.lora_target_module(engine, tgt))
            sc = GD.get_adapter_scale(mod) if mod is not None else {}
            cur = sc.get("_mean", 1.0)
            lines.append(f"- `{t}` · 当前强度 **{cur:.2f}**")
        return "\n".join(lines)

    # ---------- 合并 ----------
    def on_merge(run, ckpt, scale, out_path):
        if not (run or "").strip():
            return T.err("选一个 run。"), gr.update()
        try:
            d, arch = MG.resolve_mount_dir(run.strip(), ckpt or "best")
        except FileNotFoundError as e:
            return T.err(str(e)), gr.update()
        out = (out_path or "").strip()
        if not out:
            out = os.path.join("outputs", "merged",
                               f"{arch}_merged_{os.path.basename(run.strip())}.pth")
        opts = MG.MergeOptions(adapter_dir=d, arch=arch, out_path=out,
                               scale=float(scale))

        def fn(progress, should_stop):
            progress(0.1, "加载底座 + 合并 ΔW…")
            rep = MG.merge_lora_to_checkpoint(opts)
            progress(1.0, "完成" if rep.ok else "失败")
            return {"ok": rep.ok, "report": rep.to_dict(),
                    "markdown": rep.markdown()}

        r = runner.submit("merge", f"合并 · {run.strip()}", fn)
        if not r["ok"]:
            return T.err(r["message"]), gr.update()
        return T.tip("🧱 合并任务已启动（CPU 上做，不占显存）。"), gr.update()

    merge_btn.click(on_merge, inputs=[g_run, g_ckpt, g_scale, g_out],
                    outputs=[merge_out, merge_md])

    merge_cache: Dict[str, Any] = {"snap": None}

    def on_merge_poll():
        snap = runner.snapshot()
        if snap == merge_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update()
        merge_cache["snap"] = snap
        done = snap["ok"] is not None and not snap["running"]
        if snap["running"]:
            return gr.update(value=f'⏳ {snap["message"]}'), gr.update()
        if done and snap["kind"] == "merge":
            r = snap.get("result") or {}
            md = r.get("markdown") or ""
            return gr.update(), gr.update(value=md)
        return gr.update(), gr.update()

    merge_timer = gr.Timer(value=2.5, active=True)
    merge_timer.tick(on_merge_poll, inputs=[], outputs=[merge_out, merge_md])

    # ---------- 泛化保护 ----------
    def on_guard(rebuild: bool):
        g = GD.BaseGuard()
        try:
            if rebuild or not g.load_manifest():
                man = g.snapshot(hashes=True)
                n = len(man.get("files") or {})
                vr = g.verify(hashes=False)
                return T.tip(f"✅ 快照已{'重建' if rebuild else '建立'}"
                             f"（{n} 个文件），校验通过：底座未被改动。")
            vr = g.verify(hashes=False)
            if vr.ok:
                return T.tip(f"✅ 校验通过：{vr.checked} 个底座文件未被改动。")
            return T.err("🔴 <b>底座被改动过！</b><br>变更："
                         + "<br>".join(vr.changed[:5])
                         + ("<br>…" if len(vr.changed) > 5 else "")
                         + "<br>缺失：" + "<br>".join(vr.missing[:5])
                         + "<br>请到「模型」页重新下载，或确认改动是您主动做的。")
        except Exception as e:
            return T.err(f"校验异常：{type(e).__name__}: {e}")

    guard_btn.click(lambda: on_guard(False), inputs=[], outputs=[guard_out])
    rebuild_btn.click(lambda: on_guard(True), inputs=[], outputs=[guard_out])

    def on_page_load():
        return (gr.update(choices=DS.list_datasets()),
                gr.update(choices=_runs_choices()),
                gr.update(choices=_runs_choices()),
                gr.update(choices=_runs_choices()),
                gr.update(choices=_runs_choices()),
                gr.update(value=_mounted_markdown(eng)))

    return {
        "page_load": (on_page_load,
                      [e_ds, a_run, b_run, m_run, g_run, mounted_md]),
        "components": {"a_run": a_run, "m_run": m_run},
    }
