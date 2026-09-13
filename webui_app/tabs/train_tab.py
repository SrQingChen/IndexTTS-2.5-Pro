"""Tab：LoRA 训练（阶段 2 · L1/L2）。

一个页面管三个目标：GPT（语气韵律）/ CFM（音色音质）/ DPO（偏好对齐）。
流水线全部在 `training/` 里（预检 → 训练 → 保险库 → 记录），
这里只是把参数摆出来 + 把任务交给 runner。

互斥关系（8GB 卡的现实）：
    · 训练期间引擎必须**卸载**（runner 会在门口拦）；
    · 特征提取 / 偏好对构造 / A/B 评测需要引擎**加载**；
    · 同一时刻只允许一个后台任务。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import fsutil
from webui_app import theme as T
from webui_app.context import AppContext
from webui_app.training import dataset as DS
from webui_app.training import guard as GD
from webui_app.training import gpt_lora as GL
from webui_app.training import cfm_lora as CL
from webui_app.training import dpo as DP
from webui_app.training import runs as RN
from webui_app.training import trainer_base as TB
from webui_app.training.runner import get_runner, make_train_job

ARCHS = [
    ("GPT(T2S) · 语气与韵律", "gpt"),
    ("CFM(S2M) · 音色与音质", "cfm"),
    ("DPO · 偏好对齐（GPT）", "dpo"),
]
ARCH_LABELS = [a[0] for a in ARCHS]
ARCH_OF = {a[0]: a[1] for a in ARCHS}

PRESETS = [
    ("🟢 保守（数据 < 5 分钟首选）", "conservative"),
    ("⚖️ 均衡（默认推荐）", "balanced"),
    ("🔴 激进（数据 ≥ 30 分钟再试）", "aggressive"),
]
PRESET_LABELS = [p[0] for p in PRESETS]
PRESET_OF = {p[0]: p[1] for p in PRESETS}


def _default_options_json(arch: str) -> str:
    if arch == "cfm":
        return json.dumps(CL.CfmTrainOptions().to_dict(), ensure_ascii=False,
                          indent=1)
    if arch == "dpo":
        return json.dumps(DP.DpoTrainOptions().to_dict(), ensure_ascii=False,
                          indent=1)
    return json.dumps(GL.GptTrainOptions().to_dict(), ensure_ascii=False,
                          indent=1)


def _runs_choices(arch: Optional[str] = None) -> List[str]:
    return [r.name for r in RN.list_runs(arch=arch)]


def _ckpt_choices(run: str) -> List[str]:
    out = ["（从零开始）"]
    if run and RN.read_run(run).get("arch"):
        try:
            for c in RN.vault(run).list():
                out.append(c.name)
        except Exception:
            pass
    return out


def render(ctx: AppContext):
    eng = ctx.engine
    runner = get_runner()

    gr.HTML(T.section(
        "LoRA 训练", "🎓",
        "选目标 → 选数据集 → 选预设 → 预检 → 开跑。训练期间引擎会被要求卸载"
        "（8GB 卡放不下两份），训练记录与 checkpoint 保险库在 "
        "<code>training_runs/</code>。"))

    with gr.Row(equal_height=False):
        # ============================== 左：配置 ==============================
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("基本配置", "🎯", ""))
                arch_dd = gr.Dropdown(choices=ARCH_LABELS, value=ARCH_LABELS[0],
                                      label="训练目标")
                ds_dd = gr.Dropdown(choices=DS.list_datasets(), label="数据集",
                                    info="DPO 目标请选「偏好对数据集」（含 pairs.jsonl）")
                preset_dd = gr.Dropdown(choices=PRESET_LABELS,
                                        value=PRESET_LABELS[1], label="参数预设")
                preset_desc = gr.HTML(T.hint(GD.PRESET_NOTES["balanced"]))
                run_name_tb = gr.Textbox(label="运行名（留空自动生成）", value="")

                resume_dd = gr.Dropdown(choices=_ckpt_choices(""),
                                        value="（从零开始）",
                                        label="续训 checkpoint",
                                        info="跨 run 续训：选其他 run 的档位前，"
                                             "先把上面数据集与目标换成当时的配置")
                val_ratio = gr.Slider(0.02, 0.4, value=0.1, step=0.01,
                                      label="验证集比例（数据集未划分时自动划分）")

            with gr.Accordion("⚙️ LoRA 参数（覆盖预设）", open=False):
                rank_sl = gr.Slider(1, 64, value=0, step=1,
                                    label="rank（0=用预设值）")
                alpha_sl = gr.Slider(1, 128, value=0, step=1,
                                     label="alpha（0=用预设值）")
                lr_tb = gr.Textbox("0", label="学习率（0=用预设值）",
                                   placeholder="如 1e-4")
                epochs_sl = gr.Slider(1, 64, value=0, step=1,
                                      label="epochs（0=用预设值）")
                replay_dd = gr.Dropdown(
                    choices=["（不回放）", "通用数据集", "底座蒸馏集"],
                    value="（不回放）", label="回放源（抗遗忘，防线 6）")
                replay_ratio_sl = gr.Slider(0.0, 0.8, value=0.3, step=0.05,
                                            label="回放比例")

            with gr.Accordion("🔧 目标专属选项（JSON，留默认即可）", open=False):
                options_ta = gr.Textbox(
                    label="options JSON", value=_default_options_json("gpt"),
                    lines=10, max_lines=20)

        # ============================== 右：预检与运行 ==============================
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("预检（不加载模型，秒级）", "🧪",
                                  "把风险、显存、步数在开工前全部摆出来。"))
                pf_btn = gr.Button("🧪 预检", variant="secondary")
                pf_md = gr.Markdown("_选好配置后点「预检」。_")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("训练控制", "▶️", ""))
                start_btn = gr.Button("🚀 开始训练", variant="primary")
                stop_btn = gr.Button("⏹ 停止（当前步跑完即停）", size="sm")
                start_out = gr.HTML("")
                progress_html = gr.HTML("")
                log_ta = gr.Textbox(label="日志", lines=10, max_lines=18,
                                    interactive=False)

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("训练记录", "📚", ""))
                runs_arch_dd = gr.Dropdown(
                    choices=["全部", "GPT", "CFM", "DPO"], value="全部",
                    label="按目标筛选")
                runs_btn = gr.Button("↻ 刷新记录")
                runs_md = gr.Markdown("_暂无记录。_")
                run_detail_dd = gr.Dropdown(choices=_runs_choices(),
                                             label="查看某个 run 的详情")
                detail_md = gr.Markdown("")

    # =====================================================================
    # 回调
    # =====================================================================

    def on_preset(label):
        key = PRESET_OF.get(label, "balanced")
        return T.hint(GD.PRESET_NOTES.get(key, ""))

    preset_dd.change(on_preset, inputs=[preset_dd], outputs=[preset_desc])

    def on_arch(label, _ds):
        arch = ARCH_OF.get(label, "gpt")
        return (gr.update(value=_default_options_json(arch)),
                gr.update(choices=DS.list_datasets()),
                gr.update(choices=_runs_choices(
                    arch if arch != "dpo" else "dpo")))

    arch_dd.change(on_arch, inputs=[arch_dd, ds_dd],
                   outputs=[options_ta, ds_dd, run_detail_dd])

    def _build_cfg(arch, preset_label, rank, alpha, lr, epochs,
                   replay_dd_label, replay_ratio):
        cfg = GD.LoRAConfig.preset(PRESET_OF.get(preset_label, "balanced"))
        if arch == "cfm":                 # cfm 的默认有三处不同（见 cfm_lora）
            cfg = CL.default_config(cfg.target_preset)
        if int(rank) > 0:
            cfg.rank = int(rank)
        if int(alpha) > 0:
            cfg.alpha = int(alpha)
        try:
            lr_v = float(lr or 0)
            if lr_v > 0:
                cfg.lr = lr_v
        except ValueError:
            pass
        if int(epochs) > 0:
            cfg.epochs = int(epochs)
        # 回放（DPO 不吃回放，preflight 会声明忽略）
        if replay_dd_label == "底座蒸馏集":
            cfg.replay_source = "base_distill"
            cfg.replay_ratio = float(replay_ratio)
        elif replay_dd_label == "通用数据集":
            cfg.replay_source = "dataset"
            cfg.replay_ratio = float(replay_ratio)
        else:
            cfg.replay_source = "none"
            cfg.replay_ratio = 0.0
        return cfg

    def _build_trainer(arch, ds, preset_label, rank, alpha, lr, epochs,
                       replay_label, replay_ratio, options_json,
                       rname, resume_label, vratio):
        cfg = _build_cfg(arch, preset_label, rank, alpha, lr, epochs,
                         replay_label, replay_ratio)
        try:
            opts_dict = json.loads(options_json or "{}")
        except Exception as e:
            return None, None, f"options JSON 解析失败：{e}"
        resume = "" if (resume_label or "").startswith("（") else resume_label
        if arch == "gpt":
            tr = GL.GptTrainer(ds, cfg=cfg,
                               options=GL.GptTrainOptions.from_dict(opts_dict),
                               run_name=rname or None, val_ratio=float(vratio),
                               resume_from=resume or None)
        elif arch == "cfm":
            tr = CL.CfmTrainer(ds, cfg=cfg,
                               options=CL.CfmTrainOptions.from_dict(opts_dict),
                               run_name=rname or None, val_ratio=float(vratio),
                               resume_from=resume or None)
        else:
            tr = DP.DpoTrainer(ds, cfg=cfg,
                               options=DP.DpoTrainOptions.from_dict(opts_dict),
                               run_name=rname or None, val_ratio=float(vratio),
                               resume_from=resume or None)
        return tr, opts_dict, None

    def on_preflight(arch_label, ds, preset_label, rank, alpha, lr, epochs,
                     replay_label, replay_ratio, options_json, rname,
                     resume_label, vratio):
        arch = ARCH_OF.get(arch_label, "gpt")
        if not ds or not DS.exists(ds):
            return T.err(f"数据集 `{ds or '(未选)'}` 不存在。到「数据集」页创建。")
        tr, _opts, err = _build_trainer(
            arch, ds, preset_label, rank, alpha, lr, epochs, replay_label,
            replay_ratio, options_json, rname, resume_label, vratio)
        if err:
            return T.err(err)
        try:
            pf = tr.preflight()
        except Exception as e:
            return T.err(f"预检异常：{type(e).__name__}: {e}")
        if not pf.get("ok"):
            return ("### 🔴 预检未通过\n\n**错误**\n\n"
                    + "\n".join(f"- {m}" for m in pf.get("errors") or []))
        L = ["### 🟢 预检通过", "",
             "| 项 | 值 |", "|---|---|",
             f"| 训练/验证 | {pf.get('n_train', pf.get('n_pairs'))} / "
             f"{pf.get('n_val', 0)} |",
             f"| 总步数 | {pf.get('total_steps')}（每轮 {pf.get('steps_per_epoch')}） |",
             f"| 预估显存 | {pf.get('est_vram_gb')} GB "
             f"（空闲 {getattr(pf.get('vram'), 'free_gb', '?')} GB） |",
             f"| 回放 | {pf.get('replay_source')} × "
             f"{pf.get('n_replay_pool', 0)} 条 |"]
        if pf.get("warnings"):
            L += ["", "**⚠️ 提醒**", ""]
            L += [f"- {m}" for m in pf["warnings"]]
        if pf.get("infos"):
            L += ["", "<details><summary>ℹ️ 说明</summary>", ""]
            L += [f"- {m}" for m in pf["infos"]]
            L += ["", "</details>"]
        return "\n".join(L)

    pf_inputs = [arch_dd, ds_dd, preset_dd, rank_sl, alpha_sl, lr_tb,
                 epochs_sl, replay_dd, replay_ratio_sl, options_ta,
                 run_name_tb, resume_dd, val_ratio]
    pf_btn.click(on_preflight, inputs=pf_inputs, outputs=[pf_md])

    def on_start(*args):
        (arch_label, ds, preset_label, rank, alpha, lr, epochs, replay_label,
         replay_ratio, options_json, rname, resume_label, vratio) = args
        arch = ARCH_OF.get(arch_label, "gpt")
        if not ds or not DS.exists(ds):
            return T.err(f"数据集 `{ds or '(未选)'}` 不存在。")
        tr, opts_dict, err = _build_trainer(
            arch, ds, preset_label, rank, alpha, lr, epochs, replay_label,
            replay_ratio, options_json, rname, resume_label, vratio)
        if err:
            return T.err(err)
        cfg_dict = tr.cfg.to_dict()
        r = runner.submit(
            "train",
            f"{'GPT' if arch == 'gpt' else arch.upper()} 训练 · {tr.dataset}",
            make_train_job(arch, ds, cfg_dict, opts_dict,
                           run_name=(rname or ""), resume_from=(
                               "" if (resume_label or "").startswith("（")
                               else resume_label),
                           val_ratio=float(vratio), tracker=runner),
            require_engine="unloaded", engine=eng)
        if not r["ok"]:
            return T.err(r["message"])
        gr.Info("训练已在后台启动")
        return T.tip("🚀 训练已在后台启动。训练结束或停止后，本页与"
                     "「评测与部署」页的记录会自动刷新。")

    start_btn.click(on_start, inputs=pf_inputs, outputs=[start_out])
    stop_btn.click(lambda: runner.cancel(), inputs=[], outputs=[start_out])

    # ---------- 记录 ----------
    def on_runs(arch_label):
        arch = {"GPT": "gpt", "CFM": "cfm", "DPO": "dpo"}.get(arch_label)
        return (RN.runs_markdown(arch=arch),
                gr.update(choices=_runs_choices(arch)))

    runs_btn.click(on_runs, inputs=[runs_arch_dd], outputs=[runs_md, run_detail_dd])

    def on_detail(run):
        if not run:
            return ""
        try:
            return RN.run_detail_markdown(run)
        except Exception as e:
            return T.err(f"读取失败：{e}")

    run_detail_dd.change(on_detail, inputs=[run_detail_dd], outputs=[detail_md])

    # ---------- 轮询 ----------
    poll_cache: Dict[str, Any] = {"snap": None}

    def on_poll():
        snap = runner.snapshot()
        if snap == poll_cache["snap"] and not snap["running"]:
            return gr.update(), gr.update(), gr.update()
        poll_cache["snap"] = snap
        just_done = snap["ok"] is not None and not snap["running"]
        if snap["running"] or just_done:
            pct = int(snap["progress"] * 100)
            icon = "🎉" if (just_done and snap["ok"]) else (
                "🔴" if just_done else "▶️")
            extra = ""
            if just_done and snap["kind"] == "train":
                r = snap.get("result") or {}
                extra = (f"<br>steps={r.get('steps')} · "
                         f"val {r.get('first_val')} → {r.get('best_val')}"
                         + (f" · 早停：{r.get('stop_reason')}"
                            if r.get("stopped_early") else ""))
                if r.get("error"):
                    extra += f"<br>错误：{r.get('error')}"
            bar = (f'<div style="margin:4px 0">{icon} <b>{snap["label"]}</b> · '
                   f'{pct}% · {snap["message"]}'
                   + (" · <b>已请求停止</b>" if snap["stop_requested"] else "")
                   + f' · {snap["seconds"]}s{extra}'
                   + f'<div style="background:var(--border-color-primary);'
                     f'border-radius:6px;height:8px;margin-top:4px">'
                     f'<div style="width:{pct}%;height:8px;border-radius:6px;'
                     f'background:var(--color-accent)"></div></div></div>')
            runs_up = (RN.runs_markdown(arch=None),
                       gr.update(choices=_runs_choices())) if just_done \
                else (gr.update(), gr.update())
            return (gr.update(value=bar), gr.update(value=runner.log_text()),
                    *runs_up)
        return gr.update(), gr.update(), gr.update()

    timer = gr.Timer(value=2.5, active=True)
    timer.tick(on_poll, inputs=[],
               outputs=[progress_html, log_ta, runs_md, run_detail_dd])

    def on_page_load():
        return (gr.update(choices=DS.list_datasets()),
                gr.update(choices=_runs_choices()),
                RN.runs_markdown(arch=None))

    return {
        "page_load": (on_page_load, [ds_dd, run_detail_dd, runs_md]),
        "components": {"ds_dd": ds_dd, "arch_dd": arch_dd},
    }
