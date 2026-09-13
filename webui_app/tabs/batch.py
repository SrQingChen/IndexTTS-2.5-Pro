"""Tab：批量合成。

两种输入方式：
    ① 多行文本 —— 每行一段，共用同一个参考音频与参数
    ② JSONL 任务文件 —— 每条任务可以有独立的参考音频/情感/语言/参数

JSONL 格式与官方 examples/batch/*.jsonl 兼容，字段见下方 SCHEMA_DOC。
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from typing import Any, Dict, List, Optional

import gradio as gr

from webui_app import fsutil
from webui_app import theme as T
from webui_app import widgets as W
from webui_app.context import AppContext
from webui_app.services import inference as INF
from webui_app.services import voice_bank
from webui_app.services.engine import EngineError

SCHEMA_DOC = """
### JSONL 任务格式

每行一个 JSON 对象。**只有 `text` 是必填的**，其余字段缺省时继承界面上的全局设置。

```json
{"text": "第一段文本"}
{"text": "第二段", "lang": "EN", "spk_audio_prompt": "voice_bank/audio/host_a.wav"}
{"text": "第三段", "emo_control_method": 2, "emo_vector": [0,0,0.7,0,0,0,0,0]}
{"text": "第四段", "duration_factor": 1.2, "temperature": 0.7, "seed": 42}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | **必填**。要合成的文本 |
| `spk_audio_prompt` | str | 音色参考音频路径。缺省用界面上选的那个 |
| `lang` | str | ZH / EN / JA / AR / ES |
| `emo_control_method` | int | 0=跟随音色 1=情感音频 2=8维向量 3=情感文本 |
| `emo_audio_prompt` | str | 情感参考音频路径（模式 1） |
| `emo_vector` | [float]×8 | 情感向量（模式 2），顺序 [喜,怒,哀,惧,厌恶,低落,惊喜,平静] |
| `emo_text` | str | 情感描述文本（模式 3） |
| `emo_alpha` | float | 情感权重 |
| `use_random` | bool | 情感随机采样 |
| `duration_factor` | float | 时长系数 0.5~2.0 |
| `max_text_tokens_per_segment` | int | 分句 token 上限 |
| `interval_silence` | int | 段间静音 ms |
| `text_normalization` | bool | 文本归一化 |
| `seed` | int | 随机种子，-1 为随机 |
| `do_sample` / `top_p` / `top_k` / `temperature` | | GPT 采样参数 |
| `num_beams` / `repetition_penalty` / `length_penalty` | | GPT 采样参数 |
| `max_mel_tokens` | int | 单段最大语义 token 数 |
| `output` | str | 自定义输出文件名（不含目录） |

> 批量任务共用同一个已加载的引擎，**不会**为每条任务重新加载模型。
> 但每条任务如果换了参考音频，会触发一次声纹重算（约 1~2 秒）。
> 所以把相同音色的任务排在一起能明显提速。
"""


def render(ctx: AppContext):
    cfg = ctx.cfg
    eng = ctx.engine
    sb = ctx.component("statusbar")

    gr.HTML(T.section(
        "批量合成", "📦",
        "两种输入：多行文本（共用参数）或 JSONL 任务文件（每条可独立设参）。"
        "所有任务共用同一个已加载的引擎，不会重复加载模型。"))

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("全局设置", "🌐",
                                  "JSONL 里未指定的字段会继承这里的值。"))
                g_prompt = gr.Audio(label="全局音色参考音频", type="filepath",
                                    sources=["upload"],
                                    elem_classes=["ix-audio-compact"])
                g_voice = gr.Dropdown(choices=[""] + voice_bank.names(), value="",
                                      label="或从音色库选择",
                                      allow_custom_value=False)
                with gr.Row():
                    g_lang = gr.Dropdown(choices=cfg.languages, value="ZH",
                                         label="全局语言", scale=1) if cfg.is_v25 \
                        else gr.State("ZH")
                    g_emo_mode = gr.Dropdown(
                        choices=W.EMO_MODE_LABELS, value=W.EMO_MODE_LABELS[0],
                        label="全局情感模式", scale=2)
                with gr.Row():
                    g_dur = gr.Slider(0.5, 2.0, 1.0, step=0.01,
                                      label="全局时长系数", scale=1)
                    g_seed = gr.Number(-1, label="全局种子", precision=0,
                                       info="-1=每条随机", scale=1)
                with gr.Accordion("全局采样参数", open=False):
                    with gr.Row():
                        g_temp = gr.Slider(0.1, 2.0, 0.8, step=0.05,
                                           label="temperature", scale=1)
                        g_topp = gr.Slider(0.0, 1.0, 0.8, step=0.01,
                                           label="top_p", scale=1)
                    with gr.Row():
                        g_topk = gr.Slider(0, 100, 30, step=1, label="top_k", scale=1)
                        g_beams = gr.Slider(1, 10, 3, step=1, label="num_beams", scale=1)
                    with gr.Row():
                        g_reppen = gr.Number(10.0, label="repetition_penalty", scale=1)
                        g_maxmel = gr.Slider(50, 1815, 1500, step=10,
                                             label="max_mel_tokens", scale=1)
                    g_segtok = gr.Slider(20, 600, 120, step=2,
                                         label="分句最大 Token 数")
                    gr.HTML(T.warn(
                        "批量场景建议把 <b>num_beams 降到 1</b>：单条快 2~3 倍，"
                        "几十条任务累积下来差距很大。质量差异通常可接受。"))

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("任务输入", "📥", ""))
                mode_tab = gr.Tabs()
                with mode_tab:
                    with gr.Tab("多行文本"):
                        lines_tb = gr.Textbox(
                            lines=10, label="每行一段文本",
                            placeholder="第一段内容\n第二段内容\n第三段内容",
                            show_copy_button=True,
                        )
                        with gr.Row():
                            skip_empty = gr.Checkbox(True, label="跳过空行")
                            prefix_tb = gr.Textbox(
                                label="输出文件名前缀", value="batch", scale=2)
                    with gr.Tab("JSONL 文件"):
                        jsonl_file = gr.File(
                            label="上传 .jsonl 任务文件",
                            file_types=[".jsonl", ".json", ".txt"],
                        )
                        jsonl_tb = gr.Textbox(
                            lines=8, label="或直接粘贴 JSONL 内容",
                            placeholder='{"text": "第一段"}\n{"text": "第二段", "lang": "EN"}',
                        )
                        with gr.Row():
                            parse_btn = gr.Button("解析并校验", size="sm", scale=1)
                            template_btn = gr.Button("下载模板", size="sm", scale=1)
                        parse_out = gr.HTML("")

        with gr.Column(scale=1, min_width=400):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("执行", "▶️", ""))
                with gr.Row():
                    run_btn = gr.Button("🚀 开始批量合成", variant="primary", scale=2)
                    concat_cb = gr.Checkbox(
                        False, label="合并成一个音频", scale=1,
                        info="按顺序拼接，段间插静音")
                concat_gap = gr.Slider(0, 2000, 300, step=50, label="合并时的段间静音 (ms)",
                                       visible=False)
                progress_html = gr.HTML("")
                log_tb = gr.Textbox(lines=12, label="执行日志", interactive=False,
                                    show_copy_button=True, elem_classes=["ix-mono"])

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("结果", "🎧", ""))
                result_df = gr.Dataframe(
                    headers=["#", "状态", "时长", "耗时", "RTF", "文件"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    wrap=False, elem_classes=["ix-table"],
                    label="任务结果",
                )
                result_audio = gr.Audio(label="试听（点表格行不会自动加载，用下面的列表）")
                result_files = gr.Dropdown(choices=[], value=None,
                                           label="选择要试听的结果",
                                           allow_custom_value=False)
                with gr.Row():
                    zip_btn = gr.Button("📦 打包下载全部", variant="primary", scale=2)
                    open_dir_btn = gr.Button("打开输出目录", scale=1)
                zip_file = gr.File(label="打包结果", visible=False)

    with gr.Accordion("📄 JSONL 格式说明", open=False):
        gr.Markdown(SCHEMA_DOC)

    # =====================================================================
    # 回调
    # =====================================================================

    def on_voice_pick(name):
        e = voice_bank.get(name) if name else None
        return gr.update(value=e.audio_path if e else None)

    g_voice.change(on_voice_pick, inputs=[g_voice], outputs=[g_prompt])

    def on_concat_toggle(enabled):
        return gr.update(visible=bool(enabled))

    concat_cb.change(on_concat_toggle, inputs=[concat_cb], outputs=[concat_gap])

    # ---------- 任务解析 ----------
    def _global_values(gv: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "lang": gv["lang"],
            "duration_factor": gv["dur"],
            "seed": gv["seed"],
            "temperature": gv["temp"],
            "top_p": gv["topp"],
            "top_k": gv["topk"],
            "num_beams": gv["beams"],
            "repetition_penalty": gv["reppen"],
            "max_mel_tokens": gv["maxmel"],
            "max_text_tokens_per_segment": gv["segtok"],
            "emo_control_method": W.emo_mode_index(gv["emo_mode"]),
        }

    def parse_lines(lines: str, skip_empty: bool) -> List[Dict[str, Any]]:
        out = []
        for raw in (lines or "").splitlines():
            s = raw.strip()
            if not s and skip_empty:
                continue
            if not s:
                continue
            out.append({"text": s})
        return out

    def parse_jsonl(text: str) -> tuple[List[Dict[str, Any]], List[str]]:
        tasks, errors = [], []
        for ln, raw in enumerate((text or "").splitlines(), start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                errors.append(f"第 {ln} 行 JSON 解析失败：{e}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"第 {ln} 行不是 JSON 对象")
                continue
            if not str(obj.get("text", "")).strip():
                errors.append(f"第 {ln} 行缺少 text 字段或为空")
                continue
            tasks.append(obj)
        return tasks, errors

    def on_parse(jsonl_f, jsonl_t):
        text = jsonl_t or ""
        if jsonl_f and not text.strip():
            try:
                with open(jsonl_f, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                return T.err(f"读取文件失败：{e}")
        if not text.strip():
            return T.warn("请上传 JSONL 文件或粘贴内容。")
        tasks, errors = parse_jsonl(text)
        html = T.tip(f"解析出 <b>{len(tasks)}</b> 条有效任务。")
        if errors:
            html += T.err(f"{len(errors)} 处问题：<br>" + "<br>".join(errors[:12]))
        else:
            html += T.hint("全部行格式正确。未在 JSONL 中指定的字段会继承左侧的全局设置。")
        return html

    parse_btn.click(on_parse, inputs=[jsonl_file, jsonl_tb], outputs=[parse_out])

    def on_template():
        p = os.path.join(cfg.output_dir, "batch_template.jsonl")
        sample = [
            {"text": "这是第一条任务，只指定文本，其余继承全局设置。"},
            {"text": "This one uses English.", "lang": "EN"},
            {"text": "这条用情感向量，悲伤拉满。",
             "emo_control_method": 2, "emo_vector": [0, 0, 0.8, 0, 0, 0, 0, 0]},
            {"text": "这条放慢语速，并固定种子以便复现。",
             "duration_factor": 1.25, "seed": 42},
            {"text": "这条指定自己的参考音频。",
             "spk_audio_prompt": "examples/voice_01.wav"},
        ]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for o in sample:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        gr.Info("模板已生成")
        return p, T.tip(f"模板已写入 <code>{p}</code>，可以直接下载后修改。")

    template_file = gr.File(visible=False)
    template_btn.click(on_template, inputs=[], outputs=[template_file, parse_out])

    # ---------- 执行 ----------
    def on_run(g_prompt_v, g_lang, g_emo_mode, g_dur, g_seed, g_temp, g_topp,
               g_topk, g_beams, g_reppen, g_maxmel, g_segtok,
               lines, skip_empty, prefix, jsonl_f, jsonl_t, do_concat, gap,
               progress=gr.Progress(track_tqdm=False)):
        # Gradio 的 inputs 必须是扁平的组件列表，不接受嵌套；
        # 因此这里把全局设置逐个展开为形参。
        _fail = lambda msg: (msg, "", [], gr.update(), gr.update(), gr.update())  # noqa: E731

        # 组装任务列表
        tasks: List[Dict[str, Any]] = []
        jsonl_text = jsonl_t or ""
        if jsonl_f and not jsonl_text.strip():
            try:
                with open(jsonl_f, "r", encoding="utf-8") as f:
                    jsonl_text = f.read()
            except Exception as e:
                return _fail(T.err(f"读取 JSONL 失败：{e}"))
        if jsonl_text.strip():
            tasks, errors = parse_jsonl(jsonl_text)
            if errors and not tasks:
                return _fail(T.err("JSONL 全部行都有问题：<br>"
                                   + "<br>".join(errors[:10])))
        else:
            tasks = parse_lines(lines, skip_empty)

        if not tasks:
            return _fail(T.err("没有可执行的任务。请在「多行文本」里填内容，"
                              "或上传/粘贴 JSONL。"))

        globals_ = _global_values({
            "lang": g_lang, "dur": g_dur, "seed": g_seed, "temp": g_temp,
            "topp": g_topp, "topk": g_topk, "beams": g_beams,
            "reppen": g_reppen, "maxmel": g_maxmel, "segtok": g_segtok,
            "emo_mode": g_emo_mode,
        })

        if not eng.loaded:
            try:
                progress(0.01, desc="正在加载模型…")
                eng.load()
            except EngineError as e:
                return _fail(T.err(f"模型加载失败：{e}"))

        run_dir = os.path.join(cfg.tasks_dir, time.strftime("batch_%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

        logs: List[str] = []
        rows: List[List[Any]] = []
        produced: List[str] = []
        n = len(tasks)

        def log(msg: str):
            """追一条带时间戳的日志，并返回可直接填进 Textbox 的全文。"""
            logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            return "\n".join(logs[-400:])

        for i, task in enumerate(tasks):
            merged = dict(globals_)
            merged.update({k: v for k, v in task.items() if v is not None})
            text = str(merged.get("text", "")).strip()
            spk = merged.get("spk_audio_prompt") or g_prompt_v
            label = (text[:26] + "…") if len(text) > 26 else text

            if not spk:
                rows.append([i, "✖ 缺参考音频", "-", "-", "-", "-"])
                log(f"#{i} 跳过：没有参考音频")
                continue

            progress((i + 0.5) / n, desc=f"合成 {i+1}/{n}：{label}")
            out_name = str(merged.pop("output", "") or "").strip()
            if not out_name:
                out_name = f"{prefix or 'batch'}_{i:03d}.wav"
            out_name = os.path.basename(out_name)
            out_path = os.path.join(run_dir, out_name)

            try:
                req = INF.GenRequest.from_ui(merged, cfg)
                req.spk_audio_prompt = spk
                res = INF.generate(eng, req, progress=None, output_path=out_path)
                rows.append([
                    i, "✅ 成功", f'{res["audio_duration"]:.2f}s',
                    f'{res["seconds"]:.2f}s',
                    f'{res["rtf"]:.2f}' if res.get("rtf") else "-",
                    os.path.basename(res["path"]),
                ])
                produced.append(res["path"])
                log(f"#{i} ✅ {label} → {res['audio_duration']:.2f}s "
                    f"(RTF {res.get('rtf', 0):.2f}, seed {res['seed']})")
            except EngineError as e:
                rows.append([i, f"✖ {str(e)[:40]}", "-", "-", "-", "-"])
                log(f"#{i} ✖ {label} — {e}")
            except Exception as e:
                rows.append([i, f"✖ {type(e).__name__}", "-", "-", "-", "-"])
                log(f"#{i} ✖ {label} — {type(e).__name__}: {e}")

        ok = len(produced)
        summary = (
            f'<div class="ix-tip">✅ 完成 <b>{ok}/{n}</b> 条，'
            f'输出目录 <code>{os.path.relpath(run_dir, os.getcwd())}</code></div>'
        ) if ok == n else (
            f'<div class="ix-warn">⚠️ 成功 {ok}/{n} 条，{n-ok} 条失败，详见日志。</div>'
        )

        concat_path = None
        if do_concat and produced:
            try:
                progress(0.98, desc="合并音频…")
                concat_path = _concat(produced, os.path.join(run_dir, "_merged.wav"),
                                      int(gap))
                summary += T.hint(f"已合并为 <code>{os.path.basename(concat_path)}</code>")
            except Exception as e:
                summary += T.err(f"合并失败：{e}")

        choices = [os.path.basename(p) for p in produced]
        # 最后一个输出是 run_state_dir（gr.State）：作为 output 时要返回**值**，
        # 不是返回一个新的 gr.State 组件 —— 否则后续回调拿到的是组件对象，
        # os.path.isdir() 会直接 TypeError。
        return (summary, "\n".join(logs[-400:]), rows,
                gr.update(choices=choices, value=choices[0] if choices else None),
                gr.update(value=concat_path if concat_path else
                          (produced[0] if produced else None)),
                run_dir)

    run_state_dir = gr.State("")
    run_outputs = [progress_html, log_tb, result_df, result_files, result_audio,
                   run_state_dir]
    run_btn.click(
        on_run,
        inputs=[
            g_prompt, g_lang, g_emo_mode, g_dur, g_seed, g_temp, g_topp, g_topk,
            g_beams, g_reppen, g_maxmel, g_segtok,
            lines_tb, skip_empty, prefix_tb, jsonl_file, jsonl_tb,
            concat_cb, concat_gap,
        ],
        outputs=run_outputs,
    )

    def on_pick_result(name, run_dir):
        if not name or not run_dir:
            return gr.update()
        p = os.path.join(run_dir, name)
        return gr.update(value=p if os.path.isfile(p) else None)

    result_files.change(on_pick_result, inputs=[result_files, run_state_dir],
                        outputs=[result_audio])

    def on_zip(run_dir):
        if not run_dir or not os.path.isdir(run_dir):
            gr.Warning("还没有产出文件")
            return gr.update()
        wavs = [f for f in sorted(os.listdir(run_dir)) if f.lower().endswith(".wav")]
        if not wavs:
            gr.Warning("输出目录里没有 wav 文件")
            return gr.update()
        zp = os.path.join(cfg.output_dir,
                          f"batch_{os.path.basename(run_dir)}.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in wavs:
                z.write(os.path.join(run_dir, f), arcname=f)
        gr.Info(f"已打包 {len(wavs)} 个文件")
        return gr.update(value=zp, visible=True)

    zip_btn.click(on_zip, inputs=[run_state_dir], outputs=[zip_file])

    def on_open_dir(run_dir):
        target = run_dir or cfg.output_dir
        if fsutil.open_in_explorer(target, create=False):
            gr.Info(f"已尝试打开：{target}")
            return T.hint(f"已在文件管理器中打开：<code>{target}</code>")
        gr.Warning("无法自动打开目录，请手动复制下面的路径")
        return T.warn(f"无法自动打开（可能不在本机或无权限）。路径：<code>{target}</code>")

    open_dir_btn.click(on_open_dir, inputs=[run_state_dir], outputs=[progress_html])

    return {
        "page_load": (
            lambda: gr.update(choices=[""] + voice_bank.names()),
            [g_voice],
        ),
        "components": {"g_prompt": g_prompt},
    }


def _concat(paths: List[str], out_path: str, gap_ms: int = 300) -> str:
    """把多个 wav 按顺序拼接，段间插入静音。"""
    import numpy as np
    import soundfile as sf

    chunks = []
    sr = None
    for p in paths:
        y, s = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr is None:
            sr = s
        elif s != sr:
            import librosa
            y = librosa.resample(y, orig_sr=s, target_sr=sr)
        chunks.append(y)
    if not chunks or sr is None:
        raise RuntimeError("没有可拼接的音频")

    gap = np.zeros(int(sr * max(0, gap_ms) / 1000), dtype=np.float32)
    parts = []
    for i, c in enumerate(chunks):
        parts.append(c)
        if i < len(chunks) - 1 and len(gap):
            parts.append(gap)
    sf.write(out_path, np.concatenate(parts), sr, subtype="PCM_16")
    return out_path
