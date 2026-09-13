"""Tab：系统监控与维护。

把「这台机器现在能不能跑得动」这个问题回答清楚：
实时显存仪表 → 引擎运行统计 → 完整环境体检 → 事件日志 → 维护操作。

8 GB 显卡上显存是最容易踩的坑，所以这一页把显存放在第一位，
并且给出「一键归还」的手段（卸载引擎 / 清空 CUDA 缓存）。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List

import gradio as gr

from webui_app import theme as T
from webui_app.config import LOW_VRAM_THRESHOLD_GB, refresh_vram_free
from webui_app.context import AppContext
from webui_app.services import monitor as MON
from webui_app.services.engine import EngineError


def _bar(ratio: float, label: str) -> str:
    return T.progress_bar(max(0.0, min(1.0, ratio)), label)


def render(ctx: AppContext):
    cfg = ctx.cfg
    eng = ctx.engine
    sb = ctx.component("statusbar")

    gr.HTML(T.section(
        "系统监控与维护", "🖥",
        "实时显存、引擎统计、环境体检、事件日志。"
        f"本机低显存阈值为 <b>{LOW_VRAM_THRESHOLD_GB:.0f} GB</b>，"
        "低于它会激活官方的 <code>low_vram</code> 分支。"))

    # =====================================================================
    # 实时仪表
    # =====================================================================
    with gr.Column(elem_classes=["ix-section"]):
        gr.HTML(T.section("实时仪表", "📈", "每 2 秒刷新一次。"))
        gauge_html = gr.HTML(ctx.status_html())
        with gr.Row():
            vram_md = gr.Markdown("_尚未采样_")
            engine_md = gr.Markdown("_引擎未加载_")
        with gr.Row():
            live_cb = gr.Checkbox(True, label="自动刷新", scale=1)
            refresh_btn = gr.Button("↻ 立即刷新", size="sm", scale=1)
            gpu_test_btn = gr.Button("🧪 CUDA 自检", size="sm", scale=1)

    with gr.Row(equal_height=False):
        # =================================================================
        # 左：环境体检
        # =================================================================
        with gr.Column(scale=1, min_width=420):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("环境体检", "🩺",
                                  "硬件 / Python / 能力评估 / 依赖版本核对 / 磁盘占用。"
                                  "首次使用或报错时先看这里。"))
                env_btn = gr.Button("🩺 运行完整体检", variant="primary")
                env_md = gr.Markdown("_点上方按钮开始（约需数秒，会统计目录体积）。_")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("当前配置", "⚙️",
                                  "由命令行参数与环境推导出来的实际生效配置。"))
                cfg_btn = gr.Button("查看配置快照", size="sm")
                cfg_md = gr.Markdown("")

        # =================================================================
        # 右：日志与维护
        # =================================================================
        with gr.Column(scale=1, min_width=420):
            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("事件日志", "📜",
                                  "引擎生命周期事件（加载/卸载/推理/QwenEmotion/LoRA/错误），"
                                  "环形缓冲 300 条。"))
                with gr.Row():
                    log_btn = gr.Button("↻ 刷新日志", size="sm", scale=1)
                    log_n = gr.Slider(10, 300, 60, step=10, label="显示条数", scale=2)
                log_md = gr.Markdown(ctx.log.markdown())

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("维护操作", "🧹",
                                  "显存吃紧时按从上到下的顺序尝试，代价递增。"))
                with gr.Row():
                    empty_cache_btn = gr.Button("清空 CUDA 缓存", scale=1, size="sm")
                    clear_ref_btn = gr.Button("清参考音频缓存", scale=1, size="sm")
                with gr.Row():
                    clear_emo_btn = gr.Button("清情感向量缓存", scale=1, size="sm")
                    release_qwen_btn = gr.Button("卸载 QwenEmotion", scale=1, size="sm")
                unload_btn = gr.Button("⏏ 卸载整个引擎（归还全部显存）",
                                       variant="stop", scale=1)
                maint_out = gr.HTML("")

            with gr.Column(elem_classes=["ix-section"]):
                gr.HTML(T.section("磁盘清理", "🗑",
                                  "只清理产物目录，模型与音色库不受影响。"))
                with gr.Row():
                    clean_targets = gr.CheckboxGroup(
                        choices=["outputs/（合成产物）", "outputs/lab/（工作台中间文件）",
                                 "outputs/tasks/（批量任务）", "Gradio 临时缓存"],
                        value=[], label="选择要清理的内容",
                    )
                with gr.Row():
                    scan_btn = gr.Button("先扫描体积", size="sm", scale=1)
                    clean_btn = gr.Button("执行清理", variant="stop", size="sm", scale=1)
                clean_out = gr.HTML("")

    gr.HTML(T.section("显存速查", "💡", ""))
    gr.HTML(
        T.hint(
            "本机实测（RTX 4060 Laptop 8 GB）：<br>"
            "· 引擎<b>常驻</b> 4.94 GB · 推理<b>峰值</b> 5.24 GB · 跑完剩 1.34 GB 空闲<br>"
            "· GPT LoRA r=16 训练峰值约 <b>4.12 GB</b>（96 层 Conv1D，7.86M 参数）<br>"
            "· CFM LoRA r=16 训练极宽裕（105 层 Linear，2.90M 参数，base bf16 仅 0.18 GB）")
        + T.warn(
            "QwenEmotion（1.1 GB）<b>不能</b>与主推理引擎共存于 8 GB 显存。"
            "本项目采用<b>串行策略</b>：挂载 → 算出 8 维向量 → 立即卸载 → 走情感向量路径，"
            "并带显存预检与 CPU 自动回退。")
        + T.tip(
            "<code>torch.cuda.empty_cache()</code> 只归还未被引用的<b>缓存块</b>，"
            "不会释放仍被张量占用的显存。真想腾地方必须卸载引擎。")
    )

    # =====================================================================
    # 回调
    # =====================================================================

    def sample() -> Dict[str, Any]:
        refresh_vram_free(cfg.device)
        return ctx.snapshot()

    # 变化检测：显存/计数没动就不重绘（定时器每 2s 触发，不能无脑重绘）
    _last_gauge: Dict[str, Any] = {}

    def live_changed(snap: Dict[str, Any]) -> bool:
        key = (snap["vram_alloc_gb"], snap["vram_free_gb"], snap["infer_count"],
               snap["engine_loaded"], snap["qwen_mounted"],
               tuple(snap["lora_adapters"]))
        if _last_gauge.get("key") == key:
            return False
        _last_gauge["key"] = key
        return True

    def vram_markdown(snap: Dict[str, Any]) -> str:
        if snap["backend"] != "cuda":
            return ("### 显存\n\n当前运行在 **CPU** 上，无显存数据。\n\n"
                    + T.warn("CPU 推理慢约 3 倍，仅建议用于调试。"))
        total = max(snap["vram_total_gb"], 0.1)
        alloc = snap["vram_alloc_gb"]
        free = snap["vram_free_gb"]
        peak = snap["vram_peak_gb"]
        L = [
            "### 显存",
            "",
            _bar(alloc / total, f"PyTorch 已分配 {alloc:.2f} / {total:.1f} GB"),
            _bar(peak / total, f"历史峰值 {peak:.2f} GB"),
            _bar(max(0.0, (total - free)) / total,
                 f"整卡已占用 {total - free:.2f} GB（含其他进程）"),
            "",
            f"- 整卡空闲：**{free:.2f} GB**",
            f"- 低显存模式：**{'已激活' if snap['low_vram'] else '未激活'}**"
            f"（阈值 {LOW_VRAM_THRESHOLD_GB:.0f} GB）",
        ]
        if snap["low_vram"]:
            L.append("")
            L.append(T.warn(
                "低显存模式下官方 <code>infer()</code> 会对超过 40 字的文本"
                "用 <code>split_text_by_punctuation(max_chars=40)</code> 二次粗切，"
                "长句可能被切在不自然的位置。可在合成页把「分句最大 Token 数」调小来主动控制。"))
        if free < 1.5:
            L.append("")
            L.append(T.err(
                f"整卡空闲仅 {free:.2f} GB，继续推理有 OOM 风险。"
                "建议先关掉其他占显存的程序，或卸载引擎。"))
        return "\n".join(L)

    def engine_markdown(snap: Dict[str, Any]) -> str:
        if not snap["engine_loaded"]:
            L = ["### 引擎", "", "**未加载**。", ""]
            missing = []
            try:
                missing = eng.audit_missing()
            except Exception:
                pass
            if missing:
                L += [T.err("缺少必需文件，无法加载：<br>"
                            + "<br>".join(f"<code>{m}</code>" for m in missing)), ""]
                L.append("请到「模型资源」页下载。")
            else:
                L.append(T.tip("文件齐全，首次合成时会自动加载（约 20~30 秒）。"))
            return "\n".join(L)

        s = eng.stats
        avg = (s.infer_total_seconds / s.infer_count) if s.infer_count else 0.0
        L = [
            "### 引擎",
            "",
            f"- 加载耗时：**{s.load_seconds:.1f} s**",
            f"- 已运行：**{time.strftime('%H:%M:%S', time.gmtime(snap['uptime']))}**",
            f"- 合成次数：**{s.infer_count}**",
            f"- 最近一次：**{s.last_infer_seconds:.2f} s** · 平均：**{avg:.2f} s**",
            f"- 显存：分配 **{s.vram_alloc_gb:.2f} GB** · "
            f"保留 **{s.vram_reserved_gb:.2f} GB** · 峰值 **{s.vram_peak_gb:.2f} GB**",
            f"- QwenEmotion：**{'已挂载' if s.qwen_mounted else '未挂载'}**"
            f"（调用 {s.qwen_infer_count} 次，加载 {s.qwen_load_seconds:.1f} s）",
            f"- LoRA 适配器：**{', '.join(s.lora_adapters) if s.lora_adapters else '无'}**",
            f"- 精度：**{'BF16' if cfg.use_bf16 else ('FP16' if cfg.use_fp16 else 'FP32')}**",
        ]
        if s.error:
            L += ["", T.err(f"最近错误：{s.error}")]
        if s.notes:
            L += [""] + [f"> {n}" for n in s.notes]
        return "\n".join(L)

    def on_refresh(auto: bool):
        snap = sample()
        if auto and not live_changed(snap):
            return gr.update(), gr.update(), gr.update()
        return (gr.update(value=ctx.status_html()),
                gr.update(value=vram_markdown(snap)),
                gr.update(value=engine_markdown(snap)))

    gauge_outputs = [gauge_html, vram_md, engine_md]
    timer = gr.Timer(value=2.0, active=True)
    timer.tick(on_refresh, inputs=[live_cb], outputs=gauge_outputs)
    # 手动刷新走 auto=False（跳过变化检测，无条件重算）
    refresh_btn.click(lambda: on_refresh(False), inputs=[], outputs=gauge_outputs)

    def on_gpu_test():
        """真跑一次 CUDA 分配，确认不是只有 driver 在、runtime 坏了。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return T.err("torch.cuda.is_available() == False，CUDA 运行时不可用。")
            dev = torch.device("cuda:0")
            t0 = time.time()
            a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
            b = a @ a
            torch.cuda.synchronize()
            dt = time.time() - t0
            gflops = (2 * 2048 ** 3) / dt / 1e9
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            del a, b
            torch.cuda.empty_cache()
            return T.tip(
                f"✅ CUDA 自检通过<br>"
                f"设备：<b>{name}</b> · 算力 <b>sm_{cap[0]}{cap[1]}</b><br>"
                f"2048×2048 FP16 矩乘：<b>{dt*1000:.1f} ms</b> ≈ <b>{gflops:.0f} GFLOPS</b><br>"
                f"CUDA 构建：<b>{torch.version.cuda}</b> · cuDNN：<b>{torch.backends.cudnn.version()}</b>")
        except Exception as e:
            return T.err(f"CUDA 自检失败：{type(e).__name__}: {e}")

    gpu_test_btn.click(on_gpu_test, inputs=[], outputs=[maint_out])

    def on_env():
        try:
            return gr.update(value=MON.render_markdown(cfg))
        except Exception as e:
            return gr.update(value=T.err(f"体检失败：{type(e).__name__}: {e}"))

    env_btn.click(on_env, inputs=[], outputs=[env_md])

    def on_cfg():
        snap = ctx.snapshot()
        data = {
            "config": {
                "version": cfg.version, "model_dir": cfg.model_dir,
                "output_dir": cfg.output_dir, "host": cfg.host, "port": cfg.port,
                "share": cfg.share, "concurrency": cfg.concurrency,
                "half_precision": cfg.half_precision,
                "load_qwen_emo(构造参数)": cfg.load_qwen_emo,
                "use_qwen_emo(实际)": cfg.engine_kwargs().get("use_qwen_emo"),
                "deepspeed": cfg.deepspeed, "accel": cfg.accel,
                "cuda_kernel": cfg.cuda_kernel, "torch_compile": cfg.torch_compile,
                "is_v25": cfg.is_v25, "languages": cfg.languages,
                "low_vram阈值GB": LOW_VRAM_THRESHOLD_GB,
            },
            "runtime": snap,
            "engine_kwargs": {k: str(v) for k, v in cfg.engine_kwargs().items()},
        }
        return gr.update(value="```json\n" + json.dumps(
            data, ensure_ascii=False, indent=2, default=str) + "\n```")

    cfg_btn.click(on_cfg, inputs=[], outputs=[cfg_md])

    def on_log(n):
        return gr.update(value=ctx.log.markdown(int(n)))

    log_btn.click(on_log, inputs=[log_n], outputs=[log_md])
    log_n.change(on_log, inputs=[log_n], outputs=[log_md])

    # ---------- 维护 ----------
    def _after_maint(msg: str):
        return msg, ctx.status_html()

    def on_empty_cache():
        if cfg.device.backend != "cuda":
            return _after_maint(T.warn("当前跑在 CPU 上，无显存可清。"))
        before, after = eng.empty_cache()
        freed = before - after
        if freed < 0.05:
            return _after_maint(T.hint(
                f"已调用 <code>empty_cache()</code>，但已分配显存几乎未变"
                f"（{before:.2f} → {after:.2f} GB）。"
                "<br>这说明占用来自<b>仍被引用的模型权重</b>，不是缓存块 —— "
                "要腾地方只能卸载引擎。"))
        return _after_maint(T.tip(
            f"✅ 已归还 <b>{freed:.2f} GB</b>（{before:.2f} → {after:.2f} GB）"))

    empty_cache_btn.click(on_empty_cache, inputs=[], outputs=[maint_out, sb])

    def on_clear_ref():
        if not eng.loaded:
            return _after_maint(T.warn("引擎未加载，无缓存可清。"))
        eng.clear_reference_cache()
        return _after_maint(T.tip(
            "✅ 已清空参考音频缓存（声纹/ref_mel 等 7 项）。"
            "下次合成会重算，首句会变慢 1~2 秒。"))

    clear_ref_btn.click(on_clear_ref, inputs=[], outputs=[maint_out, sb])

    def on_clear_emo():
        n = eng.clear_emo_cache()
        return _after_maint(T.tip(f"✅ 已清空 {n} 条情感向量缓存。"))

    clear_emo_btn.click(on_clear_emo, inputs=[], outputs=[maint_out, sb])

    def on_release_qwen():
        if not eng.stats.qwen_mounted:
            return _after_maint(T.hint("QwenEmotion 当前未挂载（串行策略下用完即卸）。"))
        eng.release_qwen()
        return _after_maint(T.tip("✅ QwenEmotion 已卸载，显存已归还。"))

    release_qwen_btn.click(on_release_qwen, inputs=[], outputs=[maint_out, sb])

    def on_unload():
        if not eng.loaded:
            return _after_maint(T.warn("引擎本来就未加载。"))
        try:
            eng.unload()
        except EngineError as e:
            return _after_maint(T.err(str(e)))
        snap = sample()
        return _after_maint(T.tip(
            f"✅ 引擎已卸载。整卡空闲 <b>{snap['vram_free_gb']:.2f} GB</b>。"))

    unload_btn.click(on_unload, inputs=[], outputs=[maint_out, sb])

    # ---------- 磁盘清理 ----------
    def _clean_paths() -> Dict[str, str]:
        return {
            "outputs/（合成产物）": cfg.output_dir,
            "outputs/lab/（工作台中间文件）": os.path.join(cfg.output_dir, "lab"),
            "outputs/tasks/（批量任务）": cfg.tasks_dir,
            "Gradio 临时缓存": os.path.join(
                os.environ.get("TEMP", "/tmp"), "gradio"),
        }

    def on_scan():
        rows = ["<tr><th>目录</th><th>大小</th><th>文件数</th><th>路径</th></tr>"]
        for label, path in _clean_paths().items():
            if not os.path.isdir(path):
                rows.append(f"<tr><td>{label}</td><td>—</td>"
                            f"<td>不存在</td><td><code>{path}</code></td></tr>")
                continue
            size, nfiles = 0, 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root, f))
                        nfiles += 1
                    except OSError:
                        pass
            rows.append(
                f"<tr><td>{label}</td><td><b>{MON.human_size(size)}</b></td>"
                f"<td>{nfiles}</td><td><code>{path}</code></td></tr>")
        return ('<table class="ix-table">' + "".join(rows) + "</table>"
                + T.warn("清理不会碰 <code>checkpoints/</code>（模型）与 "
                         "<code>voice_bank/</code>（音色库）。"))

    scan_btn.click(on_scan, inputs=[], outputs=[clean_out])

    def on_clean(targets: List[str]):
        if not targets:
            return T.warn("请先勾选要清理的内容。")
        done, failed = [], []
        for label in targets:
            path = _clean_paths().get(label)
            if not path or not os.path.isdir(path):
                continue
            try:
                # 只删内容，保留目录本身（引擎与 UI 都假设目录存在）
                for entry in os.listdir(path):
                    p = os.path.join(path, entry)
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            os.remove(p)
                        except OSError as e:
                            failed.append(f"{entry}: {e}")
                done.append(label)
            except Exception as e:
                failed.append(f"{label}: {type(e).__name__}: {e}")
        msg = T.tip("✅ 已清理：" + "、".join(done)) if done else T.warn("没有可清理的目录。")
        if failed:
            msg += T.err("以下项失败（可能被占用）：<br>" + "<br>".join(failed))
        return msg

    clean_btn.click(on_clean, inputs=[clean_targets], outputs=[clean_out])

    def on_page_load():
        snap = sample()
        return (ctx.status_html(), vram_markdown(snap), engine_markdown(snap),
                ctx.log.markdown(60))

    return {
        "page_load": (on_page_load, [gauge_html, vram_md, engine_md, log_md]),
        "components": {"gauge_html": gauge_html},
    }
