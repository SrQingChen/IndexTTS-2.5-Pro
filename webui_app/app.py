"""app.py —— Blocks 组装层。

职责边界很清楚：
    · config.py    决定「用什么参数」
    · context.py   持有「运行期共享状态」（引擎、日志、跨 Tab 组件槽位）
    · services/*   做「实际的事」
    · tabs/*       画「界面」并绑定回调
    · app.py       只负责把它们**按正确顺序**拼起来

顺序很关键：
    1. 先创建顶栏 statusbar 并存入 ctx.shared —— 各 Tab 的回调要把它当 output，
       如果等到 Tab 内部再创建就会产生「幽灵组件」（每次调用新建一个隐藏控件）。
    2. 合成页必须先于预设页渲染 —— 预设页要读 ctx.shared["synthesize_apply_targets"]
       才能把「应用到合成页」的 25 个 gr.update 对准真实控件。
    3. 所有 Tab 渲染完后，再统一绑定 demo.load / tab.select。
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from webui_app import theme as T
from webui_app.config import AppConfig, config_from_args
from webui_app.context import AppContext
from webui_app.services.engine import EngineError

# Tab 定义：(id, 标题, 渲染函数路径, 一句话说明)
# 顺序即界面顺序，也是渲染顺序 —— 不要随意调换，见模块 docstring。
#
# 标题刻意做短：7 个 Tab 在笔记本屏幕上很容易超出宽度，
# Gradio 会把放不下的收进右侧「…」溢出菜单，导致功能看起来“没了”。
# 完整名称与说明在每个 Tab 自己的页首标题里给，不靠标签承担。
TAB_SPECS: List[Tuple[str, str, str, str]] = [
    ("syn",    "🎙 合成",       "webui_app.tabs.synthesize", "单条合成 · 全参数 · 实时提示"),
    ("lab",    "🔬 音频工作台", "webui_app.tabs.audio_lab",  "体检 · 智能切片 · 降噪归一 · 音色库"),
    ("batch",  "📦 批量",       "webui_app.tabs.batch",      "多行文本 / JSONL · 进度 · 打包"),
    ("preset", "💾 预设",       "webui_app.tabs.presets",    "参数快照 · 与官方 webui.py 互通"),
    ("oneclick", "🚀 一键三连", "webui_app.tabs.oneclick_tab",
     "丢音频进来 · 切片识别调参训练择优 · 全自动"),
    ("data",   "🗂 数据集",     "webui_app.tabs.dataset_tab", "建集 · 导入 · 文本 · 划分 · 特征提取"),
    ("train",  "🎓 训练",       "webui_app.tabs.train_tab",  "GPT/CFM/DPO · 预检 · 保险库 · 记录"),
    ("align",  "⚖️ 对齐",       "webui_app.tabs.align_tab",  "DPO 偏好对构造 · 浏览 · 划分"),
    ("deploy", "🏁 评测/部署",  "webui_app.tabs.deploy_tab", "A/B · 挂载强度 · 合并 · 泛化保护"),
    ("model",  "📥 模型",       "webui_app.tabs.models",     "审计 · 镜像下载 · 实时进度"),
    ("sys",    "🖥 系统",       "webui_app.tabs.system",     "显存 · 环境体检 · 日志 · 维护"),
    ("cleanup","🧹 清理",       "webui_app.tabs.cleanup_tab",
     "产物盘点 · 分类明细 · 确认弹窗 · 安全删除"),
    ("manual", "📖 手册",       "webui_app.tabs.manual",     "架构原理 · 全部参数 · 配方 · 排查"),
]


def _import_render(dotted: str) -> Callable[[AppContext], Dict[str, Any]]:
    """按字符串路径导入 Tab 的 render 函数。

    用字符串而不是直接 import，是为了让「某个 Tab 依赖缺失」不至于
    连累整个应用起不来（例如阶段 2 的训练 Tab 依赖 peft）。
    """
    import importlib
    mod = importlib.import_module(dotted)
    return mod.render


def _header_badges(cfg: AppConfig) -> List[Tuple[str, str]]:
    d = cfg.device
    badges = [("版本", f"IndexTTS-{cfg.version}")]
    if d.backend == "cuda":
        vram = f"{d.vram_total_gb:.0f}GB"
        badges.append(("GPU", (d.gpu_name or "CUDA").replace("NVIDIA ", "")[:18] + " " + vram))
    else:
        badges.append(("设备", d.backend.upper()))
    badges.append(("精度", "BF16" if cfg.use_bf16 else ("FP16" if cfg.use_fp16 else "FP32")))
    if d.low_vram:
        badges.append(("模式", "低显存"))
    return badges


def _startup_notes_html(cfg: AppConfig, missing: List[str]) -> str:
    """启动提示条：环境注意事项 + 缺失文件告警。"""
    out = []
    for n in cfg.startup_notes():
        out.append(T.hint(n))
    if missing:
        out.append(T.err(
            "<b>模型文件不完整，暂时无法推理。</b>缺失 "
            f"{len(missing)} 项：<br>"
            + "<br>".join(f"<code>{m}</code>" for m in missing[:12])
            + ("<br>…" if len(missing) > 12 else "")
            + "<br>请到 <b>「📥 模型资源」</b> 页选择「全部（推荐首次使用）」下载，"
              "约 7.9 GB，走国内镜像。"))
    return "".join(out)


def build_app(cfg: Optional[AppConfig] = None,
              do_autoload: bool = True) -> Tuple[gr.Blocks, AppContext]:
    """构建 Gradio Blocks。返回 (demo, ctx)。"""
    cfg = cfg or config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine

    # 启动前先做一次静态审计（不加载模型，只查文件），用于给出准确的启动提示
    try:
        missing = eng.audit_missing()
    except Exception:
        missing = []

    rendered: List[Tuple[Any, Dict[str, Any], str]] = []   # (gr.Tab, 返回字典, 标题)
    failures: List[str] = []

    with gr.Blocks(
        theme=T.THEME,
        css=T.CSS,
        title=f"IndexTTS-{cfg.version} Pro",
        analytics_enabled=False,
    ) as demo:

        # -----------------------------------------------------------------
        # 顶栏 + 全局状态条（必须在任何 Tab 之前创建）
        # -----------------------------------------------------------------
        gr.HTML(T.header_html(
            f"IndexTTS-{cfg.version} <span style='font-weight:400;opacity:.75'>Pro</span>",
            "模块化控制台 · 推理 / 工作台 / 批量 / 预设 / 数据集 / 训练 / 对齐 / 评测部署 / 监控",
            _header_badges(cfg),
        ))
        statusbar = gr.HTML(ctx.status_html(), elem_id="ix-statusbar")
        ctx.shared["statusbar"] = statusbar

        boot_html = _startup_notes_html(cfg, missing)
        if boot_html:
            boot_box = gr.HTML(boot_html)
        else:
            boot_box = None

        # -----------------------------------------------------------------
        # 各 Tab
        # -----------------------------------------------------------------
        with gr.Tabs():
            for tab_id, tab_label, dotted, _desc in TAB_SPECS:
                with gr.Tab(tab_label, id=tab_id) as tab_obj:
                    try:
                        render = _import_render(dotted)
                        res = render(ctx) or {}
                    except Exception as e:
                        failures.append(f"{tab_label}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        gr.HTML(T.err(
                            f"<b>本 Tab 渲染失败</b>：{type(e).__name__}: {e}<br>"
                            f"<pre style='font-size:11px;white-space:pre-wrap'>"
                            f"{traceback.format_exc()[-1600:]}</pre>"))
                        res = {}
                    rendered.append((tab_obj, res, tab_label))

        # -----------------------------------------------------------------
        # 页脚
        # -----------------------------------------------------------------
        gr.HTML(
            '<div style="text-align:center;opacity:.55;font-size:12px;margin:18px 0 6px 0">'
            f'IndexTTS-{cfg.version} Pro · 官方 <code>webui.py</code> 保持不变，'
            '本界面为独立实现 · '
            '参数行为说明均经源码核实 · '
            f'启动于 {time.strftime("%Y-%m-%d %H:%M:%S")}'
            '</div>'
        )
        if failures:
            gr.HTML(T.err(
                "<b>以下 Tab 未能正常渲染</b>（其余功能不受影响）：<br>"
                + "<br>".join(failures)))
        # 暴露给测试/诊断用：渲染失败的 Tab 不应该只靠界面上的一行红字告知
        ctx.shared["_render_failures"] = list(failures)

        # -----------------------------------------------------------------
        # 全局事件绑定（所有 Tab 渲染完之后）
        # -----------------------------------------------------------------

        def on_boot():
            """首屏：刷新状态条 + 启动提示。"""
            notes = _startup_notes_html(cfg, eng.audit_missing() if not eng.loaded else [])
            return ctx.status_html(), (gr.update(value=notes) if boot_box else gr.update())

        boot_outputs: List[Any] = [statusbar] + ([boot_box] if boot_box else [])
        demo.load(on_boot, inputs=[], outputs=boot_outputs)

        # 顶栏状态条每 3 秒自动刷新（显存/引擎状态/错误）
        global_timer = gr.Timer(value=3.0, active=True)
        global_timer.tick(lambda: ctx.status_html(), inputs=[], outputs=[statusbar])

        # 切到某个 Tab 时才刷新它的数据 —— 避免首屏一次性跑完所有重活
        # （例如「模型资源」的审计要遍历目录，「系统监控」的体检要统计磁盘体积）
        for tab_obj, res, _label in rendered:
            pl = res.get("page_load")
            if not pl:
                continue
            fn, outs = pl
            if not outs:
                continue
            tab_obj.select(fn, inputs=[], outputs=outs)

    # ---------------------------------------------------------------------
    # 引擎预加载：放到后台线程，让 UI 先出来
    # ---------------------------------------------------------------------
    if do_autoload and cfg.autoload_engine and not missing:
        def _bg_load():
            time.sleep(0.6)      # 让首屏先渲染完
            try:
                eng.load()
            except EngineError as e:
                ctx.log.push("error", f"预加载失败：{e}", "error")
            except Exception as e:
                ctx.log.push("error", f"预加载异常：{type(e).__name__}: {e}", "error")

        threading.Thread(target=_bg_load, daemon=True, name="engine-autoload").start()
    elif missing:
        ctx.log.push("error", f"模型文件缺失 {len(missing)} 项，已跳过预加载", "error")

    return demo, ctx


def launch(cfg: Optional[AppConfig] = None, **launch_kwargs) -> None:
    """构建并启动。webui_pro.py 直接调这个。"""
    cfg = cfg or config_from_args()
    demo, ctx = build_app(cfg)

    # 控制台摘要
    d = cfg.device
    if d.backend == "cuda":
        dev_line = (f"{d.gpu_name or 'CUDA'} · {d.vram_total_gb:.1f} GB · "
                    f"空闲 {d.vram_free_gb:.1f} GB")
    else:
        dev_line = d.backend.upper()
    prec = "BF16" if cfg.use_bf16 else ("FP16" if cfg.use_fp16 else "FP32")

    print("=" * 72)
    print(f"  IndexTTS-{cfg.version} Pro")
    print(f"  设备    : {dev_line}")
    print(f"  精度    : {prec}{'  (低显存模式)' if d.low_vram else ''}")
    print(f"  模型目录: {cfg.model_dir}")
    print(f"  输出目录: {cfg.output_dir}")
    print(f"  引擎    : {'启动时后台加载' if cfg.autoload_engine else '按需加载（--lazy）'}")
    print(f"  地址    : http://{cfg.host}:{cfg.port}")
    print("=" * 72)
    for n in cfg.startup_notes():
        print(f"  · {n}")
    print("=" * 72)

    demo.queue(default_concurrency_limit=cfg.concurrency)
    demo.launch(
        server_name=cfg.host,
        server_port=cfg.port,
        share=cfg.share,
        show_error=True,
        inbrowser=True,
        **launch_kwargs,
    )
