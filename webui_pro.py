#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IndexTTS Pro —— 模块化 WebUI 入口。

官方 <code>webui.py</code> 完全不动，本文件是独立实现，两者可以并存。

启动：
    双击根目录 start.bat            # 环境自检 + 启动 + 自动开浏览器
    .venv\\Scripts\\python.exe webui_pro.py
    .venv\\Scripts\\python.exe webui_pro.py --lazy            # 不预加载模型
    .venv\\Scripts\\python.exe webui_pro.py --host 0.0.0.0    # 局域网访问
    .venv\\Scripts\\python.exe webui_pro.py --port 7861
    .venv\\Scripts\\python.exe webui_pro.py --help            # 全部参数

与官方 webui.py 的区别：
    · 界面拆成 11 个 Tab，每个 Tab 一个文件，回调与业务逻辑分离
    · 36 个参数集中登记在 params 注册表，控件提示与「参数手册」共用同一份定义
    · 引擎懒加载 + 可一键卸载 + 显存实时仪表（针对 8 GB 显卡做的）
    · QwenEmotion 串行执行：挂载 → 算向量 → 立即卸载，带 CPU 自动回退与结果缓存
    · 「参考音频工作台」：体检打分 / 智能切片 / 降噪归一 / 音色库
    · 「模型资源」页：镜像优先三级回退下载 + 实时进度 + 资源审计
    · 训练体系（阶段 2）：
        数据集 → 特征离线提取 → GPT/CFM LoRA 自训练 → DPO 偏好对齐
        → 自动 A/B 评测 → 挂载/强度旋钮 → 合并回独立权重
      全程带泛化保护（底座只读快照 / 漂移体检 / 早停 / checkpoint 保险库 /
      回放抗遗忘 / 一键回滚）
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 路径与环境必须在导入 webui_app 之前处理好：
#   · 官方代码大量使用相对路径（checkpoints/、examples/），依赖 cwd == 项目根
#   · HF_ENDPOINT 要在 huggingface_hub 被导入之前设好才生效
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_HERE)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Windows 控制台默认用 GBK 代码页，会把 Python 输出的 UTF-8 字节当 GBK 解，
# 启动摘要里的中文就全变成乱码。把控制台代码页改成 65001(UTF-8)
# 与上面的 PYTHONIOENCODING 对齐。只影响本进程的控制台，不改系统设置。
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> int:
    from webui_app.app import launch
    from webui_app.config import PROJECT_ROOT, config_from_args

    cfg = config_from_args(sys.argv[1:])

    # 日志要在做任何实事之前就绪：启动阶段的问题同样需要留证据。
    # 文件落在 outputs/logs/（indextts.log 全量 + error.log 只看问题），
    # 级别可用 --log-level DEBUG 打开，界面「系统」页也能实时看与切。
    from webui_app import logging_setup as LOG

    path = LOG.setup(level=cfg.log_level, log_dir=cfg.log_dir or "",
                     console=not cfg.quiet)
    LOG.install_excepthooks()
    LOG.get_logger("main").info(
        "启动 IndexTTS-2.5 Pro · 版本 %s · 设备 %s · 低显存=%s · 日志级别 %s",
        cfg.version, cfg.device.backend, cfg.device.low_vram, LOG.level())

    # 必需目录先建好，避免运行期各处 makedirs 竞争
    for d in (cfg.output_dir, cfg.tasks_dir, cfg.voice_bank_dir,
              cfg.cache_dir, os.path.join(cfg.output_dir, "lab"),
              os.path.join(cfg.output_dir, "docs")):
        os.makedirs(d, exist_ok=True)

    launch(
        cfg,
        # 允许 UI 访问整个项目目录：音色库、outputs、examples 都在里面，
        # 「参考音频工作台」还允许直接填服务器上任意路径的音频
        allowed_paths=[PROJECT_ROOT],
        favicon_path=_pick_favicon(),
        max_file_size="512mb",
    )
    LOG.get_logger("main").info("WebUI 已退出；日志见 %s", path or "(控制台)")
    return 0


def _pick_favicon():
    """有就用，没有就返回 None（Gradio 会用默认图标）。"""
    for rel in (os.path.join("assets", "index_icon.png"),
                os.path.join("assets", "indextts_icon.png"),
                os.path.join("assets", "IndexTTS.png")):
        p = os.path.abspath(rel)
        if os.path.isfile(p):
            return p
    return None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已手动停止。")
        sys.exit(130)
