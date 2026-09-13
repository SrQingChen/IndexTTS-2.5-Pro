"""IndexTTS 模块化 WebUI。

包结构：
    config.py    全局配置与路径（单一事实来源）
    params.py    参数元数据注册表 —— 驱动 UI 控件生成 + 参数手册 + 提示信息
    theme.py     Gradio 主题与自定义 CSS
    context.py   AppContext：跨 Tab 共享的运行时上下文
    services/    业务逻辑层（引擎、推理、音频工作台、模型管理、训练）
    tabs/        视图层，每个 Tab 一个模块，互不耦合
    app.py       组装 Blocks
"""

__version__ = "1.0.0"
