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

# 进程启动时刻（首次 import 本包 ≈ WebUI 进程启动）。旁车用它对比代码
# 提交时间，自动判定「进程里跑的是不是旧代码」——2026-10-03 连续两轮
# 归因翻车（把帧量化盲区误判成旧进程）后加的追踪件：与其事后猜进程年龄，
# 不如合成时自己报。
import time as _time

PROCESS_STARTED_AT = _time.time()
