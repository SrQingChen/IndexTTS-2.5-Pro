"""文件系统小工具。

只放跨 Tab 复用的 OS 交互函数。目前只有一个「在文件管理器里打开目录」——
「模型资源」「批量合成」两个 Tab 都要用，阶段 2 的数据集/LoRA 输出目录也会用。

不放业务逻辑：音频处理在 services/audio_lab.py，体积统计在 services/monitor.py。
"""

from __future__ import annotations

import os
import subprocess
import sys


def open_in_explorer(path: str, create: bool = True) -> bool:
    """在系统文件管理器里打开目录。

    用 subprocess 而不是 os.system：后者会把路径拼进 shell 命令字符串，
    路径里带引号或特殊字符时既会失败也有注入风险。

    返回是否成功发起。失败时不抛异常 —— 打不开目录不该影响主流程，
    调用方应该把路径显示给用户让他自己复制。
    """
    try:
        if create:
            os.makedirs(path, exist_ok=True)
        if not os.path.isdir(path):
            return False
        if sys.platform.startswith("win"):
            os.startfile(path)          # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False
