"""tools/ 下验证脚本的公共引导：路径 + Windows 控制台编码。

每个探针/测试脚本开头都有同样三件事要做，而且都踩过同一个坑：
Windows 控制台默认 GBK 代码页，脚本里的 `✅` `⚠️` 这类字符
会直接抛 UnicodeEncodeError 把整个脚本打断（内容其实是对的）。
所以统一在这里处理，import 本模块即生效，不需要调用任何函数。

    import _env            # noqa: F401  —— 放在所有 webui_app 导入之前

用 `python tools\\xxx.py` 启动时，脚本所在目录会自动进 sys.path[0]，
所以这个 import 不需要额外配置。
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 与 webui_pro.py 保持一致：只改本进程的控制台代码页，不动系统设置
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# HF 走镜像，避免探针脚本在联网检查时卡住
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
