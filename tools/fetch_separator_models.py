"""预下载 AI 翻唱用的分离/去混响模型（国内镜像优先）。

用法：
    python tools/fetch_separator_models.py --list          # 看已装/缺失
    python tools/fetch_separator_models.py --all           # 下载 UI 暴露的全部模型
    python tools/fetch_separator_models.py --model UVR-DeEcho-DeReverb.pth

镜像：默认 ghproxy.net → gh-proxy.com → 直连；可用环境变量覆盖，例如
    set INDEXTTS_GH_MIRRORS=https://ghproxy.net/,
海外用户可设为 ","（仅直连）或空串（仅直连）。

说明：权重落在 checkpoints/separators/（模型资源区，不进清理页产物盘点）。
推理时缺模型也会自动走到同一套下载逻辑（webui_app/services/separation.py）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.services import separation as S  # noqa: E402


def _printer(prefix: str):
    last = {"t": 0.0}

    def _p(frac, msg):
        now = time.time()
        if now - last["t"] < 1.0 and frac < 1.0:
            return
        last["t"] = now
        print(f"\r{prefix} {frac * 100:5.1f}%  {msg}", end="", flush=True)
    return _p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="下载 UI 暴露的全部模型")
    ap.add_argument("--model", action="append", default=[],
                    help="指定模型文件名（可重复）")
    ap.add_argument("--list", action="store_true", help="只列状态")
    args = ap.parse_args()

    if args.list or not (args.all or args.model):
        print(f"模型目录: {S.MODEL_DIR}")
        print(f"镜像链  : {' -> '.join(m or '(直连)' for m in S.mirrors())}")
        for r in S.installed_models():
            print(f"  [{'✓' if r['ready'] else ' '}] {r['label']}")
            print(f"       {r['file']}  {r['mb']} MB")
        if not (args.all or args.model):
            print("\n用 --all 或 --model <文件名> 下载缺失项。")
        return 0

    if not S.available():
        print("ERROR: audio-separator 未安装。先运行：")
        print('  uv pip install --python .venv/Scripts/python.exe '
              '"audio-separator[gpu]" --default-index https://pypi.tuna.tsinghua.edu.cn/simple')
        return 2

    names = []
    if args.all:
        names += list(dict.fromkeys(list(S.VOCALS_MODELS.values()) +
                                    list(S.DEREVERB_MODELS.values())))
    names += args.model
    names = list(dict.fromkeys(names))

    print(f"模型目录: {S.MODEL_DIR}")
    print(f"镜像链  : {' -> '.join(m or '(直连)' for m in S.mirrors())}")
    ok_all = True
    for i, n in enumerate(names, 1):
        print(f"\n[{i}/{len(names)}] {n}")
        r = S.ensure_model(n, _printer("   "))
        print()
        if r.get("ok"):
            print(f"   ✓ 就绪（新下载 {len(r.get('downloaded', []))} 个文件）")
        else:
            ok_all = False
            print(f"   ✗ 失败：{r.get('error')}")
            for m in r.get("missing", []):
                print(f"     缺：{m}")
    print("\n完成。" if ok_all else "\n部分失败——可换镜像重试："
          "set INDEXTTS_GH_MIRRORS=https://ghproxy.net/,https://gh-proxy.com/,")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
