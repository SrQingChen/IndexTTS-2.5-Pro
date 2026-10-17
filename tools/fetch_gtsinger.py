"""fetch_gtsinger — GTSinger 镜像拉取 CLI 壳（核心逻辑在 webui_app/services/gts_fetch.py）。

用法：
    python tools/fetch_gtsinger.py                     # 默认 zh,en → datasets/_sources/GTSinger
    python tools/fetch_gtsinger.py --langs zh,en,ja    # 语言子集
    python tools/fetch_gtsinger.py --all               # 全量（约 40GB）

镜像：默认走 hf-mirror.com（HF_ENDPOINT），断点续传，重复执行增量补齐。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.services import gts_fetch as GF  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="GTSinger 镜像拉取（hf-mirror）")
    ap.add_argument("--dest", default=GF.DEFAULT_DEST,
                    help=f"目标目录（默认 {GF.DEFAULT_DEST}）")
    ap.add_argument("--langs", default="zh,en",
                    help="逗号分隔语言码（zh,en,ja,ko,de,fr,it,es,ru）")
    ap.add_argument("--all", action="store_true", help="拉取全量（忽略 --langs）")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    langs = [] if args.all else [x for x in args.langs.replace("，", ",").split(",")
                                 if x]
    print(f"[镜像] {GF.ensure_endpoint()}")

    def _p(frac, msg):
        print(f"[{frac:.0%}] {msg}", flush=True)

    r = GF.fetch(args.dest, langs, progress=_p, workers=args.workers)
    if not r.get("ok"):
        print(f"[{'停止' if r.get('stopped') else '失败'}] {r.get('error', '')}")
        return 2
    print(f"[完成] {r['done']} 个文件（{r.get('note', '')}）→ {r['dest']}\n"
          "下一步：向导页第③步或 CLI 导入（tools/gtsinger_import.py）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
