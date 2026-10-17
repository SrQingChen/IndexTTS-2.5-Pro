"""gtsinger_import — GTSinger 导入 CLI 壳（核心逻辑在 webui_app/services/gts_import.py）。

只做导入/整理，**不下载**数据（GTSinger 自行从 HuggingFace 获取：
https://huggingface.co/datasets/GTSinger/GTSinger ，检索显示 CC BY-NC-SA 4.0，
商用需另获授权——详见 docs/dev-plan-e1t-2026-10.md §11）。

用法示例：
    python tools/gtsinger_import.py --src D:/data/GTSinger --dataset gtsinger_zh_en \
        --langs zh,en --pitch-aug "-12,+12" --max-items 1500

UI 入口：「🎵 歌声训练（E1-T）」向导页第②步（同一份逻辑）。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.services import gts_import as GI  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="GTSinger → 本项目数据集导入")
    ap.add_argument("--src", required=True, help="GTSinger 根目录（自行下载）")
    ap.add_argument("--dataset", required=True, help="目标数据集名")
    ap.add_argument("--langs", default="", help="逗号分隔语言过滤（zh,en,…空=全部）")
    ap.add_argument("--singers", default="", help="逗号分隔歌手过滤（空=全部）")
    ap.add_argument("--copy", action="store_true",
                    help="复制音频（默认硬链接，失败退回复制）")
    ap.add_argument("--pitch-aug", default="",
                    help="导入期音高增广，如 '-12,+12'（生成移调副本并重提特征）")
    ap.add_argument("--holdout-singers", default="",
                    help="建议留出歌手（写进 split_note.md，不进 meta）")
    ap.add_argument("--holdout-technique", default="",
                    help="建议留出技巧（写进 split_note.md）")
    ap.add_argument("--max-items", type=int, default=0,
                    help="导入条数上限（0=不限；冒烟建议 1500 左右）")
    args = ap.parse_args()

    langs = [x for x in args.langs.replace("，", ",").split(",") if x]
    singers = [x for x in args.singers.replace("，", ",").split(",") if x]

    def _p(frac, msg):
        print(f"[{frac:.0%}] {msg}")

    r = GI.run_import(args.src, args.dataset, langs=langs, singers=singers,
                      copy=args.copy, pitch_aug=args.pitch_aug,
                      holdout_singers=args.holdout_singers,
                      holdout_technique=args.holdout_technique,
                      max_items=args.max_items, progress=_p)
    if not r.get("ok"):
        print(f"[错误] {r.get('error', '导入失败')}")
        return 2
    print(f"[完成] 导入 {r['imported']} 条 / 增广 {r['augmented']} 条"
          f"（移调 {r['pitch_aug']}）→ {r['out_dir']}\n"
          "下一步：向导页第③步（singing 模式提取特征）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
