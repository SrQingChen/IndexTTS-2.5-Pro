"""探针：BoN 台本 → DPO 偏好对导入（C2 桥接）。

桩数据：手造 .script.json（带 best/worst 路径与 reward）+ 真实小 wav，
导入临时数据集后核对 pairs.jsonl 行、样本文本/状态回写、margin 过滤、
缺文件跳过与追加语义。不加载引擎，秒级完成。
用法：python tools/bon_import_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from webui_app.training import dataset as DS  # noqa: E402
from webui_app.training import dpo as DP  # noqa: E402

SR = 22050
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def _wav(path):
    sf.write(path, (np.random.RandomState(0).rand(SR) * 200 - 100
                    ).astype(np.int16), SR, subtype="PCM_16")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="bon_probe_")
    DS_NAME = "probe_bon_pairs"
    if DS.exists(DS_NAME):
        DS.delete(DS_NAME)

    # ---- 造两份台本：共 3 个可导入句 + 1 个低 margin + 1 个缺文件 ----
    sidecars = []
    wavs = {}
    for s in range(2):
        lines = []
        for i in range(3 if s == 0 else 1):
            bp = os.path.join(tmp, f"s{s}_l{i}_best.wav")
            wp = os.path.join(tmp, f"s{s}_l{i}_worst.wav")
            _wav(bp)
            _wav(wp)
            wavs[bp] = wavs[wp] = True
            rb, rw = 0.9 - 0.1 * s, 0.5
            lines.append({"text": f"第{s}份第{i}句台词。", "emotion": "calm",
                          "bon": {"best_path": bp, "worst_path": wp,
                                  "reward_best": rb, "reward_worst": rw,
                                  "rewards": [rb, 0.7, rw], "chosen": 0}})
        # 第 0 份额外塞一个低 margin 句和一个缺文件句
        if s == 0:
            bp0 = os.path.join(tmp, "low_best.wav")
            wp0 = os.path.join(tmp, "low_worst.wav")
            _wav(bp0)
            _wav(wp0)
            lines.append({"text": "低margin句。", "emotion": "calm",
                          "bon": {"best_path": bp0, "worst_path": wp0,
                                  "reward_best": 0.5, "reward_worst": 0.5}})
            lines.append({"text": "缺文件句。", "emotion": "calm",
                          "bon": {"best_path": os.path.join(tmp, "nope.wav"),
                                  "worst_path": os.path.join(tmp, "nope2.wav"),
                                  "reward_best": 0.9, "reward_worst": 0.4}})
        p = os.path.join(tmp, f"spk_sidecar_{s}.script.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"backend": "rules", "lines": lines}, f, ensure_ascii=False)
        sidecars.append(p)

    print("== 1. 正常导入 ==")
    r = DP.import_bon_sidecars(DS_NAME, sidecars, min_margin=0.05)
    print(" ", {k: v for k, v in r.items() if k != "errors"} or r)
    check("ok", bool(r.get("ok")))
    check("导入 4 对（3+1，低margin与缺文件被跳过）",
          r["pairs"] == 4 and r["skipped_margin"] == 1
          and r["skipped_missing"] == 1,
          f"pairs={r['pairs']} margin={r['skipped_margin']} "
          f"missing={r['skipped_missing']}")
    pairs = DP.load_pairs(DS_NAME)
    check("pairs.jsonl 行数一致", len(pairs) == 4)
    check("source=bon 且 reward 落位",
          all(p.source == "bon" and p.reward_chosen >= p.reward_rejected
              for p in pairs))
    row0 = pairs[0]
    check("两侧样本都在数据集里且带文本",
          DS.get(DS_NAME, row0.chosen) is not None
          and DS.get(DS_NAME, row0.rejected) is not None
          and DS.get(DS_NAME, row0.chosen).text == row0.text,
          row0.text)
    check("样本状态已刷（evaluate 后非 no_text）",
          all(DS.get(DS_NAME, sid).status != "no_text"
              for p in pairs for sid in (p.chosen, p.rejected)),
          str({DS.get(DS_NAME, sid).status
               for p in pairs for sid in (p.chosen, p.rejected)}))
    n_audio = len([u for u in DS.load_meta(DS_NAME)])
    check("音频 8 条入集（4 对 × 2）", n_audio == 8, str(n_audio))

    print("== 2. 幂等与坏输入 ==")
    r2 = DP.import_bon_sidecars(DS_NAME, sidecars, min_margin=0.05)
    check("重复导入幂等（对数仍 4，计入 skipped_dup）",
          len(DP.load_pairs(DS_NAME)) == 4 and r2.get("skipped_dup") == 4,
          f"pairs={len(DP.load_pairs(DS_NAME))} dup={r2.get('skipped_dup')}")
    r3 = DP.import_bon_sidecars(DS_NAME, [os.path.join(tmp, "ghost.json")])
    check("坏台本路径 ok=True 但 0 对 + errors",
          r3.get("ok") and r3["pairs"] == 0 and r3["errors"])
    r4 = DP.import_bon_sidecars("", sidecars)
    check("空数据集名被拒", not r4.get("ok"))

    print("== 3. 清理 ==")
    DS.delete(DS_NAME)
    check("探针数据集已清", not DS.exists(DS_NAME))

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
