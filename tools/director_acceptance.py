"""真机验收：导演模式端到端（真实引擎 + 真实情感参考路由）。

覆盖桩测不到的部分：engine.infer 对逐句 kwargs 的真实接受度、
22050/int16 拼接、情感参考缓存轮换、旁车台本落盘。
会加载真实模型（约 20~30s）并合成 2 轮 × 3 句，需要 GPU 空闲。

用法：.venv\\Scripts\\python.exe tools\\director_acceptance.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


TEXT = "住手！你这家伙，到底做了什么？！哈哈，太好了。再见……走吧。"


def main() -> int:
    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import director as DR
    from webui_app.services import emotion_bank as EB
    from webui_app.services import inference as INF
    from webui_app.services import orchestrator as ORC
    from webui_app.services import voice_bank as VB

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine

    print("== 0. 准备 ==")
    spk_entry = VB.get("wujiu") or next(iter(VB.list_voices()), None)
    check("音色库有可用参考", spk_entry is not None,
          spk_entry.name if spk_entry else "音色库为空")
    if spk_entry is None:
        return finish()

    print("== 1. 引擎加载 ==")
    t0 = time.perf_counter()
    eng.load()
    print(f"  加载 {time.perf_counter() - t0:.1f}s · 显存 "
          f"{eng.stats.vram_alloc_gb:.2f} GB")

    # 给「验收角色」临时入库一条情感参考（跑完删掉，不留垃圾）
    tmp_char = "验收角色"
    added = None

    print("== 2. 规则导演 + 无路由 ==")
    sc = DR.direct(TEXT, backend="rules", seed=42)
    req = INF.GenRequest(spk_audio_prompt=spk_entry.audio_path,
                         text=TEXT, emo_alpha=0.7, seed=42)
    res1 = ORC.perform(eng, req, sc, route=False, progress=None)
    n = sc.n
    dur1 = res1["audio_duration"]
    check(f"产出音频（{n} 句）", dur1 > 3.0, f"{dur1:.2f}s")
    check("RTF 合理", res1["rtf"] is not None and res1["rtf"] < 15,
          f"rtf={res1['rtf']:.2f}")
    check("旁车台本", os.path.isfile(res1["director"]["sidecar"]))

    print("== 3. 情感路由（真实情感库） ==")
    try:
        added = EB.add(tmp_char, "angry", spk_entry.audio_path,
                       note="验收临时条目，自动删除")
    except Exception as e:
        check("临时情感参考入库", False, str(e))
        return finish(added)
    sc2 = DR.direct(TEXT, backend="rules", seed=42)
    res2 = ORC.perform(eng, req, sc2, route=True, character=tmp_char)
    check("angry 句命中路由", res2["director"]["routed"] >= 1,
          f"routed={res2['director']['routed']}/"
          f"{res2['director']['fallback']}")
    # 情感路由会合法地改变语速（愤怒参考→急促，自身参考→平缓），
    # 阈值只拦病理性 runaway（>40% 相对差），不拦表演差异
    _rel = abs(res2["audio_duration"] - dur1) / max(dur1, res2["audio_duration"])
    check("路由版时长无病理漂移（<40% 相对差）", _rel < 0.40,
          f"{res2['audio_duration']:.2f}s vs {dur1:.2f}s（相对 {_rel:.0%}）")
    data = json.load(open(res2["director"]["sidecar"], encoding="utf-8"))
    routed_lines = [l for l in data["lines"] if l["emo_ref"]]
    check("台本记录了命中的参考与逐句 alpha",
          bool(routed_lines) and all(l["emo_alpha"] for l in routed_lines),
          f"angry 句 alpha={routed_lines[0]['emo_alpha']:.3f}" if routed_lines else "")

    print("== 4. BoN 逐句择优（真实 CPU 打分器） ==")
    sc3 = DR.direct(TEXT, backend="rules", seed=42)
    res3 = ORC.perform(eng, req, sc3, route=True, character=tmp_char,
                       bon_n=2)
    d3 = json.load(open(res3["director"]["sidecar"], encoding="utf-8"))
    bon_rows = [l.get("bon") for l in d3["lines"] if l.get("bon")]
    check("每句 2 候选都有打分记录",
          len(bon_rows) == len(d3["lines"])
          and all(b["n"] == 2 and len(b["rewards"]) == 2 for b in bon_rows),
          f"{len(bon_rows)} 句 · 首句 rewards={bon_rows[0]['rewards'] if bon_rows else None}")
    check("reward 全部有效（≥0）",
          all(r >= 0 for b in bon_rows for r in b["rewards"]),
          str([b["rewards"] for b in bon_rows[:2]]))
    check("BoN 版时长合理", 5.0 < res3["audio_duration"] < 20.0,
          f"{res3['audio_duration']:.2f}s · {res3['seconds']:.0f}s 生成")

    print("== 5. 产物清单 ==")
    for r in (res1, res2, res3):
        bon_tag = f" · BoN={r['director'].get('bon_n', 0)}" \
            if r.get("director", {}).get("bon_n") else ""
        print(f"  {os.path.basename(r['path'])} · {r['audio_duration']:.2f}s · "
              f"{r['seconds']:.1f}s 生成 · {r['director']['backend']}{bon_tag}")

    return finish(added)


def finish(added=None) -> int:
    if added is not None:
        from webui_app.services import emotion_bank as EB
        EB.remove(added.name)
        print(f"  （已清理临时情感参考 {added.name}）")
    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
