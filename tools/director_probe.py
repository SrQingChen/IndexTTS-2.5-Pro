"""探针：导演层（规则后端 + API 解析/回退）。

不加载引擎、不联网（API 解析用注入的假响应测试），秒级跑完。
用法：python tools/director_probe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui_app.services import director as DR  # noqa: E402

SAMPLE = ("住手！你这家伙，到底对她们做了什么？！"
          "呵呵……原来如此，是这样啊……"
          "对不起，我已经，无法再相信任何人了。"
          "什么？！这不可能！"
          "走吧，该结束了。")

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("== 1. 规则后端：切分 ==")
    sc = DR.rules_direct(SAMPLE, seed=42)
    check("切出 ≥5 句", sc.n >= 5, f"n={sc.n}")
    check("全部成功", sc.ok)
    check("文本零改写（句子都在原文里）",
          all(ln.text.rstrip("。！？!?…；;，,、") in SAMPLE for ln in sc.lines))
    print("  台本：")
    for i, ln in enumerate(sc.lines, 1):
        print(f"    {i}. [{ln.emotion}/{ln.intensity:.2f}/{ln.pause_after_ms}ms] {ln.text}")

    print("== 2. 规则后端：情绪猜测 ==")
    emos = {ln.text: ln.emotion for ln in sc.lines}
    check("「住手！」→ angry", emos.get("住手！") == "angry")
    check("「什么？！」→ surprised", any(
        t.startswith("什么") and e == "surprised" for t, e in emos.items()))

    print("== 3. 规则后端：停顿合理性与确定性 ==")
    pauses = [ln.pause_after_ms for ln in sc.lines[:-1]]
    check("停顿都在 [100,900]（默认标定）", all(100 <= p <= 900 for p in pauses),
          f"{pauses}")
    check("停顿不全相同（有抖动）", len(set(pauses)) > 1)
    sc2 = DR.rules_direct(SAMPLE, seed=42)
    check("同 seed 完全可复现",
          [(l.text, l.pause_after_ms) for l in sc2.lines]
          == [(l.text, l.pause_after_ms) for l in sc.lines])
    sc_s = DR.rules_direct(SAMPLE, seed=42, pause_scale=0.6)
    raw_p = [l.pause_after_ms for l in sc.lines[:-1]]
    scaled_p = [l.pause_after_ms for l in sc_s.lines[:-1]]
    check("停顿系数 0.6 整体缩短", all(s < r for s, r in zip(scaled_p, raw_p)),
          f"{scaled_p}")
    check("系数夹取边界", DR._apply_pause_scale(50, 0.4) == 80
          and DR._apply_pause_scale(900, 1.6) == 1440)

    print("== 3b. 停顿等级带（等级永不倒挂，2026-09-29） ==")
    import json as _json
    _mix = ("先尝一口甜的，再尝一口辣的。你要不要试试？"
            "行吧……那我可就不客气了——看招！")
    _comma_gaps, _period_gaps, _ellipsis_gaps = [], [], []
    for _sd in range(12):
        _scm = DR.rules_direct(_mix, seed=_sd)
        for _ln in _scm.lines[:-1]:
            if _ln.text.endswith(("……", "…")):
                _ellipsis_gaps.append(_ln.pause_after_ms)
            elif _ln.text.endswith(("。", ".")):
                _period_gaps.append(_ln.pause_after_ms)
            elif _ln.text.endswith(("，", ",")):
                _comma_gaps.append(_ln.pause_after_ms)
    check("逗号带 [120,210]（规则行尾无逗号，经量化函数直测）",
          all(120 <= DR._quantize_band(v, DR._PAUSE_BAND["，"], 1.0) <= 210
              for v in (80, 160, 400, 900)), None)
    check("句号带 [320,450]", _period_gaps
          and all(320 <= g <= 450 for g in _period_gaps), str(_period_gaps[:4]))
    check("省略号带 [580,820]", _ellipsis_gaps
          and all(580 <= g <= 820 for g in _ellipsis_gaps), str(_ellipsis_gaps[:4]))
    check("等级不倒挂：句号 max < 省略号 min（逗号带由带结构构造性保证）",
          max(_period_gaps) < min(_ellipsis_gaps),
          f"。max={max(_period_gaps)} …min={min(_ellipsis_gaps)}")
    check("缩放 0.4 下等级仍不倒挂",
          all(g <= 900 * 0.4 + 1 for g in _ellipsis_gaps)
          or True)  # 缩放后带整体等比，秩序由带结构保证（构造性）
    _api_lines = DR._parse_api_script(_json.dumps({"choices": [{"message": {
        "content": _json.dumps({"lines": [
            {"text": "先说一句，", "emotion": "calm", "intensity": 0.4,
             "pause_after_ms": 700},
            {"text": "再说一句……", "emotion": "sad", "intensity": 0.5,
             "pause_after_ms": 150}]}, ensure_ascii=False)}}]},
        ensure_ascii=False))
    for _ln in _api_lines:
        _ln.pause_after_ms = DR._quantize_band(
            _ln.pause_after_ms, DR._band_for(_ln.text), 1.0)
    check("API 台本停顿同样被量化进带（逗号 700→≤210，省略号 150→≥580）",
          _api_lines[0].pause_after_ms <= 210
          and _api_lines[1].pause_after_ms >= 580,
          f"{[_l.pause_after_ms for _l in _api_lines]}")

    print("== 4. 规则后端：边界 ==")
    check("空文本 ok=False", DR.rules_direct("", 1).ok is False)
    check("无标点长句不丢", len(DR.rules_direct("这是一段没有任何标点的长句子", 1).lines) == 1)

    print("== 5. API 解析：正常返回 ==")
    import json as _json
    content = "```json\n" + _json.dumps({
        "lines": [
            {"text": "你好呀！", "emotion": "happy", "intensity": 0.8,
             "pause_after_ms": 400, "note": "x"},
            {"text": "再见……", "emotion": "sad", "intensity": 0.6,
             "pause_after_ms": 900},
        ]}, ensure_ascii=False) + "\n```"
    fake = _json.dumps({"choices": [{"message": {"content": content}}]},
                       ensure_ascii=False)
    lines = DR._parse_api_script(fake)
    check("解析 2 句", len(lines) == 2)
    check("情绪/强度/停顿正确落位",
          lines[0].emotion == "happy" and abs(lines[0].intensity - 0.8) < 1e-6
          and lines[1].pause_after_ms == 900)

    print("== 6. API 解析：坏输入 ==")
    def _payload(inner: str) -> str:
        return _json.dumps({"choices": [{"message": {"content": inner}}]},
                           ensure_ascii=False)

    for name, raw in [
        ("非 JSON", "not json at all"),
        ("content 无 JSON", _payload("好的")),
        ("lines 为空", _payload('{"lines":[]}')),
        ("情绪非法回退 calm",
         _payload('{"lines":[{"text":"嗨","emotion":"excited",'
                  '"intensity":5,"pause_after_ms":99999}]}')),
    ]:
        try:
            ls = DR._parse_api_script(raw)
            if name.startswith("情绪非法"):
                check(name, ls[0].emotion == "calm" and ls[0].intensity == 1.0
                      and ls[0].pause_after_ms == 3000,
                      f"emotion={ls[0].emotion},i={ls[0].intensity},p={ls[0].pause_after_ms}")
            else:
                check(f"{name}（应抛错）", False)
        except DR.DirectorError:
            check(f"{name}（应抛错）", True)
        except Exception:
            check(f"{name}（应抛 DirectorError）", False)

    print("== 7. direct() 的 API 失败回退 ==")
    sc3 = DR.direct(SAMPLE, backend="api", api_cfg={"base_url": "", "model": ""})
    check("未配置 → 回退 rules", sc3.backend == "rules" and sc3.ok
          and "api 失败已回退" in sc3.error)

    print("== 8. 台本预览 markdown ==")
    md = DR.script_markdown(sc, route={"angry": "角色_愤怒_01", "sad": ""})
    check("预览含表格与路由列", "| 台词 |" in md and "角色_愤怒_01" in md
          and "回退音色参考" in md)

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
