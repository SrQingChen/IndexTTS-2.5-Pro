"""标点—静音匹配：慢速首短语的词间隙不得冒领逗号保底位（2026-10-26）。

病例（16:54 祀，第一句「伤没好彻底，」）：首短语比平均语速慢，线性
估计把逗号放在 0.838s；0.718s 的 20ms「好|彻」词间隙成了最近候选，
保底位把 112ms 插在「好」后 = 用户听到的异常停顿；真逗号 1.197s/60ms
在 0.35s 窗外，被判 below_gate。

修复：词间隙屏蔽（紧邻的下一个收集段 ≤0.30s 且明显更长 → 前一段是
词间隙）+ 匹配用静音起点（不是中点）+ 窗口 0.35→0.38。本测试用合成
夹具复现同一几何：词间隙在 0.708s，真逗号在 0.958s，线性估计 0.599s。
旧代码会把保底位插在 0.708s；新代码必须让 0.958s 的真逗号入带。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui_app.services import audio_lab as AL

SR = 22050


def _tone(sec: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _runs_ms(y: np.ndarray, thresh_db: float = -30.0) -> list:
    fr = int(SR * 0.01)
    n = len(y) // fr
    db = 20 * np.log10(np.sqrt(np.mean(y[:n * fr].reshape(n, fr) ** 2,
                                        axis=1) + 1e-12) + 1e-10)
    q = db < thresh_db
    out, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            if (j - i) * 10 >= 20:
                out.append((i * 0.01, (j - i) * 10))
            i = j
        else:
            i += 1
    return out


def test_slow_first_phrase_word_gap_does_not_steal_comma():
    text = "伤没好彻底，就急着出远门。"
    # 0-0.70s 三个字（慢）| 50ms 词间隙 | 0.70-0.90s 两个字 |
    # 90ms 真逗号 | 后一短语 0.4s
    y = np.concatenate([
        _tone(0.70), np.zeros(int(0.050 * SR), np.float32),
        _tone(0.20), np.zeros(int(0.090 * SR), np.float32),
        _tone(0.40)]).astype(np.float32)
    audit: list = []
    out = AL.normalize_intra_pauses(
        y, SR, text_chars=len(text), cap_ms=210, text=text,
        punct_floor_ms=120, punct_cap_ms=210, audit_out=audit)

    punct = [a for a in audit if a["kind"] == "punct"]
    assert punct, audit
    first = punct[0]
    # 回归前提：线性估计确实落在词间隙附近（<0.70s 的真逗号之前）
    assert first["est_s"] < 0.70, first
    assert first["matched_run"] is not None, first
    assert first["matched_run_ms"] >= 50, first

    runs = [a for a in audit if a["kind"] == "run"]
    # 0.708s 的词间隙绝不能被 mandate（旧代码的病灶）
    assert not any(a.get("action", "").startswith("mandated")
                   and a.get("start_s", 0) < 0.80 for a in runs), runs
    # 真逗号（0.958s）必须被补插到带下限
    assert any(a.get("action") == "mandated_insert"
               and 0.90 <= a.get("start_s", 0) <= 1.05 for a in runs), runs

    out_runs = _runs_ms(out)
    assert any(0.90 <= s <= 1.10 and d >= 100 for s, d in out_runs), out_runs
    # 词间隙仍是短缝（没有被补成可闻停顿）
    assert not any(0.65 <= s <= 0.80 and d >= 60 for s, d in out_runs), out_runs
