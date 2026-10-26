"""末音节气泡音下压 + 韵尾延长门槛回归（2026-10-26）。

病例：用户报「请你完整了解当前项目」里的「项目」被合成成「项、目」，
「项」延音下压、「目」被压得很低。取证结论：
  · 原始 8 候选里 2 个末音节掉到 84~96Hz（中位 241~255Hz 的 0.35 倍），
    其余 6 个正常——是 GPT/CFM 双 LoRA 交互的候选级缺陷，不是停顿治理；
  · 现有 reward 不看音高，坏候选可以靠音色分进并列组；
  · 旧韵尾延长门槛 100ms 会把 100~150ms 的词间隙当成「急收」，拉长前字
    后正好放大「项、目」听感。

修复：BoN 并列组内优先无气泡音候选；韵尾延长门槛提到 150ms（与块内
停顿治理同一把尺）。本文件只测纯函数，不加载模型。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webui_app.services import audio_lab as AL
from webui_app.services.orchestrator import _choose_from_tied

SR = 22050


def _voice_like(sec: float, f0_hz: float = 220.0, amp: float = 0.5,
                seed: int = 3) -> np.ndarray:
    """类语音谐波（5Hz 颤音 + 微噪声）——harvest 对纯正弦判全无声带。"""
    t = np.arange(int(sec * SR)) / SR
    ph = 2 * np.pi * f0_hz * (t + 0.01 * np.sin(2 * np.pi * 5 * t))
    w = (amp * np.sin(ph) + 0.3 * amp * np.sin(2 * ph)
         + 0.15 * amp * np.sin(3 * ph)
         + 0.02 * np.random.default_rng(seed).normal(size=len(t)))
    return w.astype(np.float32)


def test_clean_final_syllable_scores_zero():
    y = np.concatenate([_voice_like(0.8, 225), _voice_like(0.35, 215)])
    r = AL.final_creak_score(y, SR)
    assert r["ok"] and r["score"] == 0.0, r


def test_audible_octave_down_final_flags():
    # 前段正常，末 0.2s 掉到 95Hz（约 -15 半音）——「目」被压低的实体
    y = np.concatenate([_voice_like(0.8, 230),
                        _voice_like(0.15, 225),
                        _voice_like(0.20, 95, amp=0.45)])
    r = AL.final_creak_score(y, SR)
    assert r["ok"] and r["score"] > 0.1, r
    assert r["min_ratio"] < 0.6 and r["low_ms"] >= 30, r


def test_inaudible_low_tail_is_not_flagged():
    # 数字静音/极低电平里的 harvest 噪声不是问题（真实 15:13 候选 -89dB
    # 的 88Hz 就是这种），能量门必须拦住
    y = np.concatenate([_voice_like(0.8, 230),
                        _voice_like(0.2, 95, amp=0.001)])
    r = AL.final_creak_score(y, SR)
    assert r["score"] == 0.0, r


def test_tie_break_prefers_clean_candidate_and_is_noop_when_all_clean():
    rewards = [0.98, 0.90, 0.95]
    # 0.98 是气泡音，0.90 干净且差距在并列带内 → 选干净者
    creak = {0: 0.62, 1: 0.0, 2: 0.41}
    assert _choose_from_tied([0, 1, 2], rewards, creak) == 1
    # 全干净 → 严格回到旧行为（最高奖励）
    assert _choose_from_tied([0, 1, 2], rewards, {0: 0.0, 1: 0.0, 2: 0.0}) == 0
    # 有 continuity 时，全干净仍按旧行为（最小音区跳变）
    cont = {0: 0.5, 1: 0.1, 2: 0.3}
    assert _choose_from_tied([0, 1, 2], rewards,
                             {0: 0.0, 1: 0.0, 2: 0.0},
                             lambda k: cont[k]) == 1
    # 气泡音优先于连续性：干净的 0 即使衔接最差也胜出
    assert _choose_from_tied([0, 1, 2], rewards,
                             {0: 0.0, 1: 0.5, 2: 0.2},
                             lambda k: cont[k]) == 0


def test_release_gate_ignores_word_gap_but_keeps_punctuation_pause():
    # 120ms 词间隙（「项|目」尺度）不触发；200ms 标点/块尾停顿仍触发
    gap120 = np.concatenate([_voice_like(0.9), np.zeros(int(0.12 * SR),
                                                       np.float32),
                             _voice_like(0.9)])
    _out, rel = AL.lengthen_abrupt_releases(gap120, SR)
    assert rel == [], rel

    gap200 = np.concatenate([_voice_like(0.9), np.zeros(int(0.20 * SR),
                                                       np.float32),
                             _voice_like(0.9)])
    _out2, rel2 = AL.lengthen_abrupt_releases(gap200, SR)
    assert len(rel2) == 1, rel2
