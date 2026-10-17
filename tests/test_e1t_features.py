"""E1-T 特征/训练接线单元测试（无 GPU、无模型加载）。

覆盖（会话规划 WP2/WP3）：
- F0 5ms → mel 网格对齐（长度/清音不架桥）；
- technique 段展开；tech_dom 主导技巧统计；
- SHC 八度跳变修正（错误修复 / 真实大跳保留）；
- 音区映射（分数半音、保旋律）；
- build_cfm_pair 的条件拼接（prompt 帧用参考自身 F0，r1-2）；
- truncate 同步（f0/tech 与 mel 同帧截断）；
- collate_cfm 的逐帧统计（prompt/target 各用各的，防跨说话人污染）；
- CfmTrainOptions.singing 校验。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

pyworld = pytest.importorskip("pyworld")

from webui_app.services import svc_f0  # noqa: E402
from webui_app.training import cfm_lora  # noqa: E402
from webui_app.training import features as FT  # noqa: E402

SR = 22050
MEL_FPS = 22050.0 / 256.0


def _sample(T=50, with_f0=True, tech=None):
    d = {
        "mel": torch.randn(80, T).half(),
        "mu_prompt": torch.randn(T, 512).half(),
        "mu_target": torch.randn(T, 512).half(),
        "style": torch.randn(192),
        "emo_vec": torch.randn(1280),
        "text_tokens": torch.ones(5, dtype=torch.int64),
        "mel_len": T,
        "n_codes": max(1, round(T / 3.44)),
        "n_text_tokens": 5,
    }
    if with_f0:
        d["f0"] = torch.rand(T) * 400 + 150
        d["f0"][:5] = 0                        # 头部清音
        d["f0_mean"], d["f0_std"] = 8.0, 0.4
        d["tech"] = (tech if tech is not None
                     else torch.zeros(T, dtype=torch.uint8))
    return d


# ---------------------------------------------------------------------------
# F0 网格 / technique
# ---------------------------------------------------------------------------

class TestF0Grid:
    def test_to_mel_grid_length_and_no_bridge(self):
        f0 = np.zeros(1000)                    # 5ms → 5s
        f0[:400] = 220.0                       # 前 2s 浊音，后 3s 清音
        out = FT.f0_to_mel_grid(f0, int(5 * MEL_FPS))
        assert len(out) == int(5 * MEL_FPS)
        assert (out[:int(2 * MEL_FPS) - 20] > 100).mean() > 0.9   # 前段浊音
        assert (out[int(3 * MEL_FPS):] == 0).all()                # 后段清音

    def test_expand_technique_segments(self):
        segs = [(0.0, 1.0, 2), (1.5, 2.5, 4)]
        T = int(3 * MEL_FPS)
        tech = FT.expand_technique_segments(segs, T)
        assert len(tech) == T
        assert (tech[:int(1 * MEL_FPS)] == 2).all()
        assert (tech[int(1 * MEL_FPS):int(1.5 * MEL_FPS)] == 0).all()
        assert (tech[int(1.5 * MEL_FPS):int(2.5 * MEL_FPS)] == 4).all()
        assert (tech[int(2.5 * MEL_FPS):] == 0).all()
        # 越界 id 钳位
        t2 = FT.expand_technique_segments([(0.0, 0.5, 99)], T)
        assert t2.max() <= 6

    def test_check_feature_singing_fields(self):
        d = _sample(40)
        d["tech"] = d["tech"].clone()
        d["f0"] = d["f0"].clone()
        assert FT.check_feature(d) == [] or all(
            "f0" not in p and "tech" not in p for p in FT.check_feature(d))
        bad = dict(d)
        bad["f0"] = torch.zeros(7)             # 长度错
        probs = FT.check_feature(bad)
        assert any("f0 帧数" in p for p in probs)


class TestOctaveCorrection:
    def test_fixes_short_octave_error(self):
        # 220Hz 稳态，中间 200ms 跳到 440（估计错误，随后跳回）
        n = int(3.0 / 0.005)
        f0 = np.full(n, 220.0)
        a, b = int(1.0 / 0.005), int(1.2 / 0.005)
        f0[a:b] = 440.0
        out = svc_f0.correct_octave_jumps(f0)
        assert np.allclose(out, 220.0, atol=1e-6), "短促八度跳变应被修正回 220"

    def test_preserves_real_octave_leap(self):
        # 前 1.5s 220Hz → 后 1.5s 440Hz（持续换声区=真实大跳，不回返）
        n = int(3.0 / 0.005)
        f0 = np.full(n, 220.0)
        f0[int(1.5 / 0.005):] = 440.0
        out = svc_f0.correct_octave_jumps(f0)
        assert abs(np.median(out[:int(1.0 / 0.005)]) - 220.0) < 5
        assert abs(np.median(out[int(2.0 / 0.005):]) - 440.0) < 5, \
            "持续的八度大跳必须保留（戏腔/歌剧跳进）"

    def test_passthrough_clean(self):
        f0 = np.full(500, 330.0)
        f0[::50] = 0                            # 清音
        out = svc_f0.correct_octave_jumps(f0)
        assert np.array_equal(out, f0)


class TestRegisterShift:
    def test_fractional_and_clamped(self):
        src = np.full(2000, 390.0)              # 源中位
        ref = np.full(2000, 257.0)              # 参考中位
        s = svc_f0.register_shift_semis(src, ref)
        expect = 12.0 * np.log2(257.0 / 390.0)  # ≈ -7.2
        assert abs(s - expect) < 0.01
        # +73 半音 → 钳到 +12（把源拉向高处目标音区，方向为正）
        assert svc_f0.register_shift_semis(np.full(10, 100.0),
                                           np.full(10, 2000.0)) == 12.0
        assert svc_f0.register_shift_semis(np.full(10, 0.0),
                                           np.full(10, 200.0)) == 0.0


# ---------------------------------------------------------------------------
# 配对 / 截断 / collate
# ---------------------------------------------------------------------------

class TestCfmPairConditions:
    def test_pair_carries_conditions(self):
        p, t = _sample(30), _sample(50)
        pair = FT.build_cfm_pair(p, t)
        assert "f0" in pair and "tech" in pair
        assert pair["f0"].shape[0] == 80                    # 30+50
        # r1-2：prompt 帧区段 = prompt 样本自身 f0
        assert torch.allclose(pair["f0"][:30], p["f0"].float())
        assert torch.allclose(pair["f0"][30:], t["f0"].float())
        assert pair["f0_stats_prompt"] == (p["f0_mean"], p["f0_std"])
        assert pair["f0_stats_target"] == (t["f0_mean"], t["f0_std"])

    def test_pair_without_f0_omits_conditions(self):
        p, t = _sample(30, with_f0=False), _sample(50, with_f0=False)
        pair = FT.build_cfm_pair(p, t)
        assert "f0" not in pair and "tech" not in pair

    def test_truncate_syncs_f0_tech(self):
        s = _sample(60)
        tp = cfm_lora.truncate_prompt(s, 20)
        assert tp["f0"].shape[0] == 20 and tp["tech"].shape[0] == 20
        tt = cfm_lora.truncate_target(s, 15)
        assert tt["f0"].shape[0] == 15

    def test_collate_per_region_stats(self):
        p1, t1 = _sample(20), _sample(40)
        p2, t2 = _sample(30), _sample(35)
        pa = FT.build_cfm_pair(p1, t1)
        pb = FT.build_cfm_pair(p2, t2)
        b = cfm_lora.collate_cfm([pa, pb])
        assert b["f0"].shape == (2, 65)
        assert b["technique"].dtype == torch.long
        # 样本0：prompt 区 [0:20) 用 p1 统计，target 区用 t1 统计
        assert (b["f0_mean"][0, :20] == p1["f0_mean"]).all()
        assert (b["f0_mean"][0, 20:60] == t1["f0_mean"]).all()
        # padding 区（60:65）沿用 target 统计（掩码外，无影响）
        assert b["f0_std"].min() >= 0.1

    def test_collate_without_conditions_legacy(self):
        p1, t1 = _sample(20, with_f0=False), _sample(40, with_f0=False)
        b = cfm_lora.collate_cfm([FT.build_cfm_pair(p1, t1)])
        assert "f0" not in b and "technique" not in b   # 老数据集零改动


class TestTrainOptions:
    def test_singing_validation(self):
        opt = cfm_lora.CfmTrainOptions(singing=True, tech_ce_weight=2.0)
        errs = [n.message for n in opt.validate() if n.level == "error"]
        assert any("tech_ce_weight" in e for e in errs)
        opt2 = cfm_lora.CfmTrainOptions(singing=False,
                                        singing_tech_balance=True)
        infos = [n.message for n in opt2.validate()]
        assert any("不生效" in i for i in infos)
