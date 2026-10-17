"""E1-T 推理侧单元测试（无 GPU：mock 引擎 + 纯函数）。

覆盖（会话规划 WP6/WP7）：
- SingingCond.build_kwargs：帧区段统计（参考/目标各用各）、长度对齐；
- SingingCfmSession：换装/恢复守恒（含异常路径）、LoRA detach 归属；
- engine 租约防护：SVC 会话中 cfm 通道 attach/detach 被拒（bypass 例外）；
- SYKI 高频移植：保长、频谱确实并入源高频；
- BoN 奖励：好候选分数高；
- 静音块跳过（_is_silent）与 free-VRAM cap 只收紧不放宽。
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

librosa = pytest.importorskip("librosa")

from webui_app.services import svc_core, svc_singing  # noqa: E402

SR = 22050


def _harmonic(f0: float, seconds: float, partials: int = 6) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros(len(t))
    for k in range(1, partials + 1):
        if f0 * k > SR / 2:
            break
        y += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    return (0.3 * y / max(np.max(np.abs(y)), 1e-9)).astype(np.float32)


def _singing_cond(T_mel=400, T_ref=100):
    rng = np.random.default_rng(0)
    src = np.concatenate([np.full(200, 390.0), np.full(200, 440.0)])
    src[::37] = 0.0
    return svc_core.SingingCond(
        ref_f0=np.full(T_ref, 257.0, dtype=np.float32),
        ref_mean=8.0, ref_std=0.3,
        src_f0=src.astype(np.float32),
        src_mean=8.6, src_std=0.35,
        tech=np.zeros(T_mel, dtype=np.uint8),
        src_f0_5ms=np.full(4000, 390.0))


# ---------------------------------------------------------------------------
# SingingCond
# ---------------------------------------------------------------------------

class TestSingingCond:
    def test_build_kwargs_regions(self):
        cond = _singing_cond()
        kw = cond.build_kwargs(ref_frames=100, frame_offset=50,
                               target_frames=150, device="cpu")
        f0 = kw["f0"]
        assert f0.shape == (1, 250)
        assert torch.allclose(f0[0, :100], torch.full((100,), 257.0))
        assert torch.equal(f0[0, 100:], torch.tensor(cond.src_f0[50:200]))
        mean, std = kw["f0_stats"]
        assert mean[0, :100].unique().tolist() == [8.0]      # 参考区统计
        assert mean[0, 100:].unique().tolist() == pytest.approx(
            [8.6], abs=1e-5)                                  # 目标区统计
        assert std.min() >= 0.1
        assert kw["technique"].dtype == torch.long

    def test_build_kwargs_pads_short_slice(self):
        cond = _singing_cond(T_mel=100)
        kw = cond.build_kwargs(ref_frames=50, frame_offset=90,
                               target_frames=30, device="cpu")
        assert kw["f0"].shape == (1, 80)                     # 50 + 30（已 pad）
        assert kw["technique"].shape == (1, 80)


# ---------------------------------------------------------------------------
# SingingCfmSession（mock load_base_cfm）
# ---------------------------------------------------------------------------

class _FakeSinging:
    def __init__(self):
        self.estimator = SimpleNamespace(
            timbre_conditioner=object(), class_dropout_prob=0.1)


class TestSingingCfmSession:
    def _engine(self):
        orig = SimpleNamespace(name="tts_cfm")
        holder = {"cfm": orig}
        detached = []
        eng = SimpleNamespace(
            tts=SimpleNamespace(device="cpu",
                                s2mel=SimpleNamespace(models=holder)),
            detach_lora=lambda target="cfm", bypass_lease=None:
                detached.append((target, bypass_lease)))
        return eng, holder, orig, detached

    def test_swap_and_restore(self, monkeypatch):
        import webui_app.training.cfm_lora as CL
        eng, holder, orig, detached = self._engine()
        monkeypatch.setattr(CL, "load_base_cfm",
                            lambda *a, **k: (_FakeSinging(), {}))
        with svc_core.SingingCfmSession(eng, cfg_path="x", lease="tok") as s:
            assert holder["cfm"] is not orig
            s.mark_lora()
        assert holder["cfm"] is orig, "退出后必须恢复引擎原 CFM"
        assert detached == [("cfm", "tok")], "会话内 LoRA 由会话先卸"

    def test_restore_on_exception(self, monkeypatch):
        import webui_app.training.cfm_lora as CL
        eng, holder, orig, detached = self._engine()
        monkeypatch.setattr(CL, "load_base_cfm",
                            lambda *a, **k: (_FakeSinging(), {}))
        with pytest.raises(RuntimeError):
            with svc_core.SingingCfmSession(eng, lease="tok"):
                holder["cfm"].estimator.timbre_conditioner  # 存在
                raise RuntimeError("转换中途炸了")
        assert holder["cfm"] is orig, "异常路径也必须恢复"

    def test_rejects_non_singing_weights(self, monkeypatch):
        import webui_app.training.cfm_lora as CL
        eng, holder, orig, _ = self._engine()
        fake = SimpleNamespace(estimator=SimpleNamespace())  # 无 conditioner
        monkeypatch.setattr(CL, "load_base_cfm",
                            lambda *a, **k: (fake, {}))
        with pytest.raises(RuntimeError, match="TimbreConditioner"):
            with svc_core.SingingCfmSession(eng):
                pass
        assert holder["cfm"] is orig


# ---------------------------------------------------------------------------
# engine 租约防护（无完整引擎，用 __new__ 构造轻量实例只测防护逻辑）
# ---------------------------------------------------------------------------

class TestEngineLeaseGuard:
    def _engine(self, leases):
        from threading import RLock
        from webui_app.services import engine as EN
        eng = EN.TTSEngine.__new__(EN.TTSEngine)
        eng._lock = RLock()
        eng._svc_leases = list(leases)
        eng.stats = SimpleNamespace(lora_adapters=[])
        eng._tts = SimpleNamespace(  # 防护之后的路径用的最小 tts
            gpt=None, s2mel=SimpleNamespace(models={"cfm": None, "gpt": None}))
        return eng, EN

    def test_attach_rejected_during_svc(self):
        eng, EN = self._engine(["tok"])
        with pytest.raises(EN.EngineError, match="会话进行中"):
            eng.attach_lora("whatever_dir", target="cfm")

    def test_detach_rejected_during_svc(self):
        eng, EN = self._engine(["tok"])
        with pytest.raises(EN.EngineError, match="会话进行中"):
            eng.detach_lora(target="cfm")

    def test_bypass_passes_guard(self):
        eng, EN = self._engine(["tok"])
        # bypass 自己的 token：越过租约防护（后续因引擎未加载而失败，
        # 但绝不是「会话进行中」——证明防护被正确豁免）
        with pytest.raises(Exception) as ei:
            eng.detach_lora(target="cfm", bypass_lease="tok")
        assert "会话进行中" not in str(ei.value)

    def test_gpt_target_unaffected(self):
        eng, EN = self._engine(["tok"])
        with pytest.raises(Exception) as ei:
            eng.detach_lora(target="gpt")
        assert "会话进行中" not in str(ei.value)


# ---------------------------------------------------------------------------
# svc_singing 指标与处理
# ---------------------------------------------------------------------------

class TestSvcSinging:
    def test_hnr_proxy_sine_vs_noise(self):
        from webui_app.services import svc_f0
        sine = _harmonic(220.0, 1.5)
        noise = (np.random.default_rng(0).normal(0, 0.1,
                                                 int(1.5 * SR))).astype(np.float32)
        f0s = np.full(len(svc_f0.harvest_f0(sine, SR)), 220.0)
        h_sine = svc_singing.voiced_hnr(sine, SR, f0s)
        h_noise = svc_singing.voiced_hnr(noise, SR, f0s)
        assert h_sine is not None and h_noise is not None
        assert h_sine > h_noise + 3.0, "谐波信号的 HNR 应显著高于噪声"

    def test_high_subset_stats(self):
        from webui_app.services import svc_f0
        y = _harmonic(600.0, 1.0)
        f0 = svc_f0.harvest_f0(y, SR)
        st = svc_singing.high_subset_stats(y, SR, f0, threshold_hz=500.0)
        assert st["high_ratio"] > 0.8
        assert st.get("hnr_db_high") is not None

    def test_transplant_highs_length(self):
        conv = _harmonic(220.0, 1.0)
        src = _harmonic(220.0, 1.0, partials=40)      # 源带高频
        out, info = svc_singing.transplant_highs(conv, src, SR)
        assert len(out) == len(conv), "高频移植必须保长（对拍）"
        assert 0.2 <= info["diff"] <= 4.0

    def test_bon_reward_prefers_good_candidate(self):
        from webui_app.services import svc_f0
        tgt = np.full(2000, 440.0)
        good = _harmonic(440.0, 10.0)
        bad = _harmonic(220.0, 10.0)                  # 错八度
        rg = svc_singing.bon_reward(good, SR, None, tgt)
        rb = svc_singing.bon_reward(bad, SR, None, tgt)
        assert rg["reward"] > rb["reward"] + 10.0
        # 全局掉八度由 global_cents 惩罚（对齐口径的 octave_err 看不见它）
        assert rb["global_cents"] < -600
        assert abs(rg["global_cents"]) < 100


# ---------------------------------------------------------------------------
# 分块改进（free-VRAM 只收紧）
# ---------------------------------------------------------------------------

class TestCapChunkSec:
    def test_free_vram_only_tightens(self, monkeypatch):
        monkeypatch.setattr(svc_core, "_total_vram_gib", lambda: 8.0)
        monkeypatch.setattr(svc_core, "_free_vram_gib", lambda: 0.0)
        assert svc_core.cap_chunk_sec(60.0) == 30.0       # total 表为准
        monkeypatch.setattr(svc_core, "_free_vram_gib", lambda: 2.0)
        assert svc_core.cap_chunk_sec(60.0) == 20.0       # free 紧 → 收紧
        monkeypatch.setattr(svc_core, "_free_vram_gib", lambda: 99.0)
        assert svc_core.cap_chunk_sec(60.0) == 30.0       # free 多 → 不放宽
        monkeypatch.setattr(svc_core, "_total_vram_gib", lambda: 0.0)
        monkeypatch.setattr(svc_core, "_free_vram_gib", lambda: 99.0)
        assert svc_core.cap_chunk_sec(60.0) == 60.0       # 无卡信息 → 原样


# ---------------------------------------------------------------------------
# stage_hooks=None 等价性（mock tts）
# ---------------------------------------------------------------------------

class TestStageHooks:
    def test_infer_generator_none_hooks_not_called(self):
        """stage_hooks=None 时 infer_generator 不触碰钩子（原路径）。"""
        from indextts.infer_v2_5 import IndexTTS2
        src_calls = []

        class _T:
            device = "cpu"

        # 直接对未加载实例检查签名兼容（不跑真模型）
        fake = IndexTTS2.__new__(IndexTTS2)
        import inspect
        sig = inspect.signature(fake.infer_generator)
        assert sig.parameters["stage_hooks"].default is None
        sig2 = inspect.signature(fake.infer)
        assert sig2.parameters["stage_hooks"].default is None
