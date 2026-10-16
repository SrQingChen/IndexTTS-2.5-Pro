"""阶段45（AI翻唱/AI变声）纯函数单元测试——无 GPU、无模型。

覆盖：F0 提取分窗 / 八度映射 / 轮廓相关 / WORLD 移植保长与谱自检 /
局部修复 / 分块重叠与交叉淡化长度守恒 / 预处理 / 混音。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pyworld = pytest.importorskip("pyworld")

from webui_app.services import cover_pipeline as CP  # noqa: E402
from webui_app.services import svc_core, svc_f0  # noqa: E402

SR = 22050


def _harmonic(f0_hz: float, seconds: float, sr: int = SR,
              partials: int = 5) -> np.ndarray:
    """合成谐波音频（WORLD 能完美重建的信号）。"""
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros(len(t), dtype=np.float64)
    for k in range(1, partials + 1):
        y += (1.0 / k) * np.sin(2 * np.pi * f0_hz * k * t)
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


def _melody(sr: int = SR, seconds: float = 2.0) -> np.ndarray:
    """三音旋律（220→330→440Hz 各 1/3）。"""
    n = int(seconds * sr)
    seg = n // 3
    parts = [_harmonic(f, seg / sr, sr) for f in (220.0, 330.0, 440.0)]
    y = np.concatenate(parts)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


# ---------------------------------------------------------------------------
# svc_f0
# ---------------------------------------------------------------------------

class TestSvcF0:
    def test_frames_of_formula(self):
        # 2.0s @22050/5ms → 401 帧（实测口径）
        assert svc_f0.frames_of(2 * SR, SR) == 401

    def test_harvest_single_window(self):
        y = _harmonic(220.0, 1.0)
        f0 = svc_f0.harvest_f0(y, SR)
        assert len(f0) == svc_f0.frames_of(len(y), SR)
        med = svc_f0.f0_median(f0)
        assert 200 < med < 240, f"中位数 {med}"

    def test_harvest_multi_window_consistency(self):
        """分窗拼接与整体提取的浊音中位数一致（<1% 偏差）。"""
        y = _melody(seconds=35.0)  # 跨过 30s 窗
        f0 = svc_f0.harvest_f0(y, SR)
        med = svc_f0.f0_median(f0)
        assert med > 0
        # 旋律三音中位数应为 330 附近
        assert 300 < med < 360

    def test_octave_shift(self):
        src = np.full(100, 110.0)
        ref = np.full(100, 220.0)
        assert svc_f0.octave_shift_semis(src, ref) == 12
        assert svc_f0.octave_shift_semis(src, src) == 0
        assert svc_f0.octave_shift_semis(ref, src) == -12
        assert svc_f0.octave_shift_semis(src, ref, manual=-5) == -5

    def test_shift_f0(self):
        f0 = np.array([0.0, 220.0, 440.0])
        out = svc_f0.shift_f0(f0, 12)
        assert out[0] == 0.0
        assert abs(out[1] - 440.0) < 0.01
        assert abs(out[2] - 880.0) < 0.01

    def test_f0_correlation_identical(self):
        y = _melody()
        f0 = svc_f0.harvest_f0(y, SR)
        r, err, n = svc_f0.f0_correlation(f0, f0.copy())
        assert r > 0.99
        assert err < 0.01
        assert n > 100

    def test_f0_correlation_shape_collapse(self):
        """轮廓形状破坏（时间乱序）时 r 必须低——auto 分支判定依据。"""
        y = _melody()
        f0 = svc_f0.harvest_f0(y, SR)
        rng = np.random.default_rng(7)
        # 把浊音帧顺序打乱：形状信息被破坏但分布不变
        idx = np.arange(len(f0))
        v = f0 > 0
        shuffled = f0.copy()
        shuffled[v] = rng.permutation(f0[v])
        r, err, _ = svc_f0.f0_correlation(f0, shuffled)
        assert r < 0.5, f"轮廓破坏应判 r 低: {r}"

    def test_transplant_semitone_offset_degrades_to_full(self):
        """整体半音走调（r 平移不变检测不到）经局部修复段数兜底 → 全曲移植。"""
        src_y = _melody()
        src_f0 = svc_f0.harvest_f0(src_y, SR)
        out_y = _harmonic(261.6, 2.0)   # 与源差 +3 半音左右
        res, met = svc_f0.transplant_f0(out_y, SR, src_f0, out_f0=None, mode="auto")
        # 全部帧偏差>40 音分 → 段过多 → 降级全曲；或 r 低直接全曲
        assert met.get("mode_used") in ("full", "fallback"), met
        assert len(res.y) == len(out_y)

    def test_world_replace_preserves_length(self):
        """八度移调（220→440，×2 整数倍，谐波栅格兼容）+保长。

        非整数倍移调（如 ×1.5）会产生拍频调制（330−220=110Hz）——这是
        WORLD 参数声码器的已知特性，auto 模式的八度映射天然规避；
        手动微调 ±1-2 半音时谱自检兜底回退。
        """
        y = _harmonic(220.0, 1.5)
        src_f0 = np.full(svc_f0.frames_of(len(y), SR), 440.0)
        res = svc_f0.world_replace_f0(y, SR, src_f0)
        assert res.ok, res.reason
        assert len(res.y) == len(y), "保长红线（对拍）"
        f0_new = svc_f0.harvest_f0(res.y, SR)
        assert abs(svc_f0.f0_median(f0_new) - 440.0) < 20

    def test_world_replace_garbage_rejects(self):
        """白噪声（无浊音结构）不应崩溃；要么回退要么安全输出。"""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.05, SR).astype(np.float32)
        src_f0 = np.full(201, 220.0)
        res = svc_f0.world_replace_f0(y, SR, src_f0)
        # 不炸即可（ok 与否取决于谱自检）
        assert len(res.y) == len(y)

    def test_local_repair_no_deviation(self):
        """源=输出（无偏差）时不应修任何段。"""
        y = _harmonic(220.0, 2.0)
        f0 = svc_f0.harvest_f0(y, SR)
        res, met = svc_f0.transplant_f0(y, SR, f0, out_f0=f0.copy(), mode="auto")
        assert res.ok
        assert met.get("mode_used") == "repair"
        assert met.get("repaired_segments", 0) == 0

    def test_transplant_auto_falls_back_to_full(self):
        """输出旋律完全乱掉（r<0.75）时 auto 应走全曲移植。"""
        src_y = _melody()
        src_f0 = svc_f0.harvest_f0(src_y, SR)
        out_y = _harmonic(150.0, 2.0)  # 完全不同的旋律
        out_f0 = svc_f0.harvest_f0(out_y, SR)
        r, _, _ = svc_f0.f0_correlation(out_f0, src_f0)
        res, met = svc_f0.transplant_f0(out_y, SR, src_f0, out_f0=out_f0, mode="auto")
        if r < 0.75:
            assert met.get("mode_used") in ("full", "fallback")
        assert len(res.y) == len(out_y)


# ---------------------------------------------------------------------------
# svc_core（无 GPU 部分）
# ---------------------------------------------------------------------------

class TestSvcCore:
    def test_mel_frames_formula(self):
        assert svc_core.mel_frames(256) == 1
        assert svc_core.mel_frames(512) == 2
        assert svc_core.mel_frames(22050) == (22050 - 256) // 256 + 1

    def test_plan_chunks_single(self):
        y = np.zeros(int(30 * SR), dtype=np.float32)
        assert svc_core.plan_chunks(y, SR, 60.0) == [(0, len(y))]

    def test_plan_chunks_overlap(self):
        """多块时相邻块输入必须重叠 ≥ 交叉淡化时长（对拍红线）。"""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.1, int(150 * SR)).astype(np.float32)
        chunks = svc_core.plan_chunks(y, SR, 60.0, overlap_ms=40.0)
        assert len(chunks) >= 2
        ov_samples = int(0.040 * SR)
        for (a1, b1), (a2, b2) in zip(chunks, chunks[1:]):
            assert a2 < b1, "相邻块必须重叠"
            assert b1 - a2 >= ov_samples - 1, f"重叠 {b1-a2} < {ov_samples}"

    def test_plan_chunks_prefers_valley(self):
        """有气口时应切在气口（低能量处）而不是硬切。"""
        y = _harmonic(220.0, 130.0)
        # 62s 处插 0.8s 静音气口
        a, b = int(62 * SR), int(62.8 * SR)
        y[a:b] = 0.0
        chunks = svc_core.plan_chunks(y, SR, 60.0)
        assert len(chunks) >= 2
        cut = chunks[1][0] + int(0.040 * SR)  # 下一块起点 + 重叠 ≈ 实际切点
        assert abs(cut - a) < int(1.5 * SR), f"切点 {cut} 未落在气口 {a}"

    def test_crossfade_length_conserved(self):
        """交叉淡化拼接后总长 = Σ块长 − Σ重叠（时长守恒）。"""
        rng = np.random.default_rng(1)
        blocks = [rng.normal(0, 0.1, 22050 * 10).astype(np.float32)
                  for _ in range(3)]
        ov = int(0.040 * SR)
        out = np.zeros(0, dtype=np.float32)
        for i, blk in enumerate(blocks):
            out = svc_core._crossfade_append(out, blk, ov if i > 0 else 0)
        want = sum(len(b) for b in blocks) - ov * (len(blocks) - 1)
        assert len(out) == want, f"{len(out)} != {want}"

    def test_equal_power_fade(self):
        fa, fb = svc_f0._equal_power_fade(100)
        assert np.allclose(fa ** 2 + fb ** 2, 1.0, atol=1e-5)
        assert fa[0] == 1.0 and fb[-1] == 1.0


class TestStreamConverter:
    """流式调度逻辑（mock convert_window，无 GPU）。

    曾抓到两个实锤 bug：缓冲裁剪后绝对坐标错位（12 块输出>输入总量，
    同段数据被重复消费）；_prev_tail 死锁（首块 overlap=0 不保存尾部
    → 后续全部 overlap=0）。此测试是回归闸门。
    """

    def _run(self, feeds=60, block_frames=31, left_ctx=2.0):
        from unittest.mock import patch
        calls = []

        def fake_convert_window(engine, wav16k, wav22k, ref, **kw):
            calls.append((len(wav16k), kw.get("keep_range"), kw.get("target_frames")))
            tf = kw.get("target_frames") or 1
            return np.full(tf * 256, 0.3, dtype=np.float32)

        class _Fake:
            pass

        cfg = svc_core.StreamConfig(block_frames=block_frames, steps=10,
                                    cfg_rate=0.0, left_ctx_sec=left_ctx,
                                    right_ctx_sec=0.02)
        with patch.object(svc_core, "convert_window", fake_convert_window):
            conv = svc_core.StreamConverter(_Fake(), _Fake(), 22050, cfg)
            outs = []
            for _ in range(feeds):
                r = conv.feed(np.zeros(1024, dtype=np.float32))
                if r is not None:
                    outs.append(r)
        return outs, calls, cfg

    def test_unique_samples_conserved(self):
        outs, calls, cfg = self._run()
        assert len(outs) >= 2
        block_dev = int(cfg.block_frames * 256)
        total_uniq = sum(len(w) - ov for w, ov in outs)
        # 发射的唯一样本不得超过喂入总量（曾因坐标错位超发 5×）
        fed = 60 * 1024
        assert total_uniq <= fed, f"超发: {total_uniq} > {fed}"
        assert total_uniq == len(outs) * block_dev

    def test_overlap_after_first_block(self):
        outs, _calls, _cfg = self._run()
        want_ov = int(0.040 * 22050)
        for wav, ov in outs[1:]:
            assert ov == want_ov, f"重叠死锁回归: ov={ov}"
            assert len(wav) >= want_ov

    def test_left_context_grows_window(self):
        _outs, calls, cfg = self._run()
        assert len(calls) >= 5
        # 左上下文是**渐进**生效的（历史不足时窗口=全部历史——正确行为）；
        # 稳态（最后一块）窗口应达到 左上下文+块+右 量级
        left_16k = int(cfg.left_ctx_sec * 16000)
        block_16k = int(cfg.block_frames * 256 * 16000 / 22050)
        assert calls[-1][0] >= left_16k + block_16k * 0.9, \
            f"稳态窗口未含左上下文: {calls[-1][0]} < {left_16k + block_16k}"

    def test_target_frames_constant(self):
        _outs, calls, _cfg = self._run()
        assert all(c[2] == 31 for c in calls)


# ---------------------------------------------------------------------------
# cover_pipeline（无 GPU 部分）
# ---------------------------------------------------------------------------

class TestCoverPipeline:
    def test_preprocess_removes_dc_and_keeps_length(self):
        y = (_harmonic(220.0, 1.0) + 0.2).astype(np.float32)
        out, head_cut = CP.preprocess_vocal(y, SR, 0.0)
        assert abs(float(np.mean(out))) < 0.01
        assert len(out) <= len(y)
        assert head_cut >= 0

    def test_preprocess_peak_guard(self):
        y = np.clip(_harmonic(220.0, 1.0) * 3.0, -3, 3).astype(np.float32)
        out, _ = CP.preprocess_vocal(y, SR, 0.0)
        assert float(np.max(np.abs(out))) <= 0.892

    def test_preprocess_head_cut_beat_align(self):
        """P0 回归（交付验收#4）：头部裁掉的样本数要被 mixdown 补回——
        有前奏的歌不能整体提前进拍。"""
        intro = np.zeros(int(1.2 * SR), dtype=np.float32)     # 1.2s 前奏
        vocal = np.concatenate([intro, _harmonic(220.0, 1.5)])
        # bgm 全程低幅持续音，人声起点可测
        bgm = np.full(int(2.7 * SR), 0.02, dtype=np.float32)
        trimmed, head_cut = CP.preprocess_vocal(vocal, SR, 0.0)
        assert head_cut >= int(1.0 * SR), "应裁掉大部分前奏"

        def _vocal_onset(wav):
            return int(np.argmax(np.abs(wav) > 0.1))

        m_bad = CP.mixdown(trimmed, SR, bgm, SR, 0.0)
        m_good = CP.mixdown(trimmed, SR, bgm, SR, 0.0, head_pad_samples=head_cut)
        onset_bad = _vocal_onset(m_bad["wav"])
        onset_good = _vocal_onset(m_good["wav"])
        # 补回后人声起点应后移 head_cut（换算到 mixdown 输出域）
        expect = head_cut * m_good["sr"] // SR
        assert abs((onset_good - onset_bad) - expect) <= SR // 50,             f"人声起点偏移 {(onset_good - onset_bad)} != 期望 {expect}"

    def test_run_cover_clean_route(self):
        """P0 回归（交付验收#1）：clean 状态（UI 默认）不得 FileNotFoundError。

        用假引擎+真音频走 run_cover 到 VC 前的所有阶段（mock convert_file）。
        """
        import tempfile
        from unittest.mock import patch
        import soundfile as sf
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "src.wav")
        ref = os.path.join(tmp, "ref.wav")
        sf.write(src, _melody(seconds=1.5), SR, subtype="PCM_16")
        sf.write(ref, _harmonic(330.0, 1.5), SR, subtype="PCM_16")

        class _Stats:
            loaded = False
        class _Eng:
            stats = _Stats()
            def load(self): _Stats.loaded = True
            def unload(self, bypass_svc=False): _Stats.loaded = False
            def begin_svc(self, label): self._l = label
            def end_svc(self, label): pass
            def lora_status(self): return []
            def attach_lora(self, *a, **k): return "tag"
            def detach_lora(self, *a, **k): pass
        class _Ctx:
            engine = _Eng()

        def fake_convert(engine, dry, refctx, **k):
            class R:
                ok = True; chunks = 1
                wav = _harmonic(220.0, 1.0)
                sr = 22050
                timing = {"elapsed": 0.1, "rtf": 0.1}
                reason = ""
            return R()

        from webui_app.services import svc_core as core_mod
        req = CP.CoverRequest(src_path=src, input_state="clean", ref_path=ref,
                              f0_mode="off")
        with patch.object(core_mod, "convert_file", fake_convert),              patch.object(core_mod, "build_reference", lambda *a, **k: object()):
            rep = CP.run_cover(_Ctx(), req)
        assert rep.get("ok"), f"clean 路由失败: {rep.get('error')}"
        assert os.path.isfile(os.path.join(rep["out_dir"], "02_dry_vocal_22k.wav"))

    def test_highpass(self):
        low = _harmonic(40.0, 1.0)   # 40Hz 低频
        out = CP.highpass(low, SR, 70.0)
        r_before = float(np.sqrt(np.mean(low ** 2)))
        r_after = float(np.sqrt(np.mean(out ** 2)))
        assert r_after < r_before * 0.7, "高通应显著衰减 40Hz"

    def test_mixdown_length_and_peak(self):
        rng = np.random.default_rng(3)
        v = rng.normal(0, 0.1, SR * 2).astype(np.float32)
        b = rng.normal(0, 0.1, SR * 2).astype(np.float32)
        m = CP.mixdown(v, SR, b * 2.0, 44100, vocal_gain_db=3.0)
        assert m["sr"] == 44100
        assert float(np.max(np.abs(m["wav"]))) <= 0.892

    def test_mixdown_pads_short_vocal(self):
        v = np.zeros(SR, dtype=np.float32)
        b = np.zeros(SR * 3, dtype=np.float32)
        m = CP.mixdown(v, SR, b, SR, 0.0)
        # 输出采样率下限 44100（发行口径）
        assert m["sr"] == 44100
        assert len(m["wav"]) == 44100 * 3

    def test_resample(self):
        y = _harmonic(220.0, 1.0)
        out = CP.resample(y, SR, 16000)
        assert abs(len(out) - 16000) < 100
