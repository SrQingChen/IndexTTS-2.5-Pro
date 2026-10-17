"""audio_sr 尾块补零回归测试（无 GPU：monkeypatch 推理与模型加载）。

钉住的真机事故（2026-10-17 复现）：43.8s 合成成品的第 10 块（~1.02s 尾块）
触发 AudioSR 内部 "Expected size 64 but got size 63" 形状错配。修复=尾块
补零到整块形状、按 orig_n 裁剪。本测试验证：
- 送入推理的**每一段**都是整块形状（含尾块）；
- 尾块记录了真实 orig_n（< 整块长）；
- 输出时长与输入严格一致（补零不进入成品）。
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

soundfile = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")

from webui_app.services import audio_sr as SR  # noqa: E402


def test_tail_chunk_padded_and_length_conserved(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    captured = []

    def fake_chunk(model, y, in_sr, tmp, ddim, gs, seed, orig_n=None):
        captured.append((len(y), orig_n))
        n = orig_n if orig_n is not None else len(y)
        return np.zeros(int(n * 48000 / in_sr), dtype=np.float32)

    monkeypatch.setattr(SR, "_enhance_chunk", fake_chunk)
    monkeypatch.setattr(SR, "_get_model", lambda: object())

    dur, sr = 43.8, 22050          # 事故现场的参数：10 块，尾块 ~1.02s
    y = np.zeros(int(dur * sr), dtype=np.float32)
    pin = tmp_path / "in.wav"
    soundfile.write(pin, y, sr, subtype="PCM_16")

    r = SR.enhance_file(str(pin), out_path=str(tmp_path / "out.wav"))

    chunk_n = int(SR.CHUNK_SEC * sr)
    lens = [l for l, _o in captured]
    origs = [o for _l, o in captured]
    assert len(captured) == 10
    assert all(l == chunk_n for l in lens), \
        f"所有块必须是整块形状（尾块补零）：{lens}"
    assert origs[-1] is not None and origs[-1] < chunk_n, \
        "尾块应携带真实 orig_n"
    assert all(o is None or o == chunk_n for o in origs[:-1])

    info = soundfile.info(r["path"])
    assert abs(info.duration - dur) < 0.05, \
        f"补零不得进入成品：{info.duration:.2f}s ≠ {dur}s"


def test_enhance_chunk_trims_to_orig_n():
    """_enhance_chunk 的 orig_n 裁剪语义（不跑模型，只测数学部分的调用
    约定——用伪 pipeline 验证裁剪发生在 orig_n 上）。"""
    import types

    calls = {}

    class _FakeP:
        @staticmethod
        def super_resolution(model, path, seed=0, guidance_scale=3.5,
                             ddim_steps=35):
            calls["path"] = path
            # 伪 AudioSR 行为：多吐 20% 长度（它总会补到块边界）
            return np.zeros(int(5.0 * 48000 * 1.2), dtype=np.float32)

    import sys as _sys
    fake_audiosr = types.ModuleType("audiosr")
    fake_audiosr.pipeline = _FakeP
    monkey_mods = {"audiosr": fake_audiosr}
    real_import = __builtins__.__import__ if hasattr(
        __builtins__, "__import__") else _sys.modules["builtins"].__import__

    def _guard(name, *a, **k):
        if name == "audiosr.pipeline" or name.split(".")[0] == "audiosr":
            return fake_audiosr
        return real_import(name, *a, **k)

    import builtins
    orig_import = builtins.__import__
    builtins.__import__ = _guard
    try:
        sr = 22050
        y = np.zeros(int(sr * 1.02), dtype=np.float32)   # 事故现场的尾块长度
        out = SR._enhance_chunk(object(), y, sr, "unused.wav", 35, 3.5, 42,
                                orig_n=len(y))
        assert len(out) == int(len(y) * 48000 / sr), \
            "输出必须按 orig_n（真实音频长）裁剪，而不是补零后的长度"
    finally:
        builtins.__import__ = orig_import
