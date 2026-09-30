"""探针:AudioSR 超分服务(桩模型,不下载权重、不占显存)。

覆盖:未安装路径的报错文案、懒加载单例、超分调用(桩)产 48k wav、
释放、以及 AudioSR 不可用时 UI 侧不会炸。
用法:python tools/audio_sr_probe.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from webui_app.services import audio_sr as SR  # noqa: E402

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="sr_probe_")

    print("== 1. 环境探测 ==")
    check("available() 不抛异常", isinstance(SR.available(), bool),
          f"installed={SR.available()}")

    print("== 2. 未安装时的报错(临时伪装未安装) ==")
    import importlib.util as iu
    real_find = iu.find_spec
    iu.find_spec = lambda name, *a, **k: (None if name == "audiosr"
                                          else real_find(name, *a, **k))
    try:
        SR._STATE["model"] = None
        try:
            SR._get_model()
            check("未安装时抛 AudioSRError", False)
        except SR.AudioSRError as e:
            check("未安装时抛 AudioSRError 且提示 --no-deps",
                  "--no-deps" in str(e), str(e)[:60])
    finally:
        iu.find_spec = real_find

    print("== 3. 桩模型超分(懒加载/48k/释放) ==")
    import audiosr.pipeline as AP  # 真包(已装),只桩掉重活

    class _FakeModel:
        pass

    def _fake_build(model_name="basic", device=None, **kw):
        return _FakeModel()

    def _fake_super(model, path, seed=42, guidance_scale=3.5, ddim_steps=35,
                    **kw):
        y, sr = sf.read(path, dtype="float32")
        # 桩:原信号 + 12kHz 分量(模拟"补出的高频")
        t = np.arange(len(y)) / sr
        hf = 0.01 * np.sin(2 * np.pi * 12000 * t)
        return (y + hf).astype(np.float32)

    _rb, _rs = AP.build_model, AP.super_resolution
    AP.build_model, AP.super_resolution = _fake_build, _fake_super
    # 服务内部 from audiosr import pipeline 后取属性——patch 模块属性即可
    try:
        src = os.path.join(tmp, "in.wav")
        sf.write(src, (0.2 * np.sin(2 * np.pi * 440 * np.arange(22050)
                                    / 22050)).astype(np.float32), 22050)
        r = SR.enhance_file(src, ddim_steps=35)
        check("超分返回 ok", r.get("ok") is True, str(r)[:60])
        check("输出是 _48k.wav", r["path"].endswith("_48k.wav"))
        info = sf.info(r["path"])
        check("输出 48kHz", info.samplerate == 48000, str(info.samplerate))
        check("模型被缓存(懒加载单例)", SR._STATE["model"] is not None)
        r2 = SR.enhance_file(src)
        check("第二次调用不重建模型", SR._STATE["infers"] == 2)
    finally:
        AP.build_model, AP.super_resolution = _rb, _rs
        SR.release_all()
    check("release_all 清空模型", SR._STATE["model"] is None)

    print("== 4. 输入校验 ==")
    try:
        SR.enhance_file(os.path.join(tmp, "nope.wav"))
        check("缺输入报错", False)
    except SR.AudioSRError:
        check("缺输入报错", True)

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果:{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败:{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
