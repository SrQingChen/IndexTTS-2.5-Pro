"""参考音频工作台（services/audio_lab.py + services/voice_bank.py）的验证。

本探针来自一个真实故障：用户导入长音频 → 扫描候选片段 → 「设为主素材」，
之后**再点一次导出/设为主素材就失灵了** —— 产出的是一个 44 字节的空 wav
（只有文件头），体检判「音频为空」，增强与入库跟着全失败。

根因：导出片段时把 `src_audio` 换成了刚切出来的**短片段**，而候选片段里存的
还是**相对原始长音频**的秒数偏移。再切一次就按 43~55 秒去切一个 12 秒的文件 →
越界 → 切出 0 个采样点 → 静默写出空 wav。

所以这里钉住三件事，任何一件破掉都会重现那个症状：
  1. 越界切分必须**报错**，绝不产出空文件或"位置不对"的音频；
  2. `save_audio` 拒绝写出 0 采样点；体检对空音频判 not ok；
  3. 体检不过的音频**不许进音色库**（否则会在合成页被选中然后推理失败）。

跑法：  .venv\\Scripts\\python.exe tools\\lab_probe.py
退出码：0 = 全过；1 = 有失败项。
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import os
import shutil
import tempfile

import numpy as np
import soundfile as sf

from webui_app.services import audio_lab as AL                # noqa: E402
from webui_app.services import voice_bank as VB               # noqa: E402

PASS = FAIL = 0
FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def head(t: str):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def make_long(path: str, pieces: int = 40, body: float = 2.7,
              gap: float = 0.45, sr: int = 22050) -> str:
    y = []
    for k in range(pieces):
        t = np.arange(int(body * sr)) / sr
        tone = (0.3 * np.sin(2 * np.pi * 180 * t)
                + 0.16 * np.sin(2 * np.pi * 360 * t))
        env = np.minimum(1.0, np.minimum(t / 0.05, (body - t) / 0.05))
        y.append((tone * env).astype(np.float32))
        y.append(np.zeros(int(gap * sr), dtype=np.float32))
    sf.write(path, np.concatenate(y), sr)
    return path


class _Seg:
    """手工构造的片段（避开 find_segments，专测边界）。"""

    def __init__(self, start: float, end: float):
        self.start, self.end = start, end


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="lab_probe_")
    try:
        sr = 22050
        src = make_long(os.path.join(tmp, "爱弥斯.wav"))
        dur = sf.info(src).duration
        print(f"[素材] 长音频 {dur:.0f} 秒")

        # =================================================================
        head("[1] 扫描候选片段")
        # =================================================================
        segs = AL.find_segments(src, target_sec=12.0, min_sec=6.0)
        check("扫出了候选片段", len(segs) >= 1, f"{len(segs)} 个")
        check("候选片段的偏移都落在音频范围内",
              all(0 <= s.start < s.end <= dur + 0.05 for s in segs),
              f"例：{segs[0].start:.2f}~{segs[0].end:.2f}s / 时长 {dur:.2f}s")
        check("候选片段按评分降序（#0 最高）",
              all(segs[i].score >= segs[i + 1].score
                  for i in range(len(segs) - 1)))

        # =================================================================
        head("[2] 正常导出（对照组）")
        # =================================================================
        seg0 = os.path.join(tmp, "seg0.wav")
        AL.extract_segment(src, segs[0], seg0)
        check("导出的片段时长与候选一致",
              abs(sf.info(seg0).duration - (segs[0].end - segs[0].start)) < 0.05,
              f"{sf.info(seg0).duration:.2f}s")
        check("导出的是正常大小（不是空文件）",
              os.path.getsize(seg0) > 10_000, f"{os.path.getsize(seg0)} 字节")
        check("导出后可体检通过", AL.analyze(seg0).ok)

        # =================================================================
        head("[3] 越界切分必须报错（本次故障的核心）")
        # =================================================================
        # 拿"长音频的偏移"去切"刚导出的短片段" —— 这正是用户的操作序列
        for start, end, why in [
            (segs[0].start, segs[0].end, "长音频的偏移切短片段（完全越界）"),
            (dur - 1.0, dur + 5.0, "尾部越界"),
            (dur + 10.0, dur + 20.0, "整体越界"),
        ]:
            raised = ""
            try:
                AL.extract_segment(seg0, _Seg(start, end),
                                   os.path.join(tmp, "x.wav"))
            except ValueError as e:
                raised = str(e)
            except Exception as e:
                raised = f"WRONG:{type(e).__name__}"
            check(f"越界被拦住并抛 ValueError：{why}",
                  raised.startswith("切分范围"), raised[:70])
            check(f"越界提示可操作：{why}",
                  "重新扫描" in raised, raised[-40:])

        # 明确钉住「绝不会产出空文件」：走完越界分支后没有多余文件生成
        check("越界不会留下半成品文件",
              not os.path.isfile(os.path.join(tmp, "x.wav")))

        # 取整容差内不算越界（0.001s 的溢出应当放过）
        ok_eps = os.path.join(tmp, "eps.wav")
        AL.extract_segment(seg0, _Seg(0.0, sf.info(seg0).duration + 0.01),
                           ok_eps)
        check("取整量级的端点溢出仍可接受（不被误判）",
              os.path.isfile(ok_eps) and sf.info(ok_eps).duration > 1.0,
              f"{sf.info(ok_eps).duration:.2f}s")

        # 重新扫描（主素材已经变成短片段）之后必须能正常切
        segs2 = AL.find_segments(seg0, target_sec=12.0, min_sec=6.0)
        again = os.path.join(tmp, "again.wav")
        AL.extract_segment(seg0, _Seg(segs2[0].start, segs2[0].end), again)
        check("**重新扫描后可以正常切**（用户该走的路径能走通）",
              os.path.getsize(again) > 10_000, f"{os.path.getsize(again)} 字节")

        # =================================================================
        head("[4] save_audio / analyze 对空音频的态度")
        # =================================================================
        raised = ""
        try:
            AL.save_audio(os.path.join(tmp, "empty.wav"),
                          np.array([], dtype=np.float32), sr)
        except ValueError as e:
            raised = str(e)
        check("**save_audio 拒绝写出 0 采样点**（44 字节空 wav 的根源）",
              raised.startswith("拒绝写出空音频"), raised[:60])
        check("拒绝时不会留下文件",
              not os.path.isfile(os.path.join(tmp, "empty.wav")))

        zero = os.path.join(tmp, "zero.wav")
        sf.write(zero, np.zeros(0, dtype=np.float32), sr)
        rep = AL.analyze(zero)
        check("体检对空音频判 not ok", rep.ok is False, str(rep.error))
        check("体检给出的原因是「音频为空」类说明",
              "空" in str(rep.error), str(rep.error))
        enh = AL.enhance(zero, os.path.join(tmp, "z_enh.wav"))
        check("增强对空音频返回 ok=False（不抛、也不产出文件）",
              enh.ok is False and bool(enh.error), str(enh.error)[:60])

        # =================================================================
        head("[5] 音色库：坏素材不许进库")
        # =================================================================
        for name, path, why in [
            ("测试_空文件", zero, "空音频"),
            ("测试_不存在", os.path.join(tmp, "nope.wav"), "文件不存在"),
        ]:
            raised = ""
            try:
                VB.add(name, path, note="probe")
            except ValueError as e:
                raised = str(e)
            except FileNotFoundError as e:
                raised = str(e)
            check(f"拒绝入库：{why}", bool(raised), raised[:60])
            check(f"被拒后没有在库里留条目：{why}", VB.get(name) is None)

        check("库目录里没有残留的空文件",
              not any(f.endswith("测试_空文件.wav")
                      for f in os.listdir(os.path.join(VB.BANK_DIR,
                                                       VB.AUDIO_SUBDIR))
                      if True) if os.path.isdir(
                  os.path.join(VB.BANK_DIR, VB.AUDIO_SUBDIR)) else True)

        # 正常素材要能入库，并且带上体检报告
        enh_ok = os.path.join(tmp, "enh.wav")
        r = AL.enhance(seg0, enh_ok, denoise=False, normalize=True,
                       trim_silence=True)
        check("正常素材增强成功", r.ok, str(r.error)[:60])
        e = VB.add("测试_正常音色", enh_ok, note="probe",
                   tags=["probe", " 空格会被去掉 "], lang="ZH")
        check("正常素材成功入库", e.name == "测试_正常音色", e.name)
        check("入库条目带体检评分（不是 0）", float(e.score) > 0,
              f"{e.score}（{e.grade}）")
        check("入库条目带时长", float(e.duration) > 0, f"{e.duration:.2f}s")
        check("标签被清理（去空白、去空项）",
              e.tags == ["probe", "空格会被去掉"], str(e.tags))
        check("入库文件确实落在库目录里",
              os.path.isfile(os.path.join(VB.BANK_DIR, e.audio)), e.audio)

        check("能按名字查到", VB.get("测试_正常音色") is not None)
        check("列表里能看到它",
              "测试_正常音色" in [x.name for x in VB.list_voices()])
        check("删除成功", VB.remove("测试_正常音色") is True)
        check("删除后查不到", VB.get("测试_正常音色") is None)

        # =================================================================
        head("清理")
        # =================================================================
        for nm in ("测试_空文件", "测试_不存在", "测试_正常音色"):
            try:
                VB.remove(nm)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除", not os.path.isdir(tmp))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
    print("=" * 70)
    if FAILS:
        print("失败项：")
        for f in FAILS:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
