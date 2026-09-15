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
        head("[6] 音频处理：高通 / 分频段降噪 / 抖动")
        # =================================================================
        import librosa

        def _band_pct(path, lo, hi, sr_expect=None):
            y, sr = librosa.load(path, sr=None, mono=True)
            S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
            fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
            tot = S.sum() + 1e-12
            return round(float(S[(fr >= lo) & (fr < hi)].sum() / tot * 100), 2)

        # -- 高通：低频隆隆声要被去掉，语音频段不动 --
        srx = 22050
        t = np.arange(int(4 * srx)) / srx
        rumble = 0.5 * np.sin(2 * np.pi * 30 * t)          # 30 Hz 隆隆声
        voice = 0.3 * np.sin(2 * np.pi * 1000 * t)         # 1 kHz 语音样
        noisy = os.path.join(tmp, "rumble.wav")
        sf.write(noisy, (rumble + voice).astype(np.float32), srx)
        out_hp = os.path.join(tmp, "hp.wav")
        r_hp = AL.enhance(noisy, out_hp, denoise=False, normalize=False,
                          trim_silence=False, resample=False,
                          highpass_hz=60.0, max_sec=15.0)
        check("增强链可跑通（高通档）", r_hp.ok, str(r_hp.error)[:60])
        lo_before = _band_pct(noisy, 0, 60)
        lo_after = _band_pct(out_hp, 0, 60)
        check("**高通把 60 Hz 以下的隆隆声压下去了**",
              lo_after < lo_before * 0.2, f"{lo_before}% → {lo_after}%")
        check("高通不影响语音频段（1 kHz 附近保留）",
              _band_pct(out_hp, 500, 2000) > _band_pct(noisy, 500, 2000) * 0.7,
              f"{_band_pct(noisy, 500, 2000)}% → {_band_pct(out_hp, 500, 2000)}%")
        check("高通步骤被如实记录",
              any("高通" in x for x in r_hp.steps), str(r_hp.steps))

        # -- 分频段降噪：高频细节必须保住 --
        t2 = np.arange(int(3 * srx)) / srx
        hi_tone = 0.25 * np.sin(2 * np.pi * 10000 * t2)     # 10 kHz 细节
        hiss = 0.02 * np.random.default_rng(3).normal(size=len(t2))
        y_dn = (0.3 * np.sin(2 * np.pi * 300 * t2) + hi_tone + hiss).astype(np.float32)
        src_dn = os.path.join(tmp, "dn_in.wav")
        sf.write(src_dn, y_dn, srx)
        out_dn = os.path.join(tmp, "dn_out.wav")
        r_dn = AL.enhance(src_dn, out_dn, denoise=True, denoise_strength=0.6,
                          normalize=False, trim_silence=False, resample=False,
                          highpass_hz=0.0, max_sec=15.0)
        check("增强链可跑通（降噪档）", r_dn.ok, str(r_dn.error)[:60])
        check("noisereduce 已安装（否则降噪是空操作）",
              AL.noisereduce_available() is True,
              f"探测结果={AL.noisereduce_available()}")
        # 8 kHz 以上的能量应当基本保留（降噪只作用于它以下）
        hi_before = _band_pct(src_dn, 8000, 11000)
        hi_after = _band_pct(out_dn, 8000, 11000)
        check("**降噪时 8 kHz 以上的细节被保住**（分频段，不是整带压）",
              hi_after > hi_before * 0.8, f"{hi_before}% → {hi_after}%")
        check("降噪步骤写明了「分频段」与实际保护频率",
              any("分频段降噪" in x for x in r_dn.steps), str(r_dn.steps))

        # -- 没装 noisereduce 时必须**如实说跳过**，不能谎报降噪 --
        _real_avail = AL.noisereduce_available
        AL.noisereduce_available = lambda: False
        try:
            r_skip = AL.enhance(src_dn, os.path.join(tmp, "dn_skip.wav"),
                                denoise=True, normalize=False,
                                trim_silence=False, resample=False,
                                highpass_hz=0.0)
        finally:
            AL.noisereduce_available = _real_avail
        check("**缺 noisereduce 时步骤写明「降噪已跳过」**（不再谎报）",
              any("跳过" in x for x in r_skip.steps)
              and not any("频谱降噪" in x for x in r_skip.steps),
              str(r_skip.steps))

        # -- 抖动：确定、且幅度不超过 1 LSB --
        yy = (0.5 * np.sin(2 * np.pi * 440 * np.arange(srx) / srx)).astype(np.float32)
        d1 = os.path.join(tmp, "d1.wav")
        d2 = os.path.join(tmp, "d2.wav")
        d0 = os.path.join(tmp, "d0.wav")
        AL.save_audio(d1, yy, srx, dither=True)
        AL.save_audio(d2, yy, srx, dither=True)
        AL.save_audio(d0, yy, srx, dither=False)
        check("抖动是**确定性**的（同输入同字节，可复现）",
              open(d1, "rb").read() == open(d2, "rb").read())
        a = sf.read(d1)[0]
        b = sf.read(d0)[0]
        check("抖动幅度不超过 1 LSB（不会改变听感，只打散量化失真）",
              float(np.max(np.abs(a - b))) <= 1.0 / 32768.0 + 1e-9,
              f"{float(np.max(np.abs(a - b))):.2e}")

        # =================================================================
        head("[7] 输出后处理：提亮 / 空气感")
        # =================================================================
        t3 = np.arange(int(5 * srx)) / srx
        syn = (0.35 * np.sin(2 * np.pi * 180 * t3)
               + 0.18 * np.sin(2 * np.pi * 360 * t3)
               + 0.05 * np.sin(2 * np.pi * 5200 * t3)
               + 0.03 * np.random.default_rng(5).normal(size=len(t3))).astype(np.float32)
        src_syn = os.path.join(tmp, "synth.wav")
        sf.write(src_syn, syn, srx)

        r_neutral = AL.polish(src_syn, os.path.join(tmp, "p_neutral.wav"))
        check("后处理可跑通（中性档）", r_neutral.ok, str(r_neutral.error)[:60])
        check("中性档**不动音色**（谱质心基本不变）",
              abs(r_neutral.centroid_after - r_neutral.centroid_before) < 80,
              f"{r_neutral.centroid_before} → {r_neutral.centroid_after} Hz")

        r_pres = AL.polish(src_syn, os.path.join(tmp, "p_pres.wav"),
                           presence_db=3.0)
        check("presence 提升让谱质心明显上移（更亮）",
              r_pres.centroid_after > r_neutral.centroid_after + 100,
              f"{r_neutral.centroid_after} → {r_pres.centroid_after} Hz")
        check("presence 提升抬的是高频（4-8 kHz 能量增加）",
              _band_pct(os.path.join(tmp, "p_pres.wav"), 4000, 8000)
              > _band_pct(src_syn, 4000, 8000) * 1.3,
              f"{_band_pct(src_syn, 4000, 8000)}% → "
              f"{_band_pct(os.path.join(tmp, 'p_pres.wav'), 4000, 8000)}%")

        r_exc = AL.polish(src_syn, os.path.join(tmp, "p_exc.wav"),
                          presence_db=3.0, exciter=0.12)
        check("激励进一步补上 8-11 kHz 的空气感",
              _band_pct(os.path.join(tmp, "p_exc.wav"), 8000, 11000)
              > _band_pct(os.path.join(tmp, "p_pres.wav"), 8000, 11000),
              f"{_band_pct(os.path.join(tmp, 'p_pres.wav'), 8000, 11000)}% → "
              f"{_band_pct(os.path.join(tmp, 'p_exc.wav'), 8000, 11000)}%")
        # 这一条钉的是一个自己踩过的坑：激励若用 preemphasis/deemphasis 那对，
        # 会把未预加重的主信号做 20 倍低频提升 → 谱质心从 3355 掉到 788 Hz。
        check("**激励不会把声音变闷**（谱质心不允许大幅下降）",
              r_exc.centroid_after > r_neutral.centroid_after * 0.9,
              f"中性 {r_neutral.centroid_after} → 激励 {r_exc.centroid_after} Hz")
        check("激励后峰值仍不超上限（有峰值对齐）",
              r_exc.peak_after_dbfs <= -1.0 + 0.2, f"{r_exc.peak_after_dbfs:.2f} dBFS")

        md_pol = AL.polish_markdown(r_exc)
        # 用**这份报告自己的**前后值来断言（别拿另一档的数字去比）
        check("后处理报告可渲染且含谱质心前后对比",
              "谱质心" in md_pol
              and f"{r_exc.centroid_before:.0f}" in md_pol
              and f"{r_exc.centroid_after:.0f}" in md_pol,
              f"应在报告中看到 {r_exc.centroid_before:.0f} → "
              f"{r_exc.centroid_after:.0f}")
        check("后处理报告不出裸露的 None", "None" not in md_pol)
        check("失败时报告给出错误而不是抛",
              "失败" in AL.polish_markdown(
                  AL.polish(os.path.join(tmp, "nope.wav"),
                            os.path.join(tmp, "x.wav"))))

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
