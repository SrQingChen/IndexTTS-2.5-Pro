"""阶段1 服务层集成测试。

覆盖 UI 里尚未被真实点击过的代码路径，直接调服务层，比走浏览器快且报错清晰：

    T1  参考音频增强（降噪 + 归一 + 裁静音 + 重采样 + 限长）
    T2  音色库增删查改 + 重新体检
    T3  预设保存 → 读取 → 应用值构造（对齐 on_apply 的 25 项）
    T4  批量任务解析（多行文本 / JSONL / 字段继承 / 错误行）
    T5  情感模式 2：8 维向量归一化与生效值
    T6  批量合成实跑 2 条 + 合并

跑法：
    .venv\\Scripts\\python.exe tools\\stage1_integration.py
"""

from __future__ import annotations

import json
import os
import sys

import _env                                            # noqa: F401  路径 + 控制台编码
PROJECT_ROOT = _env.PROJECT_ROOT

from webui_app.config import config_from_args          # noqa: E402
from webui_app.context import AppContext                # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    mark = "✅" if cond else "✖"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    AppContext.reset()
    cfg = config_from_args(["--lazy"])
    ctx = AppContext.get(cfg)

    from webui_app.services import audio_lab as AL
    from webui_app.services import inference as INF
    from webui_app.services import voice_bank as VB
    from webui_app.tabs import presets as PT

    src = os.path.join("examples", "voice_05.wav")
    if not os.path.isfile(src):
        print(f"✖ 缺少测试素材 {src}")
        return 1

    # ------------------------------------------------------------------
    print("\n[T1] 参考音频增强")
    rep0 = AL.analyze(src)
    check("体检可运行", rep0 is not None and rep0.duration > 0,
          f"{rep0.duration:.2f}s 评分 {rep0.score:.1f} 等级 {rep0.grade}")
    out1 = os.path.join("outputs", "lab", "_it_enh.wav")
    res = AL.enhance(src, out1, denoise=True, denoise_strength=0.45,
                     normalize=True, target_dbfs=-20.0,
                     trim_silence=True, silence_thresh_db=-45.0,
                     resample=True, target_sr=22050, max_sec=12.0)
    check("增强成功", res.ok, res.error or "")
    if res.ok:
        check("输出文件存在", os.path.isfile(out1), out1)
        check("重采样到 22050", res.sample_rate == 22050,
              f"{res.before['sample_rate']} → {res.after['sample_rate']}")
        check("限长生效", res.duration <= 12.05,
              f"{res.before['duration']:.2f}s → {res.duration:.2f}s")
        check("响度被拉向 -20dBFS",
              abs(res.after["rms_dbfs"] - (-20.0)) < 3.0,
              f"{res.before['rms_dbfs']:.1f} → {res.after['rms_dbfs']:.1f} dBFS")
        check("处理步骤有记录", len(res.steps) >= 3, str(res.steps))
        md = AL.enhance_markdown(res)
        check("增强报告可渲染", isinstance(md, str) and len(md) > 50,
              f"{len(md)} 字符")

    # ------------------------------------------------------------------
    print("\n[T2] 音色库增删查改")
    name = "_集成测试音色"
    try:
        VB.remove(name)
    except Exception:
        pass
    e = VB.add(name, out1 if os.path.isfile(out1) else src,
               note="集成测试", tags=["测试"], lang="ZH", analyze_audio=True)
    check("入库成功", e is not None and e.name == VB.safe_name(name), e.name)
    check("能在列表里找到", e.name in VB.names())
    got = VB.get(e.name)
    check("按名取回", got is not None and os.path.isfile(got.audio_path),
          got.audio_path if got else "")
    check("入库时做了体检", got is not None and got.report.get("score") is not None,
          f"score={got.report.get('score')}" if got else "")
    e2 = VB.rename(e.name, e.name + "2")
    check("重命名", e2 is not None and (e.name + "2") in VB.names())
    check("表格可渲染", "集成测试音色" in VB.table_markdown())
    check("详情可渲染", len(VB.detail_markdown(e2.name)) > 50)
    check("删除", VB.remove(e2.name) and e2.name not in VB.names())

    # ------------------------------------------------------------------
    print("\n[T3] 预设保存 / 读取 / 应用")
    from indextts.utils.presets import (delete_preset, list_presets,
                                       load_preset, preset_exists, save_preset)
    from indextts.utils.presets import safe_preset_name

    # 先验证官方的名称清洗规则：strip("._") 会剥掉首尾的点与下划线。
    # UI 里已经把这个行为告知用户（保存后会提示“名称被清洗为 xxx”），
    # 这里把它钉住，以免官方改了规则而 UI 的提示变成错的。
    check("safe_preset_name 剥除首尾下划线",
          safe_preset_name("_测试_") == "测试",
          safe_preset_name("_测试_"))
    check("safe_preset_name 空白转下划线",
          safe_preset_name("主播 A 温暖") == "主播_A_温暖",
          safe_preset_name("主播 A 温暖"))
    # 注意执行顺序：先把非法字符换成 _，最后才 strip("._")，
    # 所以以非法字符结尾的名字会把那个 _ 一并剥掉：'a/b:c*d?' → 'a_b_c_d'
    check("safe_preset_name 非法字符转下划线（末尾被 strip）",
          safe_preset_name('a/b:c*d?') == "a_b_c_d",
          safe_preset_name('a/b:c*d?'))
    check("safe_preset_name 全特殊字符时退回 untitled",
          safe_preset_name("._.") == "untitled",
          safe_preset_name("._."))

    pname = "集成测试预设"
    data = {
        "emo_control_method": 2,
        "emo_alpha": 0.7,
        "emo_vector": [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2],
        "emo_text": "",
        "use_random": False,
        "lang": "ZH",
        "seed": 42,
        "duration_factor": 1.1,
        "max_text_tokens_per_segment": 100,
        "do_sample": True, "temperature": 0.75, "top_p": 0.75, "top_k": 25,
        "num_beams": 2, "repetition_penalty": 10.0, "length_penalty": 0.0,
        "max_mel_tokens": 1400,
    }
    save_preset(pname, data, prompt_audio=src)
    check("保存成功", preset_exists(pname))
    back = load_preset(pname)
    check("读回一致", back is not None and back.get("seed") == 42,
          f"seed={back.get('seed') if back else None}")
    check("音色音频被复制进预设目录",
          bool(back.get("prompt_audio")) and os.path.isfile(back["prompt_audio"]),
          back.get("prompt_audio", ""))
    check("出现在列表里", pname in list_presets())

    # 核对 APPLY_COUNT 与实际要写入的字段数一致
    keys = ["prompt_audio", "emo_mode", "emo_audio", "lang",
            "duration_factor", "emo_alpha", "use_random", "emo_text"]
    keys += [f"vec{i}" for i in range(8)]
    keys += ["do_sample", "temperature", "top_p", "top_k", "num_beams",
             "repetition_penalty", "length_penalty", "max_mel_tokens",
             "max_text_tokens_per_segment"]
    check("APPLY_COUNT 与字段清单一致",
          len(keys) == PT.APPLY_COUNT, f"{len(keys)} vs {PT.APPLY_COUNT}")

    # 官方 webui.py 的 advanced_params 嵌套结构也要能吃下
    nested = dict(data)
    nested["advanced_params"] = {
        "temperature": 0.55, "num_beams": 7, "max_mel_tokens": 1815,
    }
    for k in ("temperature", "num_beams", "max_mel_tokens"):
        nested.pop(k, None)
    save_preset(pname + "_nested", nested)
    nb = load_preset(pname + "_nested")
    adv = (nb or {}).get("advanced_params") or {}
    check("嵌套结构可读", adv.get("num_beams") == 7,
          f"advanced_params={adv}")
    delete_preset(pname)
    delete_preset(pname + "_nested")
    check("删除预设", not preset_exists(pname))

    # ------------------------------------------------------------------
    print("\n[T4] 批量任务解析")
    from webui_app.tabs import batch as BT
    import inspect
    src_bt = inspect.getsource(BT)
    check("batch 模块可导入", "SCHEMA_DOC" in src_bt)

    # 直接复刻 parse_jsonl 的语义做一次校验
    lines = ['{"text": "第一段"}',
             '{"text": "第二段", "lang": "EN", "duration_factor": 1.3}',
             '{"bad json"',
             '{"text": "第三段", "emo_control_method": 2, '
             '"emo_vector": [0,0,0.6,0,0,0,0,0]}']
    good, bad = [], []
    for raw in lines:
        try:
            good.append(json.loads(raw))
        except json.JSONDecodeError:
            bad.append(raw)
    check("合法行被解析", len(good) == 3, f"{len(good)} 条")
    check("非法行被收集而不是抛出", len(bad) == 1, bad[0][:30])
    globals_ = {"lang": "ZH", "duration_factor": 1.0, "seed": -1}
    merged = dict(globals_)
    merged.update({k: v for k, v in good[1].items() if v is not None})
    check("字段继承：未指定的用全局值", merged["seed"] == -1)
    check("字段继承：指定的覆盖全局值",
          merged["lang"] == "EN" and merged["duration_factor"] == 1.3,
          f"lang={merged['lang']} dur={merged['duration_factor']}")
    req = INF.GenRequest.from_ui(merged, cfg)
    check("GenRequest 能承接合并结果", req.lang == "EN"
          and abs(req.duration_factor - 1.3) < 1e-6)

    # ------------------------------------------------------------------
    print("\n[T5] 情感向量归一化")
    from webui_app.config import EMO_BIAS
    from webui_app.tabs.synthesize import normalize_vec

    v = [1.0] * 8
    eff = normalize_vec(v, apply_bias=True)
    total = sum(eff)
    check("8 维全 1 被压到 0.8 上限", abs(total - 0.8) < 1e-6,
          f"总和 {total:.6f}")

    # 未超限的用例：乘完偏置后总和必须 < 0.8，否则会被等比压缩
    v2 = [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1]
    biased_sum = sum(a * b for a, b in zip(v2, EMO_BIAS))
    check("测试数据本身未超限", biased_sum < 0.8, f"偏置后总和 {biased_sum:.5f}")
    eff2 = normalize_vec(v2, apply_bias=True)
    check("未超限则只乘偏置不压缩",
          abs(sum(eff2) - biased_sum) < 1e-6,
          f"{biased_sum:.5f} → {sum(eff2):.5f}")
    check("偏置逐项生效",
          all(abs(e - a * b) < 1e-6 for e, a, b in zip(eff2, v2, EMO_BIAS)),
          f"0.5*0.9375={eff2[0]:.5f}  0.1*0.5625={eff2[7]:.5f}")

    # 超限的用例：验证压缩是**等比**的（各维比例不变，只是整体缩小）
    v3 = [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2]
    eff3 = normalize_vec(v3, apply_bias=True)
    raw3 = [a * b for a, b in zip(v3, EMO_BIAS)]
    scale = 0.8 / sum(raw3)
    check("超限时等比压缩，各维比例保持",
          all(abs(e - r * scale) < 1e-6 for e, r in zip(eff3, raw3)),
          f"原始总和 {sum(raw3):.5f} → 压缩系数 {scale:.5f}")
    check("压缩后总和恰好 0.8", abs(sum(eff3) - 0.8) < 1e-6,
          f"{sum(eff3):.6f}")

    # from_ui 的 emo_vec_N 归并路径（曾经的 bug 点）
    ui = {"text": "x", "spk_audio_prompt": src,
          "emo_control_method": 2, "emo_vec_0": 0.8, "emo_vec_2": 0.4}
    r2 = INF.GenRequest.from_ui(ui, cfg)
    check("emo_vec_N 不被过滤器丢弃", r2.emo_vector[0] == 0.8
          and r2.emo_vector[2] == 0.4, str(r2.emo_vector))
    ui2 = {"text": "x", "emo_control_method": 2,
           "emo_vector": [0.1, 0.2, 0, 0, 0, 0, 0, 0], "emo_vec_0": 0.9}
    r3 = INF.GenRequest.from_ui(ui2, cfg)
    check("显式 emo_vector 优先于分散的 emo_vec_N",
          r3.emo_vector[0] == 0.1, str(r3.emo_vector))

    # ------------------------------------------------------------------
    print("\n[T6] 批量合成实跑（2 条 + 合并）")
    eng = ctx.engine
    try:
        eng.load()
        check("引擎加载", eng.loaded,
              f"{eng.stats.load_seconds:.1f}s "
              f"显存 {eng.stats.vram_alloc_gb:.2f}GB")
    except Exception as e:
        check("引擎加载", False, f"{type(e).__name__}: {e}")
        print("\n引擎加载失败，跳过 T6")
        return _summary()

    run_dir = os.path.join(cfg.tasks_dir, "_integration")
    os.makedirs(run_dir, exist_ok=True)
    paths = []
    for i, txt in enumerate(["第一段集成测试。", "第二段集成测试，稍微长一点点。"]):
        req = INF.GenRequest.from_ui({
            "text": txt, "spk_audio_prompt": src, "lang": "ZH",
            "emo_control_method": 0, "seed": 42 + i,
            "num_beams": 1, "max_mel_tokens": 800,
            "max_text_tokens_per_segment": 120,
        }, cfg)
        out = os.path.join(run_dir, f"seg{i}.wav")
        try:
            r = INF.generate(eng, req, progress=None, output_path=out)
            ok = bool(r.get("path")) and os.path.isfile(r["path"])
            paths.append(r.get("path"))
            check(f"第 {i} 条合成", ok,
                  f"{r.get('audio_duration', 0):.2f}s / 耗时 {r.get('seconds', 0):.2f}s"
                  f" / RTF {r.get('rtf', 0):.2f}")
        except Exception as e:
            check(f"第 {i} 条合成", False, f"{type(e).__name__}: {e}")

    if len(paths) == 2 and all(p and os.path.isfile(p) for p in paths):
        cat = os.path.join(run_dir, "concat.wav")
        try:
            out = BT._concat(paths, cat, gap_ms=300)
            d = INF.audio_duration(out)
            d0 = INF.audio_duration(paths[0])
            d1 = INF.audio_duration(paths[1])
            check("合并输出", os.path.isfile(out), out)
            check("合并时长 = 各段 + 间隔",
                  abs(d - (d0 + d1 + 0.3)) < 0.15,
                  f"{d0:.2f} + {d1:.2f} + 0.30 ≈ {d:.2f}s")
        except Exception as e:
            check("合并输出", False, f"{type(e).__name__}: {e}")

    return _summary()


def _summary() -> int:
    print("\n" + "=" * 60)
    print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
    if FAIL:
        print("  失败项：")
        for f in FAIL:
            print(f"    ✖ {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
