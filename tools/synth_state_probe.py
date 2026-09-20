"""合成页参数记忆（services/synth_state.py）的单元验证。

全程 CPU、秒级完成，可以在 WebUI 运行时并行执行。
小心两件事：
    · STATE_PATH 是用户真实记忆 —— 测试前备份、finally 里恢复；
    · 配置档 roundtrip 用的是真实 outputs/presets/ —— 用唯一名，
      测完删掉，不留垃圾。
    · 临时音频造在系统 TEMP（模拟「上传的临时文件」）与 voice_bank/
      （模拟稳定路径）两种位置，验证副本与原样引用两种行为；
      voice_bank 里造的临时文件同样要回收。

    .venv\\Scripts\\python.exe tools\\synth_state_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import _env                                            # noqa: F401  路径 + 控制台编码

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from webui_app.config import PROJECT_ROOT
    from webui_app.services import synth_state as SS

    # ------------------------------------------------------------------
    # 备份真实记忆（文件 + 两个音频副本）
    # ------------------------------------------------------------------
    backups = {}
    for p in (SS.STATE_PATH,
              os.path.join(SS.STATE_DIR, SS.PROMPT_COPY),
              os.path.join(SS.STATE_DIR, SS.EMO_COPY)):
        if os.path.isfile(p):
            bak = p + ".probe_bak"
            shutil.copy2(p, bak)
            backups[p] = bak

    # 临时音频：TEMP 里的代表「上传的临时文件」，voice_bank 里的代表稳定路径
    tmp_up = os.path.join(tempfile.gettempdir(), f"zz_mem_probe_{int(time.time())}.wav")
    open(tmp_up, "wb").write(b"\0" * 512)
    from webui_app.config import config_from_args
    cfg = config_from_args(["--lazy"])
    vb_audio = os.path.join(cfg.voice_bank_dir, "audio",
                            f"zz_mem_probe_{int(time.time())}.wav")
    os.makedirs(os.path.dirname(vb_audio), exist_ok=True)
    open(vb_audio, "wb").write(b"\0" * 256)

    preset_name = f"zz_mem_probe_{int(time.time())}"

    try:
        # --------------------------------------------------------------
        # 1) sanitize：类型纠正 + 范围钳制 + 白名单
        # --------------------------------------------------------------
        s = SS.sanitize({
            "temperature": "1.4",                 # 字符串数字 → float
            "top_k": 999,                         # 越界 → 钳到 100
            "duration_factor": 99.0,              # 越界 → 钳到 2.0
            "do_sample": 1,                       # 1 → True
            "seed": None,                         # None 保留（随机）
            "lang": " zh ",                       # 清洗成 "ZH"
            "emo_mode_index": 7,                  # 钳到 3
            "voice_name": "爱弥斯",
            "polish_presence": 3.0,
            "hacker_key": "<script>",             # 白名单外 → 丢弃
            "interval_silence": "abc",            # 畸形 → 丢弃
        })
        check("字符串数字被转成 float", isinstance(s.get("temperature"), float)
              and abs(s["temperature"] - 1.4) < 1e-9)
        check("越界整数被钳制", s.get("top_k") == 100 and s.get("duration_factor") == 2.0)
        check("int 真值转 bool", s.get("do_sample") is True)
        check("None 种子保留", "seed" in s and s["seed"] is None)
        check("语言清洗为大写", s.get("lang") == "ZH")
        check("情感模式索引钳到 3", s.get("emo_mode_index") == 3)
        check("白名单外字段被丢弃", "hacker_key" not in s)
        check("畸形数值字段被丢弃", "interval_silence" not in s)

        # --------------------------------------------------------------
        # 2) save / load 往返
        # --------------------------------------------------------------
        live = {
            "lang": "EN", "duration_factor": 1.25, "seed": 42,
            "text_normalization": False,
            "max_text_tokens_per_segment": 88, "interval_silence": 300.0,
            "emo_mode_label": "使用 8 维情感向量（精确可复现）",
            "emo_mode_index": 2, "emo_alpha": 0.7,
            "emo_vec_0": 0.5, "emo_vec_3": 0.3,
            "emo_text": "平静而坚定", "use_random": True,
            "do_sample": True, "temperature": 0.7, "top_p": 0.75, "top_k": 40,
            "num_beams": 5, "repetition_penalty": 10.0,
            "length_penalty": 0.0, "max_mel_tokens": 1815,
            "voice_name": "爱弥斯",
            "lora_run": "爱弥斯_gpt", "lora_ckpt": "best", "lora_scale": 0.8,
            "polish_on": False, "polish_presence": 4.5, "polish_exciter": 0.1,
            "_remember": True,
        }
        ok = SS.save_state(live, prompt_audio=tmp_up, emo_audio=vb_audio)
        check("save_state 成功", ok and os.path.isfile(SS.STATE_PATH))

        payload = SS.load_state()
        check("load_state 读回载荷", bool(payload) and payload.get("version") == 1)
        vals = payload.get("values", {})
        expect_same = {k: live[k] for k in
                       ("lang", "duration_factor", "seed",
                        "max_text_tokens_per_segment", "interval_silence",
                        "emo_alpha", "emo_text", "temperature", "top_p",
                        "top_k", "num_beams", "max_mel_tokens", "voice_name",
                        "lora_run", "lora_ckpt", "lora_scale", "polish_presence")}
        diff = {k: (expect_same[k], vals.get(k)) for k in expect_same
                if vals.get(k) != expect_same[k]}
        check("数值/文本字段往返一致", not diff, f"差异: {diff}")
        check("bool 字段往返一致",
              vals.get("text_normalization") is False
              and vals.get("use_random") is True
              and vals.get("polish_on") is False)
        check("情感模式往返一致",
              vals.get("emo_mode_label") == live["emo_mode_label"]
              and vals.get("emo_mode_index") == 2)

        # --------------------------------------------------------------
        # 3) 音频：临时文件复制入 state，稳定路径原样引用
        # --------------------------------------------------------------
        copied = os.path.join(SS.STATE_DIR, SS.PROMPT_COPY)
        check("上传的临时音频被复制进 state 目录",
              os.path.isfile(copied)
              and os.path.realpath(payload.get("prompt_audio") or "") == copied)
        check("音色库稳定路径原样引用不复制",
              os.path.realpath(payload.get("emo_audio") or "") ==
              os.path.realpath(vb_audio))
        os.remove(tmp_up)                       # 模拟临时目录被清空
        payload2 = SS.load_state()
        check("临时文件消失后副本仍可用", os.path.isfile(copied))
        # 音频失效场景：删掉副本后 load 应丢弃该项
        os.remove(copied)
        payload3 = SS.load_state()
        check("失效音频在 load 时被丢弃",
              payload3.get("prompt_audio") is None
              and payload3.get("values"))          # 其余字段不受影响

        # --------------------------------------------------------------
        # 4) 记忆开关 / forget / 容错
        # --------------------------------------------------------------
        live_off = dict(live, _remember=False)
        SS.save_state(live_off)
        check("关闭记忆时开关随文件保存",
              SS.remember_enabled(SS.load_state()) is False)

        SS.forget()
        check("forget 清掉记忆与音频副本",
              not os.path.isfile(SS.STATE_PATH)
              and not os.path.isfile(copied)
              and SS.load_state() == {})

        os.makedirs(SS.STATE_DIR, exist_ok=True)
        with open(SS.STATE_PATH, "w", encoding="utf-8") as f:
            f.write("{ 这不是合法 json !!!")
        check("损坏的 json 静默降级为无记忆", SS.load_state() == {})

        with open(SS.STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "values": {"lang": 123}}, f)
        p4 = SS.load_state()
        check("畸形 lang（数字）被丢弃且不崩", "lang" not in p4.get("values", {}))

        # --------------------------------------------------------------
        # 5) live_to_preset_data：官方预设格式映射 + 真实 roundtrip
        # --------------------------------------------------------------
        data = SS.live_to_preset_data({
            "emo_mode_label": "使用情感描述文本（实验功能，需 QwenEmotion）",
            "emo_mode_index": 3,
            "emo_vec_0": 0.9, "emo_alpha": "0.55",
            "seed": 7, "lang": "JA", "temperature": "0.75",
            "top_k": "35",
        })
        check("情感模式标签映射为索引", data["emo_control_method"] == 3)
        check("向量补齐 8 维", len(data["emo_vector"]) == 8
              and data["emo_vector"][0] == 0.9)
        check("字符串数值被转换", data["emo_alpha"] == 0.55
              and data["temperature"] == 0.75 and data["top_k"] == 35)
        check("缺失字段用官方默认补齐",
              data["num_beams"] == 3 and data["max_mel_tokens"] == 1500
              and data["do_sample"] is True)
        check("19 个官方字段齐全",
              set(data.keys()) == {
                  "emo_control_method", "emo_alpha", "emo_vector", "emo_text",
                  "use_random", "max_text_tokens_per_segment", "duration_factor",
                  "interval_silence", "text_normalization", "lang", "seed",
                  "do_sample", "top_p", "top_k", "temperature", "num_beams",
                  "repetition_penalty", "length_penalty", "max_mel_tokens"})

        from indextts.utils.presets import (delete_preset, list_presets,
                                            load_preset, save_preset)
        save_preset(preset_name, data, prompt_audio=vb_audio)
        check("配置档经官方 save_preset 落盘",
              preset_name in list_presets())
        back = load_preset(preset_name)
        check("配置档读回字段一致",
              back is not None
              and back.get("emo_control_method") == 3
              and back.get("seed") == 7
              and back.get("lang") == "JA"
              and back.get("prompt_audio", "").endswith("prompt.wav"))
        check("配置档音频复制入档（不依赖外部路径）",
              os.path.isfile(back.get("prompt_audio") or ""))
        delete_preset(preset_name)
        check("配置档删除干净", preset_name not in list_presets()
              and not os.path.isdir(os.path.join(
                  PROJECT_ROOT, "outputs", "presets", preset_name)))

        # --------------------------------------------------------------
        # 6) 忘掉这次测试的全部痕迹
        # --------------------------------------------------------------
        SS.forget()
    finally:
        # --------------------------------------------------------------
        # 恢复真实记忆；回收临时文件与配置档残留
        # --------------------------------------------------------------
        SS.forget()
        for p, bak in backups.items():
            shutil.copy2(bak, p)
            os.remove(bak)
        for p in (tmp_up, vb_audio):
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        pd = os.path.join(PROJECT_ROOT, "outputs", "presets", preset_name)
        if os.path.isdir(pd):
            shutil.rmtree(pd, ignore_errors=True)

    print("\n" + "=" * 64)
    print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
    if FAIL:
        print("  失败项：")
        for f in FAIL:
            print(f"    FAIL {f}")
    print("=" * 64)
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
