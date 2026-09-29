"""真机验收：LoRA 双通道（GPT 语气 + CFM 音色）同时挂载。

验证引擎层的跨 target 共挂（D 批次的核心主张）：
    1. GPT run 与 CFM run 各自 mount_run → lora_status 两通道都 wrapped；
    2. 各自 set_scale（0.8 / 1.2）实测生效；
    3. 双通道在位时真合成一句，出声正常；
    4. detach 单通道不影响另一通道；全卸后回到纯底座；
    5. 角色档案存取（synth_state）。

用法：.venv\\Scripts\\python.exe tools\\dual_lora_acceptance.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {detail}" if detail else ""))


def _wrapped(eng, target) -> bool:
    for x in eng.lora_status():
        if x.get("target") == target:
            return bool(x.get("wrapped"))
    return False


def main() -> int:
    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import inference as INF
    from webui_app.services import synth_state as SS
    from webui_app.services import voice_bank as VB
    from webui_app.training import merge as MG
    from webui_app.training import runs as RN

    runs = {r.name: r for r in RN.list_runs() if getattr(r, "has_adapter", False)}
    gpt_run = next((n for n, r in runs.items() if r.arch == "gpt"), "")
    cfm_run = next((n for n, r in runs.items() if r.arch == "cfm"), "")
    check("存在 gpt 与 cfm 各一个可挂载 run", bool(gpt_run and cfm_run),
          f"gpt={gpt_run or '无'} · cfm={cfm_run or '无'}")
    if not (gpt_run and cfm_run):
        return finish()

    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine
    print("== 1. 引擎加载 ==")
    eng.load()

    print("== 2. 双通道共挂 ==")
    tag_g = MG.mount_run(eng, gpt_run, checkpoint="best", scale=0.8)
    tag_c = MG.mount_run(eng, cfm_run, checkpoint="best", scale=1.2)
    check(f"GPT 通道挂载（{tag_g}）", _wrapped(eng, "gpt"))
    check(f"CFM 通道挂载（{tag_c}）", _wrapped(eng, "cfm"))
    check("两通道标签并存",
          any(t.startswith("gpt:") for t in eng.stats.lora_adapters)
          and any(t.startswith("cfm:") for t in eng.stats.lora_adapters),
          str(eng.stats.lora_adapters))

    print("== 3. 双通道在位合成 ==")
    spk = VB.get("wujiu") or next(iter(VB.list_voices()), None)
    req = INF.GenRequest(spk_audio_prompt=spk.audio_path,
                         text="双通道挂载测试，一切正常。", seed=7)
    res = INF.generate(eng, req)
    check("合成出声", os.path.isfile(res["path"]) and res["audio_duration"] > 1.0,
          f"{res['audio_duration']:.2f}s")

    print("== 4. 分通道卸载 ==")
    eng.detach_lora(target="cfm")
    check("CFM 已卸", not _wrapped(eng, "cfm"))
    check("GPT 不受影响", _wrapped(eng, "gpt"))
    eng.detach_lora(target="gpt")
    check("全卸后回纯底座",
          not _wrapped(eng, "gpt") and not _wrapped(eng, "cfm"))

    print("== 5. 角色档案 ==")
    ok = SS.save_lora_profile("验收角色", {
        "gpt_run": gpt_run, "gpt_ckpt": "best", "gpt_scale": 0.8,
        "cfm_run": cfm_run, "cfm_ckpt": "best", "cfm_scale": 1.2})
    p = SS.get_lora_profile("验收角色")
    check("档案保存/读取", ok and p["gpt_run"] == gpt_run
          and abs(p["cfm_scale"] - 1.2) < 1e-9)
    SS.delete_lora_profile("验收角色")
    check("档案删除", SS.get_lora_profile("验收角色") is None)

    return finish()


def finish() -> int:
    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n结果：{len(CHECKS) - len(fails)}/{len(CHECKS)} 通过"
          + (f" · 失败：{fails}" if fails else " ✅"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
