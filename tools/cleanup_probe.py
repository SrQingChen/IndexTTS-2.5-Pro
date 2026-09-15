"""清理页（services/artifacts.py）的单元验证。

只造临时假产物、只删自己造的东西，**不碰真实产物目录里已有的任何文件**
（数据集与音色各造一个自己的临时条目再删掉）。
全程 CPU、秒级完成，可以在 WebUI 运行时并行执行。

覆盖四件事：
    1. 扫描：假产物能按正确的 kind / origin / 保护标记被发现；
    2. 删除：正常删、重复删（skip）、专用通道（数据集 / 训练 run / 音色）
       与通用通道都走得到；
    3. 安全：越权 key（路径穿越 / 伪造类型）一律拒绝，底模 / 代码 /
       预设 / 日志永不出现在盘点里也删不掉；
    4. 保护：status=running 的训练 run 拒绝删除。

    .venv\\Scripts\\python.exe tools\\cleanup_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

import _env                                            # noqa: F401  路径 + 控制台编码

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from webui_app.config import config_from_args
    from webui_app.services import artifacts as AR

    cfg = config_from_args(["--lazy"])

    # ------------------------------------------------------------------
    # 造假产物（全部带 zz_probe_ 前缀，结束统一回收）
    # ------------------------------------------------------------------
    stamp = time.strftime("%H%M%S")
    prefix = f"zz_probe_{stamp}"

    oc = os.path.join(cfg.output_dir, "oneclick", f"{prefix}_oc")
    ev = os.path.join(cfg.output_dir, "eval", f"{prefix}_ev")
    tk = os.path.join(cfg.tasks_dir, f"{prefix}_batch")
    lo = os.path.join(cfg.lora_dir, f"{prefix}_lora")
    ti = os.path.join(cfg.train_dir, f"{prefix}_ti")
    lab = os.path.join(cfg.output_dir, "lab", f"{prefix}_seg.wav")
    voice_wav = os.path.join(cfg.output_dir, "lab", f"{prefix}_voice.wav")
    syn = os.path.join(cfg.output_dir, f"{prefix}_out.wav")

    # 植入也放在 try 里：任何一步抛异常（比如音色体检拒绝）都不能留垃圾
    try:
        os.makedirs(oc, exist_ok=True)
        open(os.path.join(oc, "report.md"), "w", encoding="utf-8").write("# r")
        open(os.path.join(oc, "utt_0.wav"), "wb").write(b"\0" * 4096)

        os.makedirs(ev, exist_ok=True)
        open(os.path.join(ev, "a.wav"), "wb").write(b"\0" * 512)

        os.makedirs(tk, exist_ok=True)
        open(os.path.join(tk, "out.wav"), "wb").write(b"\0" * 512)

        os.makedirs(lo, exist_ok=True)
        open(os.path.join(lo, "adapter_model.safetensors"), "wb").write(b"\0" * 2048)

        os.makedirs(ti, exist_ok=True)
        open(os.path.join(ti, "x.pt"), "wb").write(b"\0" * 2048)

        open(lab, "wb").write(b"\0" * 1024)
        open(syn, "wb").write(b"\0" * 1024)

        # 假训练 run：一个 done（可删）一个 running（必须拒删）
        from webui_app.training import runs as RN
        done_run, busy_run = f"{prefix}_done", f"{prefix}_busy"
        for name, status in ((done_run, "done"), (busy_run, "running")):
            d = RN.run_dir(name, create=True)
            with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
                json.dump({"arch": "gpt", "run": name, "dataset": "none",
                           "status": status, "created_at": time.time()}, f)
            os.makedirs(os.path.join(d, "adapter"), exist_ok=True)

        # 临时数据集与音色（走真实 API 建，删的也是它们）
        from webui_app.training import dataset as DS
        ds_name = DS.create(prefix, note="清理探针临时数据集")
        open(os.path.join(DS.dir_of(ds_name), "audio", "a.wav"), "wb").write(b"\0" * 256)

        from webui_app.services import voice_bank as VB
        open(voice_wav, "wb").write(b"\0" * 512)
        VB.add(prefix, audio_path=voice_wav, note="清理探针临时音色",
               analyze_audio=False)      # 假 wav 过不了体检，探针只验索引链路

        planted = {f"oneclick:{prefix}_oc", f"eval:{prefix}_ev",
                   f"batch:{prefix}_batch", f"lora:{prefix}_lora",
                   f"train_int:{prefix}_ti", f"lab:{prefix}_seg.wav",
                   f"synth:{prefix}_out.wav", f"train_run:{done_run}",
                   f"train_run:{busy_run}", f"dataset:{ds_name}",
                   f"voice:{prefix}"}

        # --------------------------------------------------------------
        # 1) 扫描
        # --------------------------------------------------------------
        r = AR.scan(cfg)
        keys = {i.key for i in r.items}
        missing = planted - keys
        check("扫描发现全部 11 类假产物", not missing, f"缺: {sorted(missing)}")
        by = {i.key: i for i in r.items}
        check("一键三连产物带 origin 标记",
              by[f"oneclick:{prefix}_oc"].origin == "一键三连")
        check("一键三连报告目录 note 标注含报告",
              "报告" in by[f"oneclick:{prefix}_oc"].note)
        check("running 的训练 run 被标记保护",
              by[f"train_run:{busy_run}"].protected is True)
        check("done 的训练 run 不被保护",
              by[f"train_run:{done_run}"].protected is False)
        order = {k: n for n, (k, *_r) in enumerate(AR.CATEGORIES)}
        idx = [order.get(i.kind) for i in r.items]
        check("条目按 CATEGORIES 顺序分组排列", idx == sorted(idx))
        check("越权根（底模/预设/日志）绝不出现在盘点里",
              not any(i.path.startswith(("checkpoints/", "outputs/presets/",
                                         "outputs/logs/")) for i in r.items))

        # --------------------------------------------------------------
        # 2) 正常删除（专用通道 + 通用通道各验一个，其余整批走）
        # --------------------------------------------------------------
        res = AR.delete_items(cfg, [f"dataset:{ds_name}"])
        check("数据集走 DS.delete 通道删除成功",
              res["ok"] and not DS.exists(ds_name), str(res["failed"]))

        res = AR.delete_items(cfg, [f"voice:{prefix}"])
        check("音色走 voice_bank.remove 通道删除成功",
              f"voice:{prefix}" in res["deleted"]
              and prefix not in VB.names(), str(res["failed"]))

        res = AR.delete_items(cfg, [f"train_run:{done_run}"])
        check("训练 run 走 delete_run 通道删除成功",
              res["ok"] and not os.path.isdir(RN.run_dir(done_run)),
              str(res["failed"]))

        res = AR.delete_items(cfg, [f"oneclick:{prefix}_oc", f"synth:{prefix}_out.wav",
                                    f"lab:{prefix}_seg.wav", f"eval:{prefix}_ev",
                                    f"batch:{prefix}_batch", f"lora:{prefix}_lora",
                                    f"train_int:{prefix}_ti"])
        check("通用通道整批删除 7 项全部成功",
              len(res["deleted"]) == 7 and not res["failed"], str(res["failed"]))
        check("释放字节数与假产物体积一致（共 10 KB + 收编音频）",
              res["freed_bytes"] >= 10 * 1024, f"{res['freed_bytes']} B")

        # --------------------------------------------------------------
        # 3) 重复删除 → skip 而非 fail
        # --------------------------------------------------------------
        res = AR.delete_items(cfg, [f"oneclick:{prefix}_oc", f"synth:{prefix}_out.wav"])
        check("已删项重复删除记为 skip 不报错",
              res["skipped"] and not res["failed"] and not res["deleted"])

        # --------------------------------------------------------------
        # 4) 安全：越权与伪造 key
        # --------------------------------------------------------------
        bad = ["oneclick:..", "synth:../webui_pro.py", "lora:../../README.md",
               "dataset:..", "train_run:..\\checkpoints", "voice:../x",
               "cache:../..", "nonsense:x", ""]
        res = AR.delete_items(cfg, bad)
        check("全部越权 / 伪造 key 均未删成任何东西", not res["deleted"],
              f"deleted={res['deleted']}")
        check("越权 key 全部记入 failed（缺失记 skipped）",
              len(res["failed"]) + len(res["skipped"]) == len(bad))
        check("底模 / 入口 / README 原样健在",
              os.path.isdir("checkpoints") and os.path.isfile("webui_pro.py")
              and os.path.isfile("README.md"))

        res = AR.delete_items(cfg, [f"train_run:{busy_run}"])
        check("running 的训练 run 拒绝删除",
              not res["deleted"] and res["failed"]
              and os.path.isdir(RN.run_dir(busy_run)))

        # 训练 run 越界路径（safe_run_name 净化后不存在 → skip，不能删成）
        res = AR.delete_items(cfg, ["train_run:../.."])
        check("训练 run 穿越路径被净化拦下", not res["deleted"])
    finally:
        # --------------------------------------------------------------
        # 回收：清掉自己造的一切（不依赖被测函数，直接删）。
        # 植入中途抛异常时部分变量还不存在，逐个判空。
        # --------------------------------------------------------------
        for p in (oc, ev, tk, lo, ti, lab, voice_wav, syn):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        rn = locals().get("RN")
        if rn is not None:
            for name in (locals().get("done_run"), locals().get("busy_run")):
                if name:
                    shutil.rmtree(rn.run_dir(name), ignore_errors=True)
        ds = locals().get("DS")
        if ds is not None and locals().get("ds_name") and ds.exists(ds_name):
            ds.delete(ds_name)
        vb = locals().get("VB")
        if vb is not None:
            vb.remove(prefix)

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
