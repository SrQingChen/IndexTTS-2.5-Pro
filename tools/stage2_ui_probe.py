"""阶段 2 UI/Runner 集成测试。

build_check 只证明「Tab 画得出来」，这里多走两步：

  U1  全部 11 个 Tab 渲染 + 4 个新 Tab 的 page_load 回调可调；
  U2  TrainRunner：互斥门（引擎占用时训练被拦）、成功任务、
      make_train_job 的预检失败路径；
  U3  make_train_job 端到端：tiny GPT 经 runner 后台线程跑完一次
      真实训练（预检 → prepare → run → 收尾），与探针同一套断言的缩减版。

跑法：  .venv\\Scripts\\python.exe tools\\stage2_ui_probe.py
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import json
import os
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from webui_app.config import config_from_args          # noqa: E402
from webui_app.context import AppContext                # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    mark = "✅" if cond else "✖"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))


def head(t: str):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


TINY_GPT = dict(
    layers=2, model_dim=64, heads=4,
    max_text_tokens=64, max_mel_tokens=256, max_conditioning_inputs=1,
    number_text_tokens=512, number_mel_codes=256,
    start_text_token=0, stop_text_token=1,
    start_mel_token=254, stop_mel_token=255,
    checkpointing=False, types=1,
    emo_condition_module=dict(output_size=32, linear_units=32, attention_heads=4,
                              num_blocks=1, input_layer="linear", perceiver_mult=1),
)


def make_tiny_gpt(device="cpu"):
    from indextts.gpt.model_v2 import UnifiedVoice
    torch.manual_seed(0)
    return UnifiedVoice(**TINY_GPT, use_accel=False,
                        spk_cond_mode="campplus").to(device)


def make_wav(path, seconds, sr, seed):
    import soundfile as sf
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * (110 + seed * 7) * t) * 0.5
    sf.write(path, (sig + rng.standard_normal(n) * 0.002).astype(np.float32),
             sr, subtype="PCM_16")


def build_ds(name, n, sr=22050, seed=0):
    """带特征的最小数据集（与 gpt_train_probe 同一套合成方式，压缩版）。"""
    from webui_app.training import dataset as DS
    from webui_app.training import features as FT
    DS.create(name, note="stage2_ui_probe")
    d = DS.dir_of(name)
    tmpw = os.path.join(d, "_src")
    os.makedirs(tmpw, exist_ok=True)
    wavs = []
    for i in range(n):
        p = os.path.join(tmpw, f"s{i:03d}.wav")
        make_wav(p, 1.5, sr, seed + i)
        wavs.append(p)
    DS.import_audio(name, wavs, copy=True, lang="ZH")
    shutil.rmtree(tmpw, ignore_errors=True)
    touched = {}
    for i, u in enumerate(DS.load_meta(name)):
        spk = i % 3
        n_codes = 40 + (i % 4) * 10
        rng = np.random.default_rng(seed * 100 + spk)
        feat = {
            "text_tokens": torch.from_numpy(
                rng.integers(2, 512, size=(8,))).to(torch.int64),
            "codes": torch.full((n_codes,), 10 + spk * 50, dtype=torch.int64),
            "style": torch.zeros(192) + spk * 0.01,
            "emo_vec": torch.zeros(64),
            "mel": torch.zeros(80, int(n_codes * 3.44), dtype=torch.float16),
            "mu_prompt": torch.zeros(4, 512, dtype=torch.float16),
            "mu_target": torch.zeros(4, 512, dtype=torch.float16),
            "n_codes": n_codes, "n_text_tokens": 8,
            "mel_len": int(n_codes * 3.44), "lang_token": 1,
            "feature_version": FT.FEATURE_VERSION,
            "extract_seconds": 0.1, "extracted_at": time.time(),
            "source": "synthetic", "warnings": [],
        }
        torch.save(feat, FT.FeatureExtractor.feature_path(d, u.id))
        touched[u.id] = {"text": f"第{i}句文本", "has_features": True,
                         "features_at": time.time(), "n_codes": n_codes,
                         "n_text_tokens": 8, "mel_len": feat["mel_len"],
                         "lang_token": 1}
    FT._apply_meta(name, touched)
    DS.refresh_all(name, require_features=True)
    return d


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="stage2_ui_")
    from webui_app.training import dataset as DS
    from webui_app.training import runs as RN
    from webui_app.training.runner import get_runner, make_train_job
    DS.DATASETS_ROOT = os.path.join(tmp, "datasets")
    RN.ROOT = os.path.join(tmp, "runs")
    os.makedirs(DS.DATASETS_ROOT, exist_ok=True)
    os.makedirs(RN.ROOT, exist_ok=True)

    # ===================================================================
    head("[U1] Tab 渲染 + page_load")
    AppContext.reset()
    cfg = config_from_args(["--lazy"])
    from webui_app.app import TAB_SPECS, build_app
    demo, ctx = build_app(cfg, do_autoload=False)
    check("11 个主 Tab 全部渲染成功",
          not ctx.shared.get("_render_failures"))
    new_tabs = [r for r in TAB_SPECS if r[0] in ("data", "train", "align", "deploy")]
    check("阶段 2 的 4 个 Tab 都在 TAB_SPECS 里", len(new_tabs) == 4)

    # 4 个新 Tab 在新的 Blocks 上下文里还能渲染一次
    # （共享组件槽位不冲突、没有只在首次可用的隐藏状态）
    import importlib
    import gradio as gr
    n_pl = 0
    with gr.Blocks():
        for tab_id, _label, dotted, _desc in new_tabs:
            mod = importlib.import_module(dotted)
            try:
                mod.render(ctx)
            except Exception as e:
                check(f"{tab_id} 二次渲染可执行", False, str(e)[:60])
                continue
            n_pl += 1
    check("4 个新 Tab 的 render 可重复执行（无共享组件冲突）", n_pl == 4)

    # ===================================================================
    head("[U2] TrainRunner 的门与互斥")
    runner = get_runner()

    fake_loaded = SimpleNamespace(loaded=True, stats=SimpleNamespace(
        lora_adapters=[]))
    r = runner.submit("train", "测试训练", lambda p, s: {"ok": True},
                      require_engine="unloaded", engine=fake_loaded)
    check("引擎占用时训练被拦（可读错误）",
          not r["ok"] and "卸载" in r["message"], r["message"])
    time.sleep(0.1)
    check("被拦的任务没有留下 running 状态", not runner.snapshot()["running"])

    r2 = runner.submit("pairs", "测试合成", lambda p, s: {"ok": True},
                       require_engine="loaded", engine=SimpleNamespace(loaded=False))
    check("引擎未加载时合成类任务被拦", not r2["ok"] and "加载" in r2["message"])

    ok_job = runner.submit(
        "merge", "成功任务",
        lambda p, s: (time.sleep(0.6), {"ok": True, "x": 1})[1])
    check("普通任务提交成功", ok_job["ok"])
    busy = runner.submit("merge", "第二个", lambda p, s: {"ok": True})
    check("运行中第二个任务被互斥拦下", not busy["ok"], busy["message"])
    for _ in range(100):
        if not runner.snapshot()["running"]:
            break
        time.sleep(0.05)
    check("任务完成且 ok=True",
          runner.snapshot()["ok"] is True and runner.snapshot()["result"]["x"] == 1)

    # 失败任务：抛异常 → ok=False + error 有内容
    def boom(p, s):
        raise RuntimeError("炸了")
    runner.submit("merge", "失败任务", boom)
    for _ in range(100):
        if not runner.snapshot()["running"]:
            break
        time.sleep(0.05)
    snap = runner.snapshot()
    check("异常任务落为 ok=False 且带错误", snap["ok"] is False
          and "炸了" in snap["error"], snap["error"])

    # make_train_job 的预检失败路径（数据集不存在）
    job = make_train_job("gpt", "不存在的集", {"rank": 4, "alpha": 8}, {})
    res = job(lambda f, m: None, lambda: False)
    check("make_train_job 数据集不存在 → 预检错误字典",
          res.get("ok") is False and "预检" in res.get("error", ""),
          res.get("error", "")[:60])

    # ===================================================================
    head("[U3] make_train_job 端到端（tiny GPT 过 runner）")
    from webui_app.training import gpt_lora as GL
    real_load = GL.load_base_gpt
    model_dir = os.path.join(tmp, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "gpt.pth"), "wb") as f:
        f.write(b"\x00" * 64)
    with open(os.path.join(model_dir, "config.yaml"), "w",
              encoding="utf-8") as f:
        f.write("gpt_checkpoint: gpt.pth\n")
    GL.load_base_gpt = lambda device="cpu", model_dir=None, verify_keys=True: (
        make_tiny_gpt(device), {"missing": 0, "unexpected": 0,
                                "path": "<tiny>", "model_dir": model_dir})

    ds = "ui_probe_ds"
    build_ds(ds, 24)
    from webui_app.training import guard as GD
    cfgd = GD.LoRAConfig.preset("balanced").to_dict()
    cfgd.update(rank=8, alpha=16, lr=2e-3, weight_decay=0.01,
                batch_size=2, grad_accum=2, epochs=2, eval_every=5,
                val_patience=0, bf16=False, dropout=0.0, warmup_ratio=0.05,
                replay_ratio=0.0, keep_checkpoints=2)
    optsd = {"log_every": 2}
    job = make_train_job("gpt", ds, cfgd, optsd, run_name="ui_probe_run",
                         tracker=runner)
    r3 = runner.submit("train", "UI 探针训练", job,
                       require_engine="unloaded",
                       engine=SimpleNamespace(loaded=False))
    check("tiny 训练提交成功", r3["ok"], r3["message"])
    t0 = time.time()
    while runner.snapshot()["running"] and time.time() - t0 < 180:
        time.sleep(0.5)
    snap = runner.snapshot()
    check("后台训练完成（ok=True）", snap["ok"] is True,
          snap.get("error", "")[:80])
    check("结果带回 run 名与步数",
          snap["result"].get("run") == "ui_probe_run"
          and snap["result"].get("steps", 0) > 0,
          f"steps={snap['result'].get('steps')}")
    check("tracker 登记了 run（日志面板能看到训练器日志）",
          "训练器日志" in runner.log_text())
    check("run.json 已写出",
          os.path.isfile(os.path.join(RN.ROOT, "ui_probe_run", "run.json")))
    check("结果带回 markdown 报告",
          "训练" in (snap["result"].get("markdown") or ""))

    # 取消语义：一个会停的任务
    def slow(p, s):
        for i in range(1000):
            if s():
                return {"ok": True, "stopped": True}
            time.sleep(0.02)
        return {"ok": True}
    runner.submit("merge", "可取消任务", slow)
    time.sleep(0.15)
    cr = runner.cancel()
    check("cancel 返回确认", cr["ok"])
    t0 = time.time()
    while runner.snapshot()["running"] and time.time() - t0 < 10:
        time.sleep(0.05)
    check("被取消的任务优雅停下（should_stop 生效）",
          not runner.snapshot()["running"]
          and runner.snapshot()["result"].get("stopped") is True)

    GL.load_base_gpt = real_load
    shutil.rmtree(tmp, ignore_errors=True)
    check("临时目录已删除", not os.path.isdir(tmp))

    print("\n" + "=" * 70)
    if FAIL:
        print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
        for x in FAIL:
            print(f"    FAIL {x}")
    else:
        print(f"  通过 {len(PASS)} 项 · 失败 0 项")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
