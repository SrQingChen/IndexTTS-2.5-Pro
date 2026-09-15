"""合并与挂载（merge.py）的验证。

小模型（GPT 的 Conv1D / CFM 的 Linear 各一套）上做全链路：
注入 → 训练几步（拿到非零 ΔW）→ save_pretrained → 合并 →
逐层数值对账（W' = W + scale·(α/r)·B·A，Conv1D 转置由 PEFT 处理）→
合并产物能被全新底座完整加载。另钉：输出路径红线、备份、挂载助手。

跑法：  .venv\\Scripts\\python.exe tools\\merge_probe.py
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import os
import shutil
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

from webui_app.training import cfm_lora as CL               # noqa: E402
from webui_app.training import forward as FW                # noqa: E402
from webui_app.training import gpt_lora as GL               # noqa: E402
from webui_app.training import guard as GD                  # noqa: E402
from webui_app.training import merge as MG                  # noqa: E402
from webui_app.training import runs as RN                   # noqa: E402

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

TINY_S2MEL = dict(
    dit_type="DiT", reg_loss_type="l1",
    style_encoder=dict(dim=192),
    length_regulator=dict(channels=64, is_discrete=False, in_channels=64,
                          content_codebook_size=128, sampling_ratios=[1, 1, 1, 1],
                          vector_quantize=False, n_codebooks=1,
                          quantizer_dropout=0.0, f0_condition=False,
                          n_f0_bins=512),
    DiT=dict(hidden_dim=64, num_heads=4, depth=2, class_dropout_prob=0.1,
             block_size=8192, in_channels=80, style_condition=True,
             final_layer_type="wavenet", target="mel", content_dim=64,
             content_codebook_size=128, content_type="discrete",
             f0_condition=False, n_f0_bins=512, content_codebooks=1,
             is_causal=False, long_skip_connection=True,
             zero_prompt_speech_token=False, time_as_token=False,
             style_as_token=False, uvit_skip_connection=True,
             add_resblock_in_transformer=False),
    wavenet=dict(hidden_dim=64, num_layers=2, kernel_size=5, dilation_rate=1,
                 p_dropout=0.2, style_condition=True),
)


def tiny_gpt(device="cpu"):
    from indextts.gpt.model_v2 import UnifiedVoice
    torch.manual_seed(0)
    return UnifiedVoice(**TINY_GPT, use_accel=False,
                        spk_cond_mode="campplus").to(device)


def tiny_cfm(device="cpu"):
    from omegaconf import OmegaConf
    from indextts.s2mel.modules.flow_matching import CFM
    torch.manual_seed(0)
    return CFM(OmegaConf.create(TINY_S2MEL)).to(device)


def make_adapter(base, target_pats, steps=3, seed=11):
    """注入 + 随机化 B + 几步真实更新，产出有真实 ΔW 的 adapter 目录。
    返回 (peft_model, adapter_dir, 注入前的底座指纹)。"""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    GD.freeze_base(base)
    rx = GD.build_target_regex(
        GD.resolve_target_patterns(GD.scan_targets(base), target_pats))
    pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                         target_modules=rx, bias="none",
                                         task_type=None))
    with torch.no_grad():
        for _n, _l in GD.iter_lora_layers(pm):
            torch.nn.init.normal_(_l.lora_B["default"].weight, std=0.02)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],
                            lr=1e-3)
    for _ in range(steps):
        x = torch.randn(2, 80, 120)
        if "attention/wqkv" in str(target_pats):       # CFM
            loss, _ = FW.cfm_training_forward(
                pm, x, torch.LongTensor([120, 110]), torch.LongTensor([40, 40]),
                torch.randn(2, 120, 64), torch.randn(2, 192))
        else:                                          # GPT
            f = FW.gpt_training_forward(
                pm, torch.randn(2, 192), torch.randn(2, 64),
                torch.LongTensor([1, 1]),
                torch.randint(2, 500, (2, 8)), torch.LongTensor([8, 6]),
                torch.randint(2, 250, (2, 40)), torch.LongTensor([40, 36]))
            loss = f.mel_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
    d = tempfile.mkdtemp(prefix="adapter_")
    pm.save_pretrained(d)
    return pm, d


def snapshot_weights(core, pat_hint):
    """底座（requires_grad=False）在指定注入面上的权重快照。"""
    import re
    rx = re.compile(pat_hint)
    return {n: p.detach().clone() for n, p in core.named_parameters()
            if not p.requires_grad and rx.search(n)}


def adapter_scaling(adapter_dir):
    import json as _json
    import math as _math
    cfg = _json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r, a = int(cfg["r"]), float(cfg["lora_alpha"])
    return a / (_math.sqrt(r) if cfg.get("use_rslora") else r)


def main() -> int:
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="merge_probe_")
    run_root = os.path.join(tmp, "runs")
    model_dir = os.path.join(tmp, "checkpoints")
    os.makedirs(run_root); os.makedirs(model_dir)
    RN.ROOT = run_root

    # 底座目录：给 BaseGuard 与 load_base_* 的替代品一个落点
    gpt_pth = os.path.join(model_dir, "gpt.pth")
    s2mel_pth = os.path.join(model_dir, "s2mel.pth")
    with open(gpt_pth, "wb") as f:
        f.write(b"\x00" * 4096)
    with open(s2mel_pth, "wb") as f:
        f.write(b"\x00" * 4096)
    with open(os.path.join(model_dir, "config.yaml"), "w",
              encoding="utf-8") as f:
        f.write("gpt_checkpoint: gpt.pth\ns2mel_checkpoint: s2mel.pth\n")

    # 真底座权重的替身：小模型的 state_dict（gpt 带官方的 "model" 包装，
    # s2mel 带官方的 net.cfm 包装 —— 合并必须原样保留这些包装格式）
    seed_gpt = tiny_gpt("cpu")
    torch.save({"model": seed_gpt.state_dict()}, gpt_pth)
    seed_cfm = tiny_cfm("cpu")
    torch.save({"net": {"cfm": seed_cfm.state_dict()},
                "optimizer": {}}, s2mel_pth)
    del seed_gpt, seed_cfm

    GL.load_base_gpt = lambda device="cpu", model_dir=None, verify_keys=True: (
        tiny_gpt(device), {"missing": 0, "unexpected": 0, "path": gpt_pth,
                           "model_dir": model_dir})
    CL.load_base_cfm = lambda device="cpu", model_dir=None, verify_keys=True: (
        tiny_cfm(device), {"missing": 0, "unexpected": 0, "path": s2mel_pth,
                           "model_dir": model_dir})

    guard = GD.BaseGuard(model_dir=model_dir, train_root=os.path.join(tmp, "guard"))
    out_dir = os.path.join(tmp, "merged")
    os.makedirs(out_dir)

    try:
        # ==============================================================
        head("[1] GPT（Conv1D）：合并数学逐层对账")
        pm, adir = make_adapter(tiny_gpt(), ["attn/c_attn", "attn/c_proj"])
        n_layers = sum(1 for _n, m in pm.named_modules() if hasattr(m, "lora_A"))
        check(f"adapter 造好（{n_layers} 层，B 非零）",
              n_layers == 4 and os.path.isfile(
                  os.path.join(adir, "adapter_config.json")))

        # 注入面上的原权重（从 PEFT 包装里取 base_layer 的）
        before = {}
        from webui_app.training.guard import iter_lora_layers
        for name, layer in iter_lora_layers(pm):
            before[name] = layer.get_base_layer().weight.detach().clone()

        out_gpt = os.path.join(out_dir, "gpt_merged.pth")
        rep = MG.merge_lora_to_checkpoint(
            MG.MergeOptions(adapter_dir=adir, arch="gpt", out_path=out_gpt),
            guard=guard, model_dir=model_dir)
        check("GPT 合并 ok", rep.ok, "; ".join(rep.notes))
        check("合并了 4 层", rep.n_merged == 4, str(rep.n_merged))
        check("漂移体检非零（合并就是有限改动的极限）",
              rep.global_rel > 0, f"{rep.global_rel}")

        # 逐层对账：W' == W + scaling·(B@A)ᵀ（Conv1D 的 fan_in_fan_out）
        from peft import load_peft_weights
        sd_ad = load_peft_weights(adir, device="cpu")
        merged_sd = torch.load(out_gpt, map_location="cpu",
                               weights_only=False)["model"]

        def ad_key(module, part):
            for suf in (".weight", ".default.weight"):
                k = f"{module}.lora_{part}{suf}"
                if k in sd_ad:
                    return k
            raise KeyError(f"{module}.lora_{part}")

        ok_math, worst = True, 0.0
        for name, w_old in before.items():
            w_new = merged_sd[name.replace("base_model.model.", "") + ".weight"]
            a = sd_ad[ad_key(name, "A")]
            b = sd_ad[ad_key(name, "B")]
            # 合并的真实公式：W' = W + scaling·(B·A)ᵀ，scaling = α/r = 2
            delta = adapter_scaling(adir) * (b @ a).T
            diff = (w_new - w_old - delta).abs().max().item()
            worst = max(worst, diff)
            ok_math = ok_math and diff < 1e-5
        check("★ 合并数学逐层对账：W' = W + (B·A)ᵀ（Conv1D 转置正确）",
              ok_math, f"最大偏差 {worst:.2e}")
        check("合并产物保留了官方的 {'model': …} 包装",
              isinstance(torch.load(out_gpt, map_location="cpu",
                                    weights_only=False), dict)
              and "model" in torch.load(out_gpt, map_location="cpu",
                                        weights_only=False))
        del pm

        # ---- 半强度合并：ΔW 减半 ----
        out_gpt05 = os.path.join(out_dir, "gpt_merged_05.pth")
        rep05 = MG.merge_lora_to_checkpoint(
            MG.MergeOptions(adapter_dir=adir, arch="gpt",
                            out_path=out_gpt05, scale=0.5),
            guard=guard, model_dir=model_dir)
        sd05 = torch.load(out_gpt05, map_location="cpu",
                          weights_only=False)["model"]
        ok_half = all(
            (sd05[n.replace("base_model.model.", "") + ".weight"] - w_old)
            .abs().max().item()
            < 0.5 * (merged_sd[n.replace("base_model.model.", "") + ".weight"]
                     - w_old).abs().max().item() + 1e-6
            for n, w_old in before.items())
        check("scale=0.5 合并：每层的 ΔW 都减半", rep05.ok and ok_half)

        # ==============================================================
        head("[2] CFM（Linear）：合并 + 包装格式")
        pmc, adir_c = make_adapter(tiny_cfm(),
                                   ["attention/wqkv", "attention/wo"])
        before_c = {}
        for name, layer in iter_lora_layers(pmc):
            before_c[name] = layer.get_base_layer().weight.detach().clone()
        out_cfm = os.path.join(out_dir, "s2mel_merged.pth")
        repc = MG.merge_lora_to_checkpoint(
            MG.MergeOptions(adapter_dir=adir_c, arch="cfm", out_path=out_cfm),
            guard=guard, model_dir=model_dir)
        check("CFM 合并 ok", repc.ok, "; ".join(repc.notes))
        sd_ad_c = load_peft_weights(adir_c, device="cpu")
        merged_c = torch.load(out_cfm, map_location="cpu",
                              weights_only=False)

        def ad_key_c(module, part):
            for suf in (".weight", ".default.weight"):
                k = f"{module}.lora_{part}{suf}"
                if k in sd_ad_c:
                    return k
            raise KeyError(f"{module}.lora_{part}")
        check("保留了官方的 {'net': {'cfm': …}} 包装（其余键不动）",
              "net" in merged_c and "cfm" in merged_c["net"]
              and "optimizer" in merged_c)
        cfm_sd = merged_c["net"]["cfm"]
        ok_math_c, worst_c = True, 0.0
        for name, w_old in before_c.items():
            w_new = cfm_sd[name.replace("base_model.model.", "") + ".weight"]
            a = sd_ad_c[ad_key_c(name, "A")]
            b = sd_ad_c[ad_key_c(name, "B")]
            delta = adapter_scaling(adir_c) * (b @ a)   # Linear: += scaling·B·A
            diff = (w_new - w_old - delta).abs().max().item()
            worst_c = max(worst_c, diff)
            ok_math_c = ok_math_c and diff < 1e-5
        check("★ 合并数学逐层对账：W' = W + B·A（Linear 不转置）",
              ok_math_c, f"最大偏差 {worst_c:.2e}")

        # 合并产物能被全新底座完整吃下
        fresh = tiny_cfm("cpu")
        miss, unexp = fresh.load_state_dict(cfm_sd, strict=False)
        check("合并产物被全新 CFM 完整加载（missing/unexpected=0）",
              len(miss) == 0 and len(unexp) == 0,
              f"{len(miss)}/{len(unexp)}")
        del pmc, fresh

        # ==============================================================
        head("[3] 红线：输出不准写进底座目录")
        for bad in (os.path.join(model_dir, "gpt.pth"),
                    os.path.join(model_dir, "sub", "gpt_merged.pth")):
            repb = MG.merge_lora_to_checkpoint(
                MG.MergeOptions(adapter_dir=adir, arch="gpt", out_path=bad),
                guard=guard, model_dir=model_dir)
            check(f"拦下了 {os.path.relpath(bad, model_dir)}",
                  not repb.ok and any("防线" in n for n in repb.notes),
                  "; ".join(repb.notes)[:60])
        check("原 gpt.pth 一字未动（红线没破）",
              os.path.getsize(gpt_pth) > 1000)

        # 备份：覆盖已有输出前先存 .bak
        repb2 = MG.merge_lora_to_checkpoint(
            MG.MergeOptions(adapter_dir=adir, arch="gpt", out_path=out_gpt),
            guard=guard, model_dir=model_dir)
        check("覆盖输出前做了 .bak 备份",
              repb2.ok and os.path.isfile(out_gpt + ".bak"))

        # 坏输入的可读错误
        repb3 = MG.merge_lora_to_checkpoint(
            MG.MergeOptions(adapter_dir="Z:/无", arch="gpt",
                            out_path=os.path.join(out_dir, "x.pth")),
            guard=guard, model_dir=model_dir)
        check("adapter 目录不存在给可读错误", not repb3.ok)
        repb4 = MG.merge_lora_to_checkpoint(
            MG.MergeOptions(adapter_dir=adir, arch="tts",
                            out_path=os.path.join(out_dir, "x.pth")),
            guard=guard, model_dir=model_dir)
        check("arch 非法给可读错误", not repb4.ok)
        check("报告能渲染 Markdown", "合并" in repb2.markdown())

        # ==============================================================
        head("[4] 挂载助手（mount_run / set_scale / unmount）")
        RN.run_dir("runM", create=True)
        ada = RN.adapter_dir("runM", create=True)
        shutil.copytree(adir, ada, dirs_exist_ok=True)
        RN.write_run("runM", {"arch": "gpt", "run": "runM", "status": "done"})

        scales = {}

        class FakeMod:
            class _L:
                pass

        class FakeEngine:
            def __init__(self):
                self.tts = SimpleNamespace(gpt=SimpleNamespace())
                self.stats = SimpleNamespace(lora_adapters=[])
                self.attached = []

            def attach_lora(self, d, target="gpt"):
                self.attached.append((d, target))
                self.stats.lora_adapters.append(f"{target}:{os.path.basename(d)}")
                return self.stats.lora_adapters[-1]

            def detach_lora(self, target="gpt"):
                self.attached = [a for a in self.attached if a[1] != target]
                self.stats.lora_adapters = []

        # set_scale 走 GD.set_adapter_scale，需要真 LoRA 层 —— 用真 pm 验证
        pm_s, _ = make_adapter(tiny_gpt(), ["attn/c_attn", "attn/c_proj"])
        eng_real = SimpleNamespace(tts=SimpleNamespace(gpt=pm_s))
        n_set = MG.set_scale(eng_real, 0.6, "gpt")
        got = GD.get_adapter_scale(pm_s)      # 汇总：_min/_max/_mean/_layers
        check("set_scale 真的调了 LoRA 层的增益（0.6，全部 4 层一致）",
              n_set == 4 and got["_layers"] == 4
              and abs(got["_min"] - 0.6) < 1e-9
              and abs(got["_max"] - 0.6) < 1e-9,
              f"{n_set} 层 → min={got['_min']} max={got['_max']}")
        MG.set_scale(eng_real, 1.0, "gpt")
        got2 = GD.get_adapter_scale(pm_s)
        check("set_scale(1.0) 从名义值重算，不会累积漂移",
              abs(got2["_min"] - 1.0) < 1e-9 and abs(got2["_max"] - 1.0) < 1e-9)
        # 关键：替身必须用**真实类型** nn.ModuleDict。之前这里用的是普通 dict，
        # 而 dict 有 .get()、ModuleDict 没有 —— 于是
        # `tts.s2mel.models.get("cfm")` 这个会抛 AttributeError 的写法
        # 在探针里一路通过，真机上却让「挂载 CFM adapter」直接失败。
        # 替身比真实对象宽松，测试反而在掩护 bug。
        _md_empty = nn.ModuleDict()
        check("cfm 目标为空 ModuleDict 时安全返回 0（真实类型）",
              MG.set_scale(SimpleNamespace(tts=SimpleNamespace(
                  s2mel=SimpleNamespace(models=_md_empty))), 0.5, "cfm") == 0)
        check("cfm 目标缺 key 时安全返回 0",
              MG.set_scale(SimpleNamespace(tts=SimpleNamespace(
                  s2mel=SimpleNamespace(models=nn.ModuleDict(
                      {"other": nn.Linear(2, 2)})))), 0.5, "cfm") == 0)
        check("**ModuleDict 上取子模块不会抛 AttributeError**"
              "（曾经的真实故障：'ModuleDict' object has no attribute 'get'）",
              GD.lora_target_module(SimpleNamespace(tts=SimpleNamespace(
                  s2mel=SimpleNamespace(models=nn.ModuleDict(
                      {"cfm": nn.Linear(2, 2)})))), "cfm") is not None)
        check("普通 dict 也照样支持（历史调用方）",
              GD.lora_target_module(SimpleNamespace(tts=SimpleNamespace(
                  s2mel=SimpleNamespace(models={"cfm": "X"}))), "cfm") == "X")
        check("引擎未加载（tts 抛异常）时返回 None 而不是崩",
              GD.lora_target_module(SimpleNamespace(), "cfm") is None)

        fe = FakeEngine()
        tag = MG.mount_run(fe, "runM", scale=1.0)
        check("mount_run 挂到了 runM 的 adapter（gpt）",
              tag.startswith("gpt:") and fe.attached[0][0] == ada)
        MG.unmount(fe, "gpt")
        check("unmount 卸载干净", not fe.attached)
        try:
            MG.resolve_mount_dir("runX")
            check("不存在的 run 报 FileNotFoundError", False)
        except FileNotFoundError:
            check("不存在的 run 报 FileNotFoundError", True)

        # ==============================================================
        head("[5] 清理")
        shutil.rmtree(tmp, ignore_errors=True)
        check("临时目录已删除", not os.path.isdir(tmp))

    except Exception as e:
        import traceback
        traceback.print_exc()
        check(f"未捕获异常：{type(e).__name__}: {e}", False)

    print("\n" + "=" * 70)
    if FAIL == 0:
        print(f"  通过 {PASS} 项 · 失败 0 项")
    else:
        print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
        for x in FAILS:
            print(f"    FAIL {x}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
