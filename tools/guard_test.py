"""泛化保护模块（training/guard.py）的单元验证。

全程 CPU、用微型模型，几秒跑完，不占显存，可以在 WebUI 运行时并行执行。
微型模型刻意同时包含 **Conv1D 的 attn.c_proj 与 mlp.c_proj**（与真实 GPT2 同名），
用来验证注入面扫描确实能把两者区分开 —— 这是整个「缩小注入面」防线的前提。

    .venv\\Scripts\\python.exe tools\\guard_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import _env                                            # noqa: F401  路径 + 控制台编码

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 微型模型：形状与命名都对齐真实的 IndexTTS GPT2 主干
# ---------------------------------------------------------------------------

def build_tiny_model(dim: int = 32, n_blocks: int = 2, n_heads: int = 2):
    import torch
    import torch.nn as nn
    from transformers.pytorch_utils import Conv1D

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d, self.h = d, n_heads
            self.c_attn = Conv1D(3 * d, d)      # weight 形状 (in, out)
            self.c_proj = Conv1D(d, d)

        def forward(self, x):
            B, T, _ = x.shape
            q, k, v = self.c_attn(x).split(self.d, dim=2)
            hd = self.d // self.h
            sh = lambda t: t.view(B, T, self.h, hd).transpose(1, 2)   # noqa: E731
            q, k, v = sh(q), sh(k), sh(v)
            o = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5, -1) @ v
            o = o.transpose(1, 2).contiguous().view(B, T, self.d)
            return self.c_proj(o)

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c_fc = Conv1D(4 * d, d)
            self.c_proj = Conv1D(d, 4 * d)      # 与 attn.c_proj **同名**

        def forward(self, x):
            import torch as _t
            return self.c_proj(_t.relu(self.c_fc(x)))

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn, self.mlp = Attn(d), MLP(d)

        def forward(self, x):
            return x + self.mlp.forward(self.attn(x))

    class Tiny(nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.wte = nn.Embedding(64, d)
            self.h = nn.ModuleList([Block(d) for _ in range(n)])
            self.mel_head = nn.Linear(d, 48)

        def forward(self, ids):
            x = self.wte(ids)
            for b in self.h:
                x = b(x)
            return self.mel_head(x)

    return Tiny(dim, n_blocks)


def inject_lora(model, patterns, rank=4, alpha=8, dropout=0.0):
    from peft import LoraConfig as PeftLoraConfig
    from peft.tuners.lora import LoraModel
    from webui_app.training.guard import build_target_regex

    pc = PeftLoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=build_target_regex(patterns),
                        bias="none")
    return LoraModel(model, pc, adapter_name="default")


def main() -> int:
    import torch
    from webui_app.training import guard as G

    torch.manual_seed(0)

    # =======================================================================
    print("\n[1] LoRAConfig：预设 / 增益 / 序列化")
    cons = G.LoRAConfig.preset("conservative")
    bal = G.LoRAConfig.preset("balanced")
    agg = G.LoRAConfig.preset("aggressive")
    check("保守档 rank=4 + rsLoRA", cons.rank == 4 and cons.use_rslora)
    check("保守档增益 = alpha/√rank = 8/2 = 4.0",
          abs(cons.effective_scale - 4.0) < 1e-9, f"{cons.effective_scale}")
    check("均衡档增益 = 16/8 = 2.0", abs(bal.effective_scale - 2.0) < 1e-9,
          f"{bal.effective_scale}")
    check("激进档增益 = 64/32 = 2.0", abs(agg.effective_scale - 2.0) < 1e-9)
    check("保守档回放 50% / 均衡档 30% / 激进档 0%",
          cons.replay_ratio == 0.5 and bal.replay_ratio == 0.3
          and agg.replay_ratio == 0.0)
    check("预设会带入注入面清单", len(cons.target_modules) >= 2,
          str(cons.target_modules))
    check("global_batch = batch_size × grad_accum",
          bal.global_batch == bal.batch_size * bal.grad_accum, f"{bal.global_batch}")

    rt = G.LoRAConfig.from_dict(cons.to_dict())
    check("to_dict/from_dict 往返一致",
          all(getattr(rt, f) == getattr(cons, f) for f in G.LoRAConfig.__dataclass_fields__),
          str([f for f in G.LoRAConfig.__dataclass_fields__
               if getattr(rt, f) != getattr(cons, f)]))
    check("to_dict 里带了计算出的增益", "_effective_scale" in cons.to_dict())
    check("from_dict 忽略未知键",
          G.LoRAConfig.from_dict({"rank": 6, "垃圾键": 1}).rank == 6)

    print("\n[2] LoRAConfig.validate：硬错误必须拦下")
    def n_err(**kw):
        c = G.LoRAConfig(**kw)
        return [x for x in c.validate() if x.level == "error"]
    check("rank=0 报错", len(n_err(rank=0)) == 1)
    check("alpha=0 报错", len(n_err(alpha=0)) == 1)
    check("dropout=2 报错", len(n_err(dropout=2.0)) == 1)
    check("lr=0 报错", len(n_err(lr=0.0)) == 1)
    check("replay_ratio=1.0 报错", len(n_err(replay_ratio=1.0)) == 1)
    check("adapter_scale=5 报错", len(n_err(adapter_scale=5.0)) == 1)
    check("epochs=0 且 max_steps=-1 报错", len(n_err(epochs=0, max_steps=-1)) == 1)
    check("合法配置无 error", not n_err())

    print("\n[3] LoRAConfig.validate：泛化风险必须提示")
    def has(level, keyword, **kw):
        c = G.LoRAConfig(**kw)
        got = [x for x in c.validate() if x.level == level and keyword in x.message]
        check(f"{kw or '默认'} → {level} 含「{keyword}」", bool(got),
              got[0].message[:60] if got else
              str([(x.level, x.message[:34]) for x in c.validate()]))
    has("warn", "容量偏大", rank=32)
    has("warn", "旁路增益", rank=4, alpha=64)          # 增益 16
    has("warn", "没有任何随机正则", rank=16, dropout=0.0)
    has("warn", "回放比例 = 0", replay_ratio=0.0)
    has("warn", "早停已关闭", val_patience=0)
    has("warn", "全部线性层", target_preset="all_linear")
    has("info", "weight_decay=0", weight_decay=0.0)
    check("均衡档默认只给 warn 不给 error",
          not [x for x in bal.validate() if x.level == "error"])
    check("均衡档无泛化警告",
          not [x for x in bal.validate() if x.level in ("error", "warn")],
          str([(x.level, x.message[:30]) for x in bal.validate()]))

    print("\n[4] validate(n_samples)：数据量 vs 训练强度")
    e1 = [x for x in bal.validate(n_samples=10) if x.level == "error"]
    check("10 条样本直接判 error", bool(e1), e1[0].message[:50] if e1 else "")
    w1 = [x for x in bal.validate(n_samples=40) if x.level == "warn"]
    check("40 条样本给 warn + 建议保守档", any("保守" in x.message for x in w1),
          str([x.message[:34] for x in w1]))
    w2 = [x for x in bal.validate(n_samples=6) if x.level == "warn"]
    check("样本数不足两个 batch 时警告", any("global_batch" in x.message for x in w2),
          str([x.message[:40] for x in w2]))
    w3 = [x for x in G.LoRAConfig(rank=8, epochs=400, grad_accum=1,
                                 batch_size=1).validate(n_samples=1000)
          if x.level == "warn"]
    check("总步数 >20000 时警告", any("总步数" in x.message for x in w3),
          str([x.message[:40] for x in w3]))
    md = bal.risk_markdown(n_samples=120, adapter_params=7_860_000,
                           base_params=812_000_000)
    check("risk_markdown 含可训练占比", "0.97%" in md, md[-200:].replace("\n", " "))
    check("risk_markdown 含数据量与步数", "120" in md and "step" in md)

    # =======================================================================
    print("\n[5] 注入面扫描：attn.c_proj 与 mlp.c_proj 必须分开")
    tiny = build_tiny_model(dim=32, n_blocks=2)
    groups = G.scan_targets(tiny)
    pats = {g.pattern for g in groups}
    check("扫出 attn/c_proj", "attn/c_proj" in pats, str(sorted(pats)))
    check("扫出 mlp/c_proj", "mlp/c_proj" in pats, str(sorted(pats)))
    check("两者是**不同**的组", "attn/c_proj" in pats and "mlp/c_proj" in pats)
    check("扫出 attn/c_attn 与 mlp/c_fc", {"attn/c_attn", "mlp/c_fc"} <= pats)
    check("Conv1D 被识别为 Conv1D 而非 Linear",
          next(g.kind for g in groups if g.pattern == "attn/c_proj") == "Conv1D")
    check("Embedding 也被扫到", "wte" in pats,
          str([g.pattern for g in groups if g.kind == "Embedding"]))
    check("Linear 层被扫到", any(g.kind == "Linear" for g in groups))
    ga = next(g for g in groups if g.pattern == "attn/c_attn")
    check("c_attn 的 in→out 是 32→96（Conv1D 按 weight.shape）",
          (ga.in_features, ga.out_features) == (32, 96),
          f"{ga.in_features}→{ga.out_features}")
    check("层数统计正确（2 个 block）",
          next(g.count for g in groups if g.pattern == "attn/c_proj") == 2)

    rx = G.build_target_regex(["attn/c_proj"])
    import re as _re
    check("正则能匹配 h.0.attn.c_proj",
          bool(_re.fullmatch(rx, "h.0.attn.c_proj")), rx)
    check("正则**不**匹配 h.0.mlp.c_proj",
          not _re.fullmatch(rx, "h.0.mlp.c_proj"), rx)
    rx2 = G.build_target_regex(["attn/c_proj", "attn/c_attn"])
    check("多项合成后都能匹配",
          bool(_re.fullmatch(rx2, "h.1.attn.c_proj"))
          and bool(_re.fullmatch(rx2, "h.1.attn.c_attn"))
          and not _re.fullmatch(rx2, "h.1.mlp.c_fc"), rx2)
    check("顶层模块名（无点）也能匹配",
          bool(_re.fullmatch(G.build_target_regex(["mel_head"]), "mel_head")))
    check("* 通配返回 .*", G.build_target_regex(["*"]) == ".*")
    check("空清单返回空串", G.build_target_regex([]) == "")

    res = G.resolve_target_patterns(groups, ["attn/c_proj", "不存在的层", "mlp/c_fc"])
    check("resolve 丢掉模型里没有的项",
          res == ["attn/c_proj", "mlp/c_fc"], str(res))
    check("resolve 遇 * 展开成全部扫描到的 pattern（而不是把裸 * 往下传）",
          G.resolve_target_patterns(groups, ["*"])
          == [g.pattern for g in groups
              if g.pattern not in G.DEAD_MODULE_PATTERNS],
          str(G.resolve_target_patterns(groups, ["*"])))
    # 【回归】旧行为是 resolve(["*"]) → ["*"] → build_target_regex → 裸 `.*`，
    # PEFT 拿它 fullmatch **每一个**模块名 —— CFM 的 `criterion`（L1Loss）
    # 也在模块树里，于是直接抛 ValueError: Target module L1Loss() is not supported。
    # CFM 的 all_linear 预设用的就是 "*"，所以那条路必崩，不是理论风险。
    rx_star = G.build_target_regex(G.resolve_target_patterns(groups, ["*"]))
    check("【回归】展开后的正则不会匹到非注入模块（L1Loss / criterion）",
          not any(_re.fullmatch(rx_star, n) for n in
                  ("criterion", "estimator.criterion", "criterion.weight",
                   "loss_fn", "wavenet.drop")), rx_star[:70])
    check("【回归】但展开后仍能匹到真实注入面",
          all(_re.fullmatch(rx_star, n) for n in
              ("h.0.attn.c_proj", "h.1.mlp.c_fc", "wte")), rx_star[:70])
    check("resolve 去重",
          G.resolve_target_patterns(groups, ["attn/c_proj", "attn/c_proj"])
          == ["attn/c_proj"])

    est = G.estimate_adapter_params(
        [g for g in groups if g.pattern in ("attn/c_attn", "attn/c_proj")],
        G.LoRAConfig(rank=4))
    # 每 block: c_attn 4×(32+96)=512, c_proj 4×(32+32)=256 → 768，两个 block → 1536
    check("adapter 参数估算 = r×(in+out)×层数", est == 1536, f"{est}")
    smd = G.scan_targets_markdown(tiny, G.LoRAConfig(rank=4))
    check("scan_targets_markdown 可渲染", "attn/c_proj" in smd and "Conv1D" in smd,
          f"{len(smd)} 字符")

    # =======================================================================
    print("\n[6] 冻结底座 + PEFT 注入 + 强度旋钮")
    st = G.freeze_base(tiny)
    check("freeze_base 冻住了全部参数",
          st["newly_frozen"] == st["total_params"] and st["already_frozen"] == 0,
          str(st))
    check("冻结后无可训练参数", G.count_params(tiny)["trainable"] == 0)

    attn_only = ["attn/c_attn", "attn/c_proj"]
    lm = inject_lora(tiny, attn_only, rank=4, alpha=8)
    # PEFT 把 lora_B 初始化为**全零**（保证注入瞬间模型行为不变），
    # 所以不先随机化就测不出「LoRA 真的在起作用」，也测不出漂移。
    with torch.no_grad():
        for _n, _l in G.iter_lora_layers(lm):
            torch.nn.init.normal_(_l.lora_B["default"].weight, std=0.05)
    cp = G.count_params(lm)
    check("注入后有可训练参数", cp["trainable"] > 0, str(cp))
    check("可训练参数 = 估算值 1536", cp["trainable"] == 1536,
          f'{cp["trainable"]} vs 1536')
    check("可训练占比 = trainable/total",
          abs(cp["trainable_pct"] - 1536 / 30320 * 100) < 1e-6,
          f'{cp["trainable_pct"]}%（微型模型底座只有 28.8K 参数，'
          '所以占比偏高；真实 GPT 上 7.86M/812M 约 0.97%）')
    check("assert_base_frozen：底座没有被意外解冻",
          G.assert_base_frozen(lm) == [], str(G.assert_base_frozen(lm)[:3]))

    layers = list(G.iter_lora_layers(lm))
    check("iter_lora_layers 找到 4 个 LoRA 层（2 block × 2）",
          len(layers) == 4, f"{len(layers)}")
    check("只注入了注意力，MLP 干净",
          all("mlp" not in n for n, _ in layers), str([n for n, _ in layers]))
    check("nominal_scaling = alpha/r = 2.0",
          abs(G.nominal_scaling(layers[0][1], "default") - 2.0) < 1e-9,
          f'{G.nominal_scaling(layers[0][1], "default")}')

    ids = torch.randint(0, 64, (1, 6))
    lm.eval()
    with torch.no_grad():
        G.set_adapter_scale(lm, 1.0)
        out_full = lm(ids).clone()
        G.disable_adapters(lm, True)
        out_base = lm(ids).clone()
        G.disable_adapters(lm, False)
        n0 = G.set_adapter_scale(lm, 0.0)
        out_zero = lm(ids).clone()

    check("set_adapter_scale 返回受影响层数", n0 == 4, f"{n0}")
    check("factor=0 时输出与底座**完全一致**",
          torch.equal(out_zero, out_base),
          f"max|Δ| = {(out_zero - out_base).abs().max().item():.3e}")
    check("factor=1 时输出与底座不同（LoRA 确实在起作用）",
          not torch.allclose(out_full, out_base),
          f"max|Δ| = {(out_full - out_base).abs().max().item():.3e}")
    check("disable_adapters 也能回到底座", torch.equal(out_base, out_base))

    # 幂等性：反复设同一个值不能累积放大
    G.set_adapter_scale(lm, 0.5)
    s1 = G.get_adapter_scale(lm)
    G.set_adapter_scale(lm, 0.5)
    G.set_adapter_scale(lm, 0.5)
    s2 = G.get_adapter_scale(lm)
    check("反复设置不累积（幂等）",
          abs(s1["_mean"] - 0.5) < 1e-6 and abs(s2["_mean"] - 0.5) < 1e-6,
          f"{s1['_mean']:.6f} → {s2['_mean']:.6f}")
    with torch.no_grad():
        out_half_a = lm(ids).clone()
        G.set_adapter_scale(lm, 1.0); G.set_adapter_scale(lm, 0.5)
        out_half_b = lm(ids).clone()
    check("先 1.0 再 0.5 与直接 0.5 结果一致",
          torch.allclose(out_half_a, out_half_b, atol=1e-6),
          f"max|Δ| = {(out_half_a - out_half_b).abs().max().item():.3e}")
    sc = G.get_adapter_scale(lm)
    check("get_adapter_scale 反查准确",
          abs(sc["_mean"] - 0.5) < 1e-6 and sc["_layers"] == 4, str(sc))

    # =======================================================================
    print("\n[7] 漂移体检（Conv1D 需要转置对齐）")
    G.set_adapter_scale(lm, 0.0)
    d0 = G.analyze_drift(lm)
    check("factor=0 时全局漂移 = 0", d0.global_rel == 0.0, f"{d0.global_rel}")
    check("仍能列出 4 层（不是被形状检查跳过了）",
          d0.n_layers == 4, f"{d0.n_layers}")
    check("漂移 0 判定为「保守」", d0.level[0].startswith("🟢"), d0.level[0])

    G.set_adapter_scale(lm, 1.0)
    d1 = G.analyze_drift(lm)
    check("factor=1 时漂移 > 0", d1.global_rel > 0, f"{d1.global_rel:.6f}")
    check("每层 kind 记为 Conv1D", all(r.kind == "Conv1D" for r in d1.rows),
          str({r.kind for r in d1.rows}))
    check("每层 rank 记录正确", all(r.rank == 4 for r in d1.rows))
    check("max_name 指向真实层名", any(r.name == d1.max_name for r in d1.rows),
          d1.max_name)
    # 手工复核其中一层：ΔW = B@A×scaling，Conv1D 的 W 是 (in,out) 需转置
    name, layer = layers[0]
    base = layer.get_base_layer()
    A = layer.lora_A["default"].weight.detach().float()
    B = layer.lora_B["default"].weight.detach().float()
    W = base.weight.detach().float()
    dW = (B @ A) * layer.scaling["default"]
    if W.shape != dW.shape:
        dW = dW.t()
    manual = float(dW.norm()) / float(W.norm())
    row = next(r for r in d1.rows if r.name == name)
    check("手工复核与 analyze_drift 一致（Conv1D 转置处理正确）",
          abs(manual - row.rel) < 1e-5, f"手工 {manual:.6f} vs 实测 {row.rel:.6f}")
    check("base_norm 也一致", abs(float(W.norm()) - row.base_norm) < 1e-3)
    dmd = d1.markdown()
    check("drift markdown 可渲染", "权重漂移体检" in dmd and "‖ΔW‖" in dmd,
          f"{len(dmd)} 字符")
    check("空模型不崩", G.analyze_drift(build_tiny_model(dim=8)).rows == [])

    # =======================================================================
    print("\n[8] EarlyStopper")
    es = G.EarlyStopper(patience=2, min_delta=1e-3)
    r = [es.step(v, i) for i, v in enumerate([1.0, 0.90, 0.95, 0.96])]
    check("第一次必然 improved", r[0].improved)
    check("下降算改善", r[1].improved and r[1].best == 0.90)
    check("反弹计入 bad_count", not r[2].improved and r[2].bad_count == 1)
    check("连续 2 次不改善触发早停", r[3].should_stop and r[3].bad_count == 2)
    check("早停理由里带了最好成绩", "0.90000" in r[3].reason, r[3].reason)
    check("best_epoch 记录的是最好的那一轮", es.best_epoch == 1, f"{es.best_epoch}")

    es2 = G.EarlyStopper(patience=0)
    r2 = [es2.step(v, i) for i, v in enumerate([1.0, 2.0, 3.0, 4.0])]
    check("patience=0 时永不早停", not any(x.should_stop for x in r2))
    check("关闭时仍记录 best", es2.best == 1.0, f"{es2.best}")
    check("enabled 反映开关", es.enabled and not es2.enabled)

    es3 = G.EarlyStopper(patience=2, mode="max")
    r3 = [es3.step(v) for v in [0.5, 0.7, 0.6, 0.55]]
    check("mode=max 时上升算改善", r3[1].improved and es3.best == 0.7)
    check("mode=max 时下降触发早停", r3[3].should_stop)

    es4 = G.EarlyStopper(patience=1, min_delta=0.1)
    r4 = [es4.step(v) for v in [1.0, 0.95]]
    check("min_delta 内的微小改善不算改善", not r4[1].improved,
          f"best={es4.best}")
    sd = es.state_dict()
    es5 = G.EarlyStopper()
    es5.load_state_dict(sd)
    check("state_dict 往返一致",
          es5.best == es.best and es5.bad_count == es.bad_count
          and es5.history == es.history and es5.patience == 2)

    # =======================================================================
    print("\n[9] CheckpointVault：原子写入 / top-K / 回滚")
    tmp = tempfile.mkdtemp(prefix="ix_guard_")
    try:
        vault = G.CheckpointVault(os.path.join(tmp, "ckpts"), keep=2, mode="min")
        check("空保险库 list 为 []", vault.list() == [])
        check("空保险库 best 为 None", vault.best() is None)
        check("空保险库 markdown 友好", "还没有" in vault.markdown())

        def save_fn(dest):
            with open(os.path.join(dest, "adapter.bin"), "w", encoding="utf-8") as f:
                f.write("w")

        saved = []
        for ep, mt in enumerate([0.90, 0.50, 0.70, 0.60]):
            saved.append(vault.save(save_fn, epoch=ep, step=ep * 10, metric=mt,
                                    extra={"cfg": "x"}))
        items = vault.list()
        check("keep=2 时只剩 2 档", len(items) == 2,
              str([i.name for i in items]))
        keepm = sorted(i.metric for i in items)
        check("留下的是指标最好的两个（0.50 / 0.60）",
              keepm == [0.50, 0.60], str(keepm))
        check("被淘汰的目录真的删掉了",
              not os.path.isdir(saved[0].path) and not os.path.isdir(saved[2].path))
        check("extra.json 被写进去了",
              os.path.isfile(os.path.join(saved[1].path, "extra.json")))
        check("没有残留 .tmp 目录",
              not [d for d in os.listdir(vault.root) if d.endswith(".tmp")],
              str(os.listdir(vault.root)))
        b = vault.best()
        check("best 指向 0.50", b is not None and b.metric == 0.50, str(b))
        idx = vault.list()
        check("索引里 is_best 标记正确（不是陈旧值）",
              sum(1 for i in idx if i.is_best) == 1
              and next(i for i in idx if i.is_best).metric == 0.50,
              str([(i.metric, i.is_best) for i in idx]))

        # keep=1 时 best 也不能被挤掉
        v2 = G.CheckpointVault(os.path.join(tmp, "ck2"), keep=1, mode="min")
        for ep, mt in enumerate([0.20, 0.80, 0.90]):
            v2.save(save_fn, epoch=ep, step=ep, metric=mt)
        l2 = v2.list()
        check("keep=1 时仍保住 best（0.20）",
              any(abs(i.metric - 0.20) < 1e-9 for i in l2),
              str([i.metric for i in l2]))

        act = vault.rollback("best")
        check("rollback 生成 active 目录",
              act is not None and os.path.isdir(act), str(act))
        check("active 里有 adapter 文件",
              os.path.isfile(os.path.join(act, "adapter.bin")))
        check("active 里记了回滚来源",
              os.path.isfile(os.path.join(act, "rollback.json")))
        check("rollback 认得 checkpoint 名",
              vault.rollback(saved[3].name) is not None)
        check("rollback 不存在的名字返回 None", vault.rollback("没这个") is None)
        check("vault markdown 列出全部档位",
              "0.50000" in vault.markdown() and "⭐" in vault.markdown())
        check("active 不出现在 checkpoint 列表里",
              all(i.name != G.ACTIVE_SUBDIR for i in vault.list()))

        # 底座保护：输出路径安全
        fake_ckpt = os.path.join(tmp, "checkpoints")
        os.makedirs(fake_ckpt, exist_ok=True)
        with open(os.path.join(fake_ckpt, "gpt.pth"), "wb") as f:
            f.write(b"\x00" * 4096)
        with open(os.path.join(fake_ckpt, "config.yaml"), "w", encoding="utf-8") as f:
            f.write("a: 1\n")
        bg = G.BaseGuard(model_dir=fake_ckpt, train_root=os.path.join(tmp, "runs"))

        print("\n[10] BaseGuard：底座快照 / 校验 / 只读 / 路径拦截")
        man = bg.snapshot(hashes=True)
        check("快照记录了 2 个文件", len(man["files"]) == 2, str(list(man["files"])))
        check("快照算了 SHA-256",
              all(len(d["sha256"]) == 64 for d in man["files"].values()))
        check("manifest 落盘", os.path.isfile(bg.manifest_path))

        rep = bg.verify(hashes=True)
        check("未改动时校验通过", rep.ok and rep.checked == 2, rep.markdown())
        check("深度校验标记正确", rep.size_only is False)
        rep_q = bg.verify(hashes=False)
        check("快速校验也通过且标记 size_only", rep_q.ok and rep_q.size_only)

        with open(os.path.join(fake_ckpt, "gpt.pth"), "wb") as f:
            f.write(b"\x01" * 8192)          # 改了内容和大小
        rep2 = bg.verify(hashes=False)
        check("文件被改写后校验失败", not rep2.ok and "gpt.pth" in rep2.changed,
              str(rep2.changed))
        check("失败报告里有明确说明", "底座被动过" in rep2.markdown())
        # 大小一样但内容不同：快速校验发现不了，深度校验能发现。
        # 注意必须写回**快照时的尺寸**（4096），否则 size 就先对不上了，
        # 测的就不是「同尺寸篡改」这个场景。
        orig = man["files"]["gpt.pth"]
        with open(os.path.join(fake_ckpt, "gpt.pth"), "wb") as f:
            f.write(b"\x02" * orig["size"])
        os.utime(os.path.join(fake_ckpt, "gpt.pth"),
                 (orig["mtime"], orig["mtime"]))
        rep3 = bg.verify(hashes=False)
        check("同尺寸+伪造 mtime 能骗过快速校验（这是它的固有局限）", rep3.ok)
        rep4 = bg.verify(hashes=True)
        check("但骗不过 SHA-256 深度校验", not rep4.ok and "gpt.pth" in rep4.changed)

        os.remove(os.path.join(fake_ckpt, "config.yaml"))
        rep5 = bg.verify(hashes=False)
        check("文件被删后报 missing",
              not rep5.ok and "config.yaml" in rep5.missing, str(rep5.missing))

        # 只读锁
        bg.snapshot(hashes=False)               # 重建快照（config.yaml 已删）
        n_lock, failed = bg.set_read_only(True)
        check("能加只读锁", n_lock == 1 and not failed, f"{n_lock} {failed}")
        check("锁状态可查", list(bg.read_only_state().values()) == [True],
              str(bg.read_only_state()))
        try:
            with open(os.path.join(fake_ckpt, "gpt.pth"), "wb") as f:
                f.write(b"x")
            blocked = False
        except PermissionError:
            blocked = True
        check("只读锁真的挡住了写入", blocked)
        n_un, _ = bg.set_read_only(False)
        check("能解锁", n_un == 1)
        check("解锁后可写", bg.read_only_state() == {"gpt.pth": False},
              str(bg.read_only_state()))

        # 输出路径拦截
        try:
            bg.assert_safe_output(os.path.join(fake_ckpt, "merged.pth"))
            r_in = False
        except PermissionError as e:
            r_in = "拒绝写入底座目录" in str(e)
        check("拒绝写进 checkpoints/", r_in)
        try:
            bg.assert_safe_output(os.path.join(tmp, "runs", "gpt.pth"))
            r_name = False
        except PermissionError as e:
            r_name = "与底座文件同名" in str(e)
        check("拒绝生成与底座同名的文件（哪怕在别处）", r_name)
        ok_path = bg.assert_safe_output(os.path.join(tmp, "runs", "gpt_lora.pth"))
        check("合法输出路径放行", ok_path.endswith("gpt_lora.pth"), ok_path)
        check("保护清单包含关键底座文件",
              {"gpt.pth", "s2mel.pth", "config.yaml"} <= set(G.PROTECTED_FILES))

        bmd = bg.markdown()
        check("BaseGuard.markdown 可渲染",
              "受保护文件" in bmd and "gpt.pth" in bmd, f"{len(bmd)} 字符")
        empty = G.BaseGuard(model_dir=os.path.join(tmp, "不存在"),
                            train_root=os.path.join(tmp, "runs2"))
        check("没有快照时给出明确指引",
              "尚未建立底座快照" in empty.markdown())
        check("底座目录不存在时 verify 不崩",
              not empty.verify().ok and empty.verify().missing)

        print("\n[11] 回放混合 build_epoch_plan")
        plan, rp = G.build_epoch_plan(10, 100, 0.0, seed=1)
        check("ratio=0 时没有回放", rp.n_replay == 0 and len(plan) == 10,
              f"{rp.n_replay}")
        plan, rp = G.build_epoch_plan(10, 100, 0.3, seed=1)
        check("ratio=0.3 → 10 条目标配 4 条回放", rp.n_replay == 4, f"{rp.n_replay}")
        check("实际占比接近设定值",
              abs(rp.achieved - 4 / 14) < 1e-9, f"{rp.achieved:.4f}")
        check("目标样本一个不少",
              sorted(i for k, i in plan if k == "target") == list(range(10)))
        check("回放与目标被打乱混合（不是前后拼接）",
              [k for k, _ in plan] != ["target"] * 10 + ["replay"] * 4,
              "".join("T" if k == "target" else "R" for k, _ in plan))
        plan_a, _ = G.build_epoch_plan(10, 100, 0.3, seed=7)
        plan_b, _ = G.build_epoch_plan(10, 100, 0.3, seed=7)
        plan_c, _ = G.build_epoch_plan(10, 100, 0.3, seed=8)
        check("同 seed 完全可复现", plan_a == plan_b)
        check("不同 seed 顺序不同", plan_a != plan_c)
        plan, rp = G.build_epoch_plan(10, 2, 0.5, seed=1)
        check("回放池不足时启用有放回采样",
              rp.with_replacement and rp.n_replay == 10, f"{rp.n_replay}")
        check("有放回时给出提示", "有放回" in rp.note, rp.note)
        check("有放回也能凑够数量",
              sum(1 for k, _ in plan if k == "replay") == 10)
        plan, rp = G.build_epoch_plan(10, 0, 0.5, seed=1)
        check("回放池为空时降级为 0 并说明",
              rp.n_replay == 0 and "回放池为空" in rp.note, rp.note)
        plan, rp = G.build_epoch_plan(0, 100, 0.3, seed=1)
        check("没有目标数据时不生成回放", rp.n_replay == 0 and plan == [])
        plan, rp = G.build_epoch_plan(10, 100, 5.0, seed=1)
        check("ratio 超界被夹到 0.95", abs(rp.ratio - 0.95) < 1e-9, f"{rp.ratio}")

        print("\n[12] 显存余量体检")
        has_cuda = torch.cuda.is_available()
        rep = G.vram_headroom(0.0)
        if has_cuda:
            check("能看到整卡容量", rep.total_gb > 1.0, f"{rep.total_gb:.2f} GB")
            check("free + used == total",
                  abs(rep.free_gb + rep.used_gb - rep.total_gb) < 1e-6,
                  f"{rep.free_gb:.2f} + {rep.used_gb:.2f} vs {rep.total_gb:.2f}")
            check("need=0 时判定为充足", rep.ok and rep.shortfall_gb == 0.0,
                  rep.message)
            check("充足时 level 不是空串", rep.level == "显存充足", rep.level)
            # 要一个荒谬的量，必须报缺口
            big = rep.total_gb * 10
            rep2 = G.vram_headroom(big)
            check("需求超出整卡时报缺口", not rep2.ok and rep2.shortfall_gb > 0,
                  f"缺 {rep2.shortfall_gb:.1f} GB")
            check("缺口里含了安全余量",
                  abs(rep2.shortfall_gb - (big + G.VRAM_SAFETY_MARGIN_GB
                                           - rep2.free_gb)) < 1e-6,
                  f"{rep2.shortfall_gb:.2f}")
            check("不足时 markdown 会提醒静默降速",
                  "静默溢出" in rep2.markdown() and "20~30 倍" in rep2.markdown())
            check("充足时 markdown 不报缺口",
                  "缺口" not in rep.markdown())
            try:
                G.vram_preflight(big, raise_on_short=True)
                raised = False
            except RuntimeError as e:
                raised = "显存不足" in str(e)
            check("vram_preflight(raise_on_short=True) 会拦下", raised)
            check("vram_preflight(0) 不报错也不抛",
                  G.vram_preflight(0.0).ok)
            check("torch_alloc 不超过整卡已用",
                  rep.torch_alloc_gb <= rep.used_gb + 1e-6,
                  f"{rep.torch_alloc_gb:.2f} vs {rep.used_gb:.2f}")
        else:
            check("无 CUDA 时降级为 CPU 模式且不报错",
                  rep.ok and rep.level == "CPU 模式", rep.message)
            check("无 CUDA 时 vram_preflight 也不拦",
                  G.vram_preflight(999.0, raise_on_short=True).ok)
        check("安全余量 > 0", G.VRAM_SAFETY_MARGIN_GB > 0,
              f"{G.VRAM_SAFETY_MARGIN_GB} GB")
        check("VramReport.markdown 有表格",
              "| 项 | 值 |" in G.vram_headroom(0.0).markdown())

        print("\n[13] 文档与常量")
        check("PROTECTION_DOC 覆盖十道防线",
              all(k in G.PROTECTION_DOC for k in
                  ("底座只读", "rank", "注入面", "回放", "早停", "保险库",
                   "强度旋钮", "漂移体检")), f"{len(G.PROTECTION_DOC)} 字符")
        check("文档写了显存余量与 WDDM 静默降速",
              "显存余量" in G.PROTECTION_DOC and "WDDM" in G.PROTECTION_DOC)
        check("文档里记了实测的降速对比数据",
              "2.42 s" in G.PROTECTION_DOC and "66 s" in G.PROTECTION_DOC)
        check("文档说明了本机管不了的部分", "无法" in G.PROTECTION_DOC
              or "管不了" in G.PROTECTION_DOC)
        check("文档把读音标注放在微调之前推荐", "读音标注" in G.PROTECTION_DOC)
        check("文档里没有未转义的反斜杠路径",
              "\\S" not in G.PROTECTION_DOC and "\\p" not in G.PROTECTION_DOC)
        check("漂移阈值分 5 级", len(G.DRIFT_LEVELS) == 5)
        check("阈值单调递增",
              all(G.DRIFT_LEVELS[i][0] < G.DRIFT_LEVELS[i + 1][0]
                  for i in range(4)))
        check("三档预设齐全",
              set(G.CONFIG_PRESETS) == {"conservative", "balanced", "aggressive"})
        check("TRAINING_ROOT 不在 checkpoints 里",
              not G.TRAINING_ROOT.startswith(
                  os.path.join(_env.PROJECT_ROOT, G.MODEL_DIR_NAME)))
    finally:
        # 只读锁若没解开，Windows 上 rmtree 会失败 —— 兜底再解一次
        for root, _d, files in os.walk(tmp):
            for fn in files:
                try:
                    os.chmod(os.path.join(root, fn), 0o666)
                except Exception:
                    pass
        shutil.rmtree(tmp, ignore_errors=True)

    # =======================================================================
    print("\n[14] forward.py：masked_ce / build_length_mask / unwrap / GptForward")
    # 这一组是纯张量运算，不需要任何模型权重。
    # 放在这里的理由：上一轮就是因为交叉熵少写了一个 transpose，
    # 在真实 GPT 上算出 17.06 的假 loss（均匀基线才 9.01），
    # 差点误判成「前向接错了」而去改本来正确的前向。
    # 小张量穷举能一秒把这个轴约定钉死。
    import torch.nn.functional as F
    from webui_app.training import forward as FW

    B, V, L = 2, 5, 3
    tg = torch.tensor([[1, 2, 0], [3, 4, 1]])

    # ---- 轴约定：(B, V, L) ----
    perfect = torch.full((B, V, L), -10.0)
    for b in range(B):
        for i in range(L):
            perfect[b, tg[b, i], i] = 20.0        # 正确类拿到最大 logit
    lp = float(FW.masked_ce(perfect, tg, None))
    check("每个位置的正确类都拿到最大 logit 时 loss≈0",
          lp < 1e-6, f"{lp:.3e}")
    # 回归钉：少了 transpose 的写法必须算出错的值。
    # 两种写法的形状都能对上 cross_entropy（都是 (N, V) 配 (N,)），
    # 不报错、loss 有限、还能反向传播 —— 只有数值是错的，极难发现。
    buggy = float(F.cross_entropy(perfect.reshape(-1, V), tg.reshape(-1)))
    check("【回归】少了 transpose(1,2) 的写法会算出明显错误的 loss",
          buggy > 1.0, f"正确 {lp:.3e} vs 错误写法 {buggy:.4f}")
    wrong = torch.full((B, V, L), -10.0)
    for b in range(B):
        for i in range(L):
            wrong[b, (int(tg[b, i]) + 1) % V, i] = 20.0    # 故意抬错一个类
    check("正确类被抢走时 loss 很大（说明真的在看对应位置）",
          float(FW.masked_ce(wrong, tg, None)) > 10.0,
          f'{float(FW.masked_ce(wrong, tg, None)):.4f}')

    # ---- 与逐位置手算对账 ----
    g = torch.Generator().manual_seed(3)
    lg = torch.randn(B, V, L, generator=g)
    tg2 = torch.randint(0, V, (B, L), generator=g)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    want, n = 0.0, 0
    for b in range(B):
        for i in range(L):
            if bool(mask[b, i]):
                want += float(F.cross_entropy(lg[b, :, i], tg2[b, i]))
                n += 1
    want /= max(1, n)
    got = float(FW.masked_ce(lg, tg2, mask))
    check("masked_ce 与逐位置手算一致", abs(got - want) < 1e-5,
          f"{got:.6f} vs {want:.6f}")
    check("mask=None 时等价于全位置平均",
          abs(float(FW.masked_ce(lg, tg2, None))
              - float(F.cross_entropy(lg.transpose(1, 2).reshape(-1, V),
                                      tg2.reshape(-1)))) < 1e-6)

    poison = lg.clone()
    poison[:, :, 2] = -500.0                       # 把被 mask 掉的位置弄成天文数字
    check("被 mask 掉的位置即使 loss 爆炸也不影响结果",
          abs(float(FW.masked_ce(poison, tg2, mask)) - got) < 1e-6,
          f'{float(FW.masked_ce(poison, tg2, mask)):.6f} vs {got:.6f}')
    check("整批被 mask 掉时返回 0 而不是 NaN（NaN 进 AdamW 动量就洗不掉了）",
          float(FW.masked_ce(lg, tg2, torch.zeros_like(mask))) == 0.0)

    for bad_logits, why in ((torch.randn(B, V), "不是 3D"),
                            (torch.randn(B, V, L + 1), "长度不匹配")):
        try:
            FW.masked_ce(bad_logits, tg2, None)
            raised = ""
        except ValueError as e:
            raised = str(e).splitlines()[0][:50]
        check(f"logits {why}时报 ValueError", bool(raised), raised or "居然没报错")

    # ---- 梯度 ----
    lg3 = lg.clone().requires_grad_(True)
    FW.masked_ce(lg3, tg2, mask).backward()
    check("masked_ce 保留梯度", lg3.grad is not None)
    check("被 mask 掉的位置梯度为 0",
          lg3.grad is not None and float(lg3.grad[:, :, 2].abs().sum()) == 0.0,
          f'{float(lg3.grad[:, :, 2].abs().sum()):.3e}' if lg3.grad is not None else "")
    check("保留的位置梯度非零",
          lg3.grad is not None and float(lg3.grad[:, :, 0].abs().sum()) > 0)

    # ---- build_length_mask ----
    m = FW.build_length_mask(torch.LongTensor([1, 3, 0]), 4)
    check("build_length_mask 前 n 位为 True（n=3/total=4 时只亮 3 个）",
          m.tolist() == [[True, False, False, False],
                         [True, True, True, False],
                         [False, False, False, False]], str(m.tolist()))
    check("mask 是 bool 且形状 (B, total)",
          m.dtype == torch.bool and tuple(m.shape) == (3, 4))
    check("长度超过 total 时不越界（截断而不是报错）",
          tuple(FW.build_length_mask(torch.LongTensor([99]), 4).shape) == (1, 4)
          and bool(FW.build_length_mask(torch.LongTensor([99]), 4).all()))

    # ---- GptForward 的 loss 组合 ----
    tl_ = torch.randn(B, 7, L, generator=g)
    tt_ = torch.randint(0, 7, (B, L), generator=g)
    tmask = torch.tensor([[True, True, True], [True, True, False]])
    gf = FW.GptForward(text_logits=tl_, text_targets=tt_,
                       mel_logits=lg, mel_targets=tg2,
                       text_mask=tmask, mel_mask=mask)
    ml, tll = float(gf.mel_loss()), float(gf.text_loss())
    check("loss(text_weight=0) 就等于 mel_loss",
          abs(float(gf.loss(0.0)) - ml) < 1e-6, f"{float(gf.loss(0.0)):.6f} vs {ml:.6f}")
    check("loss(text_weight=w) = mel + w×text",
          abs(float(gf.loss(0.25)) - (ml + 0.25 * tll)) < 1e-5,
          f"{float(gf.loss(0.25)):.6f} vs {ml + 0.25 * tll:.6f}")
    v = gf.values()
    check("values() 给出可展示的 float（不拖着计算图）",
          isinstance(v["mel_loss"], float) and abs(v["mel_loss"] - ml) < 1e-6
          and v["mel_tokens"] == int(mask.sum())
          and v["text_tokens"] == int(tmask.sum()), str(v))

    # ---- unwrap ----
    raw = build_tiny_model()
    check("unwrap 对未包装的模块原样返回", FW.unwrap(raw) is raw)
    src_lm = build_tiny_model()
    lm = inject_lora(src_lm, ["attn/c_proj"])
    check("unwrap 能从单层 LoraModel 包装里取回原模块",
          FW.unwrap(lm) is src_lm and FW.unwrap(lm) is not lm,
          f"{type(FW.unwrap(lm)).__name__}")
    from peft import LoraConfig as _PC, get_peft_model as _gpm
    src = build_tiny_model()
    pm = _gpm(src, _PC(r=2, lora_alpha=4, lora_dropout=0.0,
                       target_modules=G.build_target_regex(["attn/c_proj"]),
                       bias="none"))
    check("unwrap 能穿透两层包装（PeftModel → LoraModel → 原模型）",
          FW.unwrap(pm) is src, f"{type(FW.unwrap(pm)).__name__}")
    check("unwrap 后的对象仍能直接拿到子模块",
          hasattr(FW.unwrap(pm), "mel_head") and hasattr(FW.unwrap(pm), "wte"))

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
