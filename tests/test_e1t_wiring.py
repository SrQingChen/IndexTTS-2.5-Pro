"""E1-T Step 4 · 模型接线单元测试（CPU 小模型，无 GPU/无大权重）。

核心断言（docs/dev-plan-e1t-2026-10.md §5.4 / 会话规划 WP1）：
1. **None 原路径逐位一致**：f0/technique 全 None 时，BASECFM.solve_euler 与
   改前实现（本测试内的冻结参照副本）在相同种子下输出 bit-exact；
2. **零初始化残差**：TimbreConditioner 初始 residual≡0 → 开条件的模型在
   训练起点与原模型逐位一致（比"接近"更强的护栏）；
3. **条件生效**：扰动 conditioner 权重后输出改变（条件不是死参数）；
4. **CFG null = 旁路 conditioner**（不是输入置零）；
5. **class_dropout 覆盖 style_t**：x_in[..., 80:] 整段清零连带条件切片；
6. **训练期条件丢弃**（cond_drop_prob）只旁路 {f0,tech}。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from indextts.s2mel.modules.conditioners import (  # noqa: E402
    NUM_TECHNIQUES, TimbreConditioner, normalize_f0)
from indextts.s2mel.modules.diffusion_transformer import DiT  # noqa: E402
from indextts.s2mel.modules.flow_matching import CFM  # noqa: E402


def _tiny_args(singing: bool = False) -> SimpleNamespace:
    """最小可跑的 s2mel 配置（结构对齐 checkpoints/config.yaml，尺寸缩小）。"""
    dit = dict(
        hidden_dim=64, num_heads=4, depth=2, class_dropout_prob=0.1,
        block_size=8192, in_channels=80, style_condition=True,
        final_layer_type="mlp", target="mel", content_dim=512,
        content_codebook_size=1024, content_type="discrete",
        f0_condition=False, n_f0_bins=512, content_codebooks=1,
        is_causal=False, long_skip_connection=True,
        zero_prompt_speech_token=False, time_as_token=False,
        style_as_token=False, uvit_skip_connection=True,
        add_resblock_in_transformer=False,
    )
    if singing:
        dit.update(technique_condition=True, timbre_dim=32, num_techniques=7)
    return SimpleNamespace(
        DiT=SimpleNamespace(**dit),
        style_encoder=SimpleNamespace(dim=192),
        dit_type="DiT",
        reg_loss_type="l1",
    )


def _seeded(n: int = 7):
    torch.manual_seed(n)
    return n


def _cond_inputs(B, T):
    f0 = torch.zeros(B, T)
    f0[:, ::2] = torch.linspace(180.0, 520.0, T)[::2]      # 一半浊音爬音阶
    tech = torch.zeros(B, T, dtype=torch.long)
    tech[:, T // 2:] = 2                                    # 后半 falsetto
    stats = (torch.full((B,), 7.5), torch.full((B,), 0.35))
    return f0, tech, stats


# ---------------------------------------------------------------------------
# TimbreConditioner 本体
# ---------------------------------------------------------------------------

class TestTimbreConditioner:
    def test_shapes_and_vocab(self):
        c = TimbreConditioner(style_dim=192, dim=64)
        B, T = 2, 17
        style = torch.randn(B, 192)
        f0, tech, stats = _cond_inputs(B, T)
        out = c(style, f0, tech, stats)
        assert out.shape == (B, T, 192)
        assert NUM_TECHNIQUES == 7

    def test_zero_init_residual_is_exactly_zero(self):
        """零初始化 + α=0.5：初始 residual 严格为 0 → style_t == style 广播。"""
        c = TimbreConditioner(style_dim=192, dim=64)
        B, T = 2, 13
        style = torch.randn(B, 192)
        f0, tech, stats = _cond_inputs(B, T)
        out = c(style, f0, tech, stats)
        ref = style[:, None, :].expand(B, T, 192)
        assert torch.equal(out, ref), "conditioner 初始应严格等价于原模型"

    def test_alpha_learnable_and_perturb_changes_output(self):
        c = TimbreConditioner(style_dim=192, dim=64)
        assert c.alpha.requires_grad
        B, T = 1, 11
        style = torch.randn(B, 192)
        f0, tech, stats = _cond_inputs(B, T)
        a = c(style, f0, tech, stats).clone()
        with torch.no_grad():
            for p in c.style_mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.05)
        b = c(style, f0, tech, stats)
        assert not torch.allclose(a, b), "扰动后输出必须变化（条件非死参数）"

    def test_cond_mask_bypass_equals_plain_style(self):
        c = TimbreConditioner(style_dim=192, dim=64)
        with torch.no_grad():
            for p in c.parameters():
                p.add_(torch.randn_like(p) * 0.1)   # 非零初始化
        B, T = 2, 9
        style = torch.randn(B, 192)
        f0, tech, stats = _cond_inputs(B, T)
        mask = torch.tensor([True, False])
        out = c(style, f0, tech, stats, cond_mask=mask)
        ref = style[:, None, :].expand(B, T, 192)
        assert torch.equal(out[1], ref[1]), "旁路样本必须回纯 style"
        assert not torch.allclose(out[0], ref[0]), "激活样本应带条件残差"

    def test_normalize_f0_semantics(self):
        B, T = 1, 8
        f0 = torch.tensor([[0.0, 200.0, 220.0, 0.0, 260.0, 280.0, 300.0, 320.0]])
        mu = torch.zeros(B).fill_(8.0)     # log2(256)
        sd = torch.ones(B)
        z, uv = normalize_f0(f0, mu, sd)
        assert torch.equal(uv, (f0 > 0).to(f0.dtype))
        v = f0 > 0
        # 清音位 z=0；浊音位 = tanh 软截断后的 (log2 f0 − 8)
        assert torch.all(z[~v] == 0)
        raw = (torch.log2(f0[v]) - 8.0)
        expect = 4.0 * torch.tanh(raw / 4.0)
        assert torch.allclose(z[v], expect, atol=1e-6)
        # 软截断：极端高音不发散、保序
        z2, _ = normalize_f0(torch.tensor([[4000.0]]), mu, sd)
        assert torch.isfinite(z2).all() and z2.abs() < 4.0 + 1e-6


# ---------------------------------------------------------------------------
# DiT 接线
# ---------------------------------------------------------------------------

class TestDiTWiring:
    def _inputs(self, B=2, T=23):
        x = torch.randn(B, 80, T)
        px = torch.randn(B, 80, T)
        lens = torch.full((B,), T, dtype=torch.long)
        t = torch.rand(B)
        style = torch.randn(B, 192)
        cond = torch.randn(B, T, 512)
        return x, px, lens, t, style, cond

    @staticmethod
    def _caches(dit, B, T):
        # gpt_fast Transformer 要求先 setup_caches（forward.py 文档的官方陷阱）；
        # use_kv_cache 被硬编码 False，只建 RoPE/causal mask，CPU 上很便宜
        dit.setup_caches(max_batch_size=max(1, B), max_seq_length=max(8, T))

    def test_default_no_conditioner_keys(self):
        dit = DiT(_tiny_args(False))
        bad = [k for k in dit.state_dict() if "timbre" in k]
        assert not bad

    def test_singing_constructs_conditioner(self):
        dit = DiT(_tiny_args(True))
        assert dit.technique_condition
        keys = [k for k in dit.state_dict() if k.startswith("timbre_conditioner")]
        assert keys, "singing config 必须构造 conditioner"

    def test_none_path_identical_statements(self):
        """f0/technique=None：kwargs 显式传 None 与不传，输出逐位一致。"""
        _seeded(3)
        dit = DiT(_tiny_args(True)).eval()
        x, px, lens, t, style, cond = self._inputs()
        self._caches(dit, x.size(0), x.size(-1))
        with torch.no_grad():
            a = dit(x, px, lens, t, style, cond)
            b = dit(x, px, lens, t, style, cond, f0=None, technique=None,
                    f0_stats=None, cond_mask=None)
        assert torch.equal(a, b)

    def test_fresh_conditioner_training_start_equals_original(self):
        """开条件（零初始化）与不开条件：前向逐位一致。"""
        _seeded(11)
        args = _tiny_args(True)
        dit = DiT(args).eval()
        x, px, lens, t, style, cond = self._inputs()
        self._caches(dit, x.size(0), x.size(-1))
        f0, tech, stats = _cond_inputs(x.size(0), x.size(-1))
        with torch.no_grad():
            a = dit(x, px, lens, t, style, cond)
            b = dit(x, px, lens, t, style, cond, f0=f0, technique=tech,
                    f0_stats=stats)
        assert torch.equal(a, b), "零初始化残差下，条件路径必须逐位等价原路径"

    def test_conditioned_output_changes_after_perturb(self):
        _seeded(5)
        dit = DiT(_tiny_args(True)).eval()
        with torch.no_grad():
            for p in dit.timbre_conditioner.style_mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.08)
        x, px, lens, t, style, cond = self._inputs()
        self._caches(dit, x.size(0), x.size(-1))
        f0, tech, stats = _cond_inputs(x.size(0), x.size(-1))
        with torch.no_grad():
            a = dit(x, px, lens, t, style, cond)
            b = dit(x, px, lens, t, style, cond, f0=f0, technique=tech,
                    f0_stats=stats)
        assert not torch.allclose(a, b, atol=1e-6)

    def test_condition_mismatch_raises(self):
        """默认模型传条件必须报错（防静默丢条件）。"""
        dit = DiT(_tiny_args(False)).eval()
        x, px, lens, t, style, cond = self._inputs()
        self._caches(dit, x.size(0), x.size(-1))
        f0, tech, stats = _cond_inputs(x.size(0), x.size(-1))
        with pytest.raises(RuntimeError):
            dit(x, px, lens, t, style, cond, f0=f0, technique=tech,
                f0_stats=stats)

    def test_class_dropout_zeroes_style_slice(self):
        """class_dropout 触发时 x_in[...,80:] 清零 → 条件残差也被覆盖。"""
        args = _tiny_args(True)
        args.DiT.class_dropout_prob = 1.0
        dit = DiT(args).train()
        with torch.no_grad():
            for p in dit.timbre_conditioner.style_mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.2)
        x, px, lens, t, style, cond = self._inputs()
        self._caches(dit, x.size(0), x.size(-1))
        f0, tech, stats = _cond_inputs(x.size(0), x.size(-1))
        a = dit(x, px, lens, t, style, cond)
        b = dit(x, px, lens, t, style, cond, f0=f0, technique=tech,
                f0_stats=stats)
        assert torch.allclose(a, b, atol=1e-6), "class_dropout 应连带清零条件"

    def test_train_cond_drop_bypasses_conditioner(self):
        """cond_drop_prob=1：训练期条件丢弃 → 输出等价无条件。"""
        args = _tiny_args(True)
        args.DiT.class_dropout_prob = 0.0
        dit = DiT(args).train()
        dit.timbre_conditioner.cond_drop_prob = 1.0
        with torch.no_grad():
            for p in dit.timbre_conditioner.style_mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.2)
        x, px, lens, t, style, cond = self._inputs()
        self._caches(dit, x.size(0), x.size(-1))
        f0, tech, stats = _cond_inputs(x.size(0), x.size(-1))
        a = dit(x, px, lens, t, style, cond)
        b = dit(x, px, lens, t, style, cond, f0=f0, technique=tech,
                f0_stats=stats)
        assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# BASECFM / solve_euler：None 原路径 bit-exact + CFG null 旁路
# ---------------------------------------------------------------------------

def _reference_solve_euler(cfm, x, x_lens, prompt, mu, style, t_span,
                           inference_cfg_rate):
    """改前 solve_euler 的冻结副本（flow_matching.py 上游版本，逐行一致）。

    用于证明：f0=None 时新实现与上游实现 bit-exact。estimator 共用同一实例，
    差异只能来自 solve_euler 自身的改动。"""
    t = t_span[0]
    sol = []
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0
    if cfm.zero_prompt_speech_token:
        mu[..., :prompt_len] = 0
    for step in range(1, len(t_span)):
        dt = t_span[step] - t_span[step - 1]
        if inference_cfg_rate > 0:
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            stacked_x = torch.cat([x, x], dim=0)
            stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
            stacked_dphi_dt = cfm.estimator(
                stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu)
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
        else:
            dphi_dt = cfm.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
        x[:, :, :prompt_len] = 0
    return sol[-1]


class TestBaseCfm:
    def _cfm(self, singing=False, T=40):
        _seeded(13)
        cfm = CFM(_tiny_args(singing)).eval()
        from webui_app.training import forward as FW
        FW.setup_cfm_caches(cfm, batch=2, seq_len=T)   # RoPE/causal mask 预建
        return cfm

    def _inputs(self, B=1, Tp=9, Tt=27):
        prompt = torch.randn(B, 80, Tp)
        mu = torch.randn(B, Tp + Tt, 512)
        style = torch.randn(B, 192)
        lens = torch.full((B,), Tp + Tt, dtype=torch.long)
        return prompt, mu, style, lens

    @torch.no_grad()
    def test_none_path_bit_exact_vs_upstream(self):
        for cfg_rate in (0.0, 0.7):
            cfm = self._cfm(False)
            prompt, mu, style, lens = self._inputs()
            _seeded(101)
            z1 = torch.randn(1, 80, mu.size(1))
            _seeded(101)
            z2 = torch.randn(1, 80, mu.size(1))
            t_span = torch.linspace(0, 1, 3)
            a = cfm.solve_euler(z1.clone(), lens, prompt, mu.clone(), style,
                                None, t_span, cfg_rate)
            b = _reference_solve_euler(cfm, z2.clone(), lens, prompt, mu.clone(),
                                       style, t_span, cfg_rate)
            assert torch.equal(a, b), f"cfg={cfg_rate}: None 路径必须与上游逐位一致"

    @torch.no_grad()
    def test_inference_positional_call_compat(self):
        """上游/svc_core 按位置传参（第 6 位 n_timesteps）不得被新参数破坏。"""
        cfm = self._cfm(True)
        prompt, mu, style, lens = self._inputs()
        _seeded(23)
        mel = cfm.inference(mu, lens, prompt, style, None, 3, 0.7)
        assert mel.shape == (1, 80, mu.size(1))

    @torch.no_grad()
    def test_cfg_null_branch_bypasses_conditioner(self):
        """CFG 堆叠调用：estimator 收到 2B 条件与 [1..1,0..0] 的 cond_mask。"""
        cfm = self._cfm(True)
        with torch.no_grad():
            for p in cfm.estimator.timbre_conditioner.style_mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.15)
        prompt, mu, style, lens = self._inputs()
        T = mu.size(1)
        f0, tech, stats = _cond_inputs(1, T)
        captured = {}
        orig = cfm.estimator.forward

        def spy(x, px, xl, t, st, cond, mask_content=False, **kw):
            captured.update(kw)
            return orig(x, px, xl, t, st, cond, mask_content, **kw)

        cfm.estimator.forward = spy
        try:
            _seeded(31)
            z = torch.randn(1, 80, T)
            t_span = torch.linspace(0, 1, 3)
            cfm.solve_euler(z, lens, prompt, mu, style, f0, t_span, 0.7,
                            technique=tech, f0_stats=stats)
        finally:
            cfm.estimator.forward = orig
        assert captured.get("f0") is not None and captured["f0"].size(0) == 2
        assert captured.get("technique").size(0) == 2
        mask = captured.get("cond_mask")
        assert mask is not None and mask.tolist() == [1.0, 0.0] or \
            (mask is not None and mask.tolist() == [1, 0]), \
            f"null 半批必须旁路 conditioner，got {mask}"

    @torch.no_grad()
    def test_forward_none_equals_reference(self):
        """BASECFM.forward（训练 loss 路径）None 条件与上游一致：estimator
        收到的 kwargs 全 None。"""
        cfm = self._cfm(False).train()
        prompt, mu, style, lens = self._inputs()
        x1 = torch.randn(1, 80, lens[0])
        captured = {}
        orig = cfm.estimator.forward

        def spy(x, px, xl, t, st, cond, mask_content=False, **kw):
            captured.update(kw)
            return orig(x, px, xl, t, st, cond, mask_content, **kw)

        cfm.estimator.forward = spy
        try:
            _seeded(41)
            loss, _y = cfm.forward(x1, lens, torch.LongTensor([prompt.size(-1)]),
                                   mu, style)
            assert torch.isfinite(loss)
        finally:
            cfm.estimator.forward = orig
        assert captured.get("f0") is None and captured.get("technique") is None

    @torch.no_grad()
    def test_forward_with_conditions_runs(self):
        cfm = self._cfm(True).train()
        cfm.estimator.class_dropout_prob = 0.0
        cfm.estimator.timbre_conditioner.cond_drop_prob = 0.0
        prompt, mu, style, lens = self._inputs()
        x1 = torch.randn(1, 80, lens[0])
        f0, tech, stats = _cond_inputs(1, lens[0])
        _seeded(43)
        loss, _y = cfm.forward(x1, lens, torch.LongTensor([prompt.size(-1)]),
                               mu, style, f0=f0, technique=tech,
                               f0_stats=stats)
        assert torch.isfinite(loss)
