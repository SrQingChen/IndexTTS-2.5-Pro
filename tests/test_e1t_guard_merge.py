"""E1-T 合并白名单与分块栅格测试（验收 P1-3 / P2-4 固化）。

白名单六断言（此前只做了会话内一次性验证，未固化——红线行为回归无护栏）：
放行两个歌声新名字（顶层）、拒绝子目录/旧名字/非白名单名、未开白名单时
全拒、训练产物目录不受影响。分块：硬切点吸附 mel 栅格。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from webui_app.services import svc_core  # noqa: E402
from webui_app.training import guard as GD  # noqa: E402

SR = 22050


class TestGuardWhitelist:
    def test_top_level_singing_names_allowed(self, tmp_path):
        g = GD.BaseGuard(model_dir=str(tmp_path), train_root=str(tmp_path / "t"),
                         allow_singing_names=True)
        for name in ("s2mel_singing.pth", "config_singing.yaml"):
            out = g.assert_safe_output(os.path.join(str(tmp_path), name))
            assert os.path.basename(out) == name

    def test_subdirectory_rejected(self, tmp_path):
        g = GD.BaseGuard(model_dir=str(tmp_path), train_root=str(tmp_path / "t"),
                         allow_singing_names=True)
        with pytest.raises(PermissionError):
            g.assert_safe_output(os.path.join(str(tmp_path), "sub",
                                              "s2mel_singing.pth"))

    def test_original_names_always_rejected(self, tmp_path):
        g = GD.BaseGuard(model_dir=str(tmp_path), train_root=str(tmp_path / "t"),
                         allow_singing_names=True)
        for bad in ("s2mel.pth", "config.yaml", "gpt.pth"):
            with pytest.raises(PermissionError):
                g.assert_safe_output(os.path.join(str(tmp_path), bad))

    def test_non_whitelist_name_rejected(self, tmp_path):
        g = GD.BaseGuard(model_dir=str(tmp_path), train_root=str(tmp_path / "t"),
                         allow_singing_names=True)
        with pytest.raises(PermissionError):
            g.assert_safe_output(os.path.join(str(tmp_path), "evil.pth"))

    def test_disabled_by_default(self, tmp_path):
        g = GD.BaseGuard(model_dir=str(tmp_path), train_root=str(tmp_path / "t"))
        with pytest.raises(PermissionError):
            g.assert_safe_output(os.path.join(str(tmp_path),
                                              "s2mel_singing.pth"))

    def test_training_root_unaffected(self, tmp_path):
        # train_root 与 model_dir 同级（嵌套会被「不许落在底座目录」正确拒绝）
        train_root = str(tmp_path.parent / (tmp_path.name + "_train"))
        g = GD.BaseGuard(model_dir=str(tmp_path), train_root=train_root,
                         allow_singing_names=True)
        out = g.assert_safe_output(os.path.join(train_root, "run1", "x.pth"))
        assert out.endswith("x.pth")


class TestMergeSingingArchGate:
    def test_arch_name_gate(self):
        """cfm_singing 产物名白名单校验（merge_lora_to_checkpoint 的前置
        分支）：直接构造 opts 验证拒绝路径返回 note 而非抛异常（UI 语义）。"""
        from webui_app.training import merge as MG
        rep = MG.merge_lora_to_checkpoint(MG.MergeOptions(
            adapter_dir=".", arch="cfm_singing",
            out_path=os.path.join("checkpoints", "wrong_name.pth")))
        assert not rep.ok
        assert any("s2mel_singing.pth" in n for n in rep.notes)


class TestHardCutGridSnap:
    def test_hard_cuts_snapped_to_mel_grid(self):
        """硬切点（找不到能量谷）必须吸附 256 样本栅格（P2-4）。"""
        rng = np.random.default_rng(3)
        y = rng.normal(0, 0.25, int(200 * SR)).astype(np.float32)
        chunks = svc_core.plan_chunks(y, SR, 60.0, overlap_ms=40.0)
        assert len(chunks) >= 2
        # 白噪声无谷 → 每个内部切点（下一块起点+重叠）应在栅格上
        for (a, b, ov) in chunks[1:]:
            cut = a + ov
            assert cut % svc_core.HOP == 0, (
                f"硬切点 {cut} 未吸附 {svc_core.HOP} 栅格")
