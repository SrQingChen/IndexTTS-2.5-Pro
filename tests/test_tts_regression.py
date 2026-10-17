"""E1-T Step 0 · TTS 回归护栏测试（无 GPU、不加载大权重）。

守住三条红线（docs/dev-plan-e1t-2026-10.md §2）：
1. 默认 TTS 三件套（s2mel.pth / config.yaml / BigVGAN）字节级不变；
2. 默认 config 永远不开 f0/technique 条件（E1-T 走 config_singing.yaml）；
3. singing config 是独立文件、指向独立权重名——两层 config 物理隔离。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

HASHES_PATH = os.path.join(os.path.dirname(__file__), "baseline_tts_hashes.json")


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestTtsFileImmutability:
    """三件套 hash 守卫。基线由 tools/e1t_hashes.py 生成（提交入库），
    文件缺失（如 CI 无权重）时 skip，不误报。"""

    @pytest.fixture(scope="class")
    def baseline(self):
        if not os.path.isfile(HASHES_PATH):
            pytest.skip("tests/baseline_tts_hashes.json 不存在")
        return json.load(open(HASHES_PATH, encoding="utf-8"))

    def test_weights_unchanged(self, baseline):
        missing, changed = [], []
        for rel, want in sorted(baseline.items()):
            p = os.path.join(ROOT, rel)
            if not os.path.isfile(p):
                missing.append(rel)
                continue
            if _md5(p) != want:
                changed.append(rel)
        assert not missing, f"基线文件缺失（可能被移动/删除）: {missing}"
        assert not changed, (
            f"默认 TTS 权重/配置被改动: {changed} —— E1-T 红线："
            "s2mel.pth / config.yaml / BigVGAN 绝不允许覆盖，歌声产物只能用"
            "s2mel_singing.pth / config_singing.yaml 等新名字")

    def test_singing_weights_are_new_names(self):
        """歌声曲线的产物名必须与默认 TTS 文件物理不同名。"""
        for name in ("s2mel.pth", "config.yaml"):
            p_sing = os.path.join(ROOT, "checkpoints",
                                  name.replace(".", "_singing."))
            if os.path.isfile(p_sing):
                assert os.path.basename(p_sing) != name


class TestDefaultConfigGating:
    """默认 config 的条件开关守卫。"""

    def _load(self, name):
        p = os.path.join(ROOT, "checkpoints", name)
        if not os.path.isfile(p):
            pytest.skip(f"checkpoints/{name} 不存在（未部署权重）")
        return open(p, encoding="utf-8").read()

    def test_default_config_f0_condition_off(self):
        txt = self._load("config.yaml")
        # 两处 f0_condition（length_regulator + DiT）都必须保持 false
        assert txt.count("f0_condition: false") == 2, \
            "默认 config.yaml 的 f0_condition 必须保持 false（上游行为）"
        assert "technique_condition" not in txt, \
            "默认 config.yaml 不得出现 technique_condition —— E1-T 只属于 config_singing.yaml"

    def test_singing_config_isolated(self):
        txt = self._load("config_singing.yaml")
        assert "technique_condition: true" in txt
        # singing 权重名必须是新名字，绝不指向 s2mel.pth
        assert "s2mel_checkpoint: s2mel_singing.pth" in txt
        assert "s2mel_checkpoint: s2mel.pth" not in txt


class TestDefaultModelNoNewKeys:
    """默认（无 technique 键）构造的 CFM：state_dict 不得出现 conditioner 键。

    这是「新模块不能无条件构造」红线的直接断言——strict=False 的加载链
    （commons.load_checkpoint2）对新键只打 Warning，靠人眼守不住。"""

    def test_default_cfm_has_no_conditioner_keys(self):
        from types import SimpleNamespace

        import yaml

        cfg_path = os.path.join(ROOT, "checkpoints", "config.yaml")
        if not os.path.isfile(cfg_path):
            pytest.skip("checkpoints/config.yaml 不存在")
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        s2 = SimpleNamespace(
            **{k: SimpleNamespace(**v) if isinstance(v, dict) else v
               for k, v in cfg["s2mel"].items()})
        # style_encoder 是 dict → 包一层
        from indextts.s2mel.modules.flow_matching import CFM
        cfm = CFM(s2)
        bad = [k for k in cfm.state_dict() if "timbre" in k or "conditioner" in k]
        assert not bad, f"默认 config 构造出了歌声条件模块: {bad}"
