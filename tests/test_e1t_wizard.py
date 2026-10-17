"""E1-T 向导服务层单元测试（无 GPU；状态检测 / run 分类 / 引导复制防护）。"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from webui_app.services import e1t_wizard as WZ  # noqa: E402


class TestRunClassification:
    """run.json 的 singing 字段 → capability / 身份 run 的分类（向导下拉的
    数据源；分错会把 capability 当身份挂到翻唱页——必须钉住）。"""

    def _fake_rn(self, runs):
        fake = SimpleNamespace(
            list_runs=lambda arch=None: [
                SimpleNamespace(name=n, has_adapter=True, arch="cfm")
                for n in runs],
            read_run=lambda n: runs[n],
        )
        return fake

    def test_capability_vs_identity_vs_plain(self, monkeypatch):
        runs = {
            "cap_v1": {"singing": {"enabled": True,
                                   "cfg_path": "checkpoints/config_singing.yaml"}},
            "cap_v2": {"singing": {"enabled": True, "cfg_path": ""}},
            "dania_id": {"singing": {"enabled": False,
                                     "cfg_path": "checkpoints/config_singing.yaml"}},
            "plain_tts": {"singing": {"enabled": False, "cfg_path": ""}},
            "no_field": {},
        }
        monkeypatch.setattr(WZ, "RN", self._fake_rn(runs))
        assert WZ.capability_runs() == ["cap_v1", "cap_v2"]
        assert WZ.singing_identity_runs() == ["dania_id"]

    def test_missing_run_json_ignored(self, monkeypatch):
        runs = {"broken": None}
        monkeypatch.setattr(WZ, "RN", self._fake_rn(runs))
        assert WZ.capability_runs() == []
        assert WZ.singing_identity_runs() == []


class TestBootstrap:
    def _paths(self, tmp_path, with_target=False):
        base = tmp_path / "s2mel.pth"
        base.write_bytes(b"x" * 1024)
        tgt = tmp_path / "s2mel_singing.pth"
        if with_target:
            tgt.write_bytes(b"y" * 2048)
        return str(base), str(tgt)

    def test_copies_when_missing(self, tmp_path, monkeypatch):
        base, tgt = self._paths(tmp_path)
        monkeypatch.setattr(WZ, "BASE_PTH", base)
        monkeypatch.setattr(WZ, "SINGING_PTH", tgt)
        out = {}
        WZ.bootstrap_fn()(lambda f, m: out.update(msg=m), lambda: False)
        assert os.path.isfile(tgt)
        assert open(tgt, "rb").read() == b"x" * 1024

    def test_refuses_to_overwrite(self, tmp_path, monkeypatch):
        base, tgt = self._paths(tmp_path, with_target=True)
        monkeypatch.setattr(WZ, "BASE_PTH", base)
        monkeypatch.setattr(WZ, "SINGING_PTH", tgt)
        r = WZ.bootstrap_fn()(lambda f, m: None, lambda: False)
        assert not r["ok"] and "不覆盖" in r["error"]
        assert open(tgt, "rb").read() == b"y" * 2048   # 原内容未动

    def test_missing_base_reports(self, tmp_path, monkeypatch):
        monkeypatch.setattr(WZ, "BASE_PTH", str(tmp_path / "nope.pth"))
        monkeypatch.setattr(WZ, "SINGING_PTH", str(tmp_path / "t.pth"))
        r = WZ.bootstrap_fn()(lambda f, m: None, lambda: False)
        assert not r["ok"]


class TestStatusMarkdown:
    def test_contains_seven_steps_and_reuse_hint(self):
        md = WZ.status_markdown()
        for kw in ("① 歌声底座", "② 公开歌声数据集", "③ 歌声特征",
                   "④ capability 训练", "⑤ 合并到歌声底座",
                   "⑥ 角色歌声身份", "可复用性"):
            assert kw in md, f"状态面板缺 {kw!r}"


class TestImportParsing:
    def test_pitch_aug_parsing(self):
        from webui_app.services import gts_import as GI
        assert GI.parse_pitch_aug("-12,+12") == [-12, 12]
        assert GI.parse_pitch_aug("−3，+5") == [-3, 5]
        assert GI.parse_pitch_aug("") == []
        assert GI.parse_pitch_aug("0, 99, abc") == []
