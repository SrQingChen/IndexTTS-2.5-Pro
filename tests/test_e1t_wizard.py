"""E1-T 向导服务层单元测试（无 GPU；状态检测 / run 分类 / 引导复制防护 /
镜像拉取过滤 / 一键全链编排 / 翻唱页接线）。"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from webui_app.services import e1t_wizard as WZ  # noqa: E402
from webui_app.services import gts_fetch as GF  # noqa: E402


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
        WZ.bootstrap_fn()(lambda f, m: None, lambda: False)
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
    def test_contains_steps_and_reuse_hint(self):
        md = WZ.status_markdown()
        for kw in ("⓪ 歌声数据源", "① 歌声底座", "② 公开歌声数据集",
                   "③ 歌声特征", "④ capability 训练", "⑤ 合并到歌声底座",
                   "⑥ 角色歌声身份", "可复用性"):
            assert kw in md, f"状态面板缺 {kw!r}"


class TestImportParsing:
    def test_pitch_aug_parsing(self):
        from webui_app.services import gts_import as GI
        assert GI.parse_pitch_aug("-12,+12") == [-12, 12]
        assert GI.parse_pitch_aug("−3，+5") == [-3, 5]
        assert GI.parse_pitch_aug("") == []
        assert GI.parse_pitch_aug("0, 99, abc") == []


# ---------------------------------------------------------------------------
# 47c：镜像拉取 / 一键全链 / 接线
# ---------------------------------------------------------------------------

class TestPlanFiles:
    """语言子集过滤（空集防御：宁全量不给假完成）。"""

    def test_lang_subset(self):
        files = ["structure.json", "README.md",
                 "zh/s1/a.wav", "zh/s1/a.json", "en/s2/b.wav",
                 "ja/s3/c.wav", "binary.bin"]
        r = GF.plan_files(files, ["zh", "en"])
        assert r["mode"] == "lang-subset"
        assert "zh/s1/a.wav" in r["files"] and "en/s2/b.wav" in r["files"]
        assert "ja/s3/c.wav" not in r["files"]
        assert "structure.json" in r["files"] and "README.md" in r["files"]
        assert "binary.bin" not in r["files"]

    def test_unknown_lang_falls_back_to_full(self):
        r = GF.plan_files(["zh/a.wav"], ["xx"])
        assert r["mode"] == "full" and len(r["files"]) == 1

    def test_no_langs_means_full(self):
        r = GF.plan_files(["a.wav", "b.json"], [])
        assert r["mode"] == "full" and len(r["files"]) == 2


class TestEndpoint:
    def test_ensure_endpoint_default(self, monkeypatch):
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        assert GF.ensure_endpoint() == "https://hf-mirror.com"

    def test_ensure_endpoint_respects_user(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
        assert GF.ensure_endpoint() == "https://huggingface.co"


class TestPipeline:
    """一键全链编排（各阶段 monkeypatch 成桩；顺序/失败即停/接线）。"""

    def _stubs(self, monkeypatch, fail_stage=None):
        calls = []

        class _FakeGF:
            DEFAULT_DEST = "fake_src"

            @staticmethod
            def source_state(dest=None):
                return {"exists": True, "wavs": 10, "langs": ["zh"],
                        "path": "x"}

            @staticmethod
            def fetch(dest, langs, progress=None, should_stop=None, workers=6):
                calls.append("fetch")
                return {"ok": fail_stage != "fetch",
                        "error": "boom" if fail_stage == "fetch" else ""}

        class _FakeGI:
            @staticmethod
            def run_import(src, dataset, langs=None, pitch_aug="", **kw):
                calls.append("import")
                if fail_stage == "import":
                    return {"ok": False, "error": "boom"}
                return {"ok": True, "imported": 5, "augmented": 2}

        class _FakeFT:
            @staticmethod
            def extract_dataset(dataset, singing=False, progress=None,
                                should_stop=None):
                calls.append("extract")
                if fail_stage == "extract":
                    return {"ok": False, "stopped": False, "failed": 3,
                            "errors": [["u", "e"]]}
                return {"ok": True, "extracted": 7, "stopped": False}

        class _FakeMG:
            @staticmethod
            def merge_lora_to_checkpoint(opts):
                calls.append("merge")
                return SimpleNamespace(ok=fail_stage != "merge",
                                       notes=["nb"] if fail_stage == "merge"
                                       else [])

        wires = []

        class _FakeState:
            @staticmethod
            def save_cover(vals):
                wires.append(dict(vals))

        def _fake_train(dataset, run_name, preset, frames, epochs, singing,
                        cfg_path, progress, should_stop):
            calls.append("cap" if singing else "identity")
            if singing and fail_stage == "capability":
                return {"ok": False, "error": "pf fail"}
            if not singing and fail_stage == "identity":
                return {"ok": False, "error": "id fail"}
            return {"ok": True, "run": run_name}

        class _FakeRN:
            @staticmethod
            def adapter_dir(name):
                return f"runs/{name}"

        class _Opts:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        _FakeMG.MergeOptions = _Opts

        monkeypatch.setattr(WZ, "GF", _FakeGF)
        monkeypatch.setattr(WZ, "GI", _FakeGI)
        monkeypatch.setattr(WZ, "FT", _FakeFT)
        monkeypatch.setattr(WZ, "MG", _FakeMG)
        monkeypatch.setattr(WZ, "svc_state", _FakeState)
        monkeypatch.setattr(WZ, "_train", _fake_train)
        monkeypatch.setattr(WZ, "RN", _FakeRN)
        return calls, wires

    def _paths(self, tmp_path):
        base = tmp_path / "fake_base.pth"
        base.write_bytes(b"x" * 32)
        monkey_targets = dict(
            BASE_PTH=str(base),
            SINGING_PTH=str(tmp_path / "fake_singing.pth"))
        return monkey_targets

    def _run(self, identity=True):
        prog = []
        fn = WZ.pipeline_fn(
            "fake_src", "ds", ["zh"], "-12,+12", 100,
            "cap_run", "conservative", 1800, 0,
            identity, "id_ds", "id_run", "conservative", 0, True)
        r = fn(lambda f, m: prog.append((round(f, 2), m)), lambda: False)
        return r, prog

    def test_full_chain_with_identity_and_wire(self, monkeypatch, tmp_path):
        calls, wires = self._stubs(monkeypatch)
        for k, v in self._paths(tmp_path).items():
            monkeypatch.setattr(WZ, k, v)
        r, prog = self._run(identity=True)
        assert r["ok"], r
        # 顺序：导入→特征→capability→合并→身份（合并在前：身份要在
        # 「已含 capability 的底座」上练）
        assert calls == ["import", "extract", "cap", "merge", "identity"]
        assert r["wire"]["lora_run"] == "id_run"
        assert wires and wires[0]["singing_enable"] is True
        assert wires[0]["lora_run"] == "id_run"
        assert prog[-1][0] == 1.0

    def test_skip_identity_still_wires_singing(self, monkeypatch, tmp_path):
        calls, wires = self._stubs(monkeypatch)
        for k, v in self._paths(tmp_path).items():
            monkeypatch.setattr(WZ, k, v)
        r, _ = self._run(identity=False)
        assert r["ok"] and "identity" not in calls
        assert wires[0]["singing_enable"] is True
        assert wires[0]["lora_run"] == ""

    def test_failure_stops_chain_at_stage(self, monkeypatch, tmp_path):
        for stage in ("import", "extract", "capability", "merge", "identity"):
            calls, _ = self._stubs(monkeypatch, fail_stage=stage)
            for k, v in self._paths(tmp_path).items():
                monkeypatch.setattr(WZ, k, v)
            r, _ = self._run(identity=True)
            assert not r["ok"] and r.get("stage") == stage, (stage, r)

    def test_wire_only_writes_e1t_keys(self, monkeypatch):
        cap = []
        monkeypatch.setattr(WZ.svc_state, "save_cover",
                            lambda v: cap.append(v))
        WZ.wire_cover_state("my_run")
        vals = cap[0]
        assert set(vals) <= {"singing_enable", "technique_mode",
                             "technique_manual", "bon_high", "hf_transplant",
                             "register_mode", "octave_fix", "lora_run",
                             "lora_ckpt"}
        assert vals["lora_run"] == "my_run" and vals["singing_enable"] is True


class TestCoverStateWhitelist:
    def test_e1t_keys_in_whitelist(self):
        # 阶段37同款「白名单静默丢键」回归钉：cover_tab 保存的键必须在表里
        from webui_app.services import svc_state
        for k in ("singing_enable", "technique_mode", "technique_manual",
                  "bon_high", "hf_transplant", "register_mode", "octave_fix"):
            assert k in svc_state._COVER_KEYS, k
