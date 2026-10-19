"""E1-T 向导服务层单元测试（无 GPU；状态检测 / run 分类 / 引导复制防护 /
镜像拉取过滤 / 一键全链编排 / 翻唱页接线）。"""

from __future__ import annotations

import json
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

    def test_lang_subset_with_real_dir_names(self):
        # 实测：仓库顶层是全名（Chinese/…），过滤用语言码也要能命中
        files = ["README.md", "dataset_license.md",
                 "Chinese/ZH-Alto-1/Breathy/s/Breathy_Group/0000.wav",
                 "Chinese/ZH-Alto-1/Breathy/s/0000.json",
                 "English/EN-Tenor-1/s/0000.wav",
                 "Japanese/s/0000.wav", "binary.bin"]
        r = GF.plan_files(files, ["zh", "en"])
        assert r["mode"] == "lang-subset"
        assert any(f.startswith("Chinese/") for f in r["files"])
        assert any(f.startswith("English/") for f in r["files"])
        assert not any(f.startswith("Japanese/") for f in r["files"])
        assert "README.md" in r["files"] and "dataset_license.md" in r["files"]
        assert "binary.bin" not in r["files"]

    def test_dir_fullname_also_accepted(self):
        r = GF.plan_files(["Chinese/a.wav"], ["Chinese"])
        assert r["mode"] == "lang-subset" and len(r["files"]) == 1

    def test_unknown_lang_falls_back_to_full(self):
        r = GF.plan_files(["Chinese/a.wav"], ["xx"])
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


# ---------------------------------------------------------------------------
# 47d：⑥ 角色数据集友好显示 + 默认 run 名
# ---------------------------------------------------------------------------

class TestCharNameOfRun:
    def test_strips_arch_suffixes(self):
        assert WZ._char_name_of_run("达妮娅_cfm") == "达妮娅"
        assert WZ._char_name_of_run("达妮娅_gpt") == "达妮娅"
        assert WZ._char_name_of_run("达妮娅_dpo") == "达妮娅"
        assert WZ._char_name_of_run("单目标名") == "单目标名"
        assert WZ._char_name_of_run("x_cfm_v2") == "x_cfm_v2"  # 只剥已知后缀


class TestDatasetLabels:
    """oneclick_<时间戳> → 角色名标签（⑥ 下拉的可用性核心）。"""

    def _setup(self, monkeypatch, tmp_path, datasets, runs, singing=()):
        for name, note in datasets:
            d = tmp_path / name
            d.mkdir()
            (d / "dataset.json").write_text(
                json.dumps({"name": name, "note": note}), encoding="utf-8")
            if name in singing:
                (d / "technique.json").write_text("{}", encoding="utf-8")

        class _FakeDS:
            @staticmethod
            def list_datasets():
                return [n for n, _ in datasets]

            @staticmethod
            def dir_of(name):
                return str(tmp_path / name)

        class _FakeRN:
            @staticmethod
            def list_runs(arch=None):
                return [SimpleNamespace(name=n, has_adapter=True)
                        for n in runs]

            @staticmethod
            def read_run(n):
                return runs[n]

            @staticmethod
            def safe_run_name(x):
                return x.replace(" ", "_")

        class _FakeFT:
            @staticmethod
            def stats(name):
                return {"ready_with_features": 12, "audio_seconds": 300.0}

        monkeypatch.setattr(WZ, "DS", _FakeDS)
        monkeypatch.setattr(WZ, "RN", _FakeRN)
        monkeypatch.setattr(WZ, "FT", _FakeFT)
        monkeypatch.setattr(WZ, "singing_datasets", lambda: list(singing))

    def test_oneclick_with_char_name(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path,
                    datasets=[("oneclick_20261011-1556", "一键三连")],
                    runs={"新锁暝_cfm": {"dataset": "oneclick_20261011-1556"},
                          "新锁暝_gpt": {"dataset": "oneclick_20261011-1556"}})
        labels = WZ.dataset_display_labels()
        assert labels["oneclick_20261011-1556"] == \
            "新锁暝 · 一键三连（oneclick_20261011-1556）"

    def test_oneclick_without_runs_falls_back(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path,
                    datasets=[("oneclick_x", "一键三连")], runs={})
        labels = WZ.dataset_display_labels()
        assert "一键三连" in labels["oneclick_x"]

    def test_plain_dataset_gets_stats(self, monkeypatch, tmp_path):
        # 无 run 映射（角色名无从反推）→ 兜底统计摘要
        self._setup(monkeypatch, tmp_path,
                    datasets=[("手动集", "")], runs={})
        labels = WZ.dataset_display_labels()
        assert "手动集（" in labels["手动集"] and "12条" in labels["手动集"]

    def test_char_named_without_oneclick_note(self, monkeypatch, tmp_path):
        # 有 run 映射但不是一键三连产物（手建数据集 + 手动训练）→ 角色名+数据集名
        self._setup(monkeypatch, tmp_path,
                    datasets=[("my_ds", "")],
                    runs={"某角色_cfm": {"dataset": "my_ds"}})
        labels = WZ.dataset_display_labels()
        assert labels["my_ds"] == "某角色（my_ds）"

    def test_identity_choices_order_and_filter(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path,
                    datasets=[("plain_a", ""), ("oneclick_b", "一键三连"),
                              ("gts_ds", "")],
                    runs={"角色B_cfm": {"dataset": "oneclick_b"}},
                    singing=("gts_ds",))
        choices = WZ.identity_dataset_choices()
        values = [v for _l, v in choices]
        assert "gts_ds" not in values            # 歌声数据不进身份下拉
        assert values[0] == "oneclick_b"          # 角色名命名的排前面
        assert values[1] == "plain_a"

    def test_identity_run_default(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path,
                    datasets=[("oneclick_1", "一键三连")],
                    runs={"达妮娅_cfm": {"dataset": "oneclick_1"}})
        assert WZ.identity_run_default("oneclick_1") == "达妮娅_singing_identity"
        # 无角色名 → 用数据集名兜底
        assert WZ.identity_run_default("随便集") == "随便集_singing_identity"
        assert WZ.identity_run_default("") .endswith("_singing_identity")

# ---------------------------------------------------------------------------
# 47e：真实仓库布局（实测）遍历与逐词 json 解析
# ---------------------------------------------------------------------------

class TestRealLayoutImport:
    """Chinese/<Singer>/<Tech>/<Song>/<Group>/NNNN.wav 布局的遍历规则。"""

    def _make_tree(self, tmp_path):
        base = tmp_path / "GTS"
        for tech, group, stem in [
                ("Breathy", "Breathy_Group", "0000"),
                ("Breathy", "Control_Group", "0000"),
                ("Vibrato", "Control_Group", "0000"),      # 跨技巧重复的 Control
                ("Breathy", "Paired_Speech_Group", "0000")  # 配对语音
                ]:
            d = base / "Chinese" / "ZH-Alto-1" / tech / "云烟成雨" / group
            d.mkdir(parents=True)
            (d / f"{stem}.wav").write_bytes(b"w" * 16)
            (d / f"{stem}.json").write_text("[]", encoding="utf-8")
        (base / "English" / "EN-Tenor-1" / "Breathy" / "song"
         / "Breathy_Group" / "0001.wav").parent.mkdir(parents=True)
        (base / "English" / "EN-Tenor-1" / "Breathy" / "song"
         / "Breathy_Group" / "0001.wav").write_bytes(b"w" * 16)
        return str(base)

    def test_iter_skips_paired_speech_and_maps_lang(self, tmp_path):
        from webui_app.services import gts_import as GI
        root = self._make_tree(tmp_path)
        got = list(GI.iter_phrases(root, langs=["zh"]))
        groups = {g["group"] for g in got}
        assert "Paired_Speech_Group" not in groups
        assert all(g["lang"] == "zh" for g in got)
        assert groups == {"Breathy_Group", "Control_Group", ""} or             groups <= {"Breathy_Group", "Control_Group"}

    def test_control_uid_dedups_across_techniques(self, tmp_path):
        from webui_app.services import gts_import as GI
        root = self._make_tree(tmp_path)
        uids = {GI._uid_of(g) for g in GI.iter_phrases(root, langs=["zh"])}
        # Vibrato/Control_Group/0000 与 Breathy/Control_Group/0000 → 同 uid
        assert len([u for u in uids if "Control_Group" in u]) == 1

    def test_find_root_accepts_real_layout(self, tmp_path):
        from webui_app.services import gts_import as GI
        assert GI.find_gtsinger_root(self._make_tree(tmp_path))
        empty = tmp_path / "not_gts"
        empty.mkdir()
        (empty / "x.wav").write_bytes(b"w")
        assert not GI.find_gtsinger_root(str(empty))


class TestRealJsonSchema:
    """逐词列表 schema（实测样本）的文本拼接与技巧段提取。"""

    def _sample(self):
        return [
            {"word": "你", "start_time": 0.0, "end_time": 0.57,
             "ph": ["n", "i"], "mix": ["0", "0"], "falsetto": ["0", "0"],
             "breathy": ["1", "1"], "pharyngeal": ["0", "0"],
             "glissando": ["0", "0"], "vibrato": ["0", "0"]},
            {"word": "好", "start_time": 0.57, "end_time": 1.10,
             "ph": ["h"], "mix": ["0"], "falsetto": ["1"],
             "breathy": ["0"], "pharyngeal": ["0"],
             "glissando": ["0"], "vibrato": ["0"]},
            {"word": "呀", "start_time": 1.10, "end_time": 1.60,
             "ph": ["a"], "mix": ["0"], "falsetto": ["0"],
             "breathy": ["0"], "pharyngeal": ["0"],
             "glissando": ["0"], "vibrato": ["0"]},
        ]

    def test_parse_text_and_segments(self, tmp_path):
        from webui_app.services import gts_import as GI
        jp = tmp_path / "0000.json"
        jp.write_text(json.dumps(self._sample()), encoding="utf-8")
        text, segs = GI.load_transcription(str(jp))
        assert text == "你好呀"                      # CJK 紧排
        assert segs == [[0.0, 0.57, 3],               # breathy=3
                        [0.57, 1.1, 2]]               # falsetto=2

    def test_english_words_spaced(self, tmp_path):
        from webui_app.services import gts_import as GI
        jp = tmp_path / "0001.json"
        jp.write_text(json.dumps([
            {"word": "hello", "start_time": 0.0, "end_time": 0.5,
             "ph": ["hh"], "mix": ["0"], "falsetto": ["0"], "breathy": ["0"],
             "pharyngeal": ["0"], "glissando": ["0"], "vibrato": ["0"]},
            {"word": "world", "start_time": 0.5, "end_time": 1.0,
             "ph": ["w"], "mix": ["0"], "falsetto": ["0"], "breathy": ["0"],
             "pharyngeal": ["0"], "glissando": ["0"], "vibrato": ["0"]},
        ]), encoding="utf-8")
        text, segs = GI.load_transcription(str(jp))
        assert text == "hello world"
        assert segs == []                              # 无技巧 → control

# ---------------------------------------------------------------------------
# 47g：拉取假失败修复（磁盘实况判定 / 重试轮 / 状态显示）
# ---------------------------------------------------------------------------

class TestFetchVerdict:
    """2026-10-19 事故回归钉：下载 90% 时报「失败」、且状态看不出缺标注。"""

    def _fake_listing(self, monkeypatch, files):
        monkeypatch.setattr(GF, "list_repo_files", lambda langs=None: list(files))
        monkeypatch.setattr(GF, "ensure_endpoint", lambda: "https://hf-mirror.com")

    def test_missing_on_disk_is_the_verdict(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"x")
        (tmp_path / "b.json").write_bytes(b"{}")
        missing = GF._missing_on_disk(str(tmp_path), ["a.wav", "b.json", "c.wav"])
        assert missing == ["c.wav"]

    def test_partial_result_writes_incomplete_list(self, tmp_path, monkeypatch):
        files = ["Chinese/s/a.wav", "Chinese/s/a.json", "Chinese/s/b.wav"]
        self._fake_listing(monkeypatch, files)
        # 宿主机上已有 a.wav；下载器把 a.json 下成功，b.wav 始终失败
        (tmp_path / "Chinese" / "s").mkdir(parents=True)
        (tmp_path / "Chinese" / "s" / "a.wav").write_bytes(b"x")

        import types
        fake = types.ModuleType("huggingface_hub")

        def _dl(repo, fname, repo_type=None, local_dir=None):
            if fname.endswith("b.wav"):
                raise ConnectionError("mirror 429")
            p = os.path.join(local_dir, fname)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(b"{}")
            return p
        fake.hf_hub_download = _dl
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

        r = GF.fetch(str(tmp_path), ["zh"])
        assert not r["ok"] and r.get("partial")
        assert r["missing"] == 1 and r["done"] == 2
        assert os.path.isfile(os.path.join(str(tmp_path), GF.INCOMPLETE_FILE))

    def test_all_present_reports_success_and_clears_list(self, tmp_path, monkeypatch):
        files = ["Chinese/s/a.wav", "Chinese/s/a.json"]
        self._fake_listing(monkeypatch, files)
        (tmp_path / "Chinese" / "s").mkdir(parents=True)
        for f in files:
            (tmp_path / f).write_bytes(b"x")
        inc = tmp_path / GF.INCOMPLETE_FILE
        inc.write_text("stale", encoding="utf-8")

        r = GF.fetch(str(tmp_path), ["zh"])
        assert r["ok"] and r["done"] == 2
        assert not inc.exists(), "补齐后必须清掉残缺名单"

    def test_source_state_counts_missing_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(GF, "DEFAULT_DEST", str(tmp_path))
        d = tmp_path / "Chinese" / "s"
        d.mkdir(parents=True)
        (d / "a.wav").write_bytes(b"x")
        (d / "a.json").write_text("[]", encoding="utf-8")
        (d / "b.wav").write_bytes(b"x")          # 缺 json
        (d / "c.wav.part").write_bytes(b"x")     # 残留分片
        st = GF.source_state(str(tmp_path))
        assert st["wavs"] == 2 and st["jsons"] == 1
        assert st["missing_json"] == 1 and st["partials"] == 1
