"""阶段45 追加：分离模型的镜像自动下载（离线单测，无网络、无 GPU）。

覆盖：
- 镜像链解析（环境变量覆盖/直连末位）
- UI 暴露模型都有完整文件对（权重+YAML）
- 已存在时 ensure_model 不重复下载
- 缺失时经 _fetch 下载并报 ok
- 推理入口 _run_sep 在创建 Separator 之前先保证模型（自动部署语义）
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.services import separation as S  # noqa: E402


class TestMirrors:
    def test_default_chain(self, monkeypatch):
        monkeypatch.delenv("INDEXTTS_GH_MIRRORS", raising=False)
        m = S.mirrors()
        assert m[0].startswith("https://ghproxy.net")
        assert "" in m[-1] or m[-1] == ""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("INDEXTTS_GH_MIRRORS", "https://a.example/,https://b.example/")
        assert S.mirrors() == ["https://a.example/", "https://b.example/"]

    def test_mirrored_order(self, monkeypatch):
        monkeypatch.setenv("INDEXTTS_GH_MIRRORS", "https://a/,https://b/,")
        urls = S._mirrored("https://github.com/x/y")
        assert urls == ["https://a/https://github.com/x/y",
                        "https://b/https://github.com/x/y",
                        "https://github.com/x/y"]

    def test_all_exposed_models_have_pairs(self):
        exposed = set(S.VOCALS_MODELS.values()) | set(S.DEREVERB_MODELS.values())
        for name in exposed:
            assert name in S.MODEL_FILES, f"缺文件清单: {name}"
            assert S.MODEL_FILES[name][0] == name, "第一个必须是权重"
        # roformer 需要 YAML，VR (.pth) 不需要
        for name in exposed:
            if name.endswith(".ckpt"):
                assert any(f.endswith(".yaml") for f in S.MODEL_FILES[name])


class TestMetadataAndAliases:
    def test_alias_for_deecho_normal(self):
        # 实锤：Release 资产名带连字符，模型标识不带
        assert S.FILE_URL_ALIASES["UVR-DeEcho-Normal.pth"] == "UVR-De-Echo-Normal.pth"

    def test_metadata_seeds_three_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "MODEL_DIR", str(tmp_path))
        seen = []

        def fake_dl(url, dest, expected, progress=None, label="", attempts=3,
                    should_stop=None):
            seen.append(label)
            open(dest, "wb").write(b"x" * 2048)
            return True

        monkeypatch.setattr(S, "_download_once", fake_dl)
        assert S.ensure_metadata()
        names = {os.path.basename(x) for x in seen}
        assert {"download_checks.json", "vr_model_data.json",
                "mdx_model_data.json"} <= names

    def test_fetch_uses_alias_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "MODEL_DIR", str(tmp_path))
        urls = []

        def fake_dl(url, dest, expected, progress=None, label="", attempts=3,
                    should_stop=None):
            urls.append(url)
            open(dest, "wb").write(b"x" * (2 << 20))
            return True

        monkeypatch.setattr(S, "_download_once", fake_dl)
        assert S._fetch("UVR-DeEcho-Normal.pth")
        assert any("UVR-De-Echo-Normal.pth" in u for u in urls)


class TestEnsureModel:
    def test_skip_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "MODEL_DIR", str(tmp_path))
        name = S.DEFAULT_VOCALS
        for f in S.MODEL_FILES[name]:
            # 权重阈值 1MiB、YAML 200B（944B 的真实 YAML 曾触发过误判）
            (tmp_path / f).write_bytes(b"x" * (2 << 20 if f.endswith(".ckpt") else 4096))
        called = {"n": 0}
        monkeypatch.setattr(S, "_fetch", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
        r = S.ensure_model(name)
        assert r["ok"] and r["downloaded"] == []
        assert called["n"] == 0

    def test_download_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(S, "ensure_metadata", lambda progress=None, should_stop=None: True)
        name = S.DEFAULT_VOCALS

        def fake_fetch(f, progress=None, rel_frac=0.0, should_stop=None):
            (tmp_path / f).write_bytes(
                b"x" * (2 << 20 if f.endswith(".ckpt") else 4096))
            return True

        monkeypatch.setattr(S, "_fetch", fake_fetch)
        r = S.ensure_model(name)
        assert r["ok"], r
        assert len(r["downloaded"]) == len(S.MODEL_FILES[name])

    def test_failure_reports_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(S, "ensure_metadata", lambda progress=None, should_stop=None: True)
        monkeypatch.setattr(S, "_fetch", lambda *a, **k: False)
        r = S.ensure_model(S.DEFAULT_DEREVERB)
        assert not r["ok"]
        assert r["missing"], "必须报告缺哪些文件"
        assert "INDEXTTS_GH_MIRRORS" in r.get("error", "")

    def test_metadata_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(S, "ensure_metadata", lambda progress=None, should_stop=None: False)
        r = S.ensure_model(S.DEFAULT_VOCALS)
        assert not r["ok"]
        assert "模型清单" in r.get("error", "")


class TestRunSepPrefetch:
    def test_run_sep_ensures_model_before_separator(self, monkeypatch, tmp_path):
        """推理入口必须先确保模型（自动部署），失败则直接返回且不建 Separator。"""
        monkeypatch.setattr(S, "available", lambda: True)
        src = tmp_path / "in.wav"
        src.write_bytes(b"RIFF" + b"\0" * 100)
        calls = {"ensure": 0, "make": 0}
        monkeypatch.setattr(S, "ensure_model",
                            lambda name, progress=None, should_stop=None:
                            (calls.__setitem__("ensure", calls["ensure"] + 1)
                             or {"ok": False, "error": "镜像不可达", "missing": [name]}))
        monkeypatch.setattr(S, "_make_separator",
                            lambda out: calls.__setitem__("make", calls["make"] + 1))
        r = S._run_sep(str(src), str(tmp_path / "out"), S.DEFAULT_VOCALS,
                       ["vocals"], ["instrumental"])
        assert not r["ok"]
        assert "自动下载失败" in r["error"]
        assert calls["ensure"] == 1
        assert calls["make"] == 0, "模型未就绪时不得创建 Separator"

    def test_run_sep_ok_after_ensure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(S, "available", lambda: True)
        src = tmp_path / "in.wav"
        src.write_bytes(b"RIFF" + b"\0" * 100)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setattr(S, "ensure_model",
                            lambda name, progress=None, should_stop=None:
                            {"ok": True, "downloaded": [], "missing": []})

        class _Sep:
            def load_model(self, model_filename):
                pass
            def separate(self, path):
                v = out_dir / "vocals.wav"
                i = out_dir / "instrumental.wav"
                v.write_bytes(b"x" * 100)
                i.write_bytes(b"y" * 100)
                return [str(v), str(i)]

        monkeypatch.setattr(S, "_make_separator", lambda out: _Sep())
        monkeypatch.setattr(S, "_vram_free_gb", lambda: 6.0)
        r = S._run_sep(str(src), str(out_dir), S.DEFAULT_VOCALS,
                       ["vocals"], ["instrumental"])
        assert r["ok"], r
        assert r["instrumental"]
