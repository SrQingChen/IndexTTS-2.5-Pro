"""自动评测台（evaluate.py）的验证。

注入假引擎与假打分器，只测评测台自己的编排：
  · A/B 逐条同种子合成（公平性）、挂/卸 adapter 的成对性
  · 胜负表 / 指标均值 / report.json + report.md / 试听文件
  · 选手目录先验、引擎状态还原、单选手体检模式
真实引擎的端到端在阶段 2 验收（b9）里跑。

跑法：  .venv\\Scripts\\python.exe tools\\eval_probe.py
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from webui_app.training import dataset as DS                # noqa: E402
from webui_app.training import evaluate as EV               # noqa: E402
from webui_app.training import guard as GD                  # noqa: E402
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


def make_wav(path: str, seconds: float, sr: int, seed: int) -> None:
    import soundfile as sf
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.sin(2 * np.pi * (110.0 + seed * 7) * t) * 0.5
    sf.write(path, (sig + rng.standard_normal(n) * 0.002).astype(np.float32),
             sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# 假引擎：记录挂载与调用，合成按「当前是否挂着 adapter」给出不同音频名
# ---------------------------------------------------------------------------
class FakeEngine:
    def __init__(self):
        self.tts = SimpleNamespace(gpt=None, s2mel=SimpleNamespace(
            models={"cfm": None}))
        self.stats = SimpleNamespace(lora_adapters=[])
        self.infer_calls = []          # (sample_id, contender_tag, seed_draw)
        self.attach_calls = []
        self.detach_calls = 0

    def attach_lora(self, adapter_dir, target="gpt"):
        name = os.path.basename(adapter_dir.rstrip("/\\"))
        self.stats.lora_adapters.append(f"{target}:{name}")
        self.attach_calls.append((adapter_dir, target))

    def detach_lora(self, target="gpt"):
        self.detach_calls += 1
        self.stats.lora_adapters = [
            t for t in self.stats.lora_adapters
            if not t.startswith(target + ":")]

    def infer(self, **kw):
        # 抓住 manual_seed 之后抽到的第一个数 —— A/B 若同种子，这里必然一致
        seed_draw = int(torch.randint(0, 2 ** 31, (1,)).item())
        sid = os.path.basename(kw["output_path"]).rsplit("_", 1)[0]
        tag = "ada" if any(t.endswith(":adapter") for t in
                           self.stats.lora_adapters) else "base"
        self.infer_calls.append((sid, tag, seed_draw))
        make_wav(kw["output_path"], 1.0, 16000, (len(sid) * 17 + len(tag)))
        return kw["output_path"]


class FakeScorer:
    """wav 文件名里带 _ada 的reward 高，_base 的低。"""
    def score(self, path, text, ref, lang=None):
        tag = "ada" if "_ada" in os.path.basename(path) else "base"
        r = {"ada": 0.9, "base": 0.5}[tag]
        return {"ok": True, "wer": 0.1 if tag == "ada" else 0.4,
                "sim": 0.8 if tag == "ada" else 0.5,
                "reward": r, "asr_text": text, "seconds": 0.0}

    def unload(self):
        pass


def main() -> int:
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="eval_probe_")
    ds_root = os.path.join(tmp, "datasets")
    run_root = os.path.join(tmp, "runs")
    out_root = os.path.join(tmp, "outputs")
    for d in (ds_root, run_root, out_root):
        os.makedirs(d)
    DS.DATASETS_ROOT = ds_root
    RN.ROOT = run_root

    DS_NAME = "probe_eval"
    N = 5

    try:
        # ==============================================================
        head("[1] 造数据集与假 run")
        DS.create(DS_NAME, note="eval_probe")
        ds_dir = DS.dir_of(DS_NAME)
        tmpw = os.path.join(ds_dir, "_src")
        os.makedirs(tmpw, exist_ok=True)
        wavs = []
        for i in range(N):
            p = os.path.join(tmpw, f"s{i:03d}.wav")
            make_wav(p, 1.5, 22050, i + 1)
            wavs.append(p)
        DS.import_audio(DS_NAME, wavs, copy=True, lang="ZH")
        shutil.rmtree(tmpw, ignore_errors=True)
        touched = {u.id: {"text": f"第{i}句评测文本。"}
                   for i, u in enumerate(DS.load_meta(DS_NAME))}
        from webui_app.training import features as FT
        FT._apply_meta(DS_NAME, touched)
        DS.refresh_all(DS_NAME, require_features=False)
        n_ready = len([u for u in DS.load_meta(DS_NAME)
                       if u.status == "ready" and u.text.strip()])
        check(f"{N} 条样本 ready 且带文本", n_ready == N, f"{n_ready}/{N}")

        RN.run_dir("runA", create=True)
        ada = RN.adapter_dir("runA", create=True)
        with open(os.path.join(ada, "adapter_model.safetensors"), "wb") as f:
            f.write(b"\x00" * 64)
        RN.write_run("runA", {"arch": "gpt", "run": "runA", "status": "done"})

        # ==============================================================
        head("[2] 配置校验与选手目录解析")
        errs = lambda vv: [x.message for x in vv.validate() if x.level == "error"]
        check("数据集不存在被拦", any("不存在" in m for m in
                                      errs(EV.EvalOptions(dataset="没有")))),
        check("温度非法被拦", any("temperature" in m for m in
                                  errs(EV.EvalOptions(dataset=DS_NAME,
                                                      temperature=0.0))))
        d, arch = EV.resolve_adapter_dir(EV.Contender("ada", run="runA"))
        check("选手目录解析到 runA 的 adapter（arch=gpt）",
              d == ada and arch == "gpt", f"{d}")
        d0, a0 = EV.resolve_adapter_dir(EV.Contender("base"))
        check("空 run 解析成纯底座", d0 == "" and a0 == "")
        try:
            EV.resolve_adapter_dir(EV.Contender("x", run="runB"))
            check("不存在的 run 报 FileNotFoundError", False)
        except FileNotFoundError:
            check("不存在的 run 报 FileNotFoundError", True)
        check("label 含 run 与强度信息",
              "@1" in EV.Contender("ada", run="runA",
                                   adapter_scale=1.0).label())

        # ==============================================================
        head("[3] A/B 对比（假引擎）")
        eng = FakeEngine()
        a = EV.Contender("ada", run="runA")
        b = EV.Contender("base")
        opts = EV.EvalOptions(dataset=DS_NAME, out_dir=os.path.join(out_root, "ab"),
                              seed=7)
        res = EV.run_eval(eng, a, b, opts, scorer=FakeScorer())
        check("评测 ok", res["ok"], str(res["errors"])[:120])
        check(f"逐条 {N} 行全部成功",
              len(res["rows"]) == N and all(r["ok"] for r in res["rows"]),
              str([r.get("ada", {}).get("error") for r in res["rows"]]))
        check("每行都有 A/B 的 reward 与 Δ",
              all("delta_reward" in r and r["delta_reward"] == 0.4
                  for r in res["rows"]))
        s = res["summary"]
        check("胜负表：A 全胜（reward 0.9 vs 0.5）",
              s["win_a"] == N and s["win_b"] == 0 and s["tie"] == 0,
              f"{s.get('win_a')}/{s.get('win_b')}/{s.get('tie')}")
        check("均值：A 的 WER 更低、相似度更高",
              s["a"]["wer"] < s["b"]["wer"] and s["a"]["sim"] > s["b"]["sim"])
        check("A 的合成发生在挂载 adapter 之后（每次 attach→infer→detach）",
              len(eng.attach_calls) == N and eng.detach_calls >= N,
              f"attach={len(eng.attach_calls)} detach={eng.detach_calls}")
        check("评测后引擎上没有残留 adapter",
              not eng.stats.lora_adapters)
        # ★ 公平性：同一样本 A/B 的种子抽样一致
        by_sid = {}
        for sid, tag, draw in eng.infer_calls:
            by_sid.setdefault(sid, {})[tag] = draw
        same_seed = all(v.get("ada") == v.get("base") for v in by_sid.values())
        check("★ 同一条样本 A/B 用同一个种子（差异只来自模型）",
              same_seed and len(by_sid) == N,
              f"{len(by_sid)} 条 · 一致 {sum(1 for v in by_sid.values() if v.get('ada') == v.get('base'))}")
        check("report.json 落盘且含 summary/rows",
              os.path.isfile(os.path.join(res["out_dir"], "report.json"))
              and json.load(open(os.path.join(res["out_dir"], "report.json"),
                                 encoding="utf-8")).get("summary", {}).get("win_a") == N)
        md = open(os.path.join(res["out_dir"], "report.md"),
                  encoding="utf-8").read()
        check("report.md 有对比表与胜负结论", "胜" in md and "| WER |" in md)
        wavs_out = [f for f in os.listdir(res["out_dir"]) if f.endswith(".wav")]
        check(f"试听文件 {2*N} 个（每条 A/B 各一）",
              len(wavs_out) == 2 * N, f"{len(wavs_out)}")

        # ==============================================================
        head("[4] 单选手体检 + 先验失败 + 还原")
        eng2 = FakeEngine()
        res1 = EV.run_eval(eng2, EV.Contender("base"),
                           None, EV.EvalOptions(
                               dataset=DS_NAME,
                               out_dir=os.path.join(out_root, "single")),
                           scorer=FakeScorer())
        check("单选手模式 ok 且无胜负表",
              res1["ok"] and "win_a" not in res1["summary"])
        md1 = open(os.path.join(res1["out_dir"], "report.md"),
                   encoding="utf-8").read()
        check("单选手报告只有一列", "| reward |" in md1 and "vs" not in md1)

        # 选手目录错 → 一条都不合成（先验）
        eng3 = FakeEngine()
        res2 = EV.run_eval(eng3, EV.Contender("x", run="runB"),
                           EV.Contender("base"),
                           EV.EvalOptions(dataset=DS_NAME),
                           scorer=FakeScorer())
        check("选手目录错：先验失败且没合成任何东西",
              not res2["ok"] and res2["errors"] and not eng3.infer_calls)

        # 引擎上原有 adapter：评测前卸载、评测后还原
        eng4 = FakeEngine()
        eng4.stats.lora_adapters.append("gpt:adapter")     # 用户挂着的
        res3 = EV.run_eval(eng4, EV.Contender("base"), None,
                           EV.EvalOptions(
                               dataset=DS_NAME,
                               out_dir=os.path.join(out_root, "restore")),
                           scorer=FakeScorer())
        check("开场卸载用户 adapter 并写了警告",
              any("临时卸载" in w for w in res3["warnings"]),
              str(res3["warnings"])[:80])
        check("评测后用户的 adapter 被重新挂回",
              eng4.stats.lora_adapters == ["gpt:adapter"],
              str(eng4.stats.lora_adapters))
        # 用户挂的 adapter 目录已不存在 → 还原失败但给了可读警告
        eng5 = FakeEngine()
        eng5.stats.lora_adapters.append("gpt:不存在的档")
        res4 = EV.run_eval(eng5, EV.Contender("base"), None,
                           EV.EvalOptions(
                               dataset=DS_NAME,
                               out_dir=os.path.join(out_root, "restore2")),
                           scorer=FakeScorer())
        check("还原失败给可读警告（而不是崩）",
              any("重新挂载" in w for w in res4["warnings"]))

        # 引擎未加载
        class Dead:
            tts = None
            stats = SimpleNamespace(lora_adapters=[])
        res5 = EV.run_eval(Dead(), EV.Contender("base"), None,
                           EV.EvalOptions(dataset=DS_NAME),
                           scorer=FakeScorer())
        check("引擎未加载给出可读错误", not res5["ok"]
              and any("加载" in e for e in res5["errors"]))

        # ==============================================================
        head("[5] n_samples 截断 + should_stop")
        eng6 = FakeEngine()
        res6 = EV.run_eval(eng6, EV.Contender("base"), None,
                           EV.EvalOptions(dataset=DS_NAME, n_samples=2,
                                          out_dir=os.path.join(out_root, "n2")),
                           scorer=FakeScorer())
        check("n_samples=2 只评 2 条", len(res6["rows"]) == 2)
        eng7 = FakeEngine()
        res7 = EV.run_eval(eng7, EV.Contender("base"), None,
                           EV.EvalOptions(
                               dataset=DS_NAME,
                               out_dir=os.path.join(out_root, "stop")),
                           scorer=FakeScorer(),
                           should_stop=lambda: True)
        check("should_stop 立即停且报告了原因",
              res7["rows"] == [] and any("停止" in w for w in res7["warnings"]))

        # ==============================================================
        head("[6] 清理")
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
