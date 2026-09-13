"""L1 特征预提取（training/features.py）的实机验证。

需要 GPU：会加载完整引擎（约 5 GB 显存），并在最后卸载。
不要在 WebUI 正在跑推理时执行。

    .venv\\Scripts\\python.exe tools\\features_probe.py

============================================================================
验证策略：为什么用「引擎自己合成的音频」当数据集
============================================================================
特征提取的正确性没法只看形状 —— 形状对但数值错，训练照样静默失败。
所以这里做一次**闭环往返**：

    已知文本 T ──引擎合成──▶ 音频 A ──features──▶ (codes, mu, mel, style, emo_vec)

因为 A 就是模型从 T 生成的，所以：
    · GPT 在 (T, codes) 上做 teacher forcing 的 loss 必须**远低于** ln(8194)≈9.01
      的均匀分布基线 —— 否则说明 text_tokens 或 codes 的编码/padding 错了
    · 把 codes 打乱后 loss 必须**明显回升** —— 否则说明前向根本没在用条件
    · CFM 在真实 (mu, style, mel) 上的 loss 必须**明显低于**把 mu/style 置零的
      无条件基线 —— 否则说明 mu 的帧对齐或 dtype 错了

这三条一起成立，才能说特征与训练前向都是对的。用 examples/ 里的现成音频做不到，
因为我们不知道它们说了什么。
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
from typing import Any, Dict

import _env                                            # noqa: F401  路径 + 控制台编码
PROJECT_ROOT = _env.PROJECT_ROOT

PASS, FAIL = [], []
# 不用下划线开头的名字：safe_dataset_name 会 strip("._")，
# 拿 "_x" 做测试会把「名称清洗」和「特征提取」两个问题混在一起。
DS_NAME = "probe_features"
REF_AUDIO = os.path.join(PROJECT_ROOT, "examples", "voice_05.wav")

SENTENCES = [
    "今天天气不错，我们出去走走吧。",
    "这个项目终于跑通了，太不容易了。",
    "请把报告发到我的邮箱，谢谢。",
]


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def head(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def vram_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def free():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 训练前向：**直接用生产代码**（webui_app/training/forward.py）
#
# 这里曾经有一份副本。副本验证通过不代表生产代码是对的 ——
# 两边会各自漂移，而探针是唯一能证明「前向接线与推理一致」的东西。
# 所以探针必须验证训练器真正会调用的那一份。
# forward.py 模块级不 import torch，所以在 sys.path 搭好之后直接导不拖慢启动。
# ---------------------------------------------------------------------------
from webui_app.training import forward as FW          # noqa: E402


def fwd(gpt, style, emo_vec, langs, text_tokens, text_lengths, codes, code_lengths,
        use_lang: bool = True) -> "FW.GptForward":
    """生产前向的简写，返回 `GptForward`（带 mask 的 loss 方法）。

    交叉熵一律走 `GptForward.mel_loss()`，不在探针里重算一遍：
    上一轮就是因为探针自带一份 `ce()`，里面少写了一个 transpose，
    把 batch/vocab/seq 三个轴的内存布局搅在一起，算出 17.06 的假 loss
    （均匀基线才 9.01），差点误判成「前向接错了」。
    `masked_ce` 本身的轴约定由 guard_test.py 在 CPU 上用小张量穷举验证。
    """
    return FW.gpt_training_forward(gpt, style, emo_vec, langs, text_tokens,
                                   text_lengths, codes, code_lengths,
                                   use_lang=use_lang)


def main() -> int:
    import torch

    head("[0] 加载引擎（复用 engine 服务，特征提取共用同一份权重）")
    from webui_app.config import config_from_args
    from webui_app.context import AppContext
    from webui_app.services import inference as INF
    from webui_app.training import dataset as DS
    from webui_app.training import features as FT
    from webui_app.training import guard as GD

    AppContext.reset()
    cfg = config_from_args([])
    ctx = AppContext.get(cfg)
    eng = ctx.engine
    t0 = time.perf_counter()
    eng.load()
    tts = eng.tts
    dev = str(getattr(tts, "device", "cuda"))
    print(f"  引擎加载 {time.perf_counter() - t0:.1f}s · 显存 {vram_gb():.2f} GB · device={dev}")
    check("引擎加载成功", tts is not None)
    check("显存占用 < 7 GB（给特征提取留余量）", vram_gb() < 7.0, f"{vram_gb():.2f} GB")

    dtypes = {n: str(next(getattr(tts, n).parameters()).dtype)
              for n in ("gpt", "semantic_codec", "campplus_model", "s2mel")
              if hasattr(tts, n)}
    print(f"  子模块 dtype: {dtypes}")

    # ==================================================================
    head("[1] 用引擎合成 3 条 (文本, 音频) 严格对齐的样本")
    if not os.path.isfile(REF_AUDIO):
        print(f"  参考音频不存在：{REF_AUDIO}")
        return 1
    made = []
    # 探针：抓下引擎自己生成时的**原始 codes 与条件**。
    # 这是验证 gpt_training_forward 接线是否正确的唯一干净手段：
    # 音频绕回（wav → w2v-bert → quantize）得到的 codes 与模型当初生成的
    # codes 并不相同，拿后者算 teacher forcing 会把「接线错」与「数据不匹配」
    # 两个完全不同的问题混成一团。
    cap: Dict[str, Any] = {}
    orig_speak = tts.gpt.inference_speech

    def spy(speech_condition, text_inputs, langs, emo_speech_condition=None,
            cond_lengths=None, emo_cond_lengths=None, emo_vec=None, **kw):
        out = orig_speak(speech_condition, text_inputs, langs, emo_speech_condition,
                         cond_lengths, emo_cond_lengths, emo_vec, **kw)
        if not cap:
            try:
                seq = out[0]
                camp = kw.get("campplus_embedding")
                cap.update(
                    codes=seq.detach().to(torch.int64).cpu(),
                    text=text_inputs.detach().to(torch.int64).cpu(),
                    langs=langs.detach().to(torch.int64).cpu(),
                    emo_vec=None if emo_vec is None else emo_vec.detach().float().cpu(),
                    style=None if camp is None else camp.detach().float().cpu(),
                )
            except Exception as e:
                cap["err"] = f"{type(e).__name__}: {e}"
        return out

    tts.gpt.inference_speech = spy
    for i, s in enumerate(SENTENCES):
        req = INF.GenRequest(spk_audio_prompt=REF_AUDIO, text=s, lang="ZH", seed=1234 + i)
        r = INF.generate(eng, req)
        made.append((s, r["path"], r["audio_duration"]))
        print(f"  [{i}] {r['audio_duration']:.2f}s  {os.path.basename(r['path'])}  {s}")
    check("3 条都合成成功", len(made) == 3)
    tts.gpt.inference_speech = orig_speak        # 拆探针
    check("探针抓到了模型自己生成的 codes",
          "codes" in cap and cap["codes"].numel() > 10,
          str(cap.get("err") or tuple(cap.get("codes", torch.empty(0)).shape)))
    check("探针同时抓到了 style / emo_vec / text / lang",
          all(cap.get(k) is not None for k in ("style", "emo_vec", "text", "langs")))
    check("音频时长都在 1~20s 的可训练区间",
          all(1.0 < d < FT.MAX_TRAIN_SEC for _, _, d in made),
          str([round(d, 2) for _, _, d in made]))
    print(f"  合成后显存 {vram_gb():.2f} GB")

    # ==================================================================
    head("[2] 建临时数据集并导入")
    # 清掉之前跑失败留下的目录（包括 dir_of/create 名称不一致时建错的那个）
    for stale in (DS_NAME, "_features_probe", "features_probe"):
        d = os.path.join(DS.DATASETS_ROOT, stale)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  清理残留目录 {d}")
    created = DS.create(DS_NAME, note="features_probe 自动生成")
    check("create 返回的名字与 dir_of 一致",
          os.path.isdir(DS.dir_of(created)) and DS.exists(created),
          f"create->{created}  dir_of->{DS.dir_of(created)}")
    check("名字不含非法字符时不被改写", created == DS_NAME,
          f"{created} vs {DS_NAME}")
    imp = DS.import_audio(created, [p for _, p, _ in made], copy=True, lang="ZH")
    print(f"  导入 {imp}")
    check("导入 3 条", imp.get("added") == 3, str(imp))
    for (txt, _p, _d), u in zip(made, DS.load_meta(created)):
        DS.update(created, u.id, text=txt)
    st = DS.refresh_all(created, require_features=False)
    print(f"  体检（不要求特征）：{st}")
    ready = [u for u in DS.load_meta(created) if u.status == "ready"]
    check("3 条都通过体检成为 ready", len(ready) == 3,
          str([(u.id, u.status, u.problems) for u in DS.load_meta(created)]))
    check("数据集里就这 3 条（没有混入旧数据）",
          len(DS.load_meta(created)) == 3, f"{len(DS.load_meta(created))} 条")
    check("文本已写入 meta", all(u.text for u in DS.load_meta(created)))
    st_feat = DS.refresh_all(created, require_features=True)
    nofeat = [u for u in DS.load_meta(created) if u.status == "no_features"]
    check("要求特征时，未提取的样本被标为 no_features", len(nofeat) == 3,
          f"{st_feat}")

    # ==================================================================
    head("[3] 离线特征提取")
    t0 = time.perf_counter()
    prog = []
    res = FT.extract_dataset(DS_NAME, tts=tts, device=dev,
                             progress=lambda f, d: prog.append((round(f, 2), d)))
    dt = time.perf_counter() - t0
    print(f"  提取 {res['extracted']} 条 / 跳过 {res['skipped']} / 失败 {res['failed']}"
          f" · {dt:.1f}s · {res['bytes'] / 1e6:.2f} MB")
    check("3 条全部提取成功", res["extracted"] == 3 and res["failed"] == 0,
          str(res["errors"]))
    check("进度回调被调用过", len(prog) >= 3, f"{len(prog)} 次")
    check("提取后显存没有失控（< 7.5 GB）", vram_gb() < 7.5, f"{vram_gb():.2f} GB")
    if res["warnings"]:
        print(f"  警告：{res['warnings']}")

    ds_dir = DS.dir_of(DS_NAME)
    feats = {}
    for u in DS.load_meta(DS_NAME):
        p = FT.FeatureExtractor.feature_path(ds_dir, u.id)
        check(f"{u.id} 的特征文件已落盘", os.path.isfile(p), p)
        feats[u.id] = torch.load(p, map_location="cpu", weights_only=False)
    check("meta 里 has_features 已回写",
          all(u.has_features for u in DS.load_meta(DS_NAME)))
    st2 = DS.refresh_all(DS_NAME, require_features=True)
    check("提取后 3 条都回到 ready",
          len([u for u in DS.load_meta(DS_NAME) if u.status == "ready"]) == 3, f"{st2}")

    # ==================================================================
    head("[4] 特征字段逐项核对")
    f0 = next(iter(feats.values()))
    check("schema 里的字段全都在", all(k in f0 for k in FT.FEATURE_SCHEMA),
          str(sorted(f0.keys())))
    check("feature_version 写入了", f0["feature_version"] == FT.FEATURE_VERSION)
    for uid, f in feats.items():
        probs = FT.check_feature(f)
        check(f"{uid} 通过一致性自检", not probs, "；".join(probs))

    for uid, f in feats.items():
        print(f"\n  -- {uid}  ({f['duration']:.2f}s)")
        for k in FT.FEATURE_SCHEMA:
            v = f[k]
            print(f"     {k:14s} {str(tuple(v.shape)):16s} {str(v.dtype).replace('torch.', ''):8s} "
                  f"min={float(v.float().min()):+.4f} max={float(v.float().max()):+.4f}")
        # 关键不变式
        check(f"{uid}: mel 是 (80, mel_len)",
              tuple(f["mel"].shape) == (80, f["mel_len"]), str(tuple(f["mel"].shape)))
        check(f"{uid}: mu_prompt/mu_target 帧数 == mel_len",
              f["mu_prompt"].shape[0] == f["mel_len"] == f["mu_target"].shape[0])
        check(f"{uid}: style 192 维且非零",
              f["style"].shape == (192,) and float(f["style"].abs().sum()) > 0)
        check(f"{uid}: emo_vec 1280 维且非零",
              f["emo_vec"].shape == (1280,) and float(f["emo_vec"].abs().sum()) > 0)
        check(f"{uid}: codes 在 [0, 8194) 且不是常量",
              int(f["codes"].min()) >= 0 and int(f["codes"].max()) < 8194
              and int(f["codes"].unique().numel()) > 5,
              f"unique={int(f['codes'].unique().numel())} max={int(f['codes'].max())}")
        check(f"{uid}: codes 帧数 ≈ duration×25",
              abs(f["n_codes"] - f["duration"] * FT.CODE_FPS) / max(1, f["n_codes"]) < 0.10,
              f'{f["n_codes"]} vs {f["duration"] * FT.CODE_FPS:.0f}')
        check(f"{uid}: mel 帧数 ≈ duration×86.13",
              abs(f["mel_len"] - f["duration"] * FT.MEL_FPS) / max(1, f["mel_len"]) < 0.05,
              f'{f["mel_len"]} vs {f["duration"] * FT.MEL_FPS:.0f}')
        check(f"{uid}: mu_prompt ≠ mu_target（量化确实有损，两条不能是同一个）",
              not torch.equal(f["mu_prompt"], f["mu_target"]),
              f'L1 差 = {float((f["mu_prompt"].float() - f["mu_target"].float()).abs().mean()):.5f}')
        check(f"{uid}: text_tokens 非空且以语言前缀开头",
              f["n_text_tokens"] >= 3 and int(f["lang_token"]) >= 0,
              f'{f["n_text_tokens"]} tokens, lang_token={f["lang_token"]}')
        check(f"{uid}: mel 没有 NaN/Inf",
              bool(torch.isfinite(f["mel"].float()).all()))
        check(f"{uid}: mu 没有 NaN/Inf",
              bool(torch.isfinite(f["mu_prompt"].float()).all())
              and bool(torch.isfinite(f["mu_target"].float()).all()))

    # 文本编码本身要能对上官方链路
    ids, warns = FT.encode_text(tts, SENTENCES[0], "ZH", True)
    u0 = DS.load_meta(DS_NAME)[0]
    check("encode_text 与缓存的 text_tokens 一致",
          ids == feats[u0.id]["text_tokens"].tolist(),
          f"{len(ids)} vs {feats[u0.id]['n_text_tokens']}")
    ids2, _ = FT.encode_text(tts, "他在银<行|XING2>里办事", "ZH", True)
    check("发音标注在特征提取链路里也被展开了", len(ids2) == 13, f"{len(ids2)} tokens")

    # ==================================================================
    head("[5] CFM(S2M) 训练前向：真实特征 + 基线对比")
    ids_list = list(feats)
    pair = FT.build_cfm_pair(feats[ids_list[0]], feats[ids_list[1]])
    check("build_cfm_pair 组装成功", pair is not None)
    if pair:
        Tp, Tt = feats[ids_list[0]]["mel_len"], feats[ids_list[1]]["mel_len"]
        check("x1 = [prompt_mel | target_mel]",
              tuple(pair["x1"].shape) == (80, Tp + Tt), str(tuple(pair["x1"].shape)))
        check("mu 与 x1 逐帧对齐", pair["mu"].shape[0] == pair["x1"].shape[1])
        check("prompt_len 正确", pair["prompt_len"] == Tp, f"{pair['prompt_len']} vs {Tp}")
        check("style 取自 prompt", torch.equal(pair["style"], feats[ids_list[0]]["style"]))
        check("max_frames 超限时返回 None",
              FT.build_cfm_pair(feats[ids_list[0]], feats[ids_list[1]],
                                max_frames=10) is None)
        # 自提示（prompt 与 target 是同一条）也必须能拼
        self_pair = FT.build_cfm_pair(feats[ids_list[0]], feats[ids_list[0]])
        check("自提示对也能拼（prompt=target）",
              self_pair is not None and self_pair["prompt_len"] * 2 == self_pair["total_len"])

        cfm = tts.s2mel.models["cfm"]
        est = cfm.estimator
        # ★ 必须先初始化 RoPE / mask 缓存，否则 gpt_fast/model.py:169 直接断言失败。
        # 注意缓存住在 `DiT.transformer` 上，`DiT.setup_caches` 只是个转发。
        # 往 DiT 本身写 max_seq_length 会创建同名影子属性，静默失效。
        tr = FW._cfm_transformer(cfm)
        check("缓存属性确实在 DiT.transformer 上，而不在 DiT 上",
              hasattr(tr, "max_seq_length") and "max_seq_length" not in est.__dict__,
              f"DiT.__dict__ 里的同名键：{[k for k in est.__dict__ if 'seq' in k]}")
        print(f"  推理入口留下的缓存：max_seq_length={tr.max_seq_length} "
              f"max_batch_size={tr.max_batch_size} use_kv_cache={tr.use_kv_cache}")
        check("DiT.setup_caches 把 use_kv_cache 硬编码成 False（所以永远没有 KVCache）",
              tr.use_kv_cache is False
              and all(getattr(b.attention, "kv_cache", None) is None for b in tr.layers))

        # [1] 已经跑过推理，infer_v2_5.py:200 那次 setup_caches(1, 8192) 留下了
        # 一个 8192² 的 causal_mask（67 MB）。先清干净，才能验证缩小真的生效。
        mask_bytes_before = tr.causal_mask.numel() if tr.causal_mask is not None else 0
        n_kv = FW.invalidate_cfm_caches(cfm)
        print(f"  invalidate：释放 {mask_bytes_before / 1e6:.1f} MB 的 causal_mask，"
              f"清掉 {n_kv} 份 KVCache")
        check("invalidate 后 Transformer 上的尺寸标记归位、mask 已释放",
              tr.max_seq_length == -1 and tr.max_batch_size == -1
              and tr.freqs_cis is None and getattr(tr, "causal_mask", None) is None)
        check("invalidate 没在 DiT 上留下影子属性（写对地方了）",
              "max_seq_length" not in est.__dict__ and "freqs_cis" not in est.__dict__)
        check("KVCache 本来就是 0 份（DiT 硬编码 use_kv_cache=False）", n_kv == 0, f"{n_kv}")

        need = Tp + Tt + 8
        FW.setup_cfm_caches(cfm, batch=1, seq_len=need)
        check("setup_cfm_caches 后 freqs_cis / causal_mask 已建好",
              tr.freqs_cis is not None and getattr(tr, "causal_mask", None) is not None)
        check("缓存长度向上取整到 8 的倍数且刚好够用（不照抄官方的 8192）",
              tr.max_seq_length % 8 == 0 and tr.max_seq_length >= need
              and tr.max_seq_length - need < 8,
              f"{tr.max_seq_length} vs 需要 {need}")
        check("causal_mask 真的缩小了（O(L²)，官方 8192 是 67 MB）",
              tr.causal_mask.numel() < mask_bytes_before,
              f"{tr.causal_mask.numel() / 1e6:.2f} MB vs {mask_bytes_before / 1e6:.1f} MB")
        len_before = tr.max_seq_length
        FW.setup_cfm_caches(cfm, batch=1, seq_len=16)
        check("setup_cfm_caches 幂等：给更小的尺寸不会把缓存缩小",
              tr.max_seq_length == len_before, f"{tr.max_seq_length}")
        FW.setup_cfm_caches(cfm, batch=1, seq_len=len_before + 100)
        check("给更大的尺寸会重建（不会因早退而在 forward 时越界）",
              tr.max_seq_length >= len_before + 100, f"{tr.max_seq_length}")
        FW.setup_cfm_caches(cfm, batch=1, seq_len=need)
        x1 = pair["x1"].unsqueeze(0).to(dev)
        mu = pair["mu"].unsqueeze(0).to(dev)
        style = pair["style"].unsqueeze(0).to(dev)
        x_lens = torch.LongTensor([pair["total_len"]]).to(dev)
        p_lens = torch.LongTensor([pair["prompt_len"]]).to(dev)
        print(f"  x1 {tuple(x1.shape)}  mu {tuple(mu.shape)}  "
              f"mel 均值={float(x1.mean()):+.4f} 标准差={float(x1.std()):.4f}")

        # ---- 基线对比：单次采样的 t 是随机的，方差极大（实测只差 3%），
        #      必须多 seed 平均才能看出条件到底有没有接上。
        N_SEED = 8

        # ★★★ 官方陷阱：BASECFM.forward 是**训练**函数，不能在 eval() 下调用。
        #   DiT.forward 的第 7 个形参是 `mask_content`，而 BASECFM.forward
        #   （flow_matching.py:153）把 `prompt_lens` 传进了这个位置。
        #   eval 模式下 `not self.training and mask_content` 成立 → class_dropout=True
        #   → `x_in[..., in_channels:] *= 0`，prompt_x / cond(mu) / style **全部被清零**。
        #   后果：早停用的 val loss 会变成与条件无关的常数，早停完全失效。
        #   → CFM 的 val loss 必须在 train() 下算，并把 class_dropout_prob 临时置 0。
        cfm.eval()
        try:
            FW.cfm_training_forward(cfm, x1.clone(), x_lens, p_lens,
                                    mu.clone(), style.clone())
            refused = ""
        except RuntimeError as e:
            refused = str(e).splitlines()[0][:60]
        check("cfm_training_forward 在 eval() 下直接报错拦住（不让静默学到错的东西）",
              bool(refused), refused or "居然没报错")
        with torch.no_grad():
            torch.manual_seed(0)
            le_a, _ = cfm.forward(x1.clone(), x_lens, p_lens, mu.clone(), style.clone())
            torch.manual_seed(0)
            le_b, _ = cfm.forward(x1.clone(), x_lens, p_lens,
                                  torch.zeros_like(mu), torch.zeros_like(style))
        print(f"  [陷阱复现] eval() 下 真实mu={float(le_a):.5f}  零mu={float(le_b):.5f}")
        check("【官方陷阱】eval() 下 cfm.forward 会把条件全部清零（两者完全相等）",
              float(le_a) == float(le_b), f"{float(le_a):.5f} vs {float(le_b):.5f}")

        saved_cdp = float(cfm.estimator.class_dropout_prob)
        cfm.train()

        def avg_loss(mu_in, style_in):
            """多 seed 平均。deterministic=True 把 class_dropout_prob 临时置 0，
            走的正是生产代码里算 val loss 的那条路径。"""
            vals = []
            with torch.no_grad():
                for s in range(N_SEED):
                    torch.manual_seed(s)
                    v, _ = FW.cfm_training_forward(
                        cfm, x1.clone(), x_lens, p_lens,
                        mu_in.clone(), style_in.clone(), deterministic=True)
                    vals.append(float(v))
            return sum(vals) / len(vals), min(vals), max(vals)

        # 解析基线：模型如果什么都不会，最好的常量预测就是 0，
        # 此时 loss = mean|u|。用与 BASECFM.forward 完全相同的 RNG 调用顺序，
        # 所以同 seed 下 z / t 是一致的，两者可直接相减。
        sig = float(cfm.sigma_min)
        pl, xl = int(p_lens[0]), int(x_lens[0])
        triv = []
        for s in range(N_SEED):
            torch.manual_seed(s)
            t = torch.rand([1, 1, 1], device=dev, dtype=x1.dtype)
            z = torch.randn_like(x1)
            u = x1 - (1 - sig) * z
            triv.append(float(u[0, :, pl:xl].abs().mean()))
        loss_trivial = sum(triv) / len(triv)

        loss_c, lo_c, hi_c = avg_loss(mu, style)
        loss_z, lo_z, hi_z = avg_loss(torch.zeros_like(mu), torch.zeros_like(style))
        perm = torch.randperm(mu.size(1), generator=torch.Generator().manual_seed(11))
        loss_s, lo_s, hi_s = avg_loss(mu[:, perm].contiguous(), style)
        print(f"  有条件      loss = {loss_c:.5f}  (单 seed 区间 {lo_c:.4f}~{hi_c:.4f})")
        print(f"  mu/style=0  loss = {loss_z:.5f}  (单 seed 区间 {lo_z:.4f}~{hi_z:.4f})")
        print(f"  mu 帧序打乱 loss = {loss_s:.5f}  (单 seed 区间 {lo_s:.4f}~{hi_s:.4f})")
        print(f"  常量0 解析基线   = {loss_trivial:.5f}")
        check("CFM loss 是标量且有限", math.isfinite(loss_c), f"loss={loss_c:.5f}")
        check("有条件 loss 低于「mu/style 置零」基线",
              loss_c < loss_z * 0.97, f"{loss_c:.5f} vs {loss_z:.5f}"
              f"（比值 {loss_c / max(1e-9, loss_z):.3f}）")
        check("有条件 loss 低于「mu 帧序打乱」基线（证明逐帧对齐真的被用到）",
              loss_c < loss_s * 0.97, f"{loss_c:.5f} vs {loss_s:.5f}"
              f"（比值 {loss_c / max(1e-9, loss_s):.3f}）")
        check("有条件 loss 明显低于「什么都不会」的常量0基线",
              loss_c < loss_trivial * 0.85,
              f"{loss_c:.5f} vs {loss_trivial:.5f}"
              f"（降低 {(1 - loss_c / max(1e-9, loss_trivial)) * 100:.1f}%）")
        if hi_c - lo_c > 0.02:
            print(f"  → 单 seed 区间宽 {hi_c - lo_c:.4f}，所以必须多 seed 平均，"
                  "拿单次结果下结论会误判")

        # ---- 梯度：同样在 train() 模式下跑（训练器就是这么用的）----
        # 解冻整个 estimator（DiT 只 98M，fp32 梯度约 392 MB，8GB 卡装得下）。
        # 比“只解冻一个最小参数”可靠：最小参数可能正好在没走到的分支上，
        # 那样 grad 为 None 也说不清是接错了还是选错了参数。
        est_params = list(cfm.estimator.parameters())
        for p in est_params:
            p.requires_grad_(True)
        cfm.train()                       # class_dropout_prob=0.1 只在 training 下生效（CFG 必需）
        torch.manual_seed(0)
        # deterministic=False：训练时必须保留 CFG 随机丢弃，这里走生产路径
        loss_g, y_c = FW.cfm_training_forward(cfm, x1.clone(), x_lens, p_lens,
                                              mu.clone(), style.clone())
        check("deterministic=False 时 class_dropout_prob 保持原值（CFG 训练必需）",
              abs(float(cfm.estimator.class_dropout_prob) - saved_cdp) < 1e-9,
              f"{cfm.estimator.class_dropout_prob}")
        check("train() 模式下 loss 仍有限", bool(torch.isfinite(loss_g)),
              f"loss={float(loss_g):.5f}")
        check("forward 返回的 y 与 x1 同形", tuple(y_c.shape) == tuple(x1.shape),
              str(tuple(y_c.shape)))
        cfm.zero_grad(set_to_none=True)
        loss_g.backward()
        n_grad = sum(1 for p in est_params
                     if p.grad is not None and float(p.grad.abs().sum()) > 0)
        print(f"  有非零梯度的参数：{n_grad}/{len(est_params)}")
        check("梯度能回传到 estimator 的绝大部分参数",
              n_grad > len(est_params) * 0.8, f"{n_grad}/{len(est_params)}")
        # prompt 区域不计 loss：BASECFM.forward 里 y[:, :, :prompt_lens] = 0 且
        # criterion 只算 [prompt_lens:x_lens]，所以把 prompt_len 推到全长时应无可算区间
        for p in est_params:
            p.requires_grad_(False)
        cfm.zero_grad(set_to_none=True)
        cfm.eval()
        check("class_dropout_prob 已恢复",
              abs(float(cfm.estimator.class_dropout_prob) - saved_cdp) < 1e-9,
              f"{cfm.estimator.class_dropout_prob}")
        # 训练用的 seq_len 比推理入口的 8192 小得多，而 setup_caches 只增不减：
        # 不清的话这个 67 MB 级别的 causal_mask 会一直挂在卡上，
        # 而训练器退出后引擎还要接着做推理 / 评测，显存本来就紧。
        FW.invalidate_cfm_caches(cfm)
        check("训练结束后 invalidate：尺寸标记与 O(L²) 的 mask 都已归还",
              tr.max_seq_length == -1
              and tr.max_batch_size == -1
              and tr.freqs_cis is None
              and getattr(tr, "causal_mask", None) is None)
        del x1, mu, style, loss_g, y_c, est_params
        free()

    # ==================================================================
    head("[6] GPT(T2S) 训练前向：teacher forcing + 基线对比")
    gpt = tts.gpt

    # ---- 6a. 先用引擎自己生成的 codes 验证接线 ----
    # 这一步把「前向接线对不对」与「绕回特征匹不匹配」分开。
    # 模型自己采样出的 codes 在 teacher forcing 下 loss 必须很低，
    # 否则就是 gpt_training_forward 接错了。
    uniform = math.log(gpt.number_mel_codes)
    if cap.get("codes") is not None:
        c_codes = cap["codes"]
        if c_codes.ndim == 2:
            c_codes = c_codes[0]
        # 去掉末尾的 stop_mel_token（训练目标里 stop 由 forward 自己补）
        stop = int(gpt.stop_mel_token)
        nz = (c_codes == stop).nonzero(as_tuple=False)
        if nz.numel():
            c_codes = c_codes[:int(nz[0][0])]
        c_txt = cap["text"]
        if c_txt.ndim == 2:
            c_txt = c_txt[0]
        # 推理传进来的 text 已经带了一个尾部 stop（infer:725），剔掉
        while c_txt.numel() and int(c_txt[-1]) == int(gpt.stop_text_token):
            c_txt = c_txt[:-1]
        print(f"  [自发 codes] text={tuple(c_txt.shape)} codes={tuple(c_codes.shape)} "
              f"style={tuple(cap['style'].shape)} emo={tuple(cap['emo_vec'].shape)} "
              f"lang={cap['langs'].tolist()}")
        with torch.no_grad():
            fc = fwd(gpt,
                     cap["style"].to(dev), cap["emo_vec"].to(dev), cap["langs"].to(dev),
                     c_txt.unsqueeze(0).to(dev),
                     torch.LongTensor([c_txt.numel()]).to(dev),
                     c_codes.unsqueeze(0).to(dev),
                     torch.LongTensor([c_codes.numel()]).to(dev))
        loss_self = float(fc.mel_loss())
        loss_text = float(fc.text_loss())
        uniform_text = math.log(gpt.text_embedding.num_embeddings)
        print(f"  自发 codes 的 mel loss  = {loss_self:.4f}   均匀基线 = {uniform:.4f}")
        print(f"  同一次前向的 text loss = {loss_text:.4f}   均匀基线 = {uniform_text:.4f}")
        check("★ 接线验证：在模型自己生成的 codes 上 loss 远低于均匀基线",
              loss_self < uniform * 0.6,
              f"{loss_self:.4f} vs {uniform:.4f}（低 {(1 - loss_self / uniform) * 100:.1f}%）")
        check("text 分支的 loss 也远低于它自己的均匀基线（两个头都接对了）",
              loss_text < uniform_text * 0.6,
              f"{loss_text:.4f} vs {uniform_text:.4f}")

        # ---- mask：只保留「真实长度 + 1」个位置 ----
        # 那 +1 是「预测 stop」的位置，必须学；再往后的全是
        # set_*_padding 造出来的 stop，属于噪声目标。
        check("mel_mask 保留 n_codes+1 个位置",
              int(fc.mel_mask.sum()) == int(c_codes.numel()) + 1,
              f'{int(fc.mel_mask.sum())} vs {int(c_codes.numel()) + 1}')
        check("text_mask 保留 n_text+1 个位置",
              int(fc.text_mask.sum()) == int(c_txt.numel()) + 1,
              f'{int(fc.text_mask.sum())} vs {int(c_txt.numel()) + 1}')
        check("mask 是前缀连续的（前 True 后 False，中间不能空洞）",
              bool(fc.mel_mask[0, :int(fc.mel_mask.sum())].all())
              and not bool(fc.mel_mask[0, int(fc.mel_mask.sum()):].any()))
        check("logits / targets / mask 三者长度一致",
              fc.mel_logits.size(2) == fc.mel_targets.size(1) == fc.mel_mask.size(1)
              and fc.text_logits.size(2) == fc.text_targets.size(1) == fc.text_mask.size(1))
        check("GptForward.values() 能拿到可展示的 float",
              abs(fc.values()["mel_loss"] - loss_self) < 1e-6, str(fc.values()))

        # ---- lang_embedding 消融：官方 forward() 漏了它，生产前向补上了 ----
        # 不能直接拿官方 forward() 对账：它在 use_bf16=True 时会 RuntimeError。
        # 原因是 model_v2.py:633 的 `torch.zeros(b, 2, d).to(device)` 没指定 dtype，
        # 默认 fp32；torch.cat 会把 bf16 的条件一并提升成 fp32，然后撞上 bf16 的
        # LayerNorm 权重。先把这个陷阱本身钉死，再用「只改 lang_embedding」的
        # 同一份生产前向做消融 —— 变量唯一，结论才可靠。
        p_dtype = next(gpt.spk_emb_proj.parameters()).dtype
        if p_dtype != torch.float32:
            with torch.no_grad():
                spk_proj = gpt.spk_emb_proj(cap["style"].to(dev).to(p_dtype)).unsqueeze(1)
                try:
                    gpt.forward(spk_proj, c_txt.unsqueeze(0).to(dev),
                                torch.LongTensor([c_txt.numel()]).to(dev),
                                c_codes.unsqueeze(0).to(dev),
                                torch.LongTensor([c_codes.numel()]).to(dev),
                                None, emo_vec=cap["emo_vec"].to(dev).to(p_dtype))
                    broke = ""
                except RuntimeError as e:
                    broke = str(e).splitlines()[0][:80]
            print(f"  [陷阱复现] 官方 forward() 在 bf16 下 → {broke or '居然没报错'}")
            check("【官方陷阱】use_bf16 时 UnifiedVoice.forward() 直接 RuntimeError"
                  "（torch.zeros 未指定 dtype → cat 提升成 fp32 → 撞 bf16 LayerNorm）",
                  bool(broke), broke)

        with torch.no_grad():
            fnl = fwd(gpt,
                      cap["style"].to(dev), cap["emo_vec"].to(dev), cap["langs"].to(dev),
                      c_txt.unsqueeze(0).to(dev),
                      torch.LongTensor([c_txt.numel()]).to(dev),
                      c_codes.unsqueeze(0).to(dev),
                      torch.LongTensor([c_codes.numel()]).to(dev),
                      use_lang=False)
        loss_nolang = float(fnl.mel_loss())
        print(f"  去掉 lang_embedding 的 mel loss = {loss_nolang:.4f}"
              f"（带 lang = {loss_self:.4f}）")
        check("去掉 lang_embedding 后 targets / mask / 形状完全不变（消融只动了这一个变量）",
              fnl.mel_targets.shape == fc.mel_targets.shape
              and bool((fnl.mel_targets == fc.mel_targets).all())
              and bool((fnl.mel_mask == fc.mel_mask).all()))
        check("lang_embedding 确实参与前向（去掉它 loss 会变）",
              abs(loss_nolang - loss_self) > 1e-6,
              f"{loss_nolang:.6f} vs {loss_self:.6f}")

    # ---- 6b. 再用绕回提取的特征跑一次 ----
    s = FT.build_gpt_sample(feats[ids_list[0]])
    print(f"  text_tokens={tuple(s['text_tokens'].shape)} codes={tuple(s['codes'].shape)} "
          f"style={tuple(s['style'].shape)} emo_vec={tuple(s['emo_vec'].shape)}")
    check("build_gpt_sample 字段齐全",
          all(k in s for k in ("text_tokens", "codes", "style", "emo_vec", "lang_token")))

    dev0 = dev
    txt = s["text_tokens"].unsqueeze(0).to(dev0)
    codes = s["codes"].unsqueeze(0).to(dev0)
    style = s["style"].unsqueeze(0).to(dev0)
    emo = s["emo_vec"].unsqueeze(0).to(dev0)
    langs = torch.LongTensor([s["lang_token"]]).to(dev0)
    tl = torch.LongTensor([txt.size(1)]).to(dev0)
    cl = torch.LongTensor([codes.size(1)]).to(dev0)

    check("codes 长度 + 2 不超过 mel_pos_embedding 容量",
          codes.size(1) + 2 <= gpt.mel_pos_embedding.emb.num_embeddings,
          f'{codes.size(1) + 2} / {gpt.mel_pos_embedding.emb.num_embeddings}')
    check("text 长度 + 2 不超过 text_pos_embedding 容量",
          txt.size(1) + 2 <= gpt.text_pos_embedding.emb.num_embeddings,
          f'{txt.size(1) + 2} / {gpt.text_pos_embedding.emb.num_embeddings}')

    mh = gpt.mel_head
    for p in gpt.parameters():
        p.requires_grad_(False)
    mh.weight.requires_grad_(True)
    if mh.bias is not None:
        mh.bias.requires_grad_(True)

    with torch.no_grad():
        f0 = fwd(gpt, style, emo, langs, txt, tl, codes, cl)
    check("text_logits 形状 (B, V_text, L)",
          f0.text_logits.shape[1] == gpt.number_text_tokens + 1,
          str(tuple(f0.text_logits.shape)))
    check("mel_logits 形状 (B, 8194, L)",
          f0.mel_logits.shape[1] == gpt.number_mel_codes, str(tuple(f0.mel_logits.shape)))
    check("targets 与 logits 的 L 对齐",
          f0.text_targets.shape[1] == f0.text_logits.shape[2]
          and f0.mel_targets.shape[1] == f0.mel_logits.shape[2])
    check("logits 已转回 fp32（bf16 的 8 位尾数算 8194 类 log-softmax 会失真）",
          f0.mel_logits.dtype == torch.float32 and f0.text_logits.dtype == torch.float32,
          f'{f0.mel_logits.dtype} / {f0.text_logits.dtype}')

    m_loss = float(f0.mel_loss())
    t_loss = float(f0.text_loss())
    uniform = math.log(gpt.number_mel_codes)
    print(f"  mel loss = {m_loss:.4f}   均匀分布基线 ln(8194) = {uniform:.4f}")
    print(f"  text loss = {t_loss:.4f}  均匀分布基线 ln(60510) = {math.log(gpt.number_text_tokens + 1):.4f}")
    check("mel loss 有限", math.isfinite(m_loss))
    check("mel loss 远低于均匀基线（说明 codes/条件接线正确）",
          m_loss < uniform * 0.8, f"{m_loss:.4f} vs {uniform:.4f}")
    check("loss(text_weight=0.3) = mel + 0.3×text（双头加权真的生效）",
          abs(float(f0.loss(0.3)) - (m_loss + 0.3 * t_loss)) < 1e-4,
          f'{float(f0.loss(0.3)):.4f} vs {m_loss + 0.3 * t_loss:.4f}')

    # 打乱 codes：条件失效，loss 必须回升
    # 注意 codes 是 (1, L)，要用 codes[:, perm]；写成 codes[perm] 会广播成 (L, L)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(codes.size(1), generator=g)
    shuffled = codes[:, perm].contiguous()
    check("打乱后长度不变", shuffled.shape == codes.shape, str(tuple(shuffled.shape)))
    check("打乱确实改变了序列", not torch.equal(shuffled, codes))
    with torch.no_grad():
        f1 = fwd(gpt, style, emo, langs, txt, tl, shuffled, cl)
    m_loss_shuffled = float(f1.mel_loss())
    print(f"  打乱 codes 后 mel loss = {m_loss_shuffled:.4f}")
    check("打乱 codes 后 loss 明显回升（说明模型确实在用 codes 做 teacher forcing）",
          m_loss_shuffled > m_loss * 1.05,
          f"{m_loss_shuffled:.4f} vs {m_loss:.4f}")

    # 换一条不匹配的文本：loss 也应该回升
    s_other = FT.build_gpt_sample(feats[ids_list[1]])
    txt_o = s_other["text_tokens"].unsqueeze(0).to(dev0)
    with torch.no_grad():
        f2 = fwd(gpt, style, emo, langs, txt_o,
                 torch.LongTensor([txt_o.size(1)]).to(dev0), codes, cl)
    m_loss_mismatch = float(f2.mel_loss())
    print(f"  文本错配后 mel loss = {m_loss_mismatch:.4f}")
    check("文本错配后 loss 回升（说明文本条件真的在起作用）",
          m_loss_mismatch > m_loss, f"{m_loss_mismatch:.4f} vs {m_loss:.4f}")

    # lang_embedding 的作用：官方 forward() 漏了它，必须证明加上后结果不同。
    # 这里只比 text_emb 本身，不重算 conds —— 后者涉及 dtype 对齐，
    # 与「lang_embedding 有没有用」这个论点无关。
    with torch.no_grad():
        ti_probe, _ = gpt.build_aligned_inputs_and_targets(
            txt.clone(), gpt.start_text_token, gpt.stop_text_token)
        base_emb = gpt.text_embedding(ti_probe) + gpt.text_pos_embedding(ti_probe)
        with_lang = base_emb + gpt.lang_embedding(langs).unsqueeze(1)
    check("lang_embedding 确实改变了 text_emb（所以官方 forward 不能直接用于训练）",
          not torch.allclose(with_lang.float(), base_emb.float()),
          f"L1 差 = {float((with_lang.float() - base_emb.float()).abs().mean()):.5f}")
    check("lang_embedding 输出是 (B, dim)，批量版必须 unsqueeze(1) 才能广播",
          tuple(gpt.lang_embedding(langs).shape) == (1, 1280),
          str(tuple(gpt.lang_embedding(langs).shape)))

    # 反向传播（用生产代码的 loss，不再在探针里重算一遍交叉熵）
    fb = fwd(gpt, style, emo, langs, txt, tl, codes, cl)
    loss = fb.mel_loss()
    gpt.zero_grad(set_to_none=True)
    loss.backward()
    check("mel_head 收到非零梯度",
          mh.weight.grad is not None and float(mh.weight.grad.abs().sum()) > 0,
          f'‖grad‖ = {float(mh.weight.grad.abs().sum()):.3e}'
          if mh.weight.grad is not None else "None")
    n_grad = sum(1 for p in gpt.parameters() if p.grad is not None)
    check("只有被解冻的参数有梯度（底座确实是冻结的）", n_grad <= 2, f"{n_grad} 个")
    bad = GD.assert_base_frozen(gpt)
    check("assert_base_frozen 只报出上面故意解冻的 mel_head 两项",
          sorted(bad) == ["mel_head.bias", "mel_head.weight"], str(bad))
    mh.weight.requires_grad_(False)
    if mh.bias is not None:
        mh.bias.requires_grad_(False)
    gpt.zero_grad(set_to_none=True)
    free()
    print(f"  GPT 前向后显存 {vram_gb():.2f} GB")

    # ==================================================================
    head("[7] guard 的注入面扫描在真实模型上的结果")
    g_gpt = GD.scan_targets(gpt)
    print(GD.scan_targets_markdown(gpt))
    pats = {g.pattern for g in g_gpt}
    check("真实 GPT 里 attn.c_proj 与 mlp.c_proj 被区分开",
          "attn/c_proj" in pats and "mlp/c_proj" in pats, str(sorted(pats))[:300])
    n_conv1d = sum(g.count for g in g_gpt if g.kind == "Conv1D")
    check("Conv1D 层数 = 96（24 层 × 4）", n_conv1d == 96, f"{n_conv1d}")
    attn_only = [g for g in g_gpt if g.pattern in ("attn/c_attn", "attn/c_proj")]
    all_conv = [g for g in g_gpt if g.kind == "Conv1D"]
    est = GD.estimate_adapter_params(attn_only, GD.LoRAConfig(rank=16))
    est_all = GD.estimate_adapter_params(all_conv, GD.LoRAConfig(rank=16))
    # 手算基准：GPT2 的 Conv1D 每层 r×(in+out)，24 层×4 类
    #   attn.c_attn  1280→3840      attn.c_proj 1280→1280
    #   mlp.c_fc     1280→5120      mlp.c_proj  5120→1280
    want_attn = 24 * 16 * ((1280 + 3840) + (1280 + 1280))
    want_all = want_attn + 2 * 24 * 16 * (1280 + 5120)
    print(f"  仅注入注意力 + rank=16 的估算参数量：{est / 1e6:.2f} M（手算 {want_attn / 1e6:.2f} M）")
    print(f"  全部 96 层 Conv1D + rank=16：{est_all / 1e6:.2f} M（手算 {want_all / 1e6:.2f} M）")
    check("仅注意力的估算值等于手算基准", est == want_attn, f"{est} vs {want_attn}")
    check("全 Conv1D 的估算值等于手算基准（也就是 lora_probe 实测的 7.86M）",
          est_all == want_all, f"{est_all} vs {want_all}")
    check("rank 翻倍 → 参数量翻倍（估算与 rank 线性）",
          GD.estimate_adapter_params(attn_only, GD.LoRAConfig(rank=32)) == 2 * est)

    g_cfm = GD.scan_targets(tts.s2mel.models["cfm"].estimator)
    n_lin = sum(g.count for g in g_cfm if g.kind == "Linear")
    check("CFM/DiT 的 Linear 层数 > 100", n_lin > 100, f"{n_lin}")
    est_c = GD.estimate_adapter_params(g_cfm, GD.LoRAConfig(rank=16))
    print(f"  CFM 全 Linear + rank=16 的估算参数量：{est_c / 1e6:.2f} M")

    # ==================================================================
    head("[8] 断点续提 / 校验 / 统计")
    res2 = FT.extract_dataset(DS_NAME, tts=tts, device=dev)
    check("第二次跑全部跳过（断点续提生效）",
          res2["skipped"] == 3 and res2["extracted"] == 0, f"{res2['skipped']}/{res2['extracted']}")
    res3 = FT.extract_dataset(DS_NAME, tts=tts, device=dev, overwrite=True)
    check("overwrite=True 时重新提取", res3["extracted"] == 3, f"{res3['extracted']}")

    v = FT.verify_dataset(DS_NAME)
    check("verify_dataset 全部有效", v["ok"] and v["valid"] == 3, str(v))

    # 故意破坏一个特征文件，看校验能不能抓到
    bad_path = FT.FeatureExtractor.feature_path(ds_dir, ids_list[0])
    d = torch.load(bad_path, map_location="cpu", weights_only=False)
    d["mu_prompt"] = d["mu_prompt"][:-3]
    torch.save(d, bad_path)
    v2 = FT.verify_dataset(DS_NAME)
    check("mu 帧数被改短后 verify 能抓到", not v2["ok"] and len(v2["invalid"]) == 1,
          str(v2["invalid"]))
    d["feature_version"] = 999
    torch.save(d, bad_path)
    v3 = FT.verify_dataset(DS_NAME)
    check("版本号不匹配也能抓到", any("特征版本" in m for _, m in v3["invalid"]),
          str(v3["invalid"]))
    check("版本不匹配的缓存被判定为不可用", not FT.is_usable(bad_path))
    FT.extract_dataset(DS_NAME, tts=tts, device=dev)      # 重新生成被破坏的那条
    check("破坏后重跑能自动修复", FT.verify_dataset(DS_NAME)["ok"])

    s = FT.stats(DS_NAME)
    print(f"  stats: {s}")
    check("stats 统计正确", s["ready_with_features"] == 3 and s["gb"] > 0, str(s))
    md = FT.stats_markdown(DS_NAME)
    check("stats_markdown 可渲染", "已提取特征" in md and "3 / 3" in md, md[:200])
    fsr = FT.feature_size_report(DS_NAME)
    check("feature_size_report 列出全部字段", "mu_target" in fsr and "KB" in fsr,
          f"{len(fsr)} 字符")

    split = DS.make_split(DS_NAME, val_ratio=0.34, seed=42, min_val=1)
    sp = DS.load_split(DS_NAME)
    check("make_split 成功，且返回的是**条数**而不是 id 列表",
          split.get("ok") is True and isinstance(split["train"], int)
          and split["train"] >= 1, str(split))
    check("train + val == ready 总数（没漏也没重）",
          split["train"] + split["val"] == split["total_ready"], str(split))
    check("划分结果已落盘，且与返回的条数一致",
          len(sp.get("train", [])) == split["train"]
          and len(sp.get("val", [])) == split["val"], str(sp)[:120])
    check("train / val 没有交集（泄露会让 val loss 变成训练集指标）",
          not (set(sp.get("train", [])) & set(sp.get("val", []))))
    check("同 seed 重跑得到同一份划分（可复现）",
          DS.make_split(DS_NAME, val_ratio=0.34, seed=42, min_val=1) == split)

    # ==================================================================
    head("[9] LoRA 注入 + 两个会让训练静默失效的开关（真实 GPT）")
    # 这一段放在最后：PEFT 会**原地**把 Conv1D 换成 lora 包装层，
    # 而 [7] 的 scan_targets 需要看到未被改写的原始结构。
    from peft import LoraConfig, get_peft_model

    regex = GD.build_target_regex(("attn/c_attn", "attn/c_proj"))
    print(f"  注入面正则：{regex}")
    lc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                    target_modules=regex, bias="none", task_type=None)
    pm = get_peft_model(gpt, lc)
    n_lora = sum(1 for _, m in FW.unwrap(pm).named_modules()
                 if hasattr(m, "lora_A") and hasattr(m, "scaling"))
    cnt = GD.count_params(pm)
    print(f"  注入 {n_lora} 个 LoRA 层，可训练 {cnt['trainable']/1e6:.2f} M "
          f"/ 共 {cnt['total']/1e6:.0f} M（{cnt['trainable_pct']:.3f}%）")
    check("PEFT 能识别 HF GPT2 的 Conv1D 并注入（attn 共 48 层）", n_lora == 48, f"{n_lora}")
    check("unwrap() 能从 PEFT 包装里取回 UnifiedVoice",
          FW.unwrap(pm) is not pm and hasattr(FW.unwrap(pm), "spk_emb_proj"))
    check("可训练参数占比 < 0.5%（底座确实被冻住了）",
          cnt["trainable_pct"] < 0.5, f"{cnt['trainable_pct']:.3f}%")

    # ---- 陷阱 A：reentrant 梯度检查点 + 全冻结底座 ----
    # config.yaml 里没有 `checkpointing` 项 → UnifiedVoice 默认 True →
    # GPT2Config(gradient_checkpointing=True)。而 transformers 4.52 的
    # gradient_checkpointing_enable() 在 kwargs=None 时默认 use_reentrant=True，
    # reentrant 版要求至少一个**输入** requires_grad。我们的 inputs_embeds
    # 来自冻结的 embedding，一个都不需要梯度 → backward 直接报错或什么都不回传。
    lora_params = [p for n, p in pm.named_parameters() if p.requires_grad]
    inner = FW.unwrap(pm).gpt

    def lora_grad_norm() -> float:
        tot = 0.0
        for p in lora_params:
            if p.grad is not None:
                tot += float(p.grad.abs().sum())
        return tot

    def one_step() -> str:
        """跑一次 forward+backward，返回 'ok' 或异常描述。

        捕 Exception 而不只 RuntimeError：reentrant 检查点在不同 torch 版本下
        可能报 RuntimeError、也可能静默地什么都不回传，两种都是训练失效。
        """
        pm.zero_grad(set_to_none=True)
        try:
            f = fwd(pm, style, emo, langs, txt, tl, codes, cl)
            f.mel_loss().backward()
        except Exception as e:
            return f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"
        return "ok"

    # 先手工把检查点推回官方的 reentrant=True，复现陷阱
    prev_state = FW.configure_gpt_for_training(pm, grad_checkpointing=True, base_dropout=0.0)
    inner.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True})
    bad_step = one_step()
    gn_bad = lora_grad_norm()
    print(f"  [陷阱复现] reentrant=True → {bad_step}，LoRA 梯度总量 = {gn_bad:.3e}")
    check("【陷阱】reentrant 梯度检查点下 LoRA 拿不到梯度（报错或梯度为 0）",
          bad_step != "ok" or gn_bad == 0.0, f"{bad_step} / grad={gn_bad:.3e}")

    # 再用生产代码的配置（use_reentrant=False）跑同一步
    inner.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    ok_step = one_step()
    gn_ok = lora_grad_norm()
    n_with_grad = sum(1 for p in lora_params if p.grad is not None)
    print(f"  configure(use_reentrant=False) → {ok_step}，"
          f"LoRA 梯度总量 = {gn_ok:.3e}（{n_with_grad}/{len(lora_params)} 个参数有梯度）")
    check("换成 use_reentrant=False 后 LoRA 真的拿到非零梯度",
          ok_step == "ok" and gn_ok > 0.0, f"{ok_step} / grad={gn_ok:.3e}")
    check("全部 LoRA 参数都有梯度（不是只有部分层被回传）",
          n_with_grad == len(lora_params), f"{n_with_grad}/{len(lora_params)}")

    # ---- 陷阱 B：底座 dropout ----
    check("configure_gpt_for_training 把底座 dropout 置 0",
          all(m.p == 0.0 for m in inner.modules() if type(m).__name__ == "Dropout")
          and float(inner.config.attn_pdrop) == 0.0,
          f"config.attn_pdrop={inner.config.attn_pdrop}")
    check("底座参数全部冻结（guard.assert_base_frozen）",
          GD.assert_base_frozen(pm) == [], str(GD.assert_base_frozen(pm)[:3]))

    FW.restore_gpt(pm, prev_state)
    check("restore_gpt 把 dropout 恢复回 0.1",
          float(inner.config.attn_pdrop) == 0.1
          and all(abs(m.p - 0.1) < 1e-9 for m in inner.modules()
                  if type(m).__name__ == "Dropout"),
          f"config.attn_pdrop={inner.config.attn_pdrop}")
    check("restore_gpt 把模型切回 eval（推理需要）", not FW.unwrap(pm).training)
    del pm, lora_params
    free()
    print(f"  LoRA 试验后显存 {vram_gb():.2f} GB")

    # ==================================================================
    head("[10] 清理")
    # main() 里一路攒下的局部引用（tts / gpt / cfm / 六个 GptForward）会把
    # 整个模型钉在显存里：eng.unload() 只能删引擎自己那份引用。
    # 不先把这些置 None，下面的「显存回落」永远不可能通过 ——
    # 而这恰恰是真实训练器退出时必须做对的事（否则下一次推理就溢出了）。
    tts = gpt = cfm = est = tr = mh = None
    f0 = f1 = f2 = fb = fc = fnl = None
    pair = self_pair = feats = cap = None
    # inner 是 GPT2Model 本体（占了 813M 里的 ~800M），不置 None 等于什么都没释放
    inner = prev_state = None
    # orig_speak 是 UnifiedVoice 的**绑定方法**，持有实例引用；
    # spy 的闭包又持有 orig_speak。两个不清，整个 GPT 就一直在卡上。
    orig_speak = spy = None
    eng.unload()
    # gc 必须在 unload **之后**：之前那些引用还在时，循环引用对 gc 而言是可达的，
    # 根本不会被回收。这也是真实训练器释放模型时必须遵守的顺序。
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    check("引擎卸载后显存回落", vram_gb() < 1.0, f"{vram_gb():.2f} GB")
    if vram_gb() >= 1.0:
        # 直接把还活着的 CUDA 张量按大小列出来 —— 猜不如看。
        # 这一步很重要：训练器要自己再加载一份 GPT，
        # 卸载漏掉的每一 GB 都会直接变成训练时的 WDDM 溢出。
        import gc as _gc
        live = {}
        for o in _gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda and o.numel() * o.element_size() > 4e6:
                    key = (tuple(o.shape), str(o.dtype))
                    live[key] = live.get(key, 0) + o.numel() * o.element_size()
            except ReferenceError:
                continue
        top = sorted(live.items(), key=lambda kv: -kv[1])[:12]
        print("  仍活着的 CUDA 张量（>4MB，按形状聚合）：")
        for (shp, dt), nb in top:
            print(f"    {nb / 1e6:8.1f} MB  {str(shp):28s} {dt}")
    DS.delete(DS_NAME)
    check("临时数据集已删除", not DS.exists(DS_NAME))
    for _, p, _ in made:
        try:
            os.remove(p)
        except OSError:
            pass
    check("临时音频已删除", not any(os.path.isfile(p) for _, p, _ in made))

    print("\n" + "=" * 70)
    print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
    if FAIL:
        print("  失败项：")
        for f in FAIL:
            print(f"    FAIL {f}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
