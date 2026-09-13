"""L1 · 离线特征预提取。

训练时如果每步都现场跑 w2v-BERT + codec + CAMPPlus + length_regulator，
8GB 显存根本装不下（光 w2v-BERT 前向的激活就吃掉一大块），而且慢得没法用。
所以这里把每条样本的特征**一次性算好存盘**，训练时只读张量。

============================================================================
必须与推理路径逐行对齐的地方（都已对照 infer_v2_5.py 核实）
============================================================================

1. 音频加载用 `librosa.load(path)`（**默认 sr=22050**），不是原始采样率。
   官方 `_load_and_cut_audio` 就是这么写的，后面的 Resample(22050→22050) 是恒等。

2. w2v-BERT 取 `hidden_states[17]`，再做 `(feat - mean) / std`
   （mean/std 来自 checkpoints 里的 w2v_stat）。帧率 50Hz，维度 1024。

3. 语义 codes = `semantic_codec.quantize(spk_cond_emb)`，帧率 25Hz。
   注意 `decode()` 返回 **(B, T, D)** 通道在后 —— 官方 docstring 写的
   `[B, D, T]` 是错的，按 docstring 写会静默算错长度。

4. mel = `mel_fn(audio_22k)`，返回 **(B, 80, T)**，帧率 22050/256 ≈ 86.13Hz。

5. **官方的 prompt / target 条件是不对称的**，这是最容易写错的一点：
       prompt 区：length_regulator(**原始** spk_cond_emb)        ← infer:650
       target 区：length_regulator(**decode(codes)** )            ← infer:831-837
   推理时 target 的条件来自 GPT 生成的 codes，经过 quantize→decode 是**有损**的。
   训练时如果拿原始特征当 target 条件，模型见到的条件比推理时干净，
   上线就会掉质量。所以两种都要缓存：`mu_prompt` 与 `mu_target`。

6. ylens 用**真实 mel 帧数**。推理时用 `int(S.shape[1] * 1.72)` 估算，
   实测会差 ±1 帧（探针里 688 vs 689）；训练时长度已知，没必要估算。
   注意 1.72 是 50Hz→86.13Hz 的系数，而 codes 是 25Hz，
   所以 codes→mel 的换算要再乘 2（见 MEL_PER_CODE）。

7. emo_vec 用 `gpt.get_emovec(spk_cond_emb, cond_lengths)`，其中
   `cond_lengths = spk_cond_emb.shape[-1]`（=1024，**不是**帧数）。
   这是官方 inference_speech 的实际传法（infer:775-776 传的也是 shape[-1]），
   虽然看着像 bug，但因为 1024 > 任何 ≤20s 音频的帧数，mask 恒为全 1，
   行为等价且**必须保持一致**——改成真长度反而会让 conformer 的 padding
   计算路径与推理不同。

8. 文本编码完整复刻 infer:699-724 的六步链路（含发音标注展开）。

9. **只有 GPT 是 bf16**。infer:142-143 在 use_bf16 时只把 `self.gpt` 整体
   `.bfloat16()`，w2v-BERT / codec / campplus / s2mel / bigvgan 全是 fp32。
   官方调 `merge_emovec` 时包在 autocast 里（infer:757-764）所以不会报错；
   这里直接喂 fp32 的 spk_cond_emb 给 bf16 的 GPT 会抛
   `Input type (float) and bias type (c10::BFloat16) should be the same`。
   所以 get_emovec 必须同样包 autocast，存盘前再转回 fp32。
   → **训练器里凡是碰到 gpt 的子模块（spk_emb_proj / emo_layer / text_embedding …）
     都得在 autocast 下跑**，否则同样报错。

============================================================================
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import dataset as DS
from webui_app.training.dataset import HARD_MAX_SEC, MAX_TRAIN_SEC

# 特征格式版本：字段增删或算法改动时 +1，旧缓存会被判定为失效并重算。
# 不做版本校验的话，改了提取逻辑而旧 .pt 还在，训练会静默用到错特征。
FEATURE_VERSION = 1

FEATURE_SUBDIR = DS.FEATURE_SUBDIR

# 每条样本缓存的字段 → (dtype 说明, 形状说明)
FEATURE_SCHEMA: Dict[str, str] = {
    "text_tokens": "int64 (L,)      文本 token，含 <|zh|> 前缀，不含 stop",
    "codes":       "int64 (T25,)    语义 token，25Hz，GPT 的训练目标",
    "style":       "fp32 (192,)     CAMPPlus 声纹，经 spk_emb_proj 后作为说话人条件",
    "emo_vec":     "fp32 (1280,)    情感向量，get_emovec 的输出（底座冻结模块算的）",
    "mel":         "fp16 (80, Tm)   目标 mel，CFM 的 x1",
    "mu_prompt":   "fp16 (Tm, 512)  length_regulator(原始 w2v-bert) —— 当参考音频用",
    "mu_target":   "fp16 (Tm, 512)  length_regulator(decode(codes)) —— 当训练目标用",
}

META_KEYS = ("id", "text", "lang", "lang_token", "duration", "mel_len",
             "n_codes", "n_text_tokens", "feature_version", "extract_seconds",
             "extracted_at", "source", "warnings")

# mel 帧率 = 22050 / hop(256)
MEL_FPS = 22050.0 / 256.0
CODE_FPS = 25.0
W2V_FPS = 50.0

# 帧率换算链（已实测核实）：
#   codes(25Hz) --semantic_codec.decode--> S(50Hz) --length_regulator--> mel(86.13Hz)
# 官方 infer:832 的 `int(S.shape[1] * 1.72)` 里的 1.72 是 **50Hz→86.13Hz** 这一段，
# 不是 codes→mel。把它当成 codes→mel 的系数会整整差一倍。
CODE_TO_W2V = 2                # semantic_codec 的 downsample_scale=2
W2V_TO_MEL = 1.72              # 官方硬编码的系数
MEL_PER_CODE = CODE_TO_W2V * W2V_TO_MEL       # ≈ 3.44 = 86.13 / 25


# ---------------------------------------------------------------------------
# 文本编码（复刻 infer_v2_5.py:699-724）
# ---------------------------------------------------------------------------

def encode_text(tts, text: str, lang: str = "ZH",
                text_normalization: bool = True) -> Tuple[List[int], List[str]]:
    """把文本编码成 GPT 的输入 token，返回 (token_ids, warnings)。

    与官方推理**逐行一致**，包括：
        · clean_pattern 全角标点替换
        · 中/英/中英混走 TextNormalizer，日/西走 nemo_text_normalize
        · 中日英转小写、西语转大写
        · apply_pronunciation_annotations 展开 `<字|拼音>`
        · 日语额外的 ja_text_process
        · 特殊 token 名统一大写
    """
    warns: List[str] = []
    lang = (lang or "ZH").upper()
    low = lang.lower()
    lang_prefix = f"<|{low}|> "

    from indextts.infer_v2_5 import apply_pronunciation_annotations

    t = text or ""
    t = tts.text_process.clean_pattern.sub(
        lambda x: tts.text_process.char_rep_map[x.group()], t)

    if text_normalization:
        if low in ("zh", "zhen", "en"):
            t = tts.text_process.normalize(t)
        elif low in ("ja", "es"):
            try:
                from indextts.utils.nemo_tn import nemo_text_normalize
                t = nemo_text_normalize(t, low)
            except Exception as e:
                warns.append(f"{low} 的 nemo 文本归一化不可用（{type(e).__name__}），"
                             "已跳过归一化。训练与推理会因此不一致，建议改用 ZH/EN")

    if low in ("ja", "zh", "zhen", "en"):
        t = t.lower()
    if low == "es":
        t = t.upper()

    t = apply_pronunciation_annotations(t)

    if low == "ja":
        try:
            t = tts.ja_text_process.process_ja_text(t)
        except Exception as e:
            warns.append(f"日语文本处理失败（{type(e).__name__}: {e}），已跳过")

    t = re.sub(r"<\|([^|]+)\|>", lambda m: f"<|{m.group(1).upper()}|>", t)

    ids = tts.tokenizer.encode(lang_prefix + t, allowed_special="all")
    return list(ids), warns


# ---------------------------------------------------------------------------
# 提取器
# ---------------------------------------------------------------------------

@dataclass
class ExtractResult:
    """一条样本的提取结果。"""
    id: str
    ok: bool
    path: str = ""
    seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor:
    """按需提取单条/整个数据集的特征。

    **优先复用已加载的 IndexTTS2 实例**（engine 里的 `eng.tts`）：
    那里面 semantic_model / semantic_codec / campplus_model / s2mel / gpt /
    mel_fn / tokenizer / text_process 全都有，重新加载一遍既要 3GB 显存
    又要 20 秒，还可能和推理引擎抢显存导致 OOM。

    没给 tts 时才自己建一个（`own_tts=True`，用完可以 unload）。
    """

    def __init__(self, tts=None, model_dir: Optional[str] = None,
                 device: Optional[str] = None, verbose: bool = False):
        self._tts = tts
        self._own = tts is None
        self.model_dir = model_dir or os.path.join(PROJECT_ROOT, "checkpoints")
        self.device = device
        self.verbose = verbose

    # ---------------- 生命周期 ----------------
    @property
    def tts(self):
        if self._tts is None:
            self._build()
        return self._tts

    def _build(self):
        from indextts.infer_v2_5 import IndexTTS2
        self._tts = IndexTTS2(model_dir=self.model_dir, cfg_path=os.path.join(
            self.model_dir, "config.yaml"), verbose=self.verbose)
        if self.device is None:
            self.device = str(getattr(self._tts, "device", "cpu"))

    def unload(self):
        """只有自己建的实例才卸载 —— 复用 engine 的不能动。"""
        if self._own and self._tts is not None:
            self._tts = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ---------------- 单条提取 ----------------
    def extract(self, audio_path: str, text: str, lang: str = "ZH",
                text_normalization: bool = True,
                store_w2v_feat: bool = False) -> Dict[str, Any]:
        """提取一条样本的全部特征。返回可直接 torch.save 的 dict。

        抛异常表示这条样本没法用（音频损坏、超长等），由调用方决定跳过还是中止。
        """
        import torch
        import torchaudio
        import librosa

        tts = self.tts
        dev = self.device or str(getattr(tts, "device", "cpu"))
        warns: List[str] = []
        t0 = time.perf_counter()

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频不存在：{audio_path}")

        # ---- 1. 音频加载（对齐官方 _load_and_cut_audio：librosa 默认 22050）----
        audio, sr = librosa.load(audio_path)
        audio = torch.tensor(audio).unsqueeze(0)          # (1, N)
        duration = audio.shape[1] / float(sr)
        if duration > HARD_MAX_SEC:
            raise ValueError(
                f"音频 {duration:.1f}s 超过架构硬上限 {HARD_MAX_SEC:.0f}s"
                f"（max_mel_tokens=1815 ÷ 25Hz），位置编码装不下")
        if duration > MAX_TRAIN_SEC:
            warns.append(f"时长 {duration:.1f}s 超过推荐上限 {MAX_TRAIN_SEC:.0f}s，"
                         "CFM 注意力开销随帧数平方增长，8GB 显存下可能 OOM")

        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        with torch.no_grad():
            # ---- 2. w2v-BERT 特征（50Hz, 1024）----
            inputs = tts.extract_features(audio_16k, sampling_rate=16000,
                                          return_tensors="pt")
            spk_cond_emb = tts.get_emb(
                inputs["input_features"].to(dev),
                inputs["attention_mask"].to(dev))          # (1, T50, 1024)

            # ---- 3. 语义 codes（25Hz）----
            codes = tts.get_scode(spk_cond_emb)            # (1, T25)
            codes = codes.squeeze(0).to(torch.int64).cpu()

            # ---- 4. CAMPPlus 声纹（192）----
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(dev), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = tts.campplus_model(feat.unsqueeze(0)).squeeze(0).float().cpu()

            # ---- 5. mel（80, Tm）----
            mel = tts.mel_fn(audio_22k.to(dev).float())    # (1, 80, Tm)
            mel_len = int(mel.size(2))
            mel = mel.squeeze(0).to(torch.float16).cpu()

            # ---- 6. mu：prompt 与 target 两条（见模块 docstring 第 5 条）----
            ylens = torch.LongTensor([mel_len]).to(dev)
            mu_prompt = tts.s2mel.models["length_regulator"](
                spk_cond_emb, ylens=ylens, n_quantizers=3, f0=None)[0]
            S_q = tts.semantic_codec.decode(codes.unsqueeze(0).to(dev))
            mu_target = tts.s2mel.models["length_regulator"](
                S_q, ylens=ylens, n_quantizers=3, f0=None)[0]

            # ---- 7. emo_vec（1280）----
            #    cond_lengths 传 shape[-1]（=1024）而不是帧数，与官方一致。
            #    GPT 在 use_bf16 时被整体 .bfloat16()（infer:142-143），
            #    官方是在 autocast 里调 merge_emovec 的（infer:757-764），
            #    这里必须照做，否则 fp32 输入喂 bf16 层直接 RuntimeError。
            dev_type = torch.device(dev).type
            tts_dtype = getattr(tts, "dtype", None)
            with torch.amp.autocast(dev_type, enabled=tts_dtype is not None,
                                    dtype=tts_dtype):
                emo_vec = tts.gpt.get_emovec(
                    spk_cond_emb,
                    torch.tensor([spk_cond_emb.shape[-1]], device=dev)).squeeze(0)
            emo_vec = emo_vec.float().cpu()      # 存盘统一 fp32，与 dtype 设置解耦

            # ---- 8. 文本 token ----
            toks, tw = encode_text(tts, text, lang, text_normalization)
            warns += tw

        from indextts.utils.tokenizer import lang_to_token

        out: Dict[str, Any] = {
            "text_tokens": torch.tensor(toks, dtype=torch.int64),
            "codes": codes,
            "style": style,
            "emo_vec": emo_vec,
            "mel": mel,
            "mu_prompt": mu_prompt.squeeze(0).to(torch.float16).cpu(),
            "mu_target": mu_target.squeeze(0).to(torch.float16).cpu(),
            # ---- 元信息（纯 python 类型，方便 json 化与人工检查）----
            "text": text or "",
            "lang": (lang or "ZH").upper(),
            "lang_token": int(lang_to_token(lang or "ZH")),
            "duration": round(float(duration), 4),
            "sample_rate": int(sr),
            "mel_len": mel_len,
            "n_codes": int(codes.shape[0]),
            "n_text_tokens": int(len(toks)),
            "feature_version": FEATURE_VERSION,
            "extract_seconds": round(time.perf_counter() - t0, 3),
            "extracted_at": time.time(),
            "source": os.path.basename(audio_path),
            "warnings": warns,
        }
        if store_w2v_feat:
            # 只在调试时开：w2v-bert 特征是最大的一块（100KB/秒），
            # 正常训练用不到 —— emo_vec 和 mu_prompt 都已经从它算出来了
            out["spk_cond_emb"] = spk_cond_emb.squeeze(0).to(torch.float16).cpu()
        return out

    # ---------------- 落盘 ----------------
    @staticmethod
    def feature_path(ds_dir: str, uid: str) -> str:
        return os.path.join(ds_dir, FEATURE_SUBDIR, f"{uid}.pt")

    def extract_to(self, ds_dir: str, uid: str, audio_path: str, text: str,
                   lang: str = "ZH", text_normalization: bool = True,
                   overwrite: bool = False) -> ExtractResult:
        """提取一条并存盘。已存在且版本匹配时跳过（断点续提）。"""
        import torch
        path = self.feature_path(ds_dir, uid)
        if not overwrite and is_usable(path):
            return ExtractResult(id=uid, ok=True, path=path, seconds=0.0,
                                 warnings=["已存在，跳过"], stats={"skipped": True})
        t0 = time.perf_counter()
        try:
            feat = self.extract(audio_path, text, lang, text_normalization)
        except Exception as e:
            return ExtractResult(id=uid, ok=False, error=f"{type(e).__name__}: {e}",
                                 seconds=time.perf_counter() - t0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(feat, tmp)
        os.replace(tmp, path)          # 原子替换，中断不会留下半个 .pt
        return ExtractResult(
            id=uid, ok=True, path=path, seconds=time.perf_counter() - t0,
            warnings=feat.get("warnings") or [],
            stats={"mel_len": feat["mel_len"], "n_codes": feat["n_codes"],
                   "n_text_tokens": feat["n_text_tokens"],
                   "lang_token": feat["lang_token"],
                   "duration": feat["duration"],
                   "bytes": os.path.getsize(path)})


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def is_usable(path: str) -> bool:
    """缓存文件是否存在且版本匹配（不加载张量，只看能不能用）。"""
    if not os.path.isfile(path):
        return False
    try:
        import torch
        # weights_only=False：我们存的是 tensor + 基本类型，没有自定义对象，
        # 但 torch 2.6+ 默认 weights_only=True 会对某些容器报错，显式关掉更稳
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return int(d.get("feature_version", -1)) == FEATURE_VERSION \
        and all(k in d for k in FEATURE_SCHEMA)


def check_feature(feat: Dict[str, Any]) -> List[str]:
    """一致性自检。返回问题列表（空 = 没问题）。

    这几条不变式一旦破了，训练会**静默**学到错的东西，所以每次提取后都查：
        · mu 的帧数必须等于 mel 帧数（CFM 里 mu 与 x1 是逐帧对应的）
        · codes 帧数 ≈ mel 帧数 / 1.72（25Hz → 86.13Hz）
        · style 192 维、emo_vec 1280 维（架构常量）
        · mel 80 通道
    """
    p: List[str] = []
    try:
        mel_len = int(feat["mel_len"])
        if tuple(feat["mel"].shape) != (80, mel_len):
            p.append(f"mel 形状 {tuple(feat['mel'].shape)} 与 mel_len={mel_len} 不符")
        for k in ("mu_prompt", "mu_target"):
            if feat[k].shape[0] != mel_len:
                p.append(f"{k} 帧数 {feat[k].shape[0]} ≠ mel_len {mel_len}")
            if feat[k].shape[1] != 512:
                p.append(f"{k} 维度 {feat[k].shape[1]} ≠ 512")
        if feat["style"].shape[0] != 192:
            p.append(f"style 维度 {feat['style'].shape[0]} ≠ 192")
        if feat["emo_vec"].shape[0] != 1280:
            p.append(f"emo_vec 维度 {feat['emo_vec'].shape[0]} ≠ 1280")
        n_codes = int(feat["n_codes"])
        expect = n_codes * MEL_PER_CODE
        if n_codes and abs(n_codes * MEL_PER_CODE - mel_len) / max(1.0, expect) > 0.05:
            p.append(f"codes {n_codes} 帧换算过来应是 {expect:.0f} mel 帧"
                     f"（×{MEL_PER_CODE:.2f}），实际 {mel_len}，偏差 >5%"
                     "（音频可能被截断或采样率不对）")
        if int(feat["n_text_tokens"]) < 2:
            p.append(f"文本只有 {feat['n_text_tokens']} 个 token，可能是空文本")
        if n_codes > 1815:
            p.append(f"codes {n_codes} 超过 max_mel_tokens=1815")
        if feat["mu_prompt"].shape != feat["mu_target"].shape:
            p.append("mu_prompt 与 mu_target 形状不一致")
    except Exception as e:
        p.append(f"自检异常：{type(e).__name__}: {e}")
    return p


# ---------------------------------------------------------------------------
# 数据集级批处理
# ---------------------------------------------------------------------------

def _apply_meta(name: str, touched: Dict[str, Dict[str, Any]]) -> int:
    """一次性把多个样本的字段回写 meta.jsonl。

    不用 DS.update()：那个每次都会 load+save 整个文件，在循环里调就是 O(n²)。
    """
    if not touched:
        return 0
    items = DS.load_meta(name)
    n = 0
    for u in items:
        fields = touched.get(u.id)
        if not fields:
            continue
        for k, v in fields.items():
            if hasattr(u, k):
                setattr(u, k, v)
        n += 1
    if n:
        DS.save_meta(name, items)
    return n


def extract_dataset(name: str, overwrite: bool = False, only: str = "ready",
                    text_normalization: bool = True,
                    lang_default: str = "ZH",
                    progress: Optional[Callable[[float, str], None]] = None,
                    should_stop: Optional[Callable[[], bool]] = None,
                    tts=None, device: Optional[str] = None,
                    ) -> Dict[str, Any]:
    """给整个数据集预提取特征。

    only: "ready" = 只处理**音频体检通过**的；"all" = 连有问题的也试（会记录失败）
    支持中断（should_stop）与断点续提（已存在的跳过）。

    注意 `only="ready"` 筛的是 `status in (ready, no_features)`，而不是只看
    `status == "ready"`：数据集页体检时会带 require_features=True，没特征的
    样本会被标成 `no_features`。如果只认 `ready`，就变成「要先有特征才能
    变 ready，要先 ready 才会提特征」的死循环，首次提取永远什么都不会做。
    """
    ds_dir = DS.dir_of(name)
    if not os.path.isdir(ds_dir):
        return {"ok": False, "error": f"数据集 `{name}` 不存在"}

    # 开工前先看一眼整卡显存。提取本身不占多少，但引擎已常驻 ~5.7 GB，
    # 如果其他进程又占了一两 GB，WDDM 会静默溢出到系统内存，
    # 表现为「提取慢 20 倍但不报错」—— 很难自己想到原因。
    from webui_app.training import guard as GD
    vram = GD.vram_headroom(0.0)
    vram_warn = ""
    if vram.total_gb > 0 and vram.free_gb < 1.5:
        vram_warn = (f"整卡仅剩 {vram.free_gb:.2f} GB 空闲（已被占 {vram.used_gb:.2f} GB）。"
                     "Windows 下显存不够不会 OOM，而是静默溢出到系统内存，"
                     "提取会慢 20 倍以上。建议先关掉占显存的应用。")

    us = DS.load_meta(name)
    if only == "ready":
        us = [u for u in us if u.status in ("ready", "no_features")]
    todo = [u for u in us
            if overwrite or not is_usable(FeatureExtractor.feature_path(ds_dir, u.id))]

    ex = FeatureExtractor(tts=tts, device=device)
    results: List[ExtractResult] = []
    n_ok = n_fail = 0
    # 断点续提的跳过数 = 被 `todo` 筛掉的那些。它们在进循环前就被挡下了
    # （不用加载 .pt，比进 extract_to 再判一次便宜得多），所以必须在这里先记上，
    # 否则 UI 上会看到「3 条样本，提取 0 跳过 0」这种对不上账的结果。
    n_skip = len(us) - len(todo)
    total_bytes = 0
    stopped = False
    t0 = time.perf_counter()
    n = max(1, len(todo))
    # meta 的更新攒到最后一次写：DS.update() 每次都会重写整个 meta.jsonl，
    # 在循环里逐条调是 O(n²)，1000 条的数据集会慢到无法接受。
    touched: Dict[str, Dict[str, Any]] = {}

    try:
        for i, u in enumerate(todo):
            if should_stop and should_stop():
                stopped = True
                break
            if progress:
                progress(i / n, f"[{i + 1}/{n}] {u.id} · {(u.text or '')[:18]}")
            r = ex.extract_to(ds_dir, u.id, u.audio_abs(ds_dir),
                              u.text or u.asr_text or "",
                              u.lang or lang_default, text_normalization,
                              overwrite=overwrite)
            results.append(r)
            if not r.ok:
                n_fail += 1
                # 不改 status：样本可能是 ready 但提取失败（音频损坏等），
                # 把状态改成别的会让它在数据集页上「消失」，很难排查
                note = (u.note + " | " if u.note else "") + f"特征提取失败：{r.error}"
                touched[u.id] = {"has_features": False, "note": note[:400]}
            elif r.stats.get("skipped"):
                n_skip += 1
                touched[u.id] = {"has_features": True}
            else:
                n_ok += 1
                total_bytes += int(r.stats.get("bytes") or 0)
                st = r.stats
                touched[u.id] = {
                    "has_features": True, "features_at": time.time(),
                    # 长度信息一并回写 meta：训练器靠它们排序/过滤，
                    # 不应该为了拿四个 int 而去加载整个特征文件。
                    "mel_len": int(st.get("mel_len") or 0),
                    "n_codes": int(st.get("n_codes") or 0),
                    "n_text_tokens": int(st.get("n_text_tokens") or 0),
                    "lang_token": int(st.get("lang_token") or 1),
                }
    finally:
        ex.unload()
        if touched:
            _apply_meta(name, touched)
        if progress:
            progress(1.0, "特征提取完成")

    seconds = time.perf_counter() - t0
    warns = sorted({w for r in results for w in r.warnings if w != "已存在，跳过"})
    if vram_warn:
        warns.insert(0, vram_warn)
    return {
        "ok": n_fail == 0,
        "dataset": name,
        "total": len(us),
        "todo": len(todo),
        "extracted": n_ok,
        "skipped": n_skip,
        "failed": n_fail,
        "stopped": stopped,
        "seconds": round(seconds, 2),
        "bytes": total_bytes,
        "results": results,
        "warnings": warns,
        "errors": [(r.id, r.error) for r in results if not r.ok],
        "vram": {"free_gb": round(vram.free_gb, 2),
                 "used_gb": round(vram.used_gb, 2),
                 "total_gb": round(vram.total_gb, 2),
                 "warn": vram_warn},
    }


def backfill_meta(name: str,
                  progress: Optional[Callable[[float, str], None]] = None
                  ) -> Dict[str, Any]:
    """给已有数据集补写 meta 里的长度字段。

    n_codes / n_text_tokens / mel_len / lang_token 是训练器接入时才加的，
    早先提取的数据集 meta 里没有。这里逐个读 .pt 把整数捞出来回写，
    读完立刻丢掉张量，所以内存峰值只有一条样本（约 700 KB）。

    只处理 `has_features=True` 但 `n_codes==0` 的条目，所以重复调用是幂等的，
    第二次几乎瞬间返回。
    """
    ds_dir = DS.dir_of(name)
    items = DS.load_meta(name)
    todo = [u for u in items if u.has_features and not int(u.n_codes or 0)]
    if not todo:
        return {"ok": True, "dataset": name, "backfilled": 0,
                "total": len(items), "failed": []}
    import torch

    touched: Dict[str, Dict[str, Any]] = {}
    failed: List[Tuple[str, str]] = []
    n = max(1, len(todo))
    for i, u in enumerate(todo):
        if progress:
            progress(i / n, f"[{i + 1}/{n}] 补写长度 {u.id}")
        p = FeatureExtractor.feature_path(ds_dir, u.id)
        if not is_usable(p):
            failed.append((u.id, "特征文件不存在或版本不匹配"))
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            failed.append((u.id, f"{type(e).__name__}: {e}"))
            continue
        touched[u.id] = {"mel_len": int(d.get("mel_len") or 0),
                         "n_codes": int(d.get("n_codes") or 0),
                         "n_text_tokens": int(d.get("n_text_tokens") or 0),
                         "lang_token": int(d.get("lang_token") or 1)}
        del d
    written = _apply_meta(name, touched)
    if progress:
        progress(1.0, "长度信息补写完成")
    return {"ok": not failed, "dataset": name, "backfilled": written,
            "total": len(items), "failed": failed}


def verify_dataset(name: str,
                   progress: Optional[Callable[[float, str], None]] = None
                   ) -> Dict[str, Any]:
    """加载已有特征做一致性自检（不重新提取）。"""
    import torch
    ds_dir = DS.dir_of(name)
    us = DS.load_meta(name)
    ok, bad, missing = [], [], []
    n = max(1, len(us))
    for i, u in enumerate(us):
        if progress:
            progress(i / n, f"[{i + 1}/{n}] 校验 {u.id}")
        p = FeatureExtractor.feature_path(ds_dir, u.id)
        if not os.path.isfile(p):
            missing.append(u.id)
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            bad.append((u.id, f"读取失败：{type(e).__name__}: {e}"))
            continue
        if int(d.get("feature_version", -1)) != FEATURE_VERSION:
            bad.append((u.id, f"特征版本 {d.get('feature_version')} ≠ "
                              f"当前 {FEATURE_VERSION}，需要重算"))
            continue
        probs = check_feature(d)
        if probs:
            bad.append((u.id, "；".join(probs)))
        else:
            ok.append(u.id)
    return {"ok": not bad and not missing, "total": len(us),
            "valid": len(ok), "invalid": bad, "missing": missing}


def stats(name: str) -> Dict[str, Any]:
    """数据集特征的汇总统计（不加载张量，只读 meta + 文件大小）。"""
    ds_dir = DS.dir_of(name)
    us = DS.load_meta(name)
    fdir = os.path.join(ds_dir, FEATURE_SUBDIR)
    have = set()
    total_bytes = 0
    if os.path.isdir(fdir):
        for fn in os.listdir(fdir):
            if fn.endswith(".pt"):
                have.add(fn[:-3])
                try:
                    total_bytes += os.path.getsize(os.path.join(fdir, fn))
                except OSError:
                    pass
    ready = [u for u in us if u.status == "ready"]
    done = [u for u in ready if u.id in have]
    dur = sum(u.duration or 0.0 for u in done)
    return {
        "total": len(us), "ready": len(ready), "with_features": len(have),
        "ready_with_features": len(done),
        "ready_missing": [u.id for u in ready if u.id not in have],
        "audio_seconds": round(dur, 1),
        "bytes": total_bytes,
        "gb": round(total_bytes / 1e9, 3),
        "feature_version": FEATURE_VERSION,
    }


def stats_markdown(name: str) -> str:
    s = stats(name)
    pct = (s["ready_with_features"] / s["ready"] * 100) if s["ready"] else 0.0
    L = [f"**数据集 `{name}`**", "",
         "| 项 | 值 |", "|---|---|",
         f"| 样本总数 | {s['total']} |",
         f"| 体检通过（ready） | {s['ready']} |",
         f"| 已提取特征 | {s['ready_with_features']} / {s['ready']}（{pct:.0f}%） |",
         f"| 可训练音频时长 | {s['audio_seconds']:.1f} 秒"
         f"（{s['audio_seconds'] / 60:.1f} 分钟） |",
         f"| 特征占用 | {s['gb']:.3f} GB |",
         f"| 特征格式版本 | v{s['feature_version']} |"]
    if s["ready_missing"]:
        miss = s["ready_missing"]
        L += ["", f"> ⚠️ 还有 {len(miss)} 条 ready 样本没提取特征："
                  f"`{', '.join(miss[:8])}`"
                  + ("…" if len(miss) > 8 else "")]
    mins = s["audio_seconds"] / 60
    if s["ready_with_features"] == 0:
        L += ["", "> 🔴 **还不能训练**：没有任何可用特征。"]
    elif mins < 1:
        L += ["", "> 🟠 可训练时长 < 1 分钟：极易过拟合，请用 guard 的**保守**档。"]
    elif mins < 5:
        L += ["", "> 🟡 可训练时长 < 5 分钟：够做音色微调，语气学习有限。"]
    elif mins < 30:
        L += ["", "> 🟢 可训练时长在甜区（5~30 分钟），用**均衡**档即可。"]
    else:
        L += ["", "> 🟢 数据充足（≥30 分钟），可以尝试更大 rank。"]
    return "\n".join(L)


def feature_size_report(name: str) -> str:
    """逐字段说明缓存里都有什么、各占多少（帮用户理解为什么特征文件不小）。"""
    import torch
    ds_dir = DS.dir_of(name)
    us = [u for u in DS.load_meta(name) if u.status == "ready"]
    for u in us:
        p = FeatureExtractor.feature_path(ds_dir, u.id)
        if os.path.isfile(p):
            try:
                d = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            L = [f"**样本 `{u.id}`**（{d.get('duration', 0):.2f}s）各字段占用", "",
                 "| 字段 | 形状 | dtype | 大小 | 说明 |", "|---|---|---|---|---|"]
            for k, desc in FEATURE_SCHEMA.items():
                v = d.get(k)
                if v is None or not hasattr(v, "shape"):
                    continue
                nbytes = v.numel() * v.element_size()
                L.append(f"| `{k}` | {tuple(v.shape)} | {str(v.dtype).replace('torch.', '')} "
                         f"| {nbytes / 1024:.1f} KB | {desc} |")
            if "spk_cond_emb" in d:
                v = d["spk_cond_emb"]
                L.append(f"| `spk_cond_emb` | {tuple(v.shape)} | "
                         f"{str(v.dtype).replace('torch.', '')} "
                         f"| {v.numel() * v.element_size() / 1024:.1f} KB "
                         f"| 调试用原始 w2v-BERT 特征 |")
            L += ["", f"> 文件总大小 {os.path.getsize(p) / 1024:.1f} KB"]
            if d.get("warnings"):
                L += ["", "> 提取时的警告：" + "；".join(d["warnings"])]
            return "\n".join(L)
    return "_这个数据集还没有可用的特征文件。_"


# ---------------------------------------------------------------------------
# 供训练器使用的批组装工具
# ---------------------------------------------------------------------------

def pad_stack(items: Sequence[Any], value: int = 0, dim: int = 0):
    """把长度不一的 1D/2D 张量补齐后堆叠，返回 (batch, lengths)。"""
    import torch
    lens = [int(x.shape[dim]) for x in items]
    m = max(lens) if lens else 0
    out = []
    for x, L in zip(items, lens):
        if L < m:
            pad_shape = list(x.shape)
            pad_shape[dim] = m - L
            x = torch.cat([x, torch.full(pad_shape, value, dtype=x.dtype)], dim=dim)
        out.append(x)
    return (torch.stack(out) if out else torch.empty(0)), torch.LongTensor(lens)


def build_cfm_pair(prompt: Dict[str, Any], target: Dict[str, Any],
                   max_frames: int = 0) -> Optional[Dict[str, Any]]:
    """把两条样本拼成一个 CFM 训练对。

    结构完全对齐推理（infer:839-845）：
        x1  = [prompt_mel | target_mel]        (80, Tp+Tt)
        mu  = [mu_prompt_P | mu_target_T]      (Tp+Tt, 512)
        prompt_lens = Tp
        style = prompt 的 style
    推理时 `vc_target[:, :, ref_mel.size(-1):]` 把 prompt 段切掉，
    训练时 `BASECFM.forward` 把 prompt 段的 y 置 0 且不计 loss —— 同一件事。

    prompt 与 target 可以是同一条（自提示），也可以来自同一说话人的不同句子。
    """
    import torch
    tp = int(prompt["mel"].shape[1])
    tt = int(target["mel"].shape[1])
    if max_frames and tp + tt > max_frames:
        return None
    x1 = torch.cat([prompt["mel"].float(), target["mel"].float()], dim=1)
    mu = torch.cat([prompt["mu_prompt"].float(), target["mu_target"].float()], dim=0)
    if x1.shape[1] != mu.shape[0]:
        return None                      # 理论上不会发生，check_feature 已经保证
    return {
        "x1": x1,                                    # (80, Tp+Tt)
        "mu": mu,                                    # (Tp+Tt, 512)
        "style": prompt["style"].float(),            # (192,)
        "prompt_len": tp,
        "total_len": tp + tt,
    }


def build_gpt_sample(feat: Dict[str, Any]) -> Dict[str, Any]:
    """GPT(T2S) 训练样本。

    条件 = `spk_emb_proj(style) + emo_vec`，两者都已缓存，
    所以训练时**不需要**跑 conformer/perceiver/w2v-bert，显存省一大截。

    注意：GPT 在 use_bf16 时是 bf16 的，但这里统一给 fp32，
    由训练前向自己对齐 dtype（见 features_probe.gpt_training_forward）——
    缓存里存什么精度不应该跟引擎的 bf16 开关绑定。
    """
    import torch
    return {
        "text_tokens": feat["text_tokens"].to(torch.int64),
        "codes": feat["codes"].to(torch.int64),
        "style": feat["style"].float(),
        "emo_vec": feat["emo_vec"].float(),
        "lang_token": int(feat["lang_token"]),
        "n_codes": int(feat["n_codes"]),
        "n_text_tokens": int(feat["n_text_tokens"]),
    }


def build_cfm_sample(feat: Dict[str, Any]) -> Dict[str, Any]:
    """CFM(S2M) 训练用的**单条**特征（配对由 build_cfm_pair 做）。

    故意**不转 fp32**：mu 是 (Tm, 512)，一条 700 帧的样本就 0.7 MB，
    样本池会缓存上百条 —— 缓存里转成 fp32 直接把内存翻倍，
    而 `build_cfm_pair` 在真正拼对时本来就会 `.float()`。
    """
    return {
        "mel": feat["mel"],                    # (80, Tm)  fp16
        "mu_prompt": feat["mu_prompt"],        # (Tm, 512) fp16
        "mu_target": feat["mu_target"],        # (Tm, 512) fp16
        "style": feat["style"],                # (192,)    fp32
        "mel_len": int(feat["mel_len"]),
        "n_codes": int(feat["n_codes"]),
    }
