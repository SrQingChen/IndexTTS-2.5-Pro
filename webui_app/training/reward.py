"""L2 · 奖励打分：一句话合成得好不好，拆成两个可测的量。

  · **WER（字错误率，via whisper）**——「说得对不对」。合成音频被 whisper
    转写，与目标文本算字级编辑距离。读错字、吞字、韵律崩坏导致的漏读
    都会在这里暴露。
  · **SS（声纹相似度，via campplus）**——「像不像」。合成音频与参考音频
    各自过 CAMPPlus 得到 192 维声纹，算余弦。音色漂没漂、像不像目标
    说话人，这一个数就能比。

    DPO（dpo.py）拿这两个数的组合当奖励：一对候选里分高的当 chosen、
    分低的当 rejected；评测台（evaluate）拿它们做 A/B 报告。

**为什么不端到端学一个奖励模型**：8GB 卡上没有余量再训一个打分器，
而「whisper 的字错率 + campplus 的余弦」在 TTS 文献里就是 UTMOS 之外
最常用的两组无参考指标，可解释、可分开看、坏了知道该怪谁。

**资源账**（RTX 4060 8GB 实测语境）：
  · campplus 6.85M 参数，加载 0.2s，fp32 也就 27 MB —— 常驻没问题；
  · whisper small 241M，fp16 约 0.5 GB —— 与推理引擎（4.9~5.7 GB）
    可以共存，但训练/打分密集期建议先卸载引擎；
  · 两个模型都**懒加载**，`reward.py` 被 import 本身零开销。

whisper 权重缓存在 `checkpoints/hf_cache/whisper/`（.pt 文件不在
BaseGuard 的保护扩展名里，不会污染底座快照）。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import guard as GD

__all__ = ["RewardOptions", "RewardScorer", "normalize_text", "cer",
           "asr_language"]

# whisper 档位 → (参数量 M, 显存 fp16 GB)。写死在这里是为了让 validate
# 能在下载之前就告诉用户「这一档要多大」。
# whisper 档位 → (参数量 M, **实际显存 GB**)。
#
# 这里的显存是**实测值**，不是按 fp16 估的：`openai-whisper` 的
# `load_model()` 没有 dtype 参数，**一律以 fp32 加载**（实测 medium 的权重 dtype
# 就是 torch.float32），加上 PyTorch 缓存分配器的预留，真实占用约为
# 「参数量 × 4 字节 × 1.5」。此前这个表填的是 fp16 数字，把 medium 写成 1.6 GB
# 而实际要 4.28 GB —— **低估 2.7 倍**，直接导致显存规划与告警都失准。
#
# 实测（RTX 4060 Laptop 8 GB，torch 2.8）：
#     base(72M)   allocated 0.27 / reserved 0.42 GB
#     small(241M) allocated 0.90 / reserved 1.46 GB
#     medium(762M) allocated 2.85 / reserved 4.28 GB
WHISPER_SIZES: Dict[str, Tuple[float, float]] = {
    "tiny": (39, 0.23), "base": (74, 0.42), "small": (244, 1.46),
    "medium": (769, 4.28), "large-v3": (1550, 7.70), "turbo": (809, 4.50),
}
# 默认档位从 small 升到 medium —— small 在中文上会把「西莲」写成「西蓮」、
# 长句还容易糊成一片，当训练文本用会直接教坏模型（它会照着错字学发音）。
# medium 参数量约 3.2 倍，中文准确率提升明显。
#
# 但要注意它的真实占用是 4.28 GB：与推理引擎（4.94 GB）**不可能共存**于 8 GB 卡，
# 所以一键三连在识别前会显式把引擎卸掉（见 oneclick.ensure_engine_off）。
# 显存更紧就往 small（1.46 GB）退一档。
DEFAULT_WHISPER = "medium"

# 语言 → whisper 的 initial_prompt。
#
# 这是**修繁体字的关键**：whisper 在 language="zh" 下经常输出繁体
# （实测 small 把「我是西莲,很高兴见到你」写成「我是西蓮,很高興見到你」）。
# 给一段简体中文的提示词后，输出会稳定落在简体上 —— 不换模型、零成本，
# 实测 5/5 条全部纠正。它同时能压低「莫名其妙蹦出一串乱码」的概率。
LANGUAGE_PROMPTS: Dict[str, str] = {
    "zh": "以下是普通话的句子。",
    "en": "This is a sentence in English.",
    "ja": "これは日本語の文章です。",
    "es": "Esta es una frase en español.",
    "ar": "هذه جملة باللغة العربية.",
}


def default_prompt(lang: str) -> str:
    """按语言给一段默认提示词。未知语言返回空串（不传 prompt）。"""
    return LANGUAGE_PROMPTS.get(str(lang or "").strip().lower(), "")

_PUNCT = re.compile(
    r"[\s\u3000`~!@#$%^&*()_\-+=\[\]{}\\|;:'\",.<>/?·！？…。，、；：‘’“”"
    r"（）《》〈〉【】〔〕％＃＠＆＊－＋＝｜｛｝／＼：；「」～￥]+")
_FULLWIDTH = str.maketrans("０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
                           "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_text(t: str) -> str:
    """把文本压成「可比对的字符流」：全角→半角、去标点空白、英文小写。

    whisper 的中文输出会带标点和空格，目标文本可能带注音标记或全角数字，
    不归一化直接比对会把 CER 抬高一截 —— 而奖励的**相对差**才是 DPO
    真正消费的东西，系统性偏差不可怕，随机偏差才可怕。
    """
    t = (t or "").translate(_FULLWIDTH)
    t = _PUNCT.sub("", t)
    return t.lower()


def cer(ref: str, hyp: str) -> float:
    """字错误率：编辑距离 ÷ len(ref)，∈ [0,1]。

    len(ref)==0 时：hyp 也空 → 0（完全正确），否则 1（全错）。
    分母用 ref 而不是 max(len)：一句 10 字的参考被读成 20 字，
    错误率应该能超过 1 再截到 1，而不是被稀释。
    """
    r, h = normalize_text(ref), normalize_text(hyp)
    if r == h:
        return 0.0
    if not r:
        return 0.0 if not h else 1.0
    # 经典两行滚动数组 Levenshtein，中文一句几十字，O(n·m) 足够快
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1,          # 删
                         cur[j - 1] + 1,       # 增
                         prev[j - 1] + (rc != hc))  # 换
        prev = cur
    return min(1.0, prev[-1] / len(r))


def asr_language(lang: str) -> str:
    """数据集语言（ZH/EN/JA/ES）→ whisper 的 language 代码。"""
    low = (lang or "ZH").lower()
    if low in ("zh", "zhen"):
        return "zh"
    if low in ("en", "zh_en"):
        return "en"
    return {"ja": "ja", "es": "es"}.get(low, "zh")


@dataclass
class RewardOptions:
    """打分器的全部旋钮（序列化进 DPO 的 run.json，保证可复现）。"""
    whisper_size: str = DEFAULT_WHISPER
    language: str = "zh"              # None/"" 时按 auto 走（不推荐：不稳定）
    wer_weight: float = 0.6           # reward = w1·(1-WER) + w2·SS
    sim_weight: float = 0.4
    beam_size: int = 5                # 温度 0 + beam，保证同音频两次转写一致
    fp16: bool = True                 # 只在 cuda 上生效
    device: str = ""                  # 空 = 自动（cuda 优先）
    # 转写提示词：非空则原样使用；留空时按 auto_prompt 取语言默认值。
    # 默认值的作用见 LANGUAGE_PROMPTS 的说明（修繁体字）。
    initial_prompt: str = ""
    auto_prompt: bool = True
    # ---- 2026-09 表现力升级新增（默认关闭/零权重 = 行为与旧版完全一致）----
    # 转写引擎：whisper（旧）| sensevoice（中文专名/标点/情绪标签更好，
    # 经 services/funasr_hub，不可用时自动回退 whisper 并记 warning）
    asr_engine: str = "whisper"
    # 情绪项权重：emotion2vec 嵌入的 cos(合成音频, 情绪参考音频)。
    # emo_ref_path 为空时该项跳过（不贡献分母）。
    emo_weight: float = 0.0
    # 停顿项权重：合成音频句内静音占比落在 8%~25% 带内得满分，
    # 0%（一口气赶完）或 >40%（稀碎）线性衰减 —— 只用能量，零模型。
    pause_weight: float = 0.0
    # 打分用 whisper 的独立档位（BoN 择优等「边合成边打分」场景用 small
    # 甚至 base，避免与引擎抢显存；纯离线打分可用 medium）
    score_whisper_size: str = ""      # 空 = 沿用 whisper_size

    def effective_whisper(self) -> str:
        return self.score_whisper_size or self.whisper_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []

        def err(m): n.append(GD.Notice("error", m))
        def warn(m): n.append(GD.Notice("warn", m))
        def info(m): n.append(GD.Notice("info", m))

        if self.whisper_size not in WHISPER_SIZES:
            err(f"whisper_size={self.whisper_size!r} 不在 {tuple(WHISPER_SIZES)}")
        if self.asr_engine not in ("whisper", "sensevoice"):
            err(f"asr_engine={self.asr_engine!r} 必须是 whisper 或 sensevoice")
        if self.score_whisper_size and self.score_whisper_size not in WHISPER_SIZES:
            err(f"score_whisper_size={self.score_whisper_size!r} 不在 "
                f"{tuple(WHISPER_SIZES)}")
        for name in ("emo_weight", "pause_weight"):
            if float(getattr(self, name)) < 0:
                err(f"{name} 不能为负")
        if self.asr_engine == "sensevoice":
            info("sensevoice 转写：中文专名/标点更准，附带情绪标签；"
                 "funasr 未安装时自动回退 whisper。")
        p, v = WHISPER_SIZES.get(self.whisper_size, (0, 0))
        if p >= 700:
            warn(f"whisper {self.whisper_size} 实测要 {v:.2f} GB 显存"
                 "（openai-whisper 一律 fp32 加载，`.half()` 在这个版本上会因"
                 "LayerNorm 强制转 float 而报错，所以省不下来）。它与推理引擎"
                 "（4.94 GB）在 8 GB 卡上无法共存 —— 一键三连会在识别前自动"
                 f"卸掉引擎；若显存仍紧，把识别模型降到 small"
                 f"（{WHISPER_SIZES['small'][1]:.2f} GB）。")
        w1, w2 = float(self.wer_weight), float(self.sim_weight)
        if w1 < 0 or w2 < 0 or (w1 + w2) <= 0:
            err(f"权重必须是正数且不全为零：wer={w1} sim={w2}")
        elif abs(w1 + w2 - 1.0) > 1e-6:
            info(f"权重和为 {w1+w2:.2f}（不是 1）。reward 会整体缩放，"
                 "不影响同一对候选的相对比较，但跨报告不可直接比")
        if self.language and asr_language(self.language) not in ("zh", "en", "ja", "es"):
            err(f"language={self.language!r} whisper 不支持")
        if not 1 <= int(self.beam_size) <= 10:
            err(f"beam_size={self.beam_size} 超出 1~10")
        return n


@dataclass
class ScoreResult:
    """一条音频的打分结果。ok=False 时带 error，其余字段不保证有效。"""
    ok: bool = False
    wer: float = 1.0
    sim: float = 0.0
    reward: float = 0.0
    asr_text: str = ""
    seconds: float = 0.0
    error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardScorer:
    """懒加载的打分器：whisper（WER） + campplus（SS）。

    用法::

        sc = RewardScorer(RewardOptions())
        r = sc.score("out.wav", text="今天天气不错", ref="prompt.wav", lang="ZH")
        r["wer"], r["sim"], r["reward"]

    campplus 直接读 `checkpoints/hf_cache/campplus_cn_common.bin` 独立构造，
    **不经过推理引擎** —— 打分时引擎可以在任何状态（加载/卸载/被改过），
    互不干扰，也不用在显存里多养一份 5GB 的东西。
    """

    def __init__(self, options: Optional[RewardOptions] = None,
                 model_dir: Optional[str] = None):
        self.opt = options or RewardOptions()
        self.model_dir = model_dir or os.path.join(PROJECT_ROOT, GD.MODEL_DIR_NAME)
        self._asr = None            # whisper 模型（懒加载）
        self._spk = None            # campplus（懒加载）
        self._device = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = (self.opt.device or
                            ("cuda" if torch.cuda.is_available() else "cpu"))
        return self._device

    def unload(self) -> None:
        """显式归还显存。两个模型都很小，但打完分就该还。

        一键三连的识别阶段会在用完后立刻调它 —— 引擎随后要加载，
        显存必须先腾出来（默认 medium 是 1.6 GB，不是可以无视的量）。
        sensevoice / emotion2vec 走 funasr_hub，同样在这里一起卸。
        """
        import gc

        import torch
        self._asr = None
        self._spk = None
        try:
            from webui_app.services import funasr_hub
            funasr_hub.release_all()
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def is_loaded(self) -> bool:
        """whisper 或 campplus 是否还驻留在内存/显存里。"""
        return self._asr is not None or self._spk is not None

    def vram_free_gb(self) -> float:
        """当前整卡空闲显存（GB）。非 CUDA 返回 0。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            free, _total = torch.cuda.mem_get_info(0)
            return round(free / (1024 ** 3), 2)
        except Exception:
            return 0.0

    def vram_note(self) -> str:
        p, v = WHISPER_SIZES.get(self.opt.whisper_size, (0, 0))
        gb = v if (self.fp16_active() and self.device.startswith("cuda")) else p * 4 / 1e3
        return f"whisper {self.opt.whisper_size} ≈ {gb:.2f} GB + campplus 0.03 GB"

    def fp16_active(self) -> bool:
        return bool(self.opt.fp16) and self.device.startswith("cuda")

    # ------------------------------------------------------------------
    # 两个子模型
    # ------------------------------------------------------------------
    def _whisper(self):
        if self._asr is None:
            import time

            import whisper

            from webui_app import logging_setup as LOG
            from webui_app.training import guard as GD

            log = LOG.get_logger("reward.whisper")
            cache = os.path.join(self.model_dir, "hf_cache", "whisper")
            _size = self.opt.effective_whisper()
            # 加载前先清一次显存。看着多余，其实是实测踩出来的：8 GB 卡上推理
            # 引擎常驻 4.94 GB，再叠 whisper medium 1.6 GB 就只剩几百 MB，
            # Windows 把计算挤进共享内存 → **静默降速 20~30 倍**（一条 10 秒
            # 音频的转写卡了 4 分钟以上）。清理只能归还已释放的块，所以调用方
            # 还必须把用不到的模型真的卸掉。
            info = GD.free_vram(f"加载 whisper-{_size} 前", log)
            t0 = time.perf_counter()
            log.info("加载 whisper-%s（device=%s，空闲显存 %.2f GB）",
                     _size, self.device, info.get("after_gb"))
            try:
                self._asr = whisper.load_model(
                    _size, device=self.device, download_root=cache)
            except Exception as e:
                log.error("whisper %s 加载失败", _size, exc_info=True)
                raise RuntimeError(
                    f"whisper {_size} 加载失败：{e}\n"
                    f"缓存目录 {cache}。首次使用需要联网下载"
                    f"（{WHISPER_SIZES.get(_size, (0, 0))[0]:.0f}M 参数）。") from e
            log.info("whisper-%s 就绪 · 加载 %.1fs · 之后空闲 %.2f GB",
                     _size, time.perf_counter() - t0, self.vram_free_gb())
        return self._asr

    def _campplus(self):
        if self._spk is None:
            import torch
            from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
            pth = os.path.join(self.model_dir, "hf_cache", "campplus_cn_common.bin")
            if not os.path.isfile(pth):
                raise FileNotFoundError(
                    f"找不到 campplus 权重 {pth}。到「模型」页重新校验/下载。")
            m = CAMPPlus(feat_dim=80, embedding_size=192)
            m.load_state_dict(torch.load(pth, map_location="cpu"))
            self._spk = m.to(self.device).eval()
        return self._spk

    # ------------------------------------------------------------------
    # 声纹
    # ------------------------------------------------------------------
    def embed(self, audio_path: str):
        """一条音频 → (192,) fp32 声纹。fbank 流程与 features.py 逐行一致
        （16k、80 bins、均值消减）—— 同一条音频在两处必须得到同一个向量，
        否则「训练时的 style」与「打分时的 SS」就不可比。"""
        import torch
        import torchaudio
        import librosa

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音频不存在：{audio_path}")
        audio, sr = librosa.load(audio_path)
        if len(audio) < sr * 0.2:
            raise ValueError(f"{audio_path} 短于 0.2s，fbank 没有足够帧")
        a16 = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(audio).unsqueeze(0))
        m = self._campplus()
        with torch.no_grad():
            f = torchaudio.compliance.kaldi.fbank(
                a16.to(self.device), num_mel_bins=80, dither=0,
                sample_frequency=16000)
            f = f - f.mean(dim=0, keepdim=True)
            return m(f.unsqueeze(0)).squeeze(0).float().cpu()

    @staticmethod
    def cosine(a, b) -> float:
        import torch
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    # ------------------------------------------------------------------
    # 转写
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str) -> str:
        """按 asr_engine 转写。whisper：温度 0 + 固定 beam（确定性）；
        sensevoice：经 funasr_hub（中文/标点/专名更准）。funasr 不可用时
        回退 whisper 并记 warning —— 回退不改变返回类型，调用方无感。"""
        if self.opt.asr_engine == "sensevoice":
            try:
                from webui_app.services import funasr_hub
                r = funasr_hub.transcribe(audio_path,
                                          lang=asr_language(self.opt.language))
                return r["text"]
            except Exception as e:
                from webui_app import logging_setup as LOG
                LOG.get_logger("reward.asr").warning(
                    "sensevoice 转写失败，回退 whisper：%s: %s",
                    type(e).__name__, e)
                # 落到下面的 whisper 路径
        m = self._whisper()
        lang = asr_language(self.opt.language) if self.opt.language else None
        r = m.transcribe(audio_path, language=lang, temperature=0.0,
                         beam_size=int(self.opt.beam_size),
                         initial_prompt=self.resolve_prompt() or None)
        return (r.get("text") or "").strip()

    def resolve_prompt(self) -> str:
        """本次转写实际使用的提示词。显式给了就用显式的。"""
        if (self.opt.initial_prompt or "").strip():
            return self.opt.initial_prompt.strip()
        if not self.opt.auto_prompt:
            return ""
        return default_prompt(self.opt.language)

    # ------------------------------------------------------------------
    # 停顿启发项（零模型，只用能量）
    # ------------------------------------------------------------------
    @staticmethod
    def pause_score(audio_path: str) -> float:
        """句内静音占比落在 5%~20% 带内得 1 分，向两侧线性衰减到 0。

        2026-09-29 校准（碎裂停顿事故复盘）：旧带区 8~25% 偏向奖励停顿，
        而真人语流的句内静音是**稀缺品**（<200ms 的微停为主，Goldman-Eisler
        1968 的 200~250ms 计划性停顿阈值基本都落在句界）。0% = 一口气赶完，
        >30% = 稀碎/半静音。
        """
        try:
            import librosa
            import numpy as np
            y, sr = librosa.load(audio_path, sr=None)
            if len(y) < sr * 0.3:
                return 0.5                      # 太短测不准，给中性分
            frame = int(sr * 0.025)
            n = len(y) // frame
            if n < 4:
                return 0.5
            rms = np.sqrt(np.mean(
                y[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
            thr = max(float(np.percentile(rms, 10)) * 2.0, 1e-4)
            # 只统计首/尾语音之间的静音（首尾静音已被增强链掐过，
            # 但打分对象可能是未处理的原始合成）
            speech = np.where(rms > thr)[0]
            if len(speech) < 2:
                return 0.5
            core = rms[speech[0]:speech[-1] + 1]
            ratio = float(np.mean(core <= thr))
            if 0.05 <= ratio <= 0.20:
                return 1.0
            if ratio < 0.05:
                return max(0.0, ratio / 0.05)
            return max(0.0, 1.0 - (ratio - 0.20) / 0.20)
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # 组合打分
    # ------------------------------------------------------------------
    def score(self, synth_path: str, text: str, ref_path: str,
              lang: Optional[str] = None,
              emo_ref_path: Optional[str] = None) -> Dict[str, Any]:
        """合成音频 → {wer, sim, reward, asr_text, ...}。

        reward = Σ wᵢ·sᵢ / Σ wᵢ，各项：
            wer  (1-WER)   whisper/sensevoice 回转
            sim  (SS)      campplus 声纹余弦
            emo  (EmoSim)  emotion2vec 余弦 —— 仅当 emo_weight>0 且
                           emo_ref_path 给了（合成 vs 情绪参考的情绪贴合度）
            pause          句内静音占比带内得分（pause_weight>0 时）
        任何一环失败都返回 ok=False + error，**不抛异常** ——
        DPO 批量打分时一条坏音频不该炸掉整批。
        """
        t0 = time.perf_counter()
        try:
            if lang:
                old = self.opt.language
                self.opt.language = lang
            else:
                old = None
            try:
                asr_text = self.transcribe(synth_path)
            finally:
                if old is not None:
                    self.opt.language = old
            wer = cer(text, asr_text)
            sim = self.cosine(self.embed(synth_path), self.embed(ref_path))

            terms: List[Tuple[str, float, float]] = [
                ("wer", float(self.opt.wer_weight), 1.0 - wer),
                ("sim", float(self.opt.sim_weight), sim),
            ]
            emo = None
            if float(self.opt.emo_weight) > 0 and emo_ref_path:
                try:
                    from webui_app.services import funasr_hub
                    emo = funasr_hub.emo_cosine(synth_path, emo_ref_path)
                    terms.append(("emo", float(self.opt.emo_weight), emo))
                except Exception as e:
                    # 情绪项是增强项：取不到就跳过（不进分母），不连坐
                    from webui_app import logging_setup as LOG
                    LOG.get_logger("reward.emo").warning(
                        "emotion2vec 打分跳过：%s: %s", type(e).__name__, e)
            pause = None
            if float(self.opt.pause_weight) > 0:
                pause = self.pause_score(synth_path)
                terms.append(("pause", float(self.opt.pause_weight), pause))

            tot = sum(w for _k, w, _v in terms) or 1.0
            reward = sum(w * v for _k, w, v in terms) / tot
            detail = {"whisper": self.opt.whisper_size,
                      "asr_engine": self.opt.asr_engine,
                      "lang": asr_language(self.opt.language)}
            if emo is not None:
                detail["emo"] = round(emo, 4)
            if pause is not None:
                detail["pause"] = round(pause, 4)
            return ScoreResult(
                ok=True, wer=round(wer, 4), sim=round(sim, 4),
                reward=round(reward, 4), asr_text=asr_text,
                seconds=round(time.perf_counter() - t0, 2),
                detail=detail).to_dict()
        except Exception as e:
            return ScoreResult(
                ok=False, error=f"{type(e).__name__}: {e}",
                seconds=round(time.perf_counter() - t0, 2)).to_dict()

    def score_pair(self, path_a: str, path_b: str, text: str,
                   ref_path: str) -> Dict[str, Any]:
        """DPO 的偏好判定：同文本同参考的两个候选，谁高谁 chosen。

        返回 {chosen, margin, a, b}。margin 是 reward 差 ——
        差距小于 min_margin 时这一对**不可信**（转写噪声就能造成），
        DPO 应该丢弃它而不是硬学。
        """
        a = self.score(path_a, text, ref_path)
        b = self.score(path_b, text, ref_path)
        if not (a.get("ok") and b.get("ok")):
            return {"ok": False, "a": a, "b": b,
                    "error": a.get("error") or b.get("error")}
        margin = float(a["reward"]) - float(b["reward"])
        return {"ok": True,
                "chosen": "a" if margin >= 0 else "b",
                "margin": round(margin, 4),
                "a": a, "b": b}

    # ------------------------------------------------------------------
    def scores_markdown(self, rows: Sequence[Dict[str, Any]],
                        title: str = "打分结果") -> str:
        """把 score() 的结果列表渲染成表格（评测台与 DPO 报告共用）。"""
        if not rows:
            return "_没有可打分的条目。_"
        L = [f"### {title}", "",
             "| 音频 | WER↓ | 声纹相似↑ | reward↑ | 转写 |",
             "|---|---|---|---|---|"]
        for r in rows:
            p = r.get("path") or r.get("audio") or ""
            name = os.path.basename(str(p))
            if not r.get("ok"):
                L.append(f"| `{name}` | 🔴 | 🔴 | 🔴 | {r.get('error', '')[:40]} |")
                continue
            L.append(f"| `{name}` | {r['wer']:.3f} | {r['sim']:.3f} "
                     f"| **{r['reward']:.3f}** | {str(r.get('asr_text', ''))[:24]}… |")
        ok = [r for r in rows if r.get("ok")]
        if ok:
            n = len(ok)
            L += ["", f"均值：WER {sum(r['wer'] for r in ok)/n:.3f} · "
                  f"相似度 {sum(r['sim'] for r in ok)/n:.3f} · "
                  f"reward {sum(r['reward'] for r in ok)/n:.3f}（{n} 条）"]
        return "\n".join(L)
