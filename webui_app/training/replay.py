"""回放数据生成 —— 防线 6（抗灾难性遗忘）的数据来源。

**为什么需要它**：小数据集微调里，遗忘比过拟合更常见也更隐蔽。因为 val loss
只测目标角色的数据，模型把「普通话整体发音能力」丢掉这件事在 val 上**看不出来**。
唯一的对策是训练时混入一批通用样本，不断提醒模型「这些发音方式不能忘」。

**这批通用样本从哪来**，两个选择：

    · `replay_source="dataset"`   用户自备一份通用语音数据集
    · `replay_source="base_distill"`  本模块生成 —— 让**底座自己**去读一批
      覆盖多音字/数字/中英混排的通用文本，把它的输出当成回放目标

后者的好处是不需要用户额外准备素材，而且回放目标恰好是「底座原本的行为」，
LoRA 想覆盖掉的正是这部分，所以钉住它的效果最直接。

**两条必须守住的红线**（都在下面代码里落地了，不是注释里的愿望）：

    1. 蒸馏时 adapter 的强度必须为 0。否则蒸出来的是「已经适配过的行为」，
       回放不但不能抗遗忘，反而会**强化**偏移 —— 越训越歪还看不出问题。
    2. 参考音频不能用目标角色的。用了就等于回放集里全是目标音色，
       通用能力一样保不住。这里做检查并给出警告。

生成完的音频会走 `features.py` **完全相同**的特征链路 —— 回放样本与目标样本
的特征必须同源，否则两边在训练里根本不是同一种东西。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import dataset as DS
from webui_app.training import features as FT
from webui_app.training import guard as GD

# ---------------------------------------------------------------------------
# 通用文本语料
# ---------------------------------------------------------------------------
# 每一组都对应一类「底座容易忘、目标数据里又很少见」的现象。
# 目标角色的录音通常是连续朗读，很少出现日期、单位、中英混排这些东西，
# 所以只回放同类文本是不够的 —— 要刻意覆盖训练集里**没有**的那些。
GENERIC_TEXTS: List[Tuple[str, str]] = [
    # ---- 多音字（最容易在微调后读错的一类）----
    ("polyphone", "银行行长说这一行行业的行情还行，走路要靠着右边行。"),
    ("polyphone", "这件事的 length 很长，长大以后他才明白长处在哪里。"),
    ("polyphone", "他把重物重新称了一遍，发现这个称呼并不合适。"),
    ("polyphone", "你得把这个地方清理干净，地上还有一本书。"),
    ("polyphone", "他慢慢地走着，心里还在想着那句话，看着远方。"),
    ("polyphone", "为了这件事，他为我做了很多，成为了一名教师。"),
    ("polyphone", "数学老师说，数一数这些参数，单于的名单也在里面。"),
    ("polyphone", "他觉得睡觉前的感觉最好，一觉醒来天已经亮了。"),
    ("polyphone", "差别在于参数不同，他参差地摆着，参加的人也参差不齐。"),
    ("polyphone", "这是什么意思？他十分钟前才说过，什锦糖很好吃。"),
    ("polyphone", "种子从这里发出去，他种了一棵树，种类繁多。"),
    ("polyphone", "音乐让他很快乐，乐曲里有一个快乐的乐章。"),
    ("polyphone", "他背着包往前走，背上的字写得很清楚。"),
    ("polyphone", "只有他知道，这张卡片上写着什么，写着的是地址。"),
    ("polyphone", "血红的旗帜下面，流血的人已经被送走了。"),
    ("polyphone", "他说话很快，把话说得大家都说服了，游说也很在行。"),

    # ---- 数字、日期、单位、符号 ----
    ("number", "会议定在2026年9月13日下午3点半，地点是第21号楼3层。"),
    ("number", "圆周率约等于3.14159，二分之一是0.5，百分之三十写作30%。"),
    ("number", "这台机器的显存是8GB，功耗115瓦，重量约2.4千克。"),
    ("number", "订单号是No.20240815，金额￥1280.50，联系电话+86 138 0013 8000。"),
    ("number", "今天气温零下5℃，明天升到12℃，风力3到4级。"),
    ("number", "第一名的成绩是9秒83，第二名慢了0.02秒，一共100米。"),
    ("number", "公元1840年到1949年，一共一百零九年，跨越了两个世纪。"),
    ("number", "三十五个人里有十二个是女生，占比约34.3%。"),

    # ---- 中英混排 ----
    ("mixed", "我们用 PyTorch 训练模型，然后在 GPU 上跑推理。"),
    ("number", "这个 API 的 QPS 上限是 200，超时时间设为 30 秒。"),
    ("mixed", "LoRA 的全称是 Low-Rank Adaptation，它只训练很小的旁路矩阵。"),
    ("mixed", "请把这份 PDF 转成 Markdown，再用 VS Code 打开看看。"),
    ("mixed", "AI 模型在 CPU 上跑得慢，建议开启 CUDA 加速。"),

    # ---- 疑问、感叹、语气 ----
    ("prosody", "你真的这么想吗？我可不这么觉得！"),
    ("prosody", "哎呀，这可怎么办呢……要不算了？"),
    ("prosody", "太好了！我们终于成功了，太不容易了。"),
    ("prosody", "请问，附近的地铁站应该怎么走？谢谢您。"),
    ("prosody", "嗯……让我想想，大概是这个意思吧。"),
    ("prosody", "别动！小心脚下，那里有一块碎玻璃。"),

    # ---- 停顿与长句（韵律骨架）----
    ("prosody", "他一边说话，一边比划着手势，说到关键处，还停顿了一下。"),
    ("long", "尽管大家都认为这个方案在技术上完全可行，但由于成本过高，"
             "加上时间窗口已经所剩无几，最终还是没有通过评审。"),
    ("long", "语音合成系统通常包含文本前端、声学模型和声码器三个部分，"
             "其中声学模型负责把语言学特征转换成声学特征。"),
    ("long", "如果明天不下雨，我们就按原计划出发；万一下雨，"
             "那就把行程推迟到下周六，你看这样可以吗？"),

    # ---- 日常口语 ----
    ("daily", "麻烦你把窗户关上，外面有点吵，谢谢。"),
    ("daily", "我点了两杯咖啡，一杯不加糖，一杯多加奶。"),
    ("daily", "这个东西多少钱？能不能便宜一点？"),
    ("daily", "麻烦让一让，我要在下一站下车。"),
    ("daily", "咱们周末去看电影吧，听说那部片子口碑不错。"),
    ("daily", "你把文件发我邮箱就行，不用打印出来了。"),
    ("daily", "孩子今天发烧了，我下午得早点走。"),

    # ---- 书面语与正式场合 ----
    ("formal", "兹定于本月二十日召开全体成员大会，请准时出席。"),
    ("formal", "综上所述，本研究在方法与结论两个方面均具有一定的参考价值。"),
    ("formal", "请各位旅客系好安全带，飞机即将起飞。"),
    ("formal", "本合同自双方签字盖章之日起生效，有效期为三年。"),
    ("formal", "感谢您的来电，祝您生活愉快，再见。"),

    # ---- 绕口令式的密集音节（考验吐字清晰度）----
    ("tongue", "四是四，十是十，十四是十四，四十是四十。"),
    ("tongue", "红鲤鱼与绿鲤鱼与驴，牛郎恋刘娘，刘娘念牛郎。"),
    ("tongue", "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。"),

    # ---- 儿化与轻声 ----
    ("daily", "今儿个天气真不错，咱俩去公园溜达溜达。"),
    ("daily", "这事儿你得好好琢磨琢磨，别急着下结论。"),
    ("daily", "把桌子上的那玩意儿拿过来给我瞧瞧。"),
]

TEXT_CATEGORIES: Dict[str, str] = {
    "polyphone": "多音字",
    "number": "数字/日期/单位",
    "mixed": "中英混排",
    "prosody": "语气与停顿",
    "long": "长句",
    "daily": "日常口语",
    "formal": "书面/正式",
    "tongue": "密集音节",
}


def generic_texts(limit: int = 0, categories: Optional[Sequence[str]] = None
                  ) -> List[Dict[str, str]]:
    """取通用文本。limit=0 表示全取。"""
    out = []
    for cat, txt in GENERIC_TEXTS:
        if categories and cat not in categories:
            continue
        out.append({"category": cat, "text": txt})
        if limit and len(out) >= int(limit):
            break
    return out


def coverage_markdown() -> str:
    """语料覆盖情况（UI 上给用户看「这批回放到底练了什么」）。"""
    cnt: Dict[str, int] = {}
    for cat, _t in GENERIC_TEXTS:
        cnt[cat] = cnt.get(cat, 0) + 1
    L = [f"**内置通用语料**：{len(GENERIC_TEXTS)} 条，覆盖 {len(cnt)} 类现象", "",
         "| 类别 | 条数 | 为什么必须覆盖 |", "|---|---|---|"]
    why = {
        "polyphone": "目标角色的录音里往往只有几个常用读音，微调后其它读音会被带跑",
        "number": "日期/金额/单位的读法规则复杂，是最先退化的能力之一",
        "mixed": "中英切换处的韵律最容易崩，且目标数据里通常完全没有",
        "prosody": "疑问/感叹的语气曲线决定了「像不像在说话」",
        "long": "长句的呼吸与停顿骨架，短样本训练集会把它磨平",
        "daily": "口语的松散节奏，与朗读腔差别很大",
        "formal": "书面语的正式腔调",
        "tongue": "密集音节下的吐字清晰度",
    }
    for cat, n in sorted(cnt.items(), key=lambda x: -x[1]):
        L.append(f"| {TEXT_CATEGORIES.get(cat, cat)} | {n} "
                 f"| {why.get(cat, '')} |")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class DistillOptions:
    """底座蒸馏的参数。"""
    n_texts: int = 0                 # 0 = 用全部内置语料
    categories: List[str] = field(default_factory=list)   # 空 = 全部类别
    lang: str = "ZH"

    # 参考音频：决定蒸出来的**音色**。刻意与目标角色无关 ——
    # 回放要保住的是「底座的通用发音能力」，不是「某个特定音色」。
    prompt_audio: str = ""
    emo_audio: str = ""

    # 采样参数。温度略高于默认：回放样本需要**多样性**，
    # 太确定的输出会让模型只记住一种韵律，抗遗忘效果打折。
    do_sample: bool = True
    temperature: float = 0.85
    top_p: float = 0.85
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_mel_tokens: int = 600
    duration_factor: float = 1.0
    text_normalization: bool = True
    seed: int = 42

    extract_features: bool = True
    keep_wav: bool = True            # 关掉可以省磁盘，但出问题时没法复核
    overwrite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistillOptions":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []
        texts = generic_texts(self.n_texts, self.categories or None)
        if not texts:
            n.append(GD.Notice("error", "按当前筛选条件一条文本都不剩"))
        if not (self.prompt_audio or "").strip():
            n.append(GD.Notice("error",
                               "必须指定参考音频：底座合成需要一段音色提示。"
                               "**不要用目标角色的音频**，否则回放集会变成"
                               "「加强目标音色」而不是「保住通用能力」。"))
        elif not os.path.isfile(self.prompt_audio):
            n.append(GD.Notice("error", f"参考音频不存在：{self.prompt_audio}"))
        if self.temperature <= 0.0:
            n.append(GD.Notice("error", "temperature 必须 > 0"))
        elif self.temperature < 0.6:
            n.append(GD.Notice("warn",
                               f"temperature={self.temperature} 偏低，蒸出来的样本"
                               "韵律会很单一，抗遗忘效果打折。建议 0.8~1.0。"))
        if not 8 <= int(self.max_mel_tokens) <= 1815:
            n.append(GD.Notice("error",
                               f"max_mel_tokens={self.max_mel_tokens} 超出 8~1815"))
        return n


# ---------------------------------------------------------------------------
# 参考音频的自动挑选
# ---------------------------------------------------------------------------

def candidate_prompts() -> List[Dict[str, str]]:
    """列出可用的参考音频候选（examples/ 与音色库）。"""
    out: List[Dict[str, str]] = []
    seen = set()

    def add(p: str, src: str) -> None:
        ap = os.path.abspath(p)
        if ap in seen or not os.path.isfile(ap):
            return
        seen.add(ap)
        out.append({"path": ap, "source": src, "name": os.path.basename(ap)})

    ex = os.path.join(PROJECT_ROOT, "examples")
    if os.path.isdir(ex):
        for root, _d, files in os.walk(ex):
            for fn in sorted(files):
                if fn.lower().endswith(DS.AUDIO_EXTS):
                    add(os.path.join(root, fn), "examples/")
    assets = os.path.join(PROJECT_ROOT, "assets")
    if os.path.isdir(assets):
        for fn in sorted(os.listdir(assets)):
            if fn.lower().endswith(DS.AUDIO_EXTS):
                add(os.path.join(assets, fn), "assets/")
    try:
        from webui_app.services import voice_bank as VB
        # list_voices() 返回的是 VoiceEntry **dataclass**（不是 dict），
        # 音频路径在 `.audio_path` 这个 property 上（BANK_DIR + 相对路径）。
        for v in VB.list_voices():
            p = getattr(v, "audio_path", "") or ""
            if p:
                add(p, f"音色库/{getattr(v, 'name', '')}")
    except Exception:
        pass
    return out


def auto_pick_prompt(exclude_dataset: str = "") -> Tuple[str, str]:
    """自动挑一个参考音频，返回 (路径, 说明)。挑不到返回 ("", 原因)。

    会刻意避开 `exclude_dataset` 里的音频 —— 那是目标角色的素材，
    用它做回放提示等于把回放变成目标数据的复制品。
    """
    banned: set = set()
    if exclude_dataset and DS.exists(exclude_dataset):
        try:
            d = DS.dir_of(exclude_dataset)
            for root, _dirs, files in os.walk(os.path.join(d, DS.AUDIO_SUBDIR)):
                for fn in files:
                    banned.add(os.path.abspath(os.path.join(root, fn)))
        except Exception:
            pass
    for c in candidate_prompts():
        if os.path.abspath(c["path"]) not in banned:
            return c["path"], f"自动选中 {c['source']}{c['name']}"
    if banned:
        return "", ("所有候选参考音频都属于目标数据集，已排除。"
                    "请手动指定一段**其它人**的干净语音作为回放提示。")
    return "", "没有找到任何候选参考音频（examples/ 与音色库都是空的）"


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def _adapter_scale_ctx(tts, on: bool):
    """把引擎上所有 LoRA 的强度临时置 0（或恢复），返回改了哪些。

    用 `set_adapter_scale(0)` 而不是 `detach_lora()`：
    后者会替换 `tts.gpt` 这个对象，引擎里别处缓存的引用就全断了
    （参考音频缓存、accel 引擎都持有它）。把 scaling 置 0 在数学上
    完全等价于纯底座，但模块树一个字节都不动。
    """
    touched = []
    for tag, getter in (("gpt", lambda: getattr(tts, "gpt", None)),
                        ("cfm", lambda: (getattr(getattr(tts, "s2mel", None),
                                                 "models", {}) or {}).get("cfm"))):
        m = getter()
        if m is None or not hasattr(m, "peft_config"):
            continue
        try:
            if on:
                GD.set_adapter_scale(m, 0.0)
            else:
                GD.set_adapter_scale(m, 1.0)
            touched.append(tag)
        except Exception:
            pass
    return touched


def build_distill_dataset(engine, options: Optional[DistillOptions] = None,
                          name: str = "",
                          progress: Optional[Callable[[float, str], None]] = None,
                          warn_target_prompt: Optional[str] = None
                          ) -> Dict[str, Any]:
    """用底座自合成一批通用语音，存成数据集并提取特征。

    `engine` 是已加载的 `TTSEngine`（复用它那份模型，不再加载第二份 ——
    8GB 卡上装不下两份）。`warn_target_prompt` 给目标数据集名，
    用来检查参考音频是不是误用了目标角色的素材。

    返回 dict：{ok, dataset, n_texts, synthesized, failed, features, warnings, seconds}
    """
    opts = options or DistillOptions()
    name = DS.safe_dataset_name(name or GD.DISTILL_DATASET)
    warns: List[str] = []
    out: Dict[str, Any] = {
        "ok": False, "dataset": name, "n_texts": 0, "synthesized": 0,
        "failed": 0, "features": 0, "warnings": warns, "seconds": 0.0,
        "errors": [], "prompt_audio": opts.prompt_audio,
        "adapter_muted": [],
    }
    t0 = time.perf_counter()

    notes = [x for x in opts.validate() if x.level == "error"]
    if notes:
        out["errors"] = [x.message for x in notes]
        return out

    texts = generic_texts(opts.n_texts, opts.categories or None)
    out["n_texts"] = len(texts)

    # 参考音频误用检查：这是**最常见**也最难自己发现的配置错误
    if warn_target_prompt and DS.exists(warn_target_prompt):
        try:
            d = DS.dir_of(warn_target_prompt)
            tgt = {os.path.abspath(u.audio_abs(d))
                   for u in DS.load_meta(warn_target_prompt) if u.audio}
            if os.path.abspath(opts.prompt_audio) in tgt:
                warns.append(
                    f"参考音频来自目标数据集 `{warn_target_prompt}`。"
                    "这样蒸出来的回放集全是目标角色的音色，抗遗忘效果会大打折扣 —— "
                    "回放要保住的是「底座遇到通用文本时的行为」，"
                    "请换一段其它人的干净语音。")
        except Exception:
            pass

    if engine is None:
        out["errors"] = ["引擎未提供。请先在「系统」页加载模型。"]
        return out
    tts = getattr(engine, "tts", None)
    if tts is None:
        out["errors"] = ["引擎还没加载模型。请先加载再蒸馏。"]
        return out

    tmp = tempfile.mkdtemp(prefix="distill_")
    muted: List[str] = []
    import torch
    try:
        # ---- 红线 1：蒸馏必须在纯底座上进行 ----
        muted = _adapter_scale_ctx(tts, True)
        out["adapter_muted"] = muted
        if muted:
            warns.append(
                f"检测到引擎上挂着 LoRA（{', '.join(muted)}），已把强度临时置 0。"
                "否则蒸出来的是「已经适配过的行为」，回放会强化偏移而不是抗遗忘。")

        def emit(f: float, m: str) -> None:
            if progress:
                progress(float(f), m)

        emit(0.02, f"准备合成 {len(texts)} 条通用语音")
        wavs: List[Tuple[str, str, str]] = []        # (path, text, category)
        for i, item in enumerate(texts):
            txt, cat = item["text"], item["category"]
            dst = os.path.join(tmp, f"distill_{i:03d}_{cat}.wav")
            try:
                kwargs: Dict[str, Any] = dict(
                    spk_audio_prompt=opts.prompt_audio,
                    text=txt, output_path=dst, lang=(opts.lang or "ZH").upper(),
                    verbose=False,
                    max_text_tokens_per_segment=120,
                    duration_factor=float(opts.duration_factor),
                    interval_silence=200,
                    text_normalization=bool(opts.text_normalization),
                    do_sample=bool(opts.do_sample),
                    top_p=float(opts.top_p),
                    top_k=(int(opts.top_k) if int(opts.top_k) > 0 else None),
                    temperature=float(opts.temperature),
                    repetition_penalty=float(opts.repetition_penalty),
                    max_mel_tokens=int(opts.max_mel_tokens),
                )
                if opts.emo_audio and os.path.isfile(opts.emo_audio):
                    kwargs["emo_audio_prompt"] = opts.emo_audio
                # 每条换一个种子：回放样本需要**多样性**，
                # 固定种子会让同一句文本每次蒸出完全一样的韵律。
                torch.manual_seed(int(opts.seed) + i)
                emit(0.05 + 0.75 * i / max(1, len(texts)),
                     f"合成 {i+1}/{len(texts)}：{txt[:18]}…")
                res = engine.infer(**kwargs)
                p = res if isinstance(res, str) else dst
                if p and os.path.isfile(p):
                    wavs.append((p, txt, cat))
                else:
                    out["failed"] += 1
                    out["errors"].append(f"第 {i+1} 条没有产出音频：{txt[:20]}")
            except Exception as e:
                out["failed"] += 1
                out["errors"].append(f"第 {i+1} 条失败：{type(e).__name__}: {e}")
                warns.append(f"「{txt[:16]}…」合成失败（{type(e).__name__}），已跳过")
        out["synthesized"] = len(wavs)
        if not wavs:
            out["errors"].append("一条都没合成成功，无法建立回放集")
            return out

        # ---- 建数据集 ----
        emit(0.82, f"写入数据集 {name}")
        if not DS.exists(name):
            DS.create(name, note="底座蒸馏的通用回放集（自动生成，请勿手工改动）",
                      lang_default=opts.lang or "ZH")
        elif opts.overwrite:
            DS.delete(name)
            DS.create(name, note="底座蒸馏的通用回放集（自动生成，请勿手工改动）",
                      lang_default=opts.lang or "ZH")
        else:
            warns.append(f"数据集 `{name}` 已存在，本次以**追加**方式写入"
                         "（想重建请勾上「覆盖」）")

        imp = DS.import_audio(name, [w[0] for w in wavs], copy=True,
                              lang=(opts.lang or "ZH").upper())
        out["imported"] = {k: v for k, v in imp.items() if k != "ids"}

        # 文本回填只能用 `imp["ids"]` 的**顺序**对齐，不能拿文件名去匹：
        # import_audio 在 copy=True 时会把文件重命名为 `<uid>.wav`（dataset.py:424），
        # 原始的 `distill_003_number.wav` 这个名字根本不会出现在 meta 里。
        # 也不能无条件 zip：一旦有文件被跳过，位置就整体错位，
        # 文本会张冠李戴 —— 而这种错在训练里不报错，只会默默学错对应关系。
        # 所以先确认数量对得上，对不上就直接失败。
        ids = list(imp.get("ids") or [])
        if len(ids) != len(wavs):
            out["errors"].append(
                f"导入数量对不上（写入 {len(ids)} / 合成 {len(wavs)}），"
                "无法保证文本与音频的对应关系，已放弃本次生成。"
                f"跳过：{imp.get('skipped')} · 失败：{imp.get('failed')}")
            return out
        n_txt = FT._apply_meta(name, {uid: {"text": txt}
                                      for uid, (_p, txt, _c) in zip(ids, wavs)})
        out["texts_written"] = n_txt
        if n_txt != len(wavs):
            warns.append(f"只有 {n_txt}/{len(wavs)} 条回填了文本")
        DS.set_info(name, distill_options=opts.to_dict(),
                    distilled_at=time.time(),
                    prompt_audio=os.path.abspath(opts.prompt_audio))

        # ---- 特征提取：走与目标数据完全相同的链路 ----
        if opts.extract_features:
            emit(0.88, "提取特征（与目标数据同一条链路）")
            # EngineStats 里没有 device；真正的设备字符串在 AppConfig.device.device_str
            dev = str(getattr(getattr(getattr(engine, "cfg", None), "device", None),
                              "device_str", "") or "")
            if not dev:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            r = FT.extract_dataset(name, tts=tts, device=dev,
                                   overwrite=bool(opts.overwrite),
                                   progress=lambda f, m: emit(
                                       0.88 + 0.12 * f, f"提取特征：{m}"))
            out["features"] = int(r.get("extracted") or 0)
            out["feature_failed"] = int(r.get("failed") or 0)
            out["feature_skipped"] = int(r.get("skipped") or 0)
            for w in (r.get("warnings") or [])[:5]:
                warns.append(str(w))
            if out["features"] == 0 and out["feature_skipped"] == 0:
                out["errors"].append("特征全部提取失败，回放集不可用")
                return out
        DS.refresh_all(name, require_features=bool(opts.extract_features))
        out["stats"] = DS.stats(name)
        out["ok"] = True
        emit(1.0, f"回放集就绪：{name}（{out['stats'].get('ready', 0)} 条可用）")
    finally:
        if muted:
            _adapter_scale_ctx(tts, False)         # 恢复强度，别把用户的推理也静音了
        if not opts.keep_wav:
            shutil.rmtree(tmp, ignore_errors=True)
        out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def distill_markdown(res: Dict[str, Any]) -> str:
    """把生成结果渲染成 Markdown。"""
    if not res.get("ok"):
        L = ["### 🔴 回放集生成失败", ""]
        for e in (res.get("errors") or [])[:8]:
            L.append(f"> ✖ {e}")
        return "\n".join(L)
    st = res.get("stats") or {}
    L = ["### ✅ 回放集已就绪", "",
         "| 项 | 值 |", "|---|---|",
         f"| 数据集 | `{res.get('dataset')}` |",
         f"| 文本条数 | {res.get('n_texts', 0)} |",
         f"| 合成成功 | {res.get('synthesized', 0)} |",
         f"| 合成失败 | {res.get('failed', 0)} |",
         f"| 特征提取 | {res.get('features', 0)} 条"
         f"（跳过 {res.get('feature_skipped', 0)}，失败 {res.get('feature_failed', 0)}） |",
         f"| 可用样本 | {st.get('ready', 0)} |",
         f"| 参考音频 | `{os.path.basename(res.get('prompt_audio') or '')}` |",
         f"| 被静音的 adapter | {res.get('adapter_muted') or '无（引擎是纯底座）'} |",
         f"| 耗时 | {res.get('seconds', 0.0):.0f} 秒 |", ""]
    if res.get("warnings"):
        L.append("<details open><summary>注意事项</summary>")
        L.append("")
        for w in res["warnings"]:
            L += [f"> ⚠️ {w}", ">"]
        if L[-1] == ">":
            L.pop()
        L += ["", "</details>", ""]
    if res.get("errors"):
        L += ["<details><summary>失败明细</summary>", ""]
        for e in res["errors"][:20]:
            L.append(f"- {e}")
        L += ["", "</details>", ""]
    L += ["", "> 现在可以在训练页把 `回放源` 选成 **底座蒸馏集**，",
          "> `replay_ratio` 建议 0.3（数据少于 5 分钟时提到 0.5）。"]
    return "\n".join(L)
