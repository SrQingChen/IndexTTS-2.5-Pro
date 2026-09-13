"""读音纠正服务 —— 拼音 / CMU 音素 / 日语假名标注。

IndexTTS 支持在文本里用 `<文字|发音>` 强制指定某个字/词的读法，
这是**纠正读错字最直接的手段**，比调任何采样参数都有效。

语法（已用 tools/pinyin_probe.py 实测验证，1728 条拼音全部往返无损）：

    中文  他在银<行|XING2>里办事          → 拼音 + 声调数字 1~5（5=轻声）
    英文  a <minute|M IH1 . N AH0 T>      → CMU 音素，空格分音节，数字是重音
    日语  料理が<上手|じょうず>だが        → 直接写假名

官方处理链路（infer_v2_5.py:699-725）：
    1. clean_pattern 替换全角标点
    2. TextNormalizer.normalize()  —— 内部会先保护 <字|音> 标注不被破坏
    3. 中文/英文/日文转小写
    4. apply_pronunciation_annotations() 把标注展开成特殊 token：
         文字含中文 → <|SPECIAL_TOKEN_2|>发音<|SPECIAL_TOKEN_2|>
         否则       → <|SPECIAL_TOKEN_1|>发音<|SPECIAL_TOKEN_1|>
         发音是纯假名 → 不加特殊 token，只用空格包裹
    5. 特殊 token 名统一大写
    6. tiktoken 编码（allowed_special='all'）

本模块**直接调用官方的 apply_pronunciation_annotations / TextNormalizer /
get_tokenizer**，不自己复刻 —— 这样官方改了行为我们会自动跟随，不会静默偏离。
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT

# 与官方 infer_v2_5.py:38 的 PRONUNCIATION_ANNOTATION_PATTERN 保持一致
ANNOTATION_RE = re.compile(r'<([^|>\n]+)\|([^>\n]+)>')

# 声调数字含义，给 UI 提示用
TONE_LABELS = {
    "1": "阴平（第一声，高平）",
    "2": "阳平（第二声，上升）",
    "3": "上声（第三声，降升）",
    "4": "去声（第四声，下降）",
    "5": "轻声（第五声，短弱）",
}

_cache_lock = threading.Lock()
_pinyin_vocab: Optional[frozenset] = None
_normalizer = None
_tokenizer = None


# ---------------------------------------------------------------------------
# 词表与官方组件（懒加载 + 缓存）
# ---------------------------------------------------------------------------

def pinyin_vocab(model_dir: Optional[str] = None) -> frozenset:
    """checkpoints/pinyin.vocab 里的合法拼音清单（大写，含声调数字）。

    这个文件在官方代码里**从未被引用**，只在 README 里作为「合法拼音清单」
    给人看。我们拿它做输入校验：不在清单里的拼音，模型大概率没学过。
    """
    global _pinyin_vocab
    with _cache_lock:
        if _pinyin_vocab is None:
            path = os.path.join(model_dir or os.path.join(PROJECT_ROOT, "checkpoints"),
                                "pinyin.vocab")
            words = set()
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            words.add(w.upper())
            _pinyin_vocab = frozenset(words)
        return _pinyin_vocab


def _official():
    """取官方的文本处理器与分词器（首次调用较慢，之后走缓存）。"""
    global _normalizer, _tokenizer
    with _cache_lock:
        if _normalizer is None:
            from indextts.utils.front import TextNormalizer
            tn = TextNormalizer(enable_glossary=True)
            tn.load()          # 必须显式调，否则 normalize() 返回空串
            _normalizer = tn
        if _tokenizer is None:
            from indextts.utils.tokenizer import get_tokenizer
            _tokenizer = get_tokenizer(
                multilingual=True,
                model_dir=os.path.join(PROJECT_ROOT, "checkpoints"))
        return _normalizer, _tokenizer


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

@dataclass
class Annotation:
    """一条 `<文字|发音>` 标注。"""
    word: str                 # 要纠正读法的文字
    pron: str                 # 指定发音（原始大小写）
    start: int = 0            # 在原字符串中的起始下标
    end: int = 0              # 结束下标（不含）
    kind: str = "unknown"     # pinyin | cmu | kana | suspicious

    @property
    def pron_upper(self) -> str:
        return self.pron.strip().upper()

    @property
    def tone(self) -> str:
        """中文拼音的声调数字，取不到返回 ''。"""
        p = self.pron_upper
        return p[-1] if p and p[-1] in "12345" else ""

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "pron": self.pron, "kind": self.kind,
                "start": self.start, "end": self.end}


_KANA_RE = re.compile(r'^[\u3040-\u309F\u30A0-\u30FF]+$')
_HAN_RE = re.compile(r'[\u4e00-\u9fff]')
# 拼音：纯字母 + 可选一位数字，不含空格与点。
# 注意这里故意**不限制数字必须 1~5**：写错声调（如 XING9）应该先被归为
# 拼音，再报「声调数字 9 非法」，而不是笼统地说「无法识别」——
# 后者对用户没任何帮助。
_PINYIN_BODY_RE = re.compile(r'^[A-Z]+[0-9]?$')
VALID_TONES = "12345"
# CMU 音素：字母/数字/点/空格，且至少有一个「音素+重音数字」组合
_CMU_CHARS_RE = re.compile(r'^[A-Z0-9.\s]+$')
_CMU_STRESS_RE = re.compile(r'[A-Z]{2}[0-2]')


def classify(word: str, pron: str) -> str:
    """判断这条标注属于哪种体系。

    先看**发音的格式**而不是先看文字：因为官方 apply_pronunciation_annotations
    是用「文字里有没有汉字」来决定包 SPECIAL_TOKEN_1 还是 _2 的，
    而发音格式决定的是「模型能不能看懂」。两件事得分开判断。
    """
    p = (pron or "").strip()
    if not p:
        return "suspicious"
    if _KANA_RE.fullmatch(p):
        return "kana"                      # 日语假名 → 官方不加特殊 token
    up = p.upper()
    if _PINYIN_BODY_RE.fullmatch(up):
        return "pinyin"
    if _CMU_CHARS_RE.fullmatch(up) and _CMU_STRESS_RE.search(up):
        return "cmu"
    return "suspicious"


def parse(text: str) -> List[Annotation]:
    """解析出文本里所有 `<文字|发音>` 标注。"""
    out: List[Annotation] = []
    for m in ANNOTATION_RE.finditer(text or ""):
        word, pron = m.group(1), m.group(2)
        out.append(Annotation(word=word, pron=pron, start=m.start(), end=m.end(),
                              kind=classify(word, pron)))
    return out


def strip_annotations(text: str) -> str:
    """去掉标注，只留原字（用于对照「不标注时模型会怎么读」）。"""
    return ANNOTATION_RE.sub(lambda m: m.group(1), text or "")


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    level: str              # error | warn | info
    message: str
    snippet: str = ""


def validate(text: str, lang: str = "ZH",
             model_dir: Optional[str] = None) -> List[Issue]:
    """检查标注写得对不对。返回问题列表（空 = 没问题）。

    校验项都是从官方代码的实际行为推出来的，不是凭空定的规则：
        · 尖括号是否成对（未闭合的标注会被当普通文本，静默失效）
        · 拼音是否在 pinyin.vocab 里（不在 = 模型没学过，标了也白标）
        · 声调数字是否 1~5
        · jqx 后面的 u 是否该写 v（官方 correct_pinyin 会自动改，但容易困惑）
        · 语言与标注体系是否匹配（中文文本里写 CMU 音素大概率是写错了）
    """
    issues: List[Issue] = []
    text = text or ""
    lang = (lang or "ZH").upper()
    vocab = pinyin_vocab(model_dir)

    # 1) 尖括号配对
    n_lt, n_gt = text.count("<"), text.count(">")
    if n_lt != n_gt:
        issues.append(Issue(
            "error",
            f"尖括号不配对：`<` 有 {n_lt} 个，`>` 有 {n_gt} 个。"
            "未闭合的标注会被当成普通文本读出来，等于没标。"))

    # 2) 逐个检查 <...> 片段是不是合法标注。
    #    不用 ANNOTATION_RE 来查——它要求各部分非空，所以 `<|>` `<行|>`
    #    这类写错的东西它直接不匹配，会静默漏报。
    _LANG_PREFIX = {"zh", "en", "ja", "es", "ar", "yue", "zhen"}
    _SPECIAL = re.compile(r'^\|[A-Z_0-9]+\|$')
    for m in re.finditer(r'<([^<>\n]*)>', text):
        inner = m.group(1)
        if inner.strip().lower().strip("|") in _LANG_PREFIX:
            continue                     # <|zh|> 这类语言前缀，合法
        if _SPECIAL.match(inner):
            continue                     # 用户粘了已展开的特殊 token
        if "|" not in inner:
            issues.append(Issue(
                "error",
                f"`<{inner}>` 不是合法标注 —— 缺少竖线 `|`。"
                "正确格式是 `<文字|发音>`。", inner))
            continue
        parts = inner.split("|")
        if len(parts) != 2:
            issues.append(Issue(
                "error",
                f"`<{inner}>` 里有 {len(parts) - 1} 个竖线，应该恰好 1 个。", inner))
        elif not parts[0].strip():
            issues.append(Issue(
                "error", f"`<{inner}>` 的文字部分是空的。", inner))
        elif not parts[1].strip():
            issues.append(Issue(
                "error", f"`<{inner}>` 的发音部分是空的。", inner))

    anns = parse(text)
    if not anns and n_lt == 0:
        return issues

    for a in anns:
        tag = f"`<{a.word}|{a.pron}>`"

        if a.kind == "pinyin":
            pu = a.pron_upper
            if not pu[-1:].isdigit():
                issues.append(Issue(
                    "warn",
                    f"{tag} 没有声调数字。拼音必须带 1~5 的声调"
                    "（5 = 轻声），例如 `XING2`。不带声调模型可能识别不了。",
                    a.pron))
            elif pu[-1] not in VALID_TONES:
                issues.append(Issue(
                    "error",
                    f"{tag} 的声调数字 `{pu[-1]}` 非法，只能是 "
                    f"{'/'.join(VALID_TONES)}（5 = 轻声）。", a.pron))
            if vocab and pu not in vocab:
                # 尝试给出相近的合法候选
                base = pu[:-1] if pu[-1:].isdigit() else pu
                near = sorted(w for w in vocab
                              if w.startswith(base[:2]) and (not pu[-1:].isdigit()
                                                             or w.endswith(pu[-1])))
                hint = f"，是不是想写 {' / '.join(near[:6])}？" if near else ""
                issues.append(Issue(
                    "warn",
                    f"{tag} 的拼音 `{pu}` 不在 `pinyin.vocab`（{len(vocab)} 条）里{hint}"
                    " 模型训练时可能没见过这个拼写，标注未必生效。",
                    a.pron))
            # jqx + u 的提示
            if pu and pu[0] in "JQX" and "U" in pu[1:] and "V" not in pu:
                issues.append(Issue(
                    "info",
                    f"{tag}：官方 `correct_pinyin()` 会把 jqx 后的 `U` 自动改成 `V`"
                    f"（ü），所以 `{pu}` 实际会按 "
                    f"`{_jqx_fix(pu)}` 处理。直接写 V 也可以。",
                    a.pron))

        elif a.kind == "suspicious":
            issues.append(Issue(
                "error",
                f"{tag} 的发音 `{a.pron}` 既不是合法拼音（字母+声调数字），"
                "也不是 CMU 音素或假名。请检查格式。",
                a.pron))

        elif a.kind == "cmu":
            if _HAN_RE.search(a.word):
                issues.append(Issue(
                    "warn",
                    f"{tag} 的文字是中文但发音写成了 CMU 音素。"
                    "官方会因为「文字含汉字」而把它包进 `<|SPECIAL_TOKEN_2|>`"
                    "（中文槽位），但里面装的是英文音素 —— 大概率不是你想要的。"
                    "中文请改用拼音标注。",
                    a.pron))
            elif lang in ("ZH", "ZHEN"):
                issues.append(Issue(
                    "info",
                    f"{tag} 用了 CMU 音素格式，但语言设的是 {lang}。"
                    "中文文本里混英文词是可以的，但如果这个词本该按中文读，"
                    "应该用拼音标注。",
                    a.pron))
            if not re.search(r'[0-2]', a.pron.upper()):
                issues.append(Issue(
                    "warn",
                    f"{tag} 的 CMU 音素没有重音数字（0/1/2）。"
                    "标准写法如 `M IH1 . N AH0 T`，空格分音节、数字表重音。",
                    a.pron))

        elif a.kind == "kana":
            if lang not in ("JA",):
                issues.append(Issue(
                    "info",
                    f"{tag} 用了日语假名，但语言设的是 {lang}。"
                    "假名标注只在 lang=JA 时有意义。",
                    a.pron))

    # 语言前缀与标注体系的总体一致性
    if lang == "JA":
        py = [a for a in anns if a.kind == "pinyin"]
        if py:
            issues.append(Issue(
                "warn",
                f"lang=JA 但有 {len(py)} 条拼音标注。日语请用假名标注，"
                "例如 `<上手|じょうず>`。"))
    return issues


def _jqx_fix(pu: str) -> str:
    """复刻官方 TextNormalizer.correct_pinyin 的 jqx+u→v 规则。"""
    if not pu or pu[0] not in "JQX":
        return pu
    return re.sub(r"U", "V", pu)


def validate_html(text: str, lang: str = "ZH",
                  model_dir: Optional[str] = None) -> str:
    """把 validate() 渲染成一行状态条（合成页实时校验用）。

    用 theme.py 里已有的 ix-* 类名，不引入新样式。
    """
    text = text or ""
    anns = parse(text)
    if not text.strip():
        return ""
    if not anns and "<" not in text:
        # 没有标注也没有尖括号：不给任何干扰信息，
        # 只留一个极轻的提示，告诉用户这个功能存在。
        return ('<div class="ix-hint" style="opacity:.72">'
                '没有读音标注。若某个字读错了，展开上方「🔤 读音纠正」'
                '查候选并一键插入 <code>&lt;字|拼音&gt;</code>。</div>')
    try:
        issues = validate(text, lang, model_dir)
    except Exception as e:
        return f'<div class="ix-err">✖ 校验异常：{type(e).__name__}: {e}</div>'

    n_err = sum(1 for i in issues if i.level == "error")
    n_warn = sum(1 for i in issues if i.level == "warn")
    n_info = sum(1 for i in issues if i.level == "info")

    if n_err:
        head = (f'<div class="ix-err">✖ 标注有 <b>{n_err}</b> 个错误'
                f'<span style="opacity:.7">（共识别到 {len(anns)} 条标注）</span></div>')
    elif n_warn:
        head = (f'<div class="ix-warn">⚠️ 识别到 <b>{len(anns)}</b> 条标注，'
                f'有 {n_warn} 处需留意</div>')
    elif anns:
        kinds = sorted({a.kind for a in anns})
        kn = {"pinyin": "拼音", "cmu": "CMU 音素", "kana": "假名",
              "suspicious": "未知"}
        head = (f'<div class="ix-tip">✅ <b>{len(anns)}</b> 条标注全部合法'
                f'<span style="opacity:.7">（{" / ".join(kn.get(k, k) for k in kinds)}）</span></div>')
    else:
        head = '<div class="ix-warn">⚠️ 文本里有尖括号但没解析出合法标注</div>'

    body = []
    icon = {"error": "✖", "warn": "⚠️", "info": "ℹ️"}
    for i in issues[:8]:
        cls = {"error": "ix-err", "warn": "ix-warn", "info": "ix-hint"}[i.level]
        body.append(f'<div class="{cls}" style="padding:6px 11px;margin:3px 0">'
                    f'{icon[i.level]} {i.message}</div>')
    if len(issues) > 8:
        body.append(f'<div class="ix-hint">… 另有 {len(issues) - 8} 条提示未显示</div>')
    if n_info and not n_err and not n_warn:
        body.append('<div class="ix-hint" style="opacity:.75">ℹ️ 以上为信息级提示，'
                    '不影响合成。</div>')
    return head + "".join(body)


# ---------------------------------------------------------------------------
# 候选表（给 Dataframe 用）
# ---------------------------------------------------------------------------

CAND_HEADERS = ["字", "读音", "声调", "在词表", "默认", "可直接复制的标注"]


def candidate_rows(word: str, model_dir: Optional[str] = None) -> List[List[str]]:
    """把 heteronyms() 的结果转成 Dataframe 行。

    最后一列是**单字**标注：官方只实测过单音节标注，
    把整个词写成 `<银行|YIN2HANG2>` 这类多音节拼法模型未必认，
    所以统一推荐逐字标注。
    """
    rows: List[List[str]] = []
    for c in heteronyms(word, model_dir):
        mark = {True: "✅", False: "❌", None: "—"}[c["in_vocab"]]
        ch = c["char"] or "字"
        rows.append([ch, c["pinyin"], c["tone_label"] or "—", mark,
                     "⭐" if c["is_default"] else "", f"<{ch}|{c['pinyin']}>"])
    return rows


def candidates_note(word: str, model_dir: Optional[str] = None) -> str:
    """候选表的说明文字。"""
    cands = heteronyms(word, model_dir)
    if not cands:
        return (f"_查不到 `{word}` 的读音候选。请确认输入的是中文"
                "（英文用 CMU 音素、日语用假名，需手写）。_")
    n_def = sum(1 for c in cands if c["is_default"])
    n_bad = sum(1 for c in cands if c["in_vocab"] is False)
    L = [f"**`{word}`** 共 {len(cands)} 个候选读音，"
         f"其中 {n_def} 个是默认读法。", "",
         "> **点击表格任意一行**，会自动把「字」与「读音」填进下方输入框。",
         "> ⭐ = pypinyin 认为最常用的读法。**模型读错时就选一个非 ⭐ 的**。"]
    if n_bad:
        L.append(f"> ❌ 有 {n_bad} 个读音不在 `pinyin.vocab` 里，标了大概率无效，别选。")
    if len(word) > 1:
        L.append("")
        L.append("> 词组查询走的是**词组感知**模式（已按语境消歧），"
                 "所以候选里不会列出无关的多音字；"
                 "想看某个字的全部读法，就只输入那**一个字**。")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 预览官方管线的实际输出
# ---------------------------------------------------------------------------

def preview(text: str, lang: str = "ZH", normalize: bool = True) -> Dict[str, Any]:
    """跑一遍官方的文本处理链路，返回模型真正会看到的东西。

    这是本功能里**最有诊断价值**的一项：标注写错时，
    从处理后的文本一眼就能看出标注没被展开成特殊 token。
    """
    tn, tok = _official()
    from indextts.infer_v2_5 import apply_pronunciation_annotations

    lang = (lang or "ZH").upper()
    lang_prefix = f'<|{lang.lower()}|> '
    t = text or ""

    steps: List[Tuple[str, str]] = [("原始输入", t)]
    t = tn.clean_pattern.sub(lambda x: tn.char_rep_map[x.group()], t)
    steps.append(("① 标点清理", t))
    if normalize and lang.lower() in ("zh", "zhen", "en"):
        t = tn.normalize(t)
        steps.append(("② 文本归一化", t))
    if lang.lower() in ("ja", "zh", "zhen", "en"):
        t = t.lower()
        steps.append(("③ 转小写", t))
    t = apply_pronunciation_annotations(t)
    steps.append(("④ 展开发音标注", t))
    t = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', t)
    steps.append(("⑤ 特殊 token 大写", t))

    final = lang_prefix + t
    steps.append(("⑥ 加语言前缀（最终）", final))

    ids = tok.encode(final, allowed_special="all")
    enc = tok.encoding
    try:
        toks = [enc.decode_single_token_bytes(i).decode("utf-8", "replace")
                for i in ids]
    except Exception:
        toks = []

    n_st1 = final.count("<|SPECIAL_TOKEN_1|>")
    n_st2 = final.count("<|SPECIAL_TOKEN_2|>")
    return {
        "steps": steps,
        "final": final,
        "token_ids": list(ids),
        "tokens": toks,
        "token_count": len(ids),
        "n_special_1": n_st1,
        "n_special_2": n_st2,
        "annotations": [a.to_dict() for a in parse(text or "")],
        "paired": (n_st1 % 2 == 0) and (n_st2 % 2 == 0),
    }


def preview_markdown(text: str, lang: str = "ZH", normalize: bool = True) -> str:
    """把 preview() 的结果渲染成 Markdown。"""
    try:
        p = preview(text, lang, normalize)
    except Exception as e:
        return f"**预览失败**：{type(e).__name__}: {e}"

    anns = p["annotations"]
    L: List[str] = []
    if not anns:
        L.append("> 文本里没有 `<文字|发音>` 标注。")
    else:
        L += [f"**识别到 {len(anns)} 条标注**", "",
              "| # | 文字 | 发音 | 体系 | 展开后 |", "|---|---|---|---|---|"]
        for i, a in enumerate(anns, 1):
            kind = {"pinyin": "中文拼音", "cmu": "CMU 音素",
                    "kana": "日语假名", "suspicious": "⚠️ 无法识别"}.get(a["kind"], a["kind"])
            tok = ("`<|SPECIAL_TOKEN_2|>`" if a["kind"] == "pinyin"
                   else "`<|SPECIAL_TOKEN_1|>`" if a["kind"] == "cmu"
                   else "（不加特殊 token）" if a["kind"] == "kana" else "—")
            L.append(f"| {i} | {a['word']} | `{a['pron']}` | {kind} | {tok} |")
        L.append("")

    if not p["paired"]:
        L.append("> ❌ 特殊 token 数量为奇数，说明有标注没正确闭合。")
        L.append("")

    L += ["<details open><summary>处理链路逐步结果</summary>", "",
          "| 步骤 | 结果 |", "|---|---|"]
    for name, val in p["steps"]:
        safe = (val or "").replace("|", "\\|").replace("\n", " ")
        if len(safe) > 150:
            safe = safe[:150] + "…"
        L.append(f"| {name} | `{safe or '(空)'}` |")
    L += ["", "</details>", ""]

    L += [f"**分词结果**：{p['token_count']} 个 token", ""]
    if p["tokens"]:
        shown = p["tokens"][:120]
        L.append("```")
        L.append(" ".join(shown))
        if len(p["tokens"]) > 120:
            L.append(f"… （共 {len(p['tokens'])} 个，仅显示前 120）")
        L.append("```")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 多音字候选
# ---------------------------------------------------------------------------

def heteronyms(word: str, model_dir: Optional[str] = None,
               limit: int = 24) -> List[Dict[str, Any]]:
    """给出一个词/字的合法读音候选。

    用 pypinyin 取候选（含多音字），再用 pinyin.vocab **过滤**——
    只推荐模型确实学过的拼写，避免推荐一个标了也没用的读音。
    """
    word = (word or "").strip()
    if not word:
        return []
    vocab = pinyin_vocab(model_dir)
    out: List[Dict[str, Any]] = []
    seen = set()

    try:
        from pypinyin import Style, pinyin as _py
        # heteronym=True 给出每个字的全部候选读音
        cands = _py(word, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)
    except Exception:
        cands = []

    default = []
    try:
        from pypinyin import Style, pinyin as _py2
        default = [x[0] for x in
                   _py2(word, style=Style.TONE3, neutral_tone_with_five=True)]
    except Exception:
        pass

    for zi_idx, group in enumerate(cands):
        for raw in group:
            pu = str(raw).upper().replace("Ü", "V")
            pu = _jqx_fix(pu)
            if not pu or pu in seen:
                continue
            seen.add(pu)
            tone = pu[-1] if pu[-1:].isdigit() else ""
            in_vocab = (pu in vocab) if vocab else None
            out.append({
                "char_index": zi_idx,
                "char": word[zi_idx] if zi_idx < len(word) else "",
                "pinyin": pu,
                "tone": tone,
                "tone_label": TONE_LABELS.get(tone, ""),
                "in_vocab": in_vocab,
                "is_default": bool(default) and zi_idx < len(default)
                              and pu == _jqx_fix(str(default[zi_idx]).upper()
                                                 .replace("Ü", "V")),
            })

    # 已验证过的排前面，默认读音再优先
    out.sort(key=lambda d: (not d["is_default"], d["in_vocab"] is False, d["pinyin"]))
    return out[:limit]


def heteronym_markdown(word: str, model_dir: Optional[str] = None) -> str:
    cands = heteronyms(word, model_dir)
    if not cands:
        return f"_查不到 `{word}` 的读音候选。请确认输入的是中文。_"
    by_char: Dict[int, List[Dict[str, Any]]] = {}
    for c in cands:
        by_char.setdefault(c["char_index"], []).append(c)
    L = [f"**`{word}` 的读音候选**（已用 `pinyin.vocab` 校验，✅ = 模型学过）", "",
         "| 字 | 候选读音 | 声调 | 在词表 | 默认 | 可直接复制的标注 |",
         "|---|---|---|---|---|---|"]
    for idx in sorted(by_char):
        for c in by_char[idx]:
            mark = {True: "✅", False: "❌", None: "—"}[c["in_vocab"]]
            L.append(
                f"| {c['char']} | `{c['pinyin']}` | {c['tone_label'] or '—'} "
                f"| {mark} | {'⭐' if c['is_default'] else ''} "
                f"| `<{'字' if not c['char'] else c['char']}|{c['pinyin']}>` |")
    L.append("")
    L.append("> ⭐ 是 pypinyin 认为最常用的读音。**如果模型读错了，"
             "就选一个非 ⭐ 的候选**，把最后一列的标注替换掉原文里的那个字。")
    if any(c["in_vocab"] is False for c in cands):
        L.append(">")
        L.append("> ❌ 的读音不在 `pinyin.vocab` 里，标了大概率无效，别选。")
    return "\n".join(L)


def apply_annotation(text: str, word: str, pinyin: str,
                     which: int = 0) -> Tuple[str, bool]:
    """把 text 里第 which 个 word 替换成 `<word|pinyin>`。

    返回 (新文本, 是否成功)。已有标注的 word 不会被重复包裹。
    """
    text = text or ""
    word = (word or "").strip()
    pinyin = (pinyin or "").strip().upper()
    if not word or not pinyin:
        return text, False
    tagged = f"<{word}|{pinyin}>"
    if tagged in text:
        return text, True           # 已经标过了

    # 找出所有未被标注包裹的 word 出现位置
    spans: List[int] = []
    for m in re.finditer(re.escape(word), text):
        s = m.start()
        inside = any(a.start <= s < a.end for a in parse(text))
        if not inside:
            spans.append(s)
    if not spans:
        return text, False
    which = max(0, min(which, len(spans) - 1))
    pos = spans[which]
    return text[:pos] + tagged + text[pos + len(word):], True


# ---------------------------------------------------------------------------
# 语法说明（UI 用）
# ---------------------------------------------------------------------------

SYNTAX_DOC = """
### 读音标注语法

格式统一为 **`<文字|发音>`**，竖线左边是原文里的字/词，右边是你想让它读成的音。

| 语言 | 发音写法 | 例子 | 展开成 |
|---|---|---|---|
| 中文 | 拼音 + 声调数字 1~5 | `银<行\\|XING2>` | `<\\|SPECIAL_TOKEN_2\\|>XING2<\\|SPECIAL_TOKEN_2\\|>` |
| 英文 | CMU 音素，空格分音节，数字表重音 | `<minute\\|M IH1 . N AH0 T>` | `<\\|SPECIAL_TOKEN_1\\|>...<\\|SPECIAL_TOKEN_1\\|>` |
| 日语 | 直接写假名 | `<上手\\|じょうず>` | 不加特殊 token，只用空格包裹 |

**声调数字**：1 阴平 · 2 阳平 · 3 上声 · 4 去声 · **5 轻声**

**几个容易踩的点**（都已实测核实）：

1. **大小写不敏感** —— 内部会统一转大写，`xing2` 和 `XING2` 等价
2. **jqx 后的 u 会自动改成 v** —— 官方 `correct_pinyin()` 干的，
   所以 `<女|NU3>` 保持 `NU3`，但 `<句|JU3>` 会变成 `JV3`。直接写 V 也行
3. **标注能扛过文本归一化** —— `TextNormalizer` 内部会先保护标注再归一化，
   所以 `2024年` 会读成「二零二四年」，同时 `<行|HANG2>` 不受影响
4. **`<` `>` 必须成对** —— 未闭合的标注会被当普通文本，静默失效，
   界面上不会报错但读音就是不对
5. **拼音必须在 `pinyin.vocab` 里** —— 这个文件有 1728 条合法拼音
   （五个声调：332/360/305/337/394 条）。不在里面的拼写模型没学过
6. **标注会增加 token 数** —— `<行|XING2>` 展开后是 5 个 token
   （特殊 token ×2 + BPE 拆出的拼音片段 ×3），而不标注时「行」只占 1 个。
   所以长文本大量标注时要留意「分句最大 Token 数」

> 拼音被 tiktoken 的 BPE 拆成几个片段是**正常**的，不影响效果 ——
> 特殊 token 成对出现已经把这段界定清楚了，模型学的是「这对 token 之间
> 的序列 = 指定发音」。
"""
