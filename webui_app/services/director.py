"""导演层（Director）：把一段文本变成「逐句演出台本」。

台本是编排器（orchestrator.py）的输入，每行一句台词，带：
    · text           —— 该句实际送去合成的文本（API 模式下可含表演化改写：
                        省略号、破折号、语气词；规则模式**不改写**，只切分）
    · emotion        —— 8 键之一（happy/angry/...），供情感参考路由
    · intensity      —— 0~1 表演强度，映射到该句的 emo_alpha 缩放
    · pause_after_ms —— 句后停顿（拟人化的核心：真人停顿是 150ms~1s+ 的
                        浮动分布，不是官方 infer 里固定的 interval_silence）

两个后端：
    rules —— 零配置、永不失败：标点切分 + 情绪词表 + 标点→停顿基表。
    api   —— OpenAI 兼容 chat 接口（GLM/Qwen/DeepSeek/ollama/LM Studio 都行），
              让 LLM 当配音导演逐句标注 + 停顿规划 + 表演化改写；
              任何失败（网络/解析/超时）都回退 rules，**不让合成中断**。

停顿基表参考真人统计（Campione & Véronis 2002：停顿三峰分布、中位
300~400ms；悲伤停顿数 +23%、愤怒停顿时长 -7% —— 情绪本身就是停顿信号）。
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from webui_app import logging_setup as LOG
from webui_app.config import PROJECT_ROOT, EMO_VECTOR_KEYS

# ---------------------------------------------------------------------------
# 配置持久化（API 后端）—— outputs/state/ 是配置目录，清理页不列它
# ---------------------------------------------------------------------------

STATE_DIR = os.path.join(PROJECT_ROOT, "outputs", "state")
CONFIG_PATH = os.path.join(STATE_DIR, "director_config.json")

DEFAULT_API_CONFIG: Dict[str, Any] = {
    "base_url": "",          # 例：https://open.bigmodel.cn/api/paas/v4 或 http://127.0.0.1:11434/v1
    "api_key": "",
    "model": "",
    "temperature": 0.7,
    "timeout_sec": 60,
}


def load_api_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_API_CONFIG)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in cfg:
                    if k in data:
                        cfg[k] = data[k]
        except Exception:
            pass
    return cfg


def save_api_config(cfg: Dict[str, Any]) -> bool:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        clean = {k: cfg.get(k, DEFAULT_API_CONFIG[k]) for k in DEFAULT_API_CONFIG}
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONFIG_PATH)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 台本数据结构
# ---------------------------------------------------------------------------

@dataclass
class ScriptLine:
    text: str
    emotion: str = "calm"          # 8 键之一
    intensity: float = 0.5         # 0~1
    pause_after_ms: int = 300      # 最后一句会被编排器忽略
    note: str = ""
    stress: List[str] = field(default_factory=list)   # 重读词（≤2 个）

    def clamp(self) -> "ScriptLine":
        self.emotion = self.emotion if self.emotion in EMO_VECTOR_KEYS else "calm"
        self.intensity = max(0.05, min(1.0, float(self.intensity or 0.5)))
        self.pause_after_ms = int(max(0, min(3000, int(self.pause_after_ms or 0))))
        self.text = (self.text or "").strip()
        # stress 词必须真实存在于 text（LLM 幻觉词在映射器里找不到会静默
        # 丢增益——不如入库时就滤掉）
        self.stress = [str(w) for w in (self.stress or [])
                       if w and str(w) in self.text][:2]
        return self

    def to_dict(self):
        from dataclasses import asdict as _ad
        return _ad(self)


@dataclass
class DirectorScript:
    lines: List[ScriptLine] = field(default_factory=list)
    backend: str = "rules"         # rules | api
    ok: bool = True
    error: str = ""                # api 失败原因（已回退 rules 时记录在此）
    raw: str = ""                  # api 原始返回（观测用）
    note: str = ""                 # 备注（如 cache-hit）

    @property
    def n(self) -> int:
        return len(self.lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lines"] = [asdict(x) for x in self.lines]
        return d


class DirectorError(RuntimeError):
    """导演层错误（调用方决定是否回退）。"""


# ---------------------------------------------------------------------------
# 规则后端：零配置、确定性（同 seed 同结果）、永不失败
# ---------------------------------------------------------------------------

# 情绪词表（规则模式粗分；API 模式由 LLM 细分）
_EMO_LEXICON: Dict[str, List[str]] = {
    "angry": ["可恶", "气死", "住手", "该死", "滚", "混蛋", "不可原谅", "敢",
              "休想", "找死", "岂有此理", "放肆", " 战 ", "杀"],
    "happy": ["哈哈", "太好了", "万岁", "好棒", "开心", "高兴", "终于", "成功",
              "赢了", "谢谢", "喜欢", "可爱", "嘿嘿", "呵呵"],
    "sad": ["对不起", "抱歉", "再见", "永别", "眼泪", "哭泣", "悲伤", "难过",
            "为什么", "舍不得", "失去", "再也", "怀念", "孤独"],
    "afraid": ["不要", "救命", "害怕", "怪物", "危险", "快跑", "救命啊", "恐怖",
               "吓", "颤抖", "黑暗"],
    "disgusted": ["恶心", "讨厌", "肮脏", "虚伪", "卑鄙", "无耻", "渣滓", "呕"],
    "melancholic": ["也许", "大概", "算了", "罢了", "无奈", "叹息", "孤独",
                    "沉默", "回忆", "曾经", "注定"],
    "surprised": ["什么", "怎么会", "竟然", "居然", "难道", "原来", "啊？",
                  "真的假的", "意想不到"],
}

# 标点 → 句后停顿基表（ms）。按「正常朗读/对话」标定（2026-09-28 用户
# 反馈初版偏长后整体下调）：句界 200~300ms、疑问/感叹略抬、省略号仍最长。
# 停顿系数（pause_scale）在 UI 上 0.4~1.6 连续可调，此表只是 1.0 的锚点。
_PAUSE_BASE: List[tuple] = [
    (("……", "…"), 600),
    (("——", "—"), 380),
    (("！", "!"), 300),
    (("？", "?"), 280),
    (("。", "."), 240),
    (("；", ";"), 240),
    (("：", ":"), 230),
    (("，", ","), 140),
    (("、",), 120),
]

# 情绪对停顿的修饰（悲伤拖、愤怒赶 —— 真人统计的定性版，幅度收敛）
_PAUSE_EMO_MOD: Dict[str, float] = {
    "sad": 1.20, "melancholic": 1.20, "afraid": 1.05,
    "angry": 0.90, "happy": 0.90, "surprised": 0.95,
    "disgusted": 1.00, "calm": 1.00,
}

# 抖动幅度 ±15%：有呼吸感但不至于一句快一句慢得太跳
_JITTER = (0.85, 1.15)


def _apply_pause_scale(ms: int, scale: float) -> int:
    """停顿系数统一后处理：缩放后夹到 [80, 2000]。"""
    s = max(0.1, min(3.0, float(scale or 1.0)))
    return int(max(80, min(2000, round(ms * s))))


# ---------------------------------------------------------------------------
# 停顿等级带（2026-09-29 用户反馈：逗号停顿比省略号还长——等级倒挂）
# ---------------------------------------------------------------------------
# 修复方式：每类标点有一个**互不重叠或按等级递增的取值带**，情绪修饰与
# 抖动只能在带内移动，越界即夹回。硬保证：
#     逗号上限 210 < 句号下限 320 < 句号上限 450 < 省略号下限 580
# 感叹/问号是句末标点，与句号同级相邻（260~380），允许与句号部分重叠。
# 带宽随 pause_scale 等比缩放 → 任何系数下等级顺序都不变。
_PAUSE_BAND: Dict[str, tuple] = {
    "、": (100, 170),
    "，": (120, 210),
    "；": (200, 300),
    "：": (200, 300),
    "？": (250, 370),
    "！": (260, 380),
    "。": (320, 450),
    "——": (480, 640),
    "……": (580, 820),
}
# 匹配顺序：长标点优先（…… 必须先于 … 检查）
_BAND_ORDER = [("……",), ("——",), ("…",), ("—",), ("！",), ("？",),
               ("。", "."), ("；", ";"), ("：", ":"), ("，", ","), ("、",)]


def _band_for(line_ending: str) -> tuple:
    """按句尾标点取停顿等级带 (lo, hi) ms @ scale=1.0。"""
    for marks in _BAND_ORDER:
        if any(line_ending.endswith(m) for m in marks):
            for m in marks:
                if line_ending.endswith(m) and m in _PAUSE_BAND:
                    return _PAUSE_BAND[m]
    return _PAUSE_BAND["。"]


def _quantize_band(ms: float, band: tuple, scale: float) -> int:
    """把原始停顿值夹进等级带（带随 pause_scale 等比缩放）。"""
    s = max(0.1, min(3.0, float(scale or 1.0)))
    lo = max(80, band[0] * s)
    hi = min(2000, band[1] * s)
    return int(max(lo, min(hi, ms)))


_SENT_SPLIT_RE = re.compile(r'([^。！？!?…；;\n]*[。！？!?…]+|[^。！？!?…；;\n]*[；;\n]+|[^。！？!?…；;\n]+)')


def _split_sentences(text: str) -> List[str]:
    """按句末标点切分，保留标点；空行/纯标点丢弃。"""
    out = []
    for m in _SENT_SPLIT_RE.finditer(text or ""):
        s = m.group(0).strip()
        if s and re.search(r'[\w\u4e00-\u9fffA-Za-z0-9]', s):
            out.append(s)
    return out


def _guess_emotion(line: str) -> tuple:
    """返回 (emotion, intensity)。词表命中 + 标点修饰，零依赖。"""
    scores = {k: 0.0 for k in EMO_VECTOR_KEYS}
    for emo, words in _EMO_LEXICON.items():
        for w in words:
            if w.strip() in line:
                scores[emo] += 1.0
            elif w in line:
                scores[emo] += 0.8
    best = max(scores, key=lambda k: scores[k])
    hits = scores[best]
    if hits <= 0:
        emo, inten = "calm", 0.35
    else:
        emo = best
        inten = min(0.9, 0.5 + 0.15 * hits)

    # 标点修饰：感叹→激烈情绪抬强度；问号→惊讶；省略号→低落
    if "！" in line or "!" in line:
        inten = min(0.95, inten + 0.15)
        if emo == "calm":
            emo = "happy"
    if ("？" in line or "?" in line) and emo in ("calm",):
        emo, inten = "surprised", max(inten, 0.5)
    if "…" in line and emo == "calm":
        emo, inten = "melancholic", 0.45
    return emo, round(inten, 2)


# 焦点重读词词典（规则后端的 stress 标记；中文焦点常落在否定词/
# 程度副词/指示词上——真正的语义焦点由 LLM 后端标注）
_STRESS_LEXICON = ["不", "没", "别", "很", "太", "最", "更", "必须", "一定",
                   "绝对", "永远", "都", "也", "才", "就", "竟", "居然"]


def _mark_stress(line: str) -> List[str]:
    """规则重读标记：词典词优先（≤2 个），否则不标（短语末已由模型延长）。"""
    hits = [w for w in _STRESS_LEXICON if w in line and len(line) >= 4]
    if not hits:
        return []
    hits.sort(key=lambda w: line.index(w))
    return hits[:2]


def _pause_for(line: str, emotion: str, rng: random.Random,
               scale: float = 1.0) -> int:
    """句后停顿 = 标点基表 × 情绪修饰 × ±15% 抖动，再夹进该标点的
    **等级带**（×停顿系数）——带间互不越界，等级永不倒挂。"""
    base = 240
    for marks, val in _PAUSE_BASE:
        if any(line.endswith(m) or m in line[-3:] for m in marks):
            base = val
            break
    mod = _PAUSE_EMO_MOD.get(emotion, 1.0)
    jitter = rng.uniform(*_JITTER)
    ms = base * mod * jitter
    return _quantize_band(ms, _band_for(line), scale)


def rules_direct(text: str, seed: int = 0, pause_scale: float = 1.0) -> DirectorScript:
    """规则导演：只切分/标情绪/排停顿，**不改写文本**（内容零风险）。"""
    rng = random.Random(seed)
    sents = _split_sentences(text)
    lines = []
    for i, s in enumerate(sents):
        emo, inten = _guess_emotion(s)
        lines.append(ScriptLine(
            text=s, emotion=emo, intensity=inten,
            pause_after_ms=_pause_for(s, emo, rng, scale=pause_scale),
            stress=_mark_stress(s),
            note="rules",
        ).clamp())
    if not lines and (text or "").strip():
        lines = [ScriptLine(text=text.strip(), emotion="calm", intensity=0.35,
                            pause_after_ms=0, note="rules:unsplit").clamp()]
    sc = DirectorScript(lines=lines, backend="rules", ok=True)
    if not lines:
        sc.ok = False
        sc.error = "文本为空或切不出任何句子"
    return sc


# ---------------------------------------------------------------------------
# API 后端：OpenAI 兼容 chat 接口
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一位资深配音导演，负责把剧本台词拆成逐句演出台本。
要求：
1. **尽量保持语义完整的句子，宁少拆不多拆**：一行 = 一个完整语义句
   （≤40 字）；只有真正的情绪转折/话题切换才另起一行。短句（≤8 字）
   绝不单独成行——与前后句合并或并入邻近行。逐行拆碎会让配音
   「两三个字一停」，这是最忌讳的失败形态。连续叙述标**同一个**
   emotion（情绪交替标注会强制拆块）。
2. 允许对文本做**表演化改写**：适度加入语气词（呢、啊、嘛、唉、哼）和
   少量重复（你、你说什么）。**但严禁改动原文的标点等级**：不得把逗号
   改成省略号/破折号，不得增删省略号或破折号，不得把一个长句拆成多个
   独立感叹句——停顿等级由编排系统按标点严格控制，改标点等级会造成
   停顿倒挂（逗号比省略号停得还长的事故就是这么来的）。不改变事实内容、
   每句总长变化不超过 ±30%。每行不超过 40 个字，长内容在句号处拆行。
3. 为每句标注 emotion（只能从这 8 个键选一个）：
   happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
4. 标注 intensity（0.2~1.0，表演强度：平静旁白 0.3 左右，爆发戏 0.9）。
5. 标注 pause_after_ms —— 只在**情绪转折或强边界**处给值（120~350），
   戏剧性长停顿/哽咽/欲言又止可到 400~700；情绪连续的叙述句之间给 0
   （编排器会把情绪连续的句子合并成一块连续合成，句内节奏由模型自然
   处理，句内停顿标记会被忽略）。真人配音的停顿是稀缺的、不均匀的：
   正常叙述流中 **90% 以上的行应给 0**，绝不要每句都填停顿。
6. 每句标注 stress：从该句原文里挑 1~2 个**应重读的词**（原文中的连续
   子串，通常是动词/形容词/情绪焦点词），JSON 里是数组；找不到就给 []。
只输出 JSON，不要多余文字，格式：
{"lines":[{"text":"...","emotion":"...","intensity":0.6,"pause_after_ms":350,"stress":["词"],"note":"可选备注"}]}"""


def _api_call(cfg: Dict[str, Any], text: str, character: str) -> str:
    """单次 chat 调用，返回原始文本。用 stdlib urllib，不引新依赖。"""
    url = (cfg.get("base_url") or "").rstrip("/") + "/chat/completions"
    body = {
        "model": cfg.get("model") or "",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",
             "content": f"角色：{character or '（未指定）'}\n台词：\n{text}"},
        ],
        "temperature": float(cfg.get("temperature") or 0.7),
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    if cfg.get("api_key"):
        req.add_header("Authorization", f"Bearer {cfg['api_key']}")
    timeout = int(cfg.get("timeout_sec") or 60)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_api_script(raw_response: str, max_lines: int = 400) -> List[ScriptLine]:
    """从 chat 返回里抠出台本 JSON 并校验。任何不对都抛 DirectorError。"""
    try:
        payload = json.loads(raw_response)
        content = payload["choices"][0]["message"]["content"]
    except Exception as e:
        raise DirectorError(f"chat 返回不是合法的 OpenAI 结构：{e}")

    # 从 content 里抠 JSON（模型有时会包 ```json 围栏或前后废话）
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise DirectorError("返回 content 里找不到 JSON")
    try:
        data = json.loads(m.group(0))
    except Exception as e:
        raise DirectorError(f"JSON 解析失败：{e}")

    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise DirectorError("JSON 里没有 lines 数组")
    if len(raw_lines) > max_lines:
        raise DirectorError(f"句数 {len(raw_lines)} 超过上限 {max_lines}")

    out = []
    for d in raw_lines:
        if not isinstance(d, dict):
            continue
        t = str(d.get("text") or "").strip()
        if not t:
            continue
        st = d.get("stress")
        out.append(ScriptLine(
            text=t,
            emotion=str(d.get("emotion") or "calm").strip().lower(),
            intensity=float(d.get("intensity") or 0.5),
            pause_after_ms=int(float(d.get("pause_after_ms") or 300)),
            note=str(d.get("note") or "")[:80],
            stress=[str(x) for x in st] if isinstance(st, list) else [],
        ).clamp())
    if not out:
        raise DirectorError("lines 里没有有效句子（text 全空）")
    return out


def api_direct(text: str, character: str = "",
               cfg: Optional[Dict[str, Any]] = None,
               pause_scale: float = 1.0) -> DirectorScript:
    """API 导演。失败抛 DirectorError（由 direct() 统一回退 rules）。"""
    cfg = cfg or load_api_config()
    if not (cfg.get("base_url") or "").strip():
        raise DirectorError("未配置 API base_url")
    if not (cfg.get("model") or "").strip():
        raise DirectorError("未配置模型名")

    log = LOG.get_logger("director")
    t0 = time.perf_counter()
    raw = _api_call(cfg, text, character)
    dt = time.perf_counter() - t0
    lines = _parse_api_script(raw)
    # LLM 给的停顿值同样夹进等级带（按每行句尾标点）——LLM 的标点
    # 等级感不可靠（实测会把逗号排得比省略号长），量化兜底。
    for ln in lines:
        ln.pause_after_ms = _quantize_band(
            ln.pause_after_ms, _band_for(ln.text), pause_scale)
    log.info("API 导演完成：%d 句 · %.1fs · model=%s · pause_scale=%.2f",
             len(lines), dt, cfg.get("model"), pause_scale)
    return DirectorScript(lines=lines, backend="api", ok=True, raw=raw[:4000])


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def direct(
    text: str,
    backend: str = "rules",
    character: str = "",
    api_cfg: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    pause_scale: float = 1.0,
) -> DirectorScript:
    """文本 → 台本。api 失败自动回退 rules（合成永不因导演层中断）。"""
    if backend == "api":
        try:
            # 走缓存版：同文本+同角色+同停顿系数命中即免一次 LLM 调用
            return cached_api_direct(text, character=character, cfg=api_cfg,
                                     pause_scale=pause_scale)
        except Exception as e:
            LOG.get_logger("director").warning(
                "API 导演失败，回退规则引擎：%s: %s", type(e).__name__, e)
            sc = rules_direct(text, seed=seed, pause_scale=pause_scale)
            sc.error = f"api 失败已回退 rules：{type(e).__name__}: {e}"
            return sc
    return rules_direct(text, seed=seed, pause_scale=pause_scale)


# ---------------------------------------------------------------------------
# 预览（UI 用）
# ---------------------------------------------------------------------------

def script_markdown(sc: DirectorScript, route: Optional[Dict[str, str]] = None) -> str:
    """台本表格。route: {emotion_key: 参考名} 或 {emotion_key: ""}（未命中）。"""
    from webui_app.config import EMO_VECTOR_LABELS
    zh = dict(zip(EMO_VECTOR_KEYS, EMO_VECTOR_LABELS))
    head = (f"**导演台本** · 后端 `{sc.backend}` · {sc.n} 句"
            + (f" · ⚠️ {sc.error}" if sc.error else ""))
    lines = ["", "| # | 台词 | 情绪 | 强度 | 句后停顿 | 情感参考 |",
             "|---|---|---|---|---|---|"]
    for i, ln in enumerate(sc.lines, 1):
        ref = "-"
        if route is not None:
            ref = route.get(ln.emotion) or "→ 回退音色参考"
        lines.append(
            f"| {i} | {ln.text[:48]} | {zh.get(ln.emotion, ln.emotion)} | "
            f"{ln.intensity:.2f} | {ln.pause_after_ms} ms | {ref} |"
        )
    pauses = [x.pause_after_ms for x in sc.lines[:-1]] or [0]
    avg = sum(pauses) / max(1, len(pauses))
    lines.append("")
    lines.append(f"句间停顿：均值 {avg:.0f} ms · 区间 "
                 f"{min(pauses)}~{max(pauses)} ms（拟人化=不均匀）")
    return head + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# 台本缓存（2026-09-29 · 用户需求：同文本不再调 LLM）
# ---------------------------------------------------------------------------

SCRIPT_CACHE_PATH = os.path.join(STATE_DIR, "director_cache.json")
SCRIPT_CACHE_MAX = 200          # 条目上限，超了丢最旧（按写入时间）


def _cache_key(text: str, character: str, pause_scale: float) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(f"{(text or '').strip()}|{character or ''}|"
             f"{round(float(pause_scale or 1.0), 2)}".encode("utf-8"))
    return h.hexdigest()


def _load_script_cache() -> Dict[str, Any]:
    if not os.path.isfile(SCRIPT_CACHE_PATH):
        return {}
    try:
        with open(SCRIPT_CACHE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_script_cache(cache: Dict[str, Any]) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        if len(cache) > SCRIPT_CACHE_MAX:
            keep = sorted(cache.items(), key=lambda kv: kv[1].get("_t", 0)
                          )[-SCRIPT_CACHE_MAX:]
            cache = dict(keep)
        tmp = SCRIPT_CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, SCRIPT_CACHE_PATH)
    except OSError:
        pass


def cached_api_direct(text: str, character: str = "",
                      cfg: Optional[Dict[str, Any]] = None,
                      pause_scale: float = 1.0,
                      use_cache: bool = True) -> DirectorScript:
    """api_direct 的带缓存版：命中即免一次 LLM 调用（note 标 cache）。

    缓存键 = 文本 + 角色 + 停顿系数（这些变了台本就该变）。API 失败
    照旧抛 DirectorError（由 direct() 回退 rules）。
    """
    key = _cache_key(text, character, pause_scale)
    if use_cache:
        hit = _load_script_cache().get(key)
        if hit and hit.get("lines"):
            sc = DirectorScript(
                lines=[ScriptLine(**{k: v for k, v in ln.items()
                                     if k in ScriptLine.__dataclass_fields__})
                       for ln in hit["lines"]],
                backend="api", ok=True, raw="(cache)",
                note="cache-hit")
            LOG.get_logger("director").info("台本缓存命中（%d 句）", sc.n)
            return sc
    sc = api_direct(text, character=character, cfg=cfg,
                    pause_scale=pause_scale)
    if use_cache:
        cache = _load_script_cache()
        cache[key] = {"lines": [ln.to_dict() for ln in sc.lines],
                      "_t": time.time(),
                      "character": character or "",
                      "preview": (text or "")[:60]}
        _save_script_cache(cache)
    return sc
