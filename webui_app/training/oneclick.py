"""一键三连：长音频 / 批量短音频 → 优化 → 识别对齐 → 自动调参 → 训练 → 择优。

用户只把音频丢进来，剩下七道工序全自动：

    S1 采集    导入文件（复制副本进数据集，原件不动）；长音频按声学打分切片
    S2 优化    去直流 / 掐静音 / 降噪 / 归一 / 重采样 / 截断，写回数据集
    S3 识别    逐条 whisper 转写。长音频**先切片再逐片转写**，于是
               「一片音频 ↔ 一段文本」按构造配对 —— 这就是本项目的对齐
    S4 筛选    体检复算 → 丢掉不合格 → 文本去重 → 划分 train / val
    S5 调参    按数据量 × 空闲显存 逐个 preflight 候选配置，挑出第一个真能跑的
    S6 训练    逐目标（默认 GPT + CFM）真训练，保险库只留 top-K 档位
    S7 择优    逐档位真机合成 → reward 打分 → 排序 → 激活最好的那一个

=======================================================================
三条硬约束（改这个文件之前务必先读，全部有实测依据）
=======================================================================

1. **必须用 `require_engine="none"` 提交**。runner 的 `engine_req` 全程生效：
   `"unloaded"` 会让 `engine.load()` 抛，`"loaded"`（含 `holds_engine=True`）
   会让 `engine.unload()` 抛。本流水线横跨两种引擎状态 —— 特征提取和
   择优评测要**加载**，训练要**卸载** —— 所以只能 "none"，引擎归本模块管。

2. **阶段之间必须显式 load/unload**。8 GB 卡上引擎常驻 4.9~5.7 GB，与
   训练器不能共存。Windows WDDM 下显存溢出**不报 OOM**，而是静默降速
   20~30 倍（实测 CFM 25 步 2.4s → 66s），所以这里宁可多花两次加载时间，
   也不让引擎在训练期间留在显存里。训练器自己的
   `vram_preflight(raise_on_short=True)` 是最后一道硬拦。

3. **样本数 < 20 是 `LoRAConfig.validate` 的硬错误**。宁可在这里早失败并
   给出「再录/再切几条」的行动指引，也不要跑到训练器里才炸 —— 所以
   S4 的筛选结果必须过 `MIN_SAMPLES`，不够就直接终止（不浪费后面的算力）。

「对齐」的诚实说明：工程里**没有**强制对齐器（无 whisperX / MFA / CTC
对齐，上游只有一段死代码）。本模块的对齐方式是「先按声学质量切片，再对
每片独立转写」，文本与音频一一对应。这是现有组件能给出的唯一可靠对齐，
也是长音频转写最容易翻车的地方（整段转写再按字数硬切必然错位）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app import logging_setup as LOG
from webui_app.config import OUTPUT_SAMPLE_RATE, PROJECT_ROOT
from webui_app.services import audio_lab as AL
from webui_app.training import cfm_lora as CL
from webui_app.training import dataset as DS
from webui_app.training import evaluate as EV
from webui_app.training import features as FT
from webui_app.training import gpt_lora as GL
from webui_app.training import guard as GD
from webui_app.training import parallel as PL
from webui_app.training import reward as RW
from webui_app.training import runs as RN

ONECLICK_ROOT = os.path.join(PROJECT_ROOT, "outputs", "oneclick")

# `LoRAConfig.validate` 在样本少于这个数时给 error，训练器也跑不起来。
MIN_SAMPLES = 20

# 阶段名 / 标题 / 在总进度里占的比重。比重按实测耗时估的：
# 训练占大头，识别次之（whisper small 每条约 0.3~1s）。
STAGES: List[Tuple[str, str, float]] = [
    ("ingest",   "采集与切片", 0.05),
    ("optimize", "音频优化",   0.10),
    ("asr",      "识别与对齐", 0.15),
    ("curate",   "筛选与划分", 0.08),
    ("tune",     "自动调参",   0.04),
    ("train",    "LoRA 训练",  0.45),
    ("rank",     "择优与交付", 0.13),
]
STAGE_TITLE = {k: t for k, t, _ in STAGES}

ARCH_LABELS = {"gpt": "GPT(T2S) 语气韵律", "cfm": "CFM(S2M) 音色音质"}


# ===========================================================================
# 选项
# ===========================================================================

@dataclass
class OneClickOptions:
    """一键三连的全部可调项。默认值面向「丢进来就不管」的场景。"""

    # ---- 目标 ----
    dataset_name: str = ""            # "" = 自动生成 oneclick_<时间戳>
    model_name: str = ""              # 训练记录名（「模型名称」，便于区分管理）
    lang: str = "ZH"                  # 训练语种（也决定 whisper 的转写语言）

    # ---- CPU 并行（只作用于纯 CPU 的音频处理，见 training/parallel.py）----
    cpu_workers: int = 0              # 0 = 自动（保守值）；1 = 关掉并行；上限 4

    # ---- S1 切片（只对超过 slice_over_sec 的长音频生效）----
    slice_over_sec: float = DS.MAX_TRAIN_SEC   # 20s：超过就不算「可直接训练」
    slice_target_sec: float = 12.0             # 每片目标时长
    slice_min_sec: float = 4.0                 # 太短的片段不要
    slice_max_pieces: int = 60                 # 单条长音频最多切几片

    # ---- S2 优化 ----
    enhance: bool = True              # 关掉则只做体检、不改音频
    denoise: bool = True
    denoise_strength: float = 0.6
    normalize: bool = True
    trim_silence: bool = True

    # ---- S3 识别 ----
    asr: bool = True                  # 关掉则需要自己补文本（数据集会留 no_text）
    whisper_size: str = RW.DEFAULT_WHISPER
    # S7 择优打分用的模型。**故意比 ASR 小一档**：打分只在候选之间做相对比较，
    # 而它运行时引擎还驻留在显存里 —— ASR 用 medium（1.6 GB）+ 引擎（4.94 GB）
    # 在 8 GB 卡上只剩几百 MB，实测会把扩散采样挤到共享内存，
    # s2mel 从 0.9 秒变成 25 秒（WDDM 静默降速，不报 OOM）。
    score_whisper_size: str = "small"

    # ---- S4 筛选 ----
    min_score: float = 45.0           # 音频体检分低于此值丢弃（0 = 不按分筛）
    max_text_repeats: int = 3         # 同一句话最多留几条（防重复句主导训练）；0 = 不限
    val_ratio: float = 0.1

    # ---- S5 调参 ----
    arches: str = "gpt,cfm"           # 逗号分隔，可选 gpt / cfm / dpo
    preset_mode: str = "auto"         # auto = 按数据量自动排候选；也可固定某档
    top_k: int = 3                    # 每个目标保留并参评的档位数（"三个"）

    # ---- S7 择优 ----
    rank_eval: bool = True            # 关掉则只按 val loss 排序（省一次引擎加载）
    eval_samples: int = 5             # 每个候选评几条（0 = 全量，慢）
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "OneClickOptions":
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k) and v is not None:
                setattr(o, k, v)
        return o

    def arch_list(self) -> List[str]:
        """可自动化的训练目标。

        只含 gpt / cfm：DPO 需要先有 `pairs.jsonl` 偏好对，而偏好对要靠
        「用当前模型合成候选再打分」蒸出来，属于对齐页的活，不在本流程内。
        """
        out = []
        for a in str(self.arches or "").split(","):
            a = a.strip().lower()
            if a in ("gpt", "cfm") and a not in out:
                out.append(a)
        return out

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []

        def err(m):
            n.append(GD.Notice("error", m))

        def warn(m):
            n.append(GD.Notice("warn", m))

        raw = [a.strip().lower() for a in str(self.arches or "").split(",") if a.strip()]
        if "dpo" in raw:
            err("一键三连不支持 DPO 目标：DPO 需要先用当前模型合成候选、打分、"
                "构造偏好对（pairs.jsonl），请先到「⚖️ 对齐」页蒸出偏好对，"
                "再到「🎓 训练」页选 DPO。本流程只做 GPT / CFM。")
        bad = [a for a in raw if a not in ("gpt", "cfm", "dpo")]
        if bad:
            err(f"训练目标 {bad} 不认识（可选 gpt / cfm）")
        if not self.arch_list():
            err("训练目标为空，至少要选一个（gpt / cfm）")
        if self.lang not in ("ZH", "EN", "JA", "AR", "ES"):
            err(f"lang={self.lang} 不支持（可选 ZH/EN/JA/AR/ES）")
        if float(self.slice_over_sec) < float(self.slice_min_sec):
            err("slice_over_sec 不能小于 slice_min_sec")
        if not 1.0 <= float(self.slice_target_sec) <= DS.MAX_TRAIN_SEC:
            err(f"slice_target_sec 应在 1~{DS.MAX_TRAIN_SEC} 秒之间"
                "（超过 20s 的样本不参与训练，切了也白切）")
        if float(self.slice_min_sec) < DS.MIN_SEC:
            err(f"slice_min_sec 不能小于 {DS.MIN_SEC} 秒")
        if not 1 <= int(self.slice_max_pieces) <= 500:
            err("slice_max_pieces 应在 1~500 之间")
        if not 0.0 <= float(self.min_score) <= 100.0:
            err("min_score 应在 0~100 之间")
        if not 0 <= int(self.max_text_repeats):
            err("max_text_repeats 不能为负数")
        if not 0.0 < float(self.val_ratio) < 0.5:
            err("val_ratio 应在 0~0.5 之间")
        if not 1 <= int(self.top_k) <= 8:
            err("top_k 应在 1~8 之间（保险库档位数）")
        if not 0 <= int(self.eval_samples):
            err("eval_samples 不能为负数")
        if self.whisper_size not in RW.WHISPER_SIZES:
            err(f"whisper_size={self.whisper_size} 不在 "
                f"{list(RW.WHISPER_SIZES)} 里")
        if self.score_whisper_size not in RW.WHISPER_SIZES:
            err(f"score_whisper_size={self.score_whisper_size} 不在 "
                f"{list(RW.WHISPER_SIZES)} 里")
        if not self.asr:
            warn("已关闭语音识别：数据集里的文本会是空的，"
                 "需要到「数据集」页手动补文本，否则筛选阶段一条都留不下。")
        if self.preset_mode != "auto" and self.preset_mode not in GD.CONFIG_PRESETS:
            err(f"preset_mode={self.preset_mode} 不在 "
                f"auto / {list(GD.CONFIG_PRESETS)} 里")
        if int(self.eval_samples) and int(self.eval_samples) < 3:
            warn("eval_samples < 3：择优的名次会很不稳（打分本身有转写噪声）。")
        if int(self.cpu_workers or 0) > PL.MAX_WORKERS:
            warn(f"cpu_workers={self.cpu_workers} 超过上限 {PL.MAX_WORKERS}，"
                 "会按上限裁剪。音频处理是辅助工序，抢太多 CPU 反而拖慢训练。")
        if int(self.cpu_workers or 0) < 0:
            err("cpu_workers 不能为负数（0 = 自动，1 = 关掉并行）")
        if (self.model_name or "").strip():
            got = RN.safe_run_name(self.model_name)
            if got != (self.model_name or "").strip():
                warn(f"模型名称含非法字符，将按 `{got}` 落盘"
                     "（斜杠、冒号等在 Windows 上不能做目录名）。")
        return n

    def plan_run_names(self) -> Dict[str, str]:
        """模型名称 → 每个目标实际使用的训练记录名。

        单个目标时就用用户给的名字（所见即所得）；两个目标时必须加后缀区分，
        否则 CFM 会把 GPT 的记录目录覆盖掉。
        """
        base = (self.model_name or "").strip() or (self.dataset_name or "").strip()
        arches = self.arch_list()
        if not base:
            base = time.strftime("oneclick_%Y%m%d-%H%M")
        base = RN.safe_run_name(base)
        if len(arches) <= 1:
            return {arches[0]: base} if arches else {}
        return {a: RN.safe_run_name(f"{base}_{a}") for a in arches}

    def check_run_names(self) -> str:
        """训练记录名是否已被占用。占用则返回错误文案（在开工前拦下）。"""
        clash = []
        for arch, name in self.plan_run_names().items():
            try:
                if RN.read_run(name):
                    clash.append(f"{name}（{arch}）")
            except Exception:
                pass
        if not clash:
            return ""
        return ("模型名称已被占用：" + "、".join(clash)
                + "。请换一个名称，或到「🎓 训练」页删掉旧记录后再跑"
                  "——现在拦下是为了不让你白等一整轮训练才发现名字冲突。")


# ===========================================================================
# 报告
# ===========================================================================

@dataclass
class OneClickReport:
    """流水线的可读结果。既回传 UI，也落盘成 report.json / report.md。"""

    ok: bool = False
    dataset: str = ""
    names: Dict[str, str] = field(default_factory=dict)   # arch -> 训练记录名
    started_at: float = 0.0
    seconds: float = 0.0
    options: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen: Dict[str, Any] = field(default_factory=dict)
    training: List[Dict[str, Any]] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    best: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    error: str = ""
    # 阶段开始/结束时回调 (status, title)，供上层同步 runner.phase 等
    phase_cb: Optional[Callable[[str, str], None]] = None

    def _notify(self, status: str, title: str) -> None:
        if self.phase_cb is not None:
            try:
                self.phase_cb(status, title)
            except Exception:
                pass

    def start(self, key: str, detail: str = "") -> None:
        title = STAGE_TITLE.get(key, key)
        self.stages.append({"key": key, "title": title, "detail": detail,
                            "status": "running", "seconds": 0.0, "info": {}})
        # 阶段转换写进**文件日志**：界面上的进度只在内存里，进程一退就没了，
        # 而「一键三连卡在哪一步」这类问题往往是事后才回头查的。
        self._central().info("阶段开始 · %s · %s", title, detail or "")
        self._notify("start", title)

    def finish(self, key: str, detail: str = "", status: str = "done",
               seconds: float = 0.0, **info: Any) -> None:
        for s in reversed(self.stages):
            if s["key"] == key:
                s.update(detail=detail or s["detail"], status=status,
                         seconds=round(seconds, 1), info=info)
                from webui_app import logging_setup as LOG
                lv = {"done": LOG.logging.INFO, "skipped": LOG.logging.WARNING,
                      "failed": LOG.logging.ERROR,
                      "stopped": LOG.logging.WARNING}.get(status,
                                                          LOG.logging.INFO)
                label = {"done": "完成", "skipped": "跳过", "failed": "失败",
                         "stopped": "已停止"}.get(status, status)
                self._central().log(lv, "阶段%s · %s · 用时 %.1fs · %s",
                                    label, s["title"], seconds, detail or "")
                self._notify(status, s["title"])
                return

    def mark(self, key: str, status: str) -> None:
        for s in reversed(self.stages):
            if s["key"] == key:
                s["status"] = status
                return

    def running(self) -> Dict[str, Any]:
        """当前正在跑的阶段。心跳与失败归因都用它 —— 没有它的话，
        界面只会说「任务失败」，看不出是在哪一步卡住的。"""
        for s in reversed(self.stages):
            if s.get("status") == "running":
                return s
        return {}

    @staticmethod
    def _central():
        from webui_app import logging_setup as LOG
        return LOG.get_logger("oneclick")

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)

    def stage_of(self, key: str) -> Dict[str, Any]:
        for s in reversed(self.stages):
            if s["key"] == key:
                return s
        return {}

    # -- 渲染 ---------------------------------------------------------------

    def markdown(self) -> str:
        from webui_app import theme as T

        icon = {"done": "✅", "running": "⏳", "skipped": "⏭",
                "failed": "🔴", "pending": "⏸"}

        L: List[str] = []
        head = ("✅ 一键三连完成" if self.ok else "🔴 一键三连未完成")
        L.append(T.tip(f"<b>{head}</b> · 数据集 <code>{self.dataset}</code>"
                       + (f" · 耗时 {self.seconds / 60:.1f} 分钟"
                          if self.seconds else "")))
        if self.names:
            pairs = " · ".join(f"{a} → <code>{n}</code>"
                               for a, n in self.names.items())
            L.append(T.hint(f"训练记录名：{pairs}"))
        if self.error:
            L.append(T.err(self.error))

        # ---- 阶段表 ----
        L.append("")
        L.append("| # | 阶段 | 状态 | 耗时 | 结果 |")
        L.append("|---|---|---|---|---|")
        for i, s in enumerate(self.stages, 1):
            L.append(f"| {i} | {s['title']} | "
                     f"{icon.get(s['status'], s['status'])} "
                     f"{s['status']} | "
                     f"{(str(s['seconds']) + 's') if s['seconds'] else '—'} | "
                     f"{s['detail'] or '—'} |")

        # ---- 自动调参：候选与裁决 ----
        if self.candidates:
            L.append("")
            L.append("### 自动调参：候选与裁决")
            L.append("")
            L.append("按「数据量 → 预设档位」排序，逐个 `preflight` 实跑，"
                     "第一个通过的就采用 —— 不是拍脑袋选，是**真跑一遍**再选。")
            L.append("")
            L.append("| 目标 | 预设 | rank | alpha | lr | epochs | 步数 | "
                     "评估间隔 | 预估显存 | 裁决 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for c in self.candidates:
                verdict = ("✅ **采用**" if c.get("chosen")
                           else f"❌ {c.get('reject', '')}")
                L.append(
                    f"| {c.get('arch', '')} | {c.get('preset', '')} | "
                    f"{c.get('rank', '')} | {c.get('alpha', '')} | "
                    f"{c.get('lr', '')} | {c.get('epochs', '')} | "
                    f"{c.get('total_steps', '')} | "
                    f"{c.get('eval_every', '')} | "
                    f"{c.get('est_vram_gb', '')} GB | {verdict} |")
            L.append("")
            L.append("<sub>「评估间隔」是按真实步数推导出来的：出厂预设里的 "
                     "50/100/200 是给大数据集定的，小数据集上整个 run 都触发"
                     "不到按步评估，保险库就只剩轮末兜底存下的 1 个档位 —— "
                     "那么「留 K 个档位再择优」就无档可选。这里按总步数反推，"
                     "保证能攒出约 K 个可比较的档位。</sub>")

        # ---- 训练结果 ----
        if self.training:
            L.append("")
            L.append("### 训练结果")
            L.append("")
            L.append("| 目标 | run | steps | val（底座 → 最好） | 改善 | 峰值显存 | 档位 |")
            L.append("|---|---|---|---|---|---|---|")
            for t in self.training:
                imp = t.get("improved")
                imp_s = f"**{imp * 100:.1f}%**" if isinstance(imp, float) else "—"
                L.append(
                    f"| {t.get('arch', '')} | <code>{t.get('run', '')}</code> | "
                    f"{t.get('steps', '')} | "
                    f"{t.get('first_val', '?')} → **{t.get('best_val', '?')}** | "
                    f"{imp_s} | {t.get('vram_peak_gb', '')} GB | "
                    f"{t.get('n_checkpoints', 0)} |")

        # ---- 择优 ----
        if self.ranking:
            L.append("")
            L.append("### 择优（真机合成 + reward 打分）")
            L.append("")
            L.append("reward = 0.6×(1−WER) + 0.4×声纹相似度，越高越好；"
                     "同种子合成，差异只来自模型本身。")
            L.append("")
            L.append("| 名次 | run | 档位 | val loss | reward | WER | 声纹相似 | 评价条数 |")
            L.append("|---|---|---|---|---|---|---|---|")
            for r in self.ranking:
                star = "🥇" if r["place"] == 1 else (
                    "🥈" if r["place"] == 2 else ("🥉" if r["place"] == 3 else ""))
                L.append(
                    f"| {star} {r['place']} | <code>{r.get('run', '')}</code> | "
                    f"{r.get('checkpoint', '')} | {r.get('val', '')} | "
                    f"**{r.get('reward', '')}** | {r.get('wer', '')} | "
                    f"{r.get('sim', '')} | {r.get('n', 0)} |")

        # ---- 交付 ----
        if self.best:
            L.append("")
            b = self.best
            L.append(T.tip(
                f"🏆 <b>推荐使用</b>：<code>{b.get('run', '')}</code> 的 "
                f"<code>{b.get('checkpoint', '')}</code> 档位"
                f"（reward {b.get('reward', '')}）已激活到该 run 的 "
                f"<code>adapter/</code>。<br>"
                f"到「🎙 合成」页的「LoRA 音色模型」里选这个 run 即可试听；"
                f"到「🏁 评测/部署」页可调强度旋钮或合并成独立权重。"))

        if self.notes:
            L.append("")
            L.append("### 提示")
            for m in self.notes:
                L.append(f"- {m}")

        return "\n".join(L)


# ===========================================================================
# 工具
# ===========================================================================

class _Bands:
    """把「阶段内 0~1 的进度」映射到「全局 0~1」，让进度条不来回跳。"""

    def __init__(self, stages: Sequence[Tuple[str, str, float]]):
        total = sum(w for _, _, w in stages) or 1.0
        self._off: Dict[str, Tuple[float, float]] = {}
        acc = 0.0
        for key, _, w in stages:
            self._off[key] = (acc / total, w / total)
            acc += w

    def at(self, key: str, local: float) -> float:
        off, w = self._off.get(key, (0.0, 1.0))
        local = max(0.0, min(1.0, float(local)))
        return min(1.0, off + local * w)


def _sub(progress, bands: _Bands, key: str, prefix: str, base: float = 0.0,
         span: float = 1.0):
    """把一个阶段的子进度回调，包装成全局进度 + 带前缀的日志。"""
    def cb(frac: float, msg: str) -> None:
        if not progress:
            return
        local = base + max(0.0, min(1.0, float(frac))) * span
        progress(bands.at(key, local), f"[{prefix}] {msg}")
    return cb


def _auto_dataset_name() -> str:
    return "oneclick_" + time.strftime("%Y%m%d-%H%M")


def collect_inputs(paths: Sequence[str], extra_path: str = "",
                   ) -> Dict[str, Any]:
    """收集待处理音频。

    paths      —— 上传控件给出的文件路径（单个或多个）
    extra_path —— 服务器上的路径：可以是**一个长音频文件**，也可以是一个
                  **目录**（递归找出全部支持的音频）。上传一个两小时的 wav
                  很痛苦，所以长音频走这条更实际。

    返回 {"files": [...], "missing": [...], "skipped": [...]}，files 保序去重。
    """
    files: List[str] = []
    missing: List[str] = []
    skipped: List[str] = []
    seen = set()

    # Gradio 的 gr.File 在只选了一个文件时给的是 str 而不是 list，
    # 直接迭代会把路径当成逐字符的可迭代对象。
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    def _take(p: str) -> None:
        p = str(p or "").strip().strip('"')
        if not p:
            return
        ap = os.path.abspath(p)
        key = os.path.normcase(ap)
        if key in seen:
            return
        if not os.path.isfile(ap):
            missing.append(p)
            return
        if not ap.lower().endswith(DS.AUDIO_EXTS):
            skipped.append(f"{os.path.basename(ap)}: 不是支持的音频格式")
            return
        seen.add(key)
        files.append(ap)

    for p in (paths or []):
        _take(p)

    extra = str(extra_path or "").strip().strip('"')
    if extra:
        ae = os.path.abspath(extra)
        if os.path.isdir(ae):
            for root, _dirs, names in os.walk(ae):
                for nm in sorted(names):
                    if nm.lower().endswith(DS.AUDIO_EXTS):
                        _take(os.path.join(root, nm))
        else:
            _take(ae)

    return {"files": files, "missing": missing, "skipped": skipped}


def _durations(paths: Sequence[str], workers: int = 1, progress=None,
               should_stop=None) -> Dict[str, float]:
    """批量取时长；读不出来的记为 -1。

    这里只用得上「这条算不算长音频」，所以走 `AL.probe_duration`（只读文件头）。
    原先用 `AL.analyze` 会做**整段解码 + 全量帧能量**，对一段一小时长的音频
    白等几十秒到几分钟，而且和后面 find_segments 的那次解码完全重复。
    """
    uniq = list(dict.fromkeys(paths))
    if not uniq:
        return {}

    def _one(p: str) -> float:
        d = AL.probe_duration(p)
        return float(d) if d is not None and d >= 0 else -1.0

    n = max(1, len(uniq))
    outcome = PL.map_parallel(
        _one, uniq, workers=workers, should_stop=should_stop,
        on_done=(lambda i, _r: progress((i + 1) / n, f"读取时长 {i + 1}/{n}"))
        if progress else None)
    return {p: (outcome.items[i] if outcome.items[i] is not None else -1.0)
            for i, p in enumerate(uniq)}


# ===========================================================================
# S1 · 采集与切片
# ===========================================================================

def stage_ingest(dataset: str, files: Sequence[str], opt: OneClickOptions,
                 progress=None, should_stop=None, report: Optional[OneClickReport] = None,
                 bands: Optional[_Bands] = None) -> Dict[str, Any]:
    """导入音频，并把超长的那几条切片。

    「长音频」的判据是时长 > opt.slice_over_sec（默认 20s = MAX_TRAIN_SEC），
    因为超过 20s 的样本体检直接是 too_long，不切片就永远不可训练。
    """
    bands = bands or _Bands(STAGES)
    cb = _sub(progress, bands, "ingest", "采集")

    r = DS.import_audio(dataset, list(files), copy=True, lang=opt.lang,
                        progress=lambda f, m: cb(f * 0.6, m))
    added = list(r.get("ids") or [])
    result: Dict[str, Any] = {"added": len(added), "sliced": 0, "pieces": 0,
                              "skipped": list(r.get("skipped") or []),
                              "failed": list(r.get("failed") or [])}

    # ---- 找出需要切片的长音频 ----
    items = {u.id: u for u in DS.load_meta(dataset)}
    cb(0.62, "读取音频时长（只读文件头，不解码）")
    dur = _durations([items[i].audio_abs(DS.dir_of(dataset))
                      for i in added if i in items],
                     workers=PL.default_workers(len(added), opt.cpu_workers),
                     progress=lambda f, m: cb(0.62 + 0.03 * f, m),
                     should_stop=should_stop)
    long_uids = []
    for uid in added:
        u = items.get(uid)
        if u is None:
            continue
        d = dur.get(u.audio_abs(DS.dir_of(dataset)), -1.0)
        if d > float(opt.slice_over_sec):
            long_uids.append(uid)

    if long_uids:
        longest = max((dur.get(items[i].audio_abs(DS.dir_of(dataset)), 0.0)
                       for i in long_uids if i in items), default=0.0)
        cb(0.65, f"{len(long_uids)} 条长音频需要切片（最长 {longest:.0f} 秒）")

    # ---- 逐条切片（split_long 自己管锁，且会把长原件留成 too_long 不参与训练）----
    # 切片是整条流水线里最慢的**单点**：整段解码 → 全量帧能量 → 逐窗口评分 →
    # 逐片写盘。所以这里把进度回调一路传进去（子进度映射到 65%~100%），
    # 并且把 should_stop 也传进去 —— 否则按了停止要等它整条跑完才生效，
    # 而用户看到的就是「没日志、停不下来」。
    pieces = 0
    total = max(1, len(long_uids))
    for i, uid in enumerate(long_uids):
        if should_stop and should_stop():
            result["stopped"] = True
            break
        cb(0.65 + 0.35 * (i / total),
           f"切片 {i + 1}/{len(long_uids)} · {uid}（长音频，需要一点时间）")
        sr = DS.split_long(
            dataset, uid,
            target_sec=float(opt.slice_target_sec),
            min_sec=float(opt.slice_min_sec),
            max_pieces=int(opt.slice_max_pieces),
            progress=(lambda f, m, _i=i: cb(
                0.65 + 0.35 * ((_i + f) / total), m)) if progress else None,
            should_stop=should_stop)
        if sr.get("ok"):
            pieces += int(sr.get("created") or 0)
        elif sr.get("stopped"):
            result["stopped"] = True
            break
        else:
            result.setdefault("slice_errors", []).append(
                f"{uid}: {sr.get('error', '未知错误')}")

    # 再查一次：停止请求可能是在**最后一条**切片的过程中发出的，
    # 那时循环已经不会再迭代，不补这一下就会被算成「采集阶段正常完成」，
    # 于是停止被归因到下一个阶段（实测出现过「在优化阶段停止」的误导）。
    if not result.get("stopped") and should_stop and should_stop():
        result["stopped"] = True

    result["sliced"] = len(long_uids)
    result["pieces"] = pieces
    cb(1.0, f"导入 {result['added']} 条，切出 {pieces} 片"
            + ("（已中断）" if result.get("stopped") else ""))
    if report is not None:
        report.inputs = dict(result)
    return result


# ===========================================================================
# S2 · 音频优化
# ===========================================================================

def _enhance_one(ds_dir: str, u: DS.Utterance, opt: OneClickOptions) -> Dict[str, Any]:
    """对一条样本做增强，原地替换数据集里的音频。

    增强产物一律写成 `<uid>.wav`：源文件可能是 .mp3/.m4a（librosa 能读，
    但写出来的是 wav 数据），改名并把 `audio` 字段一起更新，避免出现
    「后缀是 .mp3、内容是 wav」的迷惑文件。
    """
    src = u.audio_abs(ds_dir)
    audio_dir = os.path.join(ds_dir, DS.AUDIO_SUBDIR)
    dst = os.path.join(audio_dir, f"{u.id}.wav")
    tmp = dst + ".enh.tmp.wav"

    res = AL.enhance(src, tmp,
                     denoise=bool(opt.denoise),
                     denoise_strength=float(opt.denoise_strength),
                     normalize=bool(opt.normalize),
                     trim_silence=bool(opt.trim_silence),
                     target_sr=OUTPUT_SAMPLE_RATE,
                     max_sec=float(opt.slice_over_sec))
    if not res.ok:
        try:
            os.path.isfile(tmp) and os.remove(tmp)
        except OSError:
            pass
        return {"ok": False, "error": res.error or "增强失败"}

    try:
        os.replace(tmp, dst)
    except OSError as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 源文件名与目标不同（扩展名不同）时，删掉旧文件并改 meta
    new_rel = os.path.join(DS.AUDIO_SUBDIR, f"{u.id}.wav")
    fields: Dict[str, Any] = {"audio": new_rel}
    if os.path.normcase(os.path.abspath(src)) != os.path.normcase(dst):
        try:
            os.remove(src)
        except OSError:
            pass
    return {"ok": True, "fields": fields}


def stage_optimize(dataset: str, opt: OneClickOptions,
                   progress=None, should_stop=None, report=None,
                   bands: Optional[_Bands] = None) -> Dict[str, Any]:
    """去直流 / 掐静音 / 降噪 / 归一 / 重采样 / 截断到 slice_over_sec。"""
    bands = bands or _Bands(STAGES)
    cb = _sub(progress, bands, "optimize", "优化")
    ds_dir = DS.dir_of(dataset)
    items = DS.load_meta(dataset)
    out = {"total": len(items), "enhanced": 0, "failed": 0,
           "skipped": 0, "skipped_long": 0,
           "before_avg_score": 0.0, "after_avg_score": 0.0}

    if not opt.enhance:
        out["skipped"] = len(items)
        cb(1.0, "已关闭音频优化，跳过")
        return out

    # 待处理的条目（跳过空音频与不可训练的超长原件）
    todo: List[DS.Utterance] = []
    for u in items:
        if not u.audio:
            continue
        if float(u.duration or 0.0) > float(opt.slice_over_sec):
            out["skipped_long"] += 1
            continue
        todo.append(u)

    n = max(1, len(todo))
    touched: Dict[str, Dict[str, Any]] = {}
    scores_before, scores_after = [], []

    def _one(u: DS.Utterance) -> Dict[str, Any]:
        return _enhance_one(ds_dir, u, opt)

    def _done(i: int, res: Any) -> None:
        cb(i / n, f"优化 {i + 1}/{len(todo)} · {todo[i].id}")

    # 每条样本：读自己的音频 → 算 → 写自己的文件。彼此独立，所以并行安全，
    # 而且结果与串行逐字节一致（探针里有对账）。见 training/parallel.py。
    workers = PL.default_workers(len(todo), opt.cpu_workers)
    outcome = PL.map_parallel(_one, todo, workers=workers,
                              should_stop=should_stop, on_done=_done)
    out["workers"] = outcome.workers
    out["seconds"] = outcome.seconds
    out["stopped"] = bool(outcome.stopped)

    for i, u in enumerate(todo):
        if i in outcome.errors:
            out["failed"] += 1
            out.setdefault("errors", []).append(f"{u.id}: {outcome.errors[i]}")
            continue
        r = outcome.items[i]
        if not isinstance(r, dict) or not r.get("ok"):
            out["failed"] += 1
            continue
        try:
            scores_before.append(float(u.score or 0.0))
        except Exception:
            pass
        touched[u.id] = r["fields"]
        out["enhanced"] += 1

    if opt.denoise:
        note = AL.denoise_note()
        if note:
            # 缺 noisereduce 时 _denoise 会原样返回 —— 这是**静默跳过**，
            # 必须在报告里说出来，否则用户会以为降噪做过了。
            out["denoise_note"] = note

    FT._apply_meta(dataset, touched)
    DS.refresh_all(dataset, require_features=False,
                   workers=workers,
                   progress=lambda f, m: cb(0.9 + 0.1 * f, m))

    after = {x.id: x for x in DS.load_meta(dataset)}
    for uid in touched:
        u = after.get(uid)
        if u is not None:
            scores_after.append(float(u.score or 0.0))
    if scores_before:
        out["before_avg_score"] = round(sum(scores_before) / len(scores_before), 1)
    if scores_after:
        out["after_avg_score"] = round(sum(scores_after) / len(scores_after), 1)

    tail = f"（{workers} 线程 · {outcome.seconds}s）" if workers > 1 else ""
    if out.get("denoise_note"):
        tail += " · 降噪已跳过（未装 noisereduce）"
    cb(1.0, f"优化 {out['enhanced']} 条，平均体检分 "
            f"{out['before_avg_score']} → {out['after_avg_score']}{tail}")
    return out


# ===========================================================================
# S3 · 识别与对齐
# ===========================================================================

def stage_asr(dataset: str, opt: OneClickOptions,
              progress=None, should_stop=None, report=None,
              bands: Optional[_Bands] = None) -> Dict[str, Any]:
    """逐条 whisper 转写，写回 text / asr_text。

    长音频在 S1 已切片，所以这里是「一片转一段」—— 文本与音频天然配对，
    这就是全流程的对齐环节。已有 text 的样本不覆盖（尊重人工校对）。
    """
    bands = bands or _Bands(STAGES)
    cb = _sub(progress, bands, "asr", "识别")
    items = DS.load_meta(dataset)
    out = {"total": len(items), "transcribed": 0, "empty": 0, "failed": 0,
           "kept_existing": 0, "whisper": opt.whisper_size}

    if not opt.asr:
        out["skipped"] = True
        cb(1.0, "已关闭语音识别")
        return out

    todo = [u for u in items if not str(u.text or "").strip()]
    out["kept_existing"] = len(items) - len(todo)
    if not todo:
        cb(1.0, "全部样本都已有文本，跳过识别")
        return out

    sc = RW.RewardScorer(
        RW.RewardOptions(whisper_size=str(opt.whisper_size),
                         language=RW.asr_language(opt.lang)))
    out["prompt"] = sc.resolve_prompt()
    out["vram_before_gb"] = sc.vram_free_gb()
    touched: Dict[str, Dict[str, Any]] = {}
    n = max(1, len(todo))
    try:
        cb(0.01, f"加载 whisper-{opt.whisper_size}"
                 f"（首次会下载，空闲显存 {out['vram_before_gb']} GB）")
        for i, u in enumerate(todo):
            if should_stop and should_stop():
                out["stopped"] = True
                break
            cb(i / n, f"转写 {i + 1}/{len(todo)} · {u.id}")
            ap = u.audio_abs(DS.dir_of(dataset))
            try:
                txt = str(sc.transcribe(ap) or "").strip()
            except Exception as e:
                out["failed"] += 1
                out.setdefault("errors", []).append(
                    f"{u.id}: {type(e).__name__}: {e}")
                continue
            if not txt:
                out["empty"] += 1
                touched[u.id] = {"note": (u.note + " | ASR 无输出").strip(" |")}
                continue
            touched[u.id] = {"text": txt, "asr_text": txt,
                             "asr_model": f"whisper-{opt.whisper_size}"}
            out["transcribed"] += 1
    finally:
        # 用完立刻卸载：后面要加载推理引擎（特征是笔大开销），显存必须先腾出来。
        # medium/large 是 GB 量级，留在卡上会直接导致引擎加载失败或静默降速。
        try:
            sc.unload()
        except Exception:
            pass

    out["loaded_after_unload"] = bool(sc.is_loaded())
    out["vram_after_gb"] = sc.vram_free_gb()
    out["freed_gb"] = round(float(out["vram_after_gb"])
                            - float(out["vram_before_gb"]), 2)

    FT._apply_meta(dataset, touched)
    release = ("已卸载" if not out.get("loaded_after_unload")
               else "⚠️ 卸载后仍有驻留")
    cb(1.0, f"转写 {out['transcribed']} 条"
            f"（无输出 {out['empty']}，失败 {out['failed']}）· "
            f"whisper-{opt.whisper_size} {release}，"
            f"释放 {out.get('freed_gb')} GB")
    return out


# ===========================================================================
# S4 · 筛选与划分
# ===========================================================================

def stage_curate(dataset: str, opt: OneClickOptions, progress=None,
                 report=None, bands: Optional[_Bands] = None) -> Dict[str, Any]:
    """体检复算 → 按状态/评分/文本去重筛选 → 划分 train/val。"""
    bands = bands or _Bands(STAGES)
    cb = _sub(progress, bands, "curate", "筛选")

    cb(0.1, "重新体检全部样本")
    DS.refresh_all(dataset, require_features=False,
                   progress=lambda f, m: cb(0.1 + 0.5 * f, m))

    items = DS.load_meta(dataset)
    out: Dict[str, Any] = {"total": len(items), "dropped": {},
                           "ready_before": 0, "ready_after": 0}

    by_status: Dict[str, int] = {}
    for u in items:
        by_status[u.status] = by_status.get(u.status, 0) + 1
    out["by_status"] = by_status
    out["ready_before"] = int(by_status.get("ready", 0))

    # ---- 逐条裁决 ----
    drop: Dict[str, str] = {}
    seen_text: Dict[str, int] = {}
    for u in items:
        if u.status != "ready":
            drop[u.id] = f"状态 {u.status}"
            continue
        if float(opt.min_score) > 0 and float(u.score or 0) < float(opt.min_score):
            drop[u.id] = f"体检分 {u.score:.0f} < {opt.min_score:.0f}"
            continue
        cap = int(opt.max_text_repeats)
        if cap > 0:
            key = " ".join(str(u.text).split()).lower()
            seen_text[key] = seen_text.get(key, 0) + 1
            if seen_text[key] > cap:
                drop[u.id] = f"同文本重复超过 {cap} 条"
                continue

    for reason in drop.values():
        k = reason.split(" ")[0]
        out["dropped"][k] = out["dropped"].get(k, 0) + 1

    if drop:
        DS.remove_utterances(dataset, list(drop.keys()))
    out["dropped_total"] = len(drop)

    items = DS.load_meta(dataset)
    ready = [u for u in items if u.status == "ready"]
    out["ready_after"] = len(ready)
    out["minutes"] = round(sum(float(u.duration or 0) for u in ready) / 60.0, 2)

    # ---- 样本量闸门：宁可在这里停，也不进训练器再炸 ----
    if len(ready) < MIN_SAMPLES:
        out["ok"] = False
        out["error"] = (
            f"筛选后只剩 {len(ready)} 条可用样本，低于训练下限 {MIN_SAMPLES} 条。"
            "可能原因：音频太短/太长、信噪比不足、识别失败、或体检分门槛太高。"
            "建议：多录几段干净人声，或把 min_score 调低、"
            "slice_target_sec 调小以切出更多片段。")
        cb(1.0, out["error"])
        return out

    cb(0.85, f"划分 train/val（{len(ready)} 条）")
    sp = DS.make_split(dataset, val_ratio=float(opt.val_ratio),
                       seed=int(opt.seed))
    out["ok"] = bool(sp.get("ok"))
    out["split"] = {"train": sp.get("train", 0), "val": sp.get("val", 0)}
    if not out["ok"]:
        out["error"] = str(sp.get("error") or "划分失败")

    cb(1.0, f"留用 {len(ready)} 条 · 约 {out['minutes']} 分钟 · "
            f"train {out['split']['train']} / val {out['split']['val']}")
    return out


# ===========================================================================
# S5 · 自动调参
# ===========================================================================

def preset_ladder(minutes: float) -> List[str]:
    """按数据量排出候选预设顺序。

    依据 `GD.PRESET_NOTES` 的出厂建议：<5 分钟用保守档，≥30 分钟才考虑激进档。
    第一个通过的候选会被采用，所以顺序即偏好。
    """
    m = float(minutes or 0.0)
    if m < 5.0:
        return ["conservative", "balanced"]
    if m < 30.0:
        return ["balanced", "conservative", "aggressive"]
    return ["balanced", "aggressive", "conservative"]


def eval_every_for(total_steps: int, top_k: int) -> int:
    """按真实总步数推导「每几步评估一次」。

    出厂预设的 eval_every（50/100/200）是给大数据集定的。小数据集上
    一个 run 总共才几个 step，整个 run 都触发不到一次按步评估，保险库
    最终只有「轮末兜底评估」存下的 1 个档位 —— 那么「留 K 个档位再择优」
    就无档可选，只能退化成拿 1 个候选自说自话。

    这里按总步数反推，保证能攒出大约 K 个可比较的档位；总步数大的长训练
    自然得到大的间隔，不会把评估次数抬到病态（评估要跑完整 val 集）。
    下限 1：再小也只能是每步一评。
    """
    steps = int(total_steps or 0)
    if steps <= 0:
        return 0
    return max(1, steps // (max(1, int(top_k)) + 1))


def build_cfg(arch: str, preset: str, top_k: int, epochs: Optional[int] = None
              ) -> Tuple[GD.LoRAConfig, Any]:
    """(LoRAConfig, TrainOptions)。CFM 走它自己的 default_config ——
    `LoRAConfig.preset()` 硬编码了 gpt 的注入面与 bf16，直接拿来训练
    CFM 会注入错层。"""
    if arch == "cfm":
        cfg = CL.default_config(preset)
        opts = CL.CfmTrainOptions()
    else:
        cfg = GD.LoRAConfig.preset(preset)
        cfg.apply_target_preset(str(cfg.target_preset), "gpt")
        opts = GL.GptTrainOptions()
    cfg.keep_checkpoints = int(top_k)     # 保险库只留 top-K，正是「三个」的来源
    if epochs is not None:
        cfg.epochs = int(epochs)
    return cfg, opts


def _make_trainer(arch: str, dataset: str, cfg: GD.LoRAConfig, opts: Any,
                  run_name: str, val_ratio: float):
    if arch == "cfm":
        return CL.CfmTrainer(dataset, cfg=cfg, options=opts,
                             run_name=run_name, val_ratio=val_ratio)
    from webui_app.training import dpo as DP
    if arch == "dpo":
        return DP.DpoTrainer(dataset, cfg=cfg, options=opts,
                             run_name=run_name, val_ratio=val_ratio)
    return GL.GptTrainer(dataset, cfg=cfg, options=opts,
                         run_name=run_name, val_ratio=val_ratio)


def stage_tune(dataset: str, opt: OneClickOptions, progress=None,
               report=None, bands: Optional[_Bands] = None) -> Dict[str, Any]:
    """逐个 preflight 候选配置，挑出第一个真能跑起来的。

    这是「选择最合适的参数进行调整」的实现：不是查表，而是拿真实的
    数据集与空闲显存，把候选配置**实际跑一遍预检**（读 meta、建样本池、
    估显存、查注入面、试划分），通过的才采用。被拒的候选连原因一起
    记录，在 UI 上摆出来。
    """
    bands = bands or _Bands(STAGES)
    cb = _sub(progress, bands, "tune", "调参")

    st = DS.stats(dataset)
    minutes = float(st.get("ready_minutes") or 0.0)
    n_ready = int(st.get("ready") or 0)

    vr = GD.vram_headroom()
    free_gb = float(getattr(vr, "free_gb", 0.0) or 0.0)

    arches = opt.arch_list()
    ladder = ([str(opt.preset_mode)] if opt.preset_mode != "auto"
              else preset_ladder(minutes))

    out: Dict[str, Any] = {"n_ready": n_ready, "minutes": round(minutes, 2),
                           "free_vram_gb": round(free_gb, 2),
                           "preset_order": ladder, "arches": arches,
                           "candidates": [], "chosen": {}}

    total = max(1, len(arches) * len(ladder))
    step = 0
    for arch in arches:
        for preset in ladder:
            step += 1
            cb(step / total, f"预检 {arch} · {preset}")
            row: Dict[str, Any] = {"arch": arch, "preset": preset}

            try:
                cfg, opts = build_cfg(arch, preset, opt.top_k)
            except Exception as e:
                row["reject"] = f"配置构造失败 {type(e).__name__}: {e}"
                out["candidates"].append(row)
                continue

            row.update(rank=cfg.rank, alpha=cfg.alpha, lr=cfg.lr,
                       epochs=cfg.epochs, target_preset=cfg.target_preset)

            # 配置自检（含「数据量 vs 训练强度」）—— 有 error 就没必要预检了
            notices = cfg.validate(n_ready)
            errs = [x.message for x in notices if x.level == "error"]
            if errs:
                row["reject"] = errs[0]
                out["candidates"].append(row)
                continue
            row["warnings"] = [x.message for x in notices if x.level == "warn"]

            # 真预检：读 meta、建样本池、估显存、试划分
            try:
                tr = _make_trainer(arch, dataset, cfg, opts,
                                   run_name=f"__tune_{arch}_{preset}",
                                   val_ratio=float(opt.val_ratio))
                pf = tr.preflight()
            except Exception as e:
                row["reject"] = f"预检异常 {type(e).__name__}: {e}"
                out["candidates"].append(row)
                continue

            row["total_steps"] = pf.get("total_steps")
            row["n_train"] = pf.get("n_train")
            row["n_val"] = pf.get("n_val")
            row["est_vram_gb"] = pf.get("est_vram_gb")

            if not pf.get("ok"):
                bad = (pf.get("errors") or ["预检未通过"])[0]
                row["reject"] = str(bad)
                out["candidates"].append(row)
                continue

            est = float(pf.get("est_vram_gb") or 0.0)
            # 留 0.6 GB 余量（与 guard.VRAM_SAFETY_MARGIN_GB 同口径）
            if free_gb > 0 and est + GD.VRAM_SAFETY_MARGIN_GB > free_gb:
                row["reject"] = (f"预估 {est:.2f} GB + 余量 "
                                 f"{GD.VRAM_SAFETY_MARGIN_GB} GB 超过空闲 "
                                 f"{free_gb:.2f} GB")
                out["candidates"].append(row)
                continue

            # ---- 按真实步数推导评估频率 ----
            # 出厂预设的 eval_every（50/100/200）是给大数据集定的。小数据集上
            # 一个 run 总共才几个 step，整个 run 都触发不到一次按步评估，
            # 保险库最终只有「轮末兜底评估」存下的 1 个档位 —— 那么 top-K 择优
            # 就名存实亡（只剩 1 个候选可比）。这里按总步数反推，保证能攒出
            # 大约 top_k 个可比较的档位；总步数大的长训练自然得到大的间隔，
            # 不会把评估次数抬到病态。
            steps = int(pf.get("total_steps") or 0)
            cfg.eval_every = eval_every_for(steps, opt.top_k)
            row["eval_every"] = cfg.eval_every

            row["chosen"] = True
            out["candidates"].append(row)
            out["chosen"][arch] = {"preset": preset, "cfg": cfg, "options": opts,
                                   "preflight": {k: v for k, v in pf.items()
                                                 if k != "vram"},
                                   "total_steps": pf.get("total_steps"),
                                   "est_vram_gb": est}
            break        # 这个目标已选定，换下一个目标

    missing = [a for a in arches if a not in out["chosen"]]
    if missing:
        out["ok"] = False
        out["error"] = (
            f"目标 {'/'.join(missing)} 没有任何候选配置通过预检。"
            "最可能的原因是样本太少或空闲显存不足 —— 看下方候选表的「裁决」列。")
    else:
        out["ok"] = True
        picks = "、".join(f"{a}→{v['preset']}" for a, v in out["chosen"].items())
        cb(1.0, f"选定：{picks}")
    return out


# ===========================================================================
# S6 · 训练
# ===========================================================================

def stage_train(dataset: str, engine, tuning: Dict[str, Any],
                opt: OneClickOptions, progress=None, should_stop=None,
                tracker=None, report=None, bands: Optional[_Bands] = None
                ) -> Dict[str, Any]:
    """逐目标真训练。**训练前必须卸载引擎**（8 GB 卡放不下两份）。"""
    bands = bands or _Bands(STAGES)
    chosen: Dict[str, Any] = tuning.get("chosen") or {}
    out: Dict[str, Any] = {"runs": [], "ok": False}
    if not chosen:
        out["error"] = "没有选定的配置，跳过训练"
        return out

    # 训练记录名：用户给的「模型名称」优先。单目标就用原名（所见即所得），
    # 多目标必须加 `_arch` 后缀 —— 否则 CFM 会把 GPT 的记录目录覆盖掉。
    names = opt.plan_run_names()
    out["run_names"] = dict(names)
    n = len(chosen)
    results = []
    for i, (arch, pick) in enumerate(chosen.items()):
        if should_stop and should_stop():
            out["stopped"] = True
            break

        # ---- 引擎必须卸载 ----
        if getattr(engine, "loaded", False):
            progress and progress(bands.at("train", i / n),
                                  "[训练] 卸载引擎，腾出显存给训练器…")
            engine.unload()

        run_name = names.get(arch) or RN.safe_run_name(f"{arch}_{time.strftime('%H%M%S')}")
        cfg = pick["cfg"]
        opts = pick["options"]
        cb = _sub(progress, bands, "train", f"训练·{arch}",
                  base=i / n, span=1.0 / n)

        cb(0.01, f"{arch} 准备（名称 {run_name} · rank={cfg.rank} · "
                f"preset={pick['preset']}）")
        tr = _make_trainer(arch, dataset, cfg, opts, run_name=run_name,
                           val_ratio=float(opt.val_ratio))
        # 名字冲突在 prepare 里才暴露，这里提前挡住并换个名字
        conflict = tr.name_conflict()
        if conflict:
            run_name = RN.suggest_run_name(arch, dataset)
            tr = _make_trainer(arch, dataset, cfg, opts, run_name=run_name,
                               val_ratio=float(opt.val_ratio))
            cb(0.02, f"运行名冲突，改用 {run_name}")

        if tracker is not None:
            try:
                tracker.track_run(tr.run_name)
            except Exception:
                pass

        try:
            rep = tr.run(progress=cb, should_stop=should_stop)
        except Exception as e:
            results.append({"arch": arch, "run": tr.run_name, "ok": False,
                            "error": f"{type(e).__name__}: {e}"})
            continue

        cks = []
        try:
            cks = [{"name": c.name, "metric": float(c.metric or 0.0),
                    "epoch": c.epoch, "step": c.step}
                   for c in RN.list_checkpoints(tr.run_name)]
        except Exception:
            pass

        row = {"arch": arch, "run": tr.run_name, "preset": pick["preset"],
               "ok": bool(getattr(rep, "ok", False)),
               "steps": getattr(rep, "steps", 0),
               "epochs": getattr(rep, "epochs", 0),
               "first_val": (round(rep.first_val, 4)
                             if getattr(rep, "first_val", None) is not None else None),
               "best_val": (round(rep.best_val, 4)
                            if getattr(rep, "best_val", None) is not None else None),
               "improved": (round(rep.improved, 4)
                            if getattr(rep, "improved", None) is not None else None),
               "stopped_early": bool(getattr(rep, "stopped_early", False)),
               "stop_reason": getattr(rep, "stop_reason", "") or "",
               "vram_peak_gb": getattr(rep, "vram_peak_gb", 0.0),
               "n_checkpoints": len(cks),
               "checkpoints": cks,
               "seconds": round(getattr(rep, "seconds", 0.0), 1),
               "error": getattr(rep, "error", "") or ""}
        results.append(row)
        cb(1.0, f"{arch} 完成：val "
                f"{row['first_val']} → {row['best_val']}"
                + ("（早停）" if row["stopped_early"] else ""))

        # adapter 已由训练器同步到 <run>/adapter；保险库档位留给 S7 参评
        out[f"{arch}_run"] = tr.run_name

    out["runs"] = results
    good = [x for x in results if x.get("ok") and x.get("n_checkpoints")]
    out["ok"] = bool(good)
    if not out["ok"]:
        out["error"] = "训练全部失败或没有产出任何可用档位。"
    return out


# ===========================================================================
# S7 · 择优与交付
# ===========================================================================

def stage_rank(dataset: str, engine, training: Dict[str, Any],
               opt: OneClickOptions, progress=None, should_stop=None,
               report=None, bands: Optional[_Bands] = None) -> Dict[str, Any]:
    """逐档位真机合成 → reward 打分 → 排序 → 激活最优。

    val loss 只能说明「拟合得好不好」，说明不了「像不像本人」——
    后者只能靠合成出来再打分。所以这一步必须加载引擎。
    """
    bands = bands or _Bands(STAGES)
    cb = _sub(progress, bands, "rank", "择优")
    rows = [r for r in (training.get("runs") or []) if r.get("ok")]
    out: Dict[str, Any] = {"candidates": [], "ranking": [], "ok": False}

    if not rows:
        out["error"] = "没有训练成功的 run，无法择优"
        return out

    # ---- 收集候选：每个 run 的 top-K 档位（没有档位就退回 adapter/best）----
    cands: List[Tuple[str, str, float, str]] = []      # (run, ckpt, val, arch)
    for r in rows:
        cks = sorted(r.get("checkpoints") or [], key=lambda c: c["metric"])
        for c in cks[:max(1, int(opt.top_k))]:
            cands.append((r["run"], c["name"], float(c["metric"]), r["arch"]))
        if not cks:
            cands.append((r["run"], "best", float("nan"), r["arch"]))
    out["n_candidates"] = len(cands)

    # ---- 关掉择优评测时：只按 val loss 排序 ----
    if not opt.rank_eval:
        for i, (run, ck, val, arch) in enumerate(cands, 1):
            out["ranking"].append({"place": i, "run": run, "checkpoint": ck,
                                   "arch": arch, "val": round(val, 4)
                                   if val == val else None,
                                   "reward": None, "wer": None, "sim": None,
                                   "n": 0})
        out["ranking"].sort(key=lambda x: (x["val"] is None, x["val"]))
        for i, x in enumerate(out["ranking"], 1):
            x["place"] = i
        out["ok"] = True
        out["best"] = dict(out["ranking"][0])
        cb(1.0, "按 val loss 排序（已关闭择优评测）")
        return out

    # ---- 真机评测：每个候选与纯底座同种子对比，取 reward ----
    if not getattr(engine, "loaded", False):
        cb(0.02, "加载引擎用于评测…")
        engine.load()

    st = DS.stats(dataset)
    scored: List[Dict[str, Any]] = []
    n = max(1, len(cands))

    # 一个 scorer 服务全部候选：run_eval 只卸载自己创建的 scorer，
    # 外面传进去的由这里收尾。否则每个候选都要「加载 1.6 GB whisper + 卸掉」，
    # 6 个候选就是 6 个来回 —— 白等好几分钟，还反复挤压引擎的显存。
    shared = RW.RewardScorer(RW.RewardOptions(
        whisper_size=str(opt.score_whisper_size),
        language=RW.asr_language(opt.lang)))

    # ---- 显存余量体检：这一步是引擎与打分器**同时驻留**的唯一阶段 ----
    # 实测 8 GB 卡上：引擎 4.94 + whisper medium 1.6 + campplus ≈ 7.8/8.2 GB，
    # 只剩几百 MB 余量。溢出不会报 OOM，而是在 WDDM 下静默降速 20~30 倍
    # （CFM 25 步 2.4s → 66s 那种），所以宁可提前说清楚。
    if getattr(engine, "loaded", False):
        est = RW.WHISPER_SIZES.get(str(opt.score_whisper_size), (0, 0))[1]
        try:
            vr = GD.vram_headroom(need_gb=float(est))
            if not vr.ok:
                msg = (f"评分阶段引擎与 whisper-{opt.score_whisper_size}"
                       f"（约 {est:.2f} GB）需要同时驻留，"
                       f"当前空闲 {vr.free_gb} GB、缺口 {vr.shortfall_gb} GB —— "
                       "可能触发 Windows 显存溢出静默降速（不是 OOM）。"
                       "建议把「打分模型」换成 small，"
                       "或关掉浏览器等占显存的程序。")
                out.setdefault("warnings", []).append(msg)
                if progress:
                    progress(bands.at("rank", 0.02), "[择优] ⚠️ " + msg)
        except Exception:
            pass
    out["vram_headroom"] = {}
    try:
        _vr = GD.vram_headroom()
        out["vram_headroom"] = {"free_gb": _vr.free_gb, "need_gb": _vr.need_gb,
                                "level": _vr.level}
    except Exception:
        pass

    try:
        for i, (run, ck, val, arch) in enumerate(cands):
            if should_stop and should_stop():
                out["stopped"] = True
                break
            cb(0.05 + 0.9 * (i / n), f"评测 {i + 1}/{len(cands)} · {run}/{ck}")
            opts = EV.EvalOptions(
                dataset=dataset,
                out_dir=os.path.join(ONECLICK_ROOT,
                                     f"{time.strftime('%Y%m%d-%H%M%S')}"
                                     f"_rank_{run}_{ck}"),
                n_samples=int(opt.eval_samples or 0),
                seed=int(opt.seed),
                whisper_size=str(opt.score_whisper_size),
                language=RW.asr_language(opt.lang),
            )
            try:
                a = EV.Contender(name=f"{run}:{ck}", run=run, checkpoint=ck)
                res = EV.run_eval(engine, a, None, opts,
                                  progress=lambda f, m: cb(
                                      0.05 + 0.9 * ((i + f) / n), m),
                                  should_stop=should_stop,
                                  scorer=shared)
            except Exception as e:
                scored.append({"run": run, "checkpoint": ck, "arch": arch,
                               "val": val, "reward": None, "wer": None,
                               "sim": None, "n": 0,
                               "error": f"{type(e).__name__}: {e}"})
                continue

            summ = (res or {}).get("summary", {}) or {}
            aa = summ.get("a") or {}
            scored.append({
                "run": run, "checkpoint": ck, "arch": arch,
                "val": round(val, 4) if val == val else None,
                "reward": aa.get("reward"), "wer": aa.get("wer"),
                "sim": aa.get("sim"), "n": int(aa.get("n") or 0),
                "out_dir": (res or {}).get("out_dir", ""),
                "ok": bool((res or {}).get("ok")),
                "error": ("；".join((res or {}).get("errors") or []) or "")[:200],
            })
    finally:
        # 无论成功、失败还是被停止，打分器都要还回去（medium 是 1.6 GB）。
        try:
            shared.unload()
        except Exception:
            pass
        out["scorer_released"] = not shared.is_loaded()
        out["vram_after_gb"] = shared.vram_free_gb()

    out["candidates"] = scored
    rankable = [x for x in scored if x.get("reward") is not None and x.get("n")]
    # 一次都没评成 → 退回 val loss 排序，别把流水线整体判死
    if not rankable:
        for i, x in enumerate(sorted(
                [s for s in scored], key=lambda y: (y["val"] is None, y["val"])), 1):
            x["place"] = i
        out["ranking"] = scored
        out["ok"] = bool(scored)
        out["degraded"] = True
        if scored:
            out["best"] = dict(scored[0])
        cb(1.0, "全部候选都评不出分，已退回按 val loss 排序")
        return out

    rankable.sort(key=lambda x: (-float(x["reward"]),
                                 x["val"] if x["val"] is not None else 9e9))
    for i, x in enumerate(rankable, 1):
        x["place"] = i
    out["ranking"] = rankable
    out["ok"] = True
    out["best"] = dict(rankable[0])

    # ---- 激活最优档位：把它的权重同步到 <run>/adapter，推理页就能直接选 ----
    b = rankable[0]
    try:
        act = RN.activate(b["run"], which=b["checkpoint"])
        b["activated"] = bool(act)
        out["best"] = dict(b)
        cb(0.98, f"已激活 {b['run']}/{b['checkpoint']}")
    except Exception as e:
        b["activate_error"] = f"{type(e).__name__}: {e}"
        out["best"] = dict(b)
        out.setdefault("warnings", []).append(
            f"无法自动激活最优档位：{b['activate_error']}")

    cb(1.0, f"最优：{b['run']}/{b['checkpoint']}（reward {b['reward']}）")
    return out


# ===========================================================================
# 主流程
# ===========================================================================

def run_oneclick(engine, options: OneClickOptions,
                 paths: Sequence[str] = (), extra_path: str = "",
                 progress: Optional[Callable[[float, str], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None,
                 tracker=None) -> Dict[str, Any]:
    """跑完整条流水线。由 runner 以 require_engine="none" 提交。

    返回 {"ok", "dataset", "markdown", "report", "out_dir", ...}；
    runner 只在 ok is False 时判失败，所以任何提前终止都必须把 ok 置 False。
    """
    t0 = time.perf_counter()
    bands = _Bands(STAGES)
    rep = OneClickReport(started_at=t0, options=options.to_dict())
    log = LOG.get_logger("oneclick")

    # ------------------------------------------------------------------
    # 持续可观测性：进度落盘 + 心跳
    #
    # 之前只有「阶段完成」才会在界面上留一行，而切片这种**单点几步跑几分钟**
    # 的工序中间什么都看不到 —— 用户没法区分「在工作」和「卡死了」，
    # 事后翻文件日志也只有任务的开始和结束两条。所以：
    #   · 每条进度都写进文件日志（DEBUG 级，不刷屏的级别下也能按需打开）；
    #   · 一条**心跳线程**：只要超过 HEARTBEAT_IDLE 秒没有新进度，
    #     就按 INFO 报一次「仍在进行 + 当前阶段 + 已静默多久 + 最后一条进度」。
    #     静默本身就是信息 —— 它能把「长音频在算」和「真的卡住」分开。
    # ------------------------------------------------------------------
    HEARTBEAT_IDLE = 12.0
    HEARTBEAT_PERIOD = 5.0
    # 先抓住调用方给的进度回调：下面会把 `progress` 这个名字重绑成包装版，
    # 而 Python 闭包捕获的是**变量** —— 不先存一份就会自己调自己，无限递归。
    _user_progress = progress
    last = {"t": time.time(), "msg": "启动", "frac": 0.0}
    hb_stop = threading.Event()

    def _progress(frac: float, msg: str) -> None:
        last.update(t=time.time(), msg=msg, frac=float(frac))
        log.debug("[%3.0f%%] %s", float(frac) * 100, msg)
        if _user_progress is not None:
            _user_progress(frac, msg)

    def _heartbeat() -> None:
        while not hb_stop.wait(HEARTBEAT_PERIOD):
            idle = time.time() - last["t"]
            if idle < HEARTBEAT_IDLE:
                continue
            cur = rep.running()
            log.info("…仍在进行 · 阶段「%s」· 已静默 %.0f 秒 · 最后进度：%s",
                     cur.get("title", "启动"), idle, last["msg"])
            if _user_progress is not None:
                # 让界面上那行进度也动一下，否则看起来像死了
                _user_progress(last["frac"],
                               f"仍在进行（{cur.get('title', '')}）· "
                               f"最后进度：{last['msg']}")

    def _phase_cb(status: str, title: str) -> None:
        # 让 runner 也记下当前阶段：任务失败时它会把这一项打进日志，
        # 归因到具体阶段而不是一句「未知」。
        try:
            if tracker is not None:
                tracker.phase = title if status == "start" else ""
        except Exception:
            pass

    rep.phase_cb = _phase_cb
    progress = _progress          # 之后所有阶段都用包装版（落盘 + 心跳）
    threading.Thread(target=_heartbeat, daemon=True,
                     name="oneclick-heartbeat").start()

    def set_req(req: str) -> None:
        """告诉 runner「本流水线此刻需要引擎处于什么状态」。

        本任务是以 require_engine="none" 提交的（否则切换引擎状态会被自己
        的要求挡死），代价是训练期间 engine.load() 失去保护。这里在每个
        阶段边界把真实需求同步给 runner，双向互斥因此全程有效：
          · 特征提取 / 择优评测 → "loaded"（引擎必须常驻）
          · 训练 → "unloaded"（引擎必须让出显存，且期间禁止别人加载）
        见 runner.TrainRunner.set_engine_req 的说明。
        """
        try:
            if tracker is not None and hasattr(tracker, "set_engine_req"):
                tracker.set_engine_req(req)
        except Exception:
            pass

    def done(status: str, error: str = "") -> Dict[str, Any]:
        hb_stop.set()          # 心跳必须停，否则它会在任务结束后继续写日志
        rep.ok = status == "done"
        rep.error = error
        rep.seconds = round(time.perf_counter() - t0, 1)
        out_dir = os.path.join(ONECLICK_ROOT, time.strftime("%Y%m%d-%H%M%S"))
        try:
            os.makedirs(out_dir, exist_ok=True)
            rep_md = rep.markdown()
            with open(os.path.join(out_dir, "report.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"report": asdict(rep)}, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "report.md"), "w",
                      encoding="utf-8") as f:
                f.write(rep_md)
        except Exception:
            rep_md = rep.markdown()
        res = {"ok": rep.ok, "dataset": rep.dataset, "out_dir": out_dir,
               "markdown": rep_md, "report": asdict(rep),
               "seconds": rep.seconds, "error": error}
        # 把交付物一起回传，UI 好直接跳转/预选
        res["best"] = rep.best
        res["runs"] = [r.get("run") for r in rep.training if r.get("run")]
        return res

    def stop_checked() -> bool:
        return bool(should_stop and should_stop())

    # ---- 选项自检 ----
    notices = options.validate()
    errs = [x.message for x in notices if x.level == "error"]
    if errs:
        rep.start("ingest", "选项自检")
        rep.finish("ingest", errs[0], status="failed")
        return done("failed", "；".join(errs))
    for x in notices:
        if x.level != "error":
            rep.add_note(x.message)

    if not options.dataset_name:
        options.dataset_name = _auto_dataset_name()
    dataset = options.dataset_name
    rep.dataset = dataset

    # ---- 训练记录名预检：名字撞了就在**开工前**拦下 ----
    # 放到 ingest 之前是刻意的：否则用户要等切片 / 识别 / 特征提取全都跑完，
    # 才在训练器里发现「这个 run 名已存在」，白等十几分钟。
    clash = options.check_run_names()
    if clash:
        rep.start("ingest", "训练记录名预检")
        rep.finish("ingest", clash, status="failed")
        return done("failed", clash)
    rep.names = dict(options.plan_run_names())

    # ---- 输入收集 ----
    got = collect_inputs(paths, extra_path)
    files = got["files"]
    if got["missing"]:
        rep.add_note(f"以下路径不存在，已跳过：{got['missing'][:3]}"
                     + ("…" if len(got["missing"]) > 3 else ""))
    if got["skipped"]:
        rep.add_note(f"跳过非音频文件 {len(got['skipped'])} 个")
    if not files:
        rep.start("ingest", "输入检查")
        rep.finish("ingest", "没有找到任何可用音频", status="failed")
        return done("failed",
                    "没有找到任何可用音频：请上传音频文件，"
                    "或在「服务器路径」里填一个音频文件 / 目录的路径。")

    if progress:
        progress(bands.at("ingest", 0.0),
                 f"[一键] 收到 {len(files)} 个音频文件，开始处理")
    if DS.exists(dataset):
        # 不覆盖已有数据集：改名，避免把用户原先的数据搅进来
        dataset = dataset + "_" + time.strftime("%H%M%S")
        options.dataset_name = dataset
        rep.dataset = dataset
        rep.add_note(f"同名数据集已存在，本次改用 `{dataset}`")
    DS.create(dataset, note="一键三连")

    try:
        # ---- S1 采集 ----
        rep.start("ingest", f"{len(files)} 个输入文件")
        t = time.perf_counter()
        st1 = stage_ingest(dataset, files, options, progress, should_stop,
                          rep, bands)
        rep.finish("ingest",
                   f"导入 {st1['added']} 条 · 长音频 {st1['sliced']} 条切成 "
                   f"{st1['pieces']} 片", seconds=time.perf_counter() - t,
                   added=st1.get("added"), sliced=st1.get("sliced"),
                   pieces=st1.get("pieces"),
                   skipped=len(st1.get("skipped") or []),
                   failed=len(st1.get("failed") or []))
        if st1.get("stopped"):
            rep.mark("ingest", "skipped")
            return done("stopped", "用户在采集阶段停止")

        # ---- S2 优化 ----
        rep.start("optimize", "降噪 / 归一 / 掐静音")
        t = time.perf_counter()
        st2 = stage_optimize(dataset, options, progress, should_stop, rep, bands)
        rep.finish("optimize",
                   f"优化 {st2['enhanced']} 条 · 平均体检分 "
                   f"{st2['before_avg_score']} → {st2['after_avg_score']}",
                   seconds=time.perf_counter() - t,
                   enhanced=st2.get("enhanced"), failed=st2.get("failed"),
                   before_avg_score=st2.get("before_avg_score"),
                   after_avg_score=st2.get("after_avg_score"),
                   workers=st2.get("workers"),
                   denoise_skipped=bool(st2.get("denoise_note")))
        if st2.get("stopped"):
            rep.mark("optimize", "skipped")
            return done("stopped", "用户在优化阶段停止")
        if st2.get("denoise_note"):
            rep.add_note("音频优化：" + str(st2["denoise_note"]))
        if int(st2.get("workers") or 1) > 1:
            rep.add_note(f"CPU 并行生效：音频优化用 {st2['workers']} 线程，"
                         f"{st2.get('seconds')} 秒处理完 {st2.get('enhanced')} 条"
                         "（结果与串行逐字节一致，见 tools/oneclick_probe.py）")

        # ---- S3 识别 ----
        rep.start("asr", f"whisper {options.whisper_size} 逐条转写")
        t = time.perf_counter()
        st3 = stage_asr(dataset, options, progress, should_stop, rep, bands)
        rep.finish("asr",
                   f"转写 {st3['transcribed']} 条（无输出 {st3['empty']} · "
                   f"失败 {st3['failed']}）", seconds=time.perf_counter() - t,
                   whisper=st3.get("whisper"), prompt=st3.get("prompt"),
                   transcribed=st3.get("transcribed"),
                   empty=st3.get("empty"), failed=st3.get("failed"),
                   vram_before_gb=st3.get("vram_before_gb"),
                   vram_after_gb=st3.get("vram_after_gb"),
                   freed_gb=st3.get("freed_gb"),
                   loaded_after_unload=st3.get("loaded_after_unload"))
        if st3.get("stopped"):
            rep.mark("asr", "skipped")
            return done("stopped", "用户在识别阶段停止")

        # ---- S4 筛选 ----
        rep.start("curate", "体检复算 → 筛选 → 划分")
        t = time.perf_counter()
        st4 = stage_curate(dataset, options, progress, rep, bands)
        rep.data = {k: v for k, v in st4.items() if k != "ok"}
        if not st4.get("ok"):
            rep.finish("curate", st4.get("error", "筛选未通过"), status="failed",
                       seconds=time.perf_counter() - t)
            return done("failed", st4.get("error", "筛选未通过"))
        rep.finish("curate",
                   f"留用 {st4['ready_after']} 条 · 约 {st4['minutes']} 分钟 · "
                   f"train {st4['split']['train']} / val {st4['split']['val']}",
                   seconds=time.perf_counter() - t)

        # ---- 提取特征（要引擎加载；与训练互斥）----
        if progress:
            progress(bands.at("curate", 1.0),
                     "[特征] 加载引擎提取离线特征（训练时不再碰这些大模型）")
        set_req("loaded")
        if not getattr(engine, "loaded", False):
            engine.load()
        ex = FT.extract_dataset(dataset, overwrite=False, only="ready",
                                tts=engine.tts,
                                device=str(getattr(engine.tts, "device", "cuda")),
                                progress=_sub(progress, bands, "curate",
                                              "特征", base=0.0, span=1.0),
                                should_stop=should_stop)
        rep.add_note(f"特征提取：成功 {ex.get('extracted', 0)} · "
                     f"跳过 {ex.get('skipped', 0)} · 失败 {ex.get('failed', 0)}")
        DS.refresh_all(dataset, require_features=True)

        # ---- 特征提取用完了，立刻卸载引擎 ----
        # 关键顺序：调参必须看到**训练时真实的空闲显存**。引擎占着 4.9~5.7 GB
        # 的话，vram_headroom 只报出一两个 GB，本来完全跑得动的配置会被全部
        # 判成「显存不足」而拒掉 —— 真机验收就是在这里抓到整条流水线停住的。
        # 卸载自己也要先放开要求（req=="loaded" 时 unload 会被自己挡住）。
        set_req("none")
        if getattr(engine, "loaded", False):
            if progress:
                progress(bands.at("curate", 1.0),
                         "[调参] 卸载引擎，按训练时的真实显存评估候选配置")
            engine.unload()
        set_req("unloaded")     # 调参 + 训练期间禁止任何人加载引擎

        # ---- S5 调参 ----
        rep.start("tune", "按数据量 × 显存逐个预检候选")
        t = time.perf_counter()
        st5 = stage_tune(dataset, options, progress, rep, bands)
        rep.candidates = st5.get("candidates") or []
        if not st5.get("ok"):
            rep.finish("tune", st5.get("error", "调参未通过"), status="failed",
                       seconds=time.perf_counter() - t)
            return done("failed", st5.get("error", "调参未通过"))
        picks = "、".join(f"{a}→{v['preset']}" for a, v in st5["chosen"].items())
        rep.chosen = {a: {"preset": v["preset"], "rank": v["cfg"].rank,
                          "alpha": v["cfg"].alpha, "lr": v["cfg"].lr,
                          "epochs": v["cfg"].epochs,
                          "eval_every": v["cfg"].eval_every,
                          "keep_checkpoints": v["cfg"].keep_checkpoints,
                          "target_preset": v["cfg"].target_preset,
                          "total_steps": v.get("total_steps"),
                          "est_vram_gb": v.get("est_vram_gb")}
                      for a, v in st5["chosen"].items()}
        rep.finish("tune", f"选定 {picks}", seconds=time.perf_counter() - t)

        # ---- S6 训练 ----
        rep.start("train", "逐目标真训练（引擎已在调参前卸载）")
        t = time.perf_counter()
        st6 = stage_train(dataset, engine, st5, options, progress, should_stop,
                          tracker, rep, bands)
        rep.training = st6.get("runs") or []
        if not st6.get("ok"):
            rep.finish("train", st6.get("error", "训练失败"), status="failed",
                       seconds=time.perf_counter() - t)
            return done("failed", st6.get("error", "训练失败"))
        rep.finish("train",
                   f"{len([r for r in rep.training if r.get('ok')])} 个 run 完成",
                   seconds=time.perf_counter() - t)
        if st6.get("stopped"):
            return done("stopped", "用户在训练阶段停止")

        # ---- S7 择优 ----
        rep.start("rank", "逐档位真机合成 → reward 打分 → 激活最优")
        t = time.perf_counter()
        set_req("loaded")
        st7 = stage_rank(dataset, engine, st6, options, progress, should_stop,
                         rep, bands)
        rep.ranking = st7.get("ranking") or []
        rep.best = st7.get("best") or {}
        if not st7.get("ok"):
            rep.finish("rank", st7.get("error", "择优失败"), status="failed",
                       seconds=time.perf_counter() - t)
            return done("failed", st7.get("error", "择优失败"))
        rep.finish("rank",
                   f"{len(rep.ranking)} 个候选已排序，最优 "
                   f"{rep.best.get('run', '')}/{rep.best.get('checkpoint', '')}",
                   seconds=time.perf_counter() - t)

        return done("done")

    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=6)
        rep.add_note(tb)
        for k, _t, _w in STAGES:
            if rep.stage_of(k).get("status") == "running":
                rep.finish(k, f"{type(e).__name__}: {e}", status="failed")
        return done("failed", f"{type(e).__name__}: {e}")
