"""L2 · DPO 偏好对齐：让模型在「同样的话」里更喜欢说得好的那一次。

SFT 教的是「照着本人的音频学」，它只有一个方向：更像训练数据。
但训练数据里也有说得好和说得差的时刻，而且「像」不等于「好」——
读错字、吞字、韵律崩坏的样本 SFT 照学不误。DPO 补上第二个方向：

  1. **构造偏好对**（`build_pairs`）：同一段文本、同一个参考音色，
     用当前模型合成多个候选（不同种子/温度），`reward.py` 打分 ——
     分高的当 chosen、分低的当 rejected。差距不够大的对**丢弃**：
     转写噪声就能造成的差距，学它等于学噪声。
  2. **DPO 训练**（`DpoTrainer`）：拉大 policy 对 chosen / rejected 的
     **相对**对数似然差（相对参考策略 = 纯底座）。

**为什么只在 GPT(T2S) 上做**：DPO 需要序列的**对数似然**，GPT 的语义
token 序列有，CFM 的连续 mel 输出没有（flow matching 是回归，不是密度）。
「读错字/吞字」恰好也都是 GPT 侧的毛病 —— 目标和方法是对得上的。

**参考策略不另开一份模型**：LoRA 的 B 矩阵零贡献时，`enable_adapters(False)`
下的前向**逐位等于**纯底座。所以 ref logprob 就是「同一棵树、关掉旁路」
再跑一遍前向 —— 不多占 1.6 GB，也没有两份权重不同步的风险。
这个等价性在 dpo_probe 里有钉：step-0 的 DPO loss 必须恰好等于 ln 2。

前向复用 `forward.py::gpt_training_forward`（lang_embedding / mask /
reentrant 检查点那几个陷阱不用再趟一遍），流水线复用
`trainer_base.BaseTrainer`（登记 / 保险库 / 早停 / 续训 / 释放）。
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from webui_app.config import PROJECT_ROOT
from webui_app.training import dataset as DS
from webui_app.training import features as FT
from webui_app.training import forward as FW
from webui_app.training import gpt_lora as GL
from webui_app.training import guard as GD
from webui_app.training import reward as RW
from webui_app.training import runs as RN
from webui_app.training import trainer_base as TB

ARCH = "dpo"

TrainReport = TB.TrainReport

PAIRS_FILE = "pairs.jsonl"
PAIR_SPLIT_FILE = "pair_split.json"

LN2 = float(math.log(2.0))       # step-0 的 DPO loss（logits 全 0 时）


# ===========================================================================
# 1. 偏好对的存储
# ===========================================================================

@dataclass
class PairRow:
    """一条偏好对。chosen / rejected 是数据集里两个**已有特征**的样本 id。

    margin 是构造时两个候选的 reward 差 —— 训练时不用它，但报告里要看：
    一批对的 margin 都贴着 min_margin，说明候选之间没什么真差别，
    这批对学不到东西（学的是转写噪声）。
    """
    id: str
    text: str
    chosen: str
    rejected: str
    prompt_id: str = ""
    margin: float = 0.0
    reward_chosen: float = 0.0
    reward_rejected: float = 0.0
    source: str = "manual"        # synth（构造器生成）| manual（手工标注）
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PairRow":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def pairs_path(ds_name: str) -> str:
    return os.path.join(DS.dir_of(ds_name), PAIRS_FILE)


def load_pairs(ds_name: str) -> List[PairRow]:
    p = pairs_path(ds_name)
    if not os.path.isfile(p):
        return []
    out: List[PairRow] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(PairRow.from_dict(json.loads(line)))
            except Exception:
                continue                      # 坏行跳过：一批对不该被一行毁掉
    return out


def save_pairs(ds_name: str, rows: Sequence[PairRow]) -> int:
    """整表重写（原子）。构造器的覆盖模式用这个。"""
    p = pairs_path(ds_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return len(rows)


def append_pairs(ds_name: str, rows: Sequence[PairRow]) -> int:
    have = load_pairs(ds_name)
    ids = {r.id for r in have}
    rows = [r for r in rows if r.id not in ids]
    return save_pairs(ds_name, have + list(rows))


def split_pairs(ds_name: str, val_ratio: float = 0.1, seed: int = 42
                ) -> Dict[str, List[str]]:
    """把**对**切成 train/val（不是把样本切 —— 一对的两个样本必须同侧，
    否则 val 里会出现「训练时见过 chosen 的另一半」的泄漏）。

    写在 pair_split.json 而不是 split.json：后者是 SFT 的样本级划分，
    两套语义混在一个文件里迟早互相覆盖。
    """
    import random

    rows = load_pairs(ds_name)
    ids = [r.id for r in rows]
    rnd = random.Random(int(seed))
    rnd.shuffle(ids)
    n_val = int(round(len(ids) * float(val_ratio)))
    val = ids[:n_val]
    sp = {"train": ids[n_val:], "val": val, "seed": int(seed),
          "val_ratio": float(val_ratio)}
    p = os.path.join(DS.dir_of(ds_name), PAIR_SPLIT_FILE)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=1)
    return sp


def load_pair_split(ds_name: str) -> Dict[str, List[str]]:
    p = os.path.join(DS.dir_of(ds_name), PAIR_SPLIT_FILE)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ===========================================================================
# 2. 序列对数似然（DPO 的原料）
# ===========================================================================

def mel_seq_logprob(gf: FW.GptForward, normalize: bool = False):
    """从一次 GptForward 里取每条样本的 mel 序列对数似然 (B,)。

    · 只算 mel 头：DPO 的偏好来自「这句话说得好不好」，mel token 序列
      就是这句话的韵律实现；text 头是下一文本 token，与好坏无关。
    · `normalize=True` 时按有效 token 数取平均 —— 句子长短差很多时，
      sum 会让长句的差距天然更大（它有更多位置可以拉开），
      DPO 会被「偏好长句」这个假信号带偏。
    """
    import torch
    import torch.nn.functional as F

    logp = F.log_softmax(gf.mel_logits.float(), dim=1)          # (B, V, L)
    tgt = gf.mel_targets.unsqueeze(1)                            # (B, 1, L)
    tok = torch.gather(logp, 1, tgt).squeeze(1)                  # (B, L)
    mask = gf.mel_mask.to(tok.dtype)
    tok = tok * mask
    if normalize:
        return tok.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return tok.sum(dim=1)


@contextlib.contextmanager
def adapters_off(pm):
    """临时关掉全部 LoRA 层 —— 这就是「参考策略」（逐位等于纯底座）。"""
    GD.disable_adapters(pm, True)
    try:
        yield
    finally:
        GD.disable_adapters(pm, False)


# ===========================================================================
# 3. 训练配置
# ===========================================================================

@dataclass
class DpoTrainOptions:
    """DPO 训练专属的旋钮（通用的那些在 `guard.LoRAConfig` 里）。"""
    beta: float = 0.1              # DPO 温度：越大越贴近 SFT，越小越敢拉差距
    sft_weight: float = 0.1        # chosen 的 NLL 锚：防止似然整体崩塌
    length_normalize: bool = True  # 按句长归一（见 mel_seq_logprob 的说明）
    max_codes: int = 900
    max_text_tokens: int = 500
    sort_by_length: bool = True
    base_dropout: float = 0.0
    min_lr_ratio: float = 0.1
    log_every: int = 10
    save_optimizer_state: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DpoTrainOptions":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []

        def err(m): n.append(GD.Notice("error", m))
        def warn(m): n.append(GD.Notice("warn", m))
        def info(m): n.append(GD.Notice("info", m))

        if not 0.01 <= float(self.beta) <= 1.0:
            err(f"beta={self.beta} 超出 0.01~1.0")
        if not 0.0 <= float(self.sft_weight) <= 1.0:
            err(f"sft_weight={self.sft_weight} 超出 0~1")
        if not 16 <= int(self.max_codes) <= 1816:
            err(f"max_codes={self.max_codes} 超出 16~1816")
        if not 4 <= int(self.max_text_tokens) <= 600:
            err(f"max_text_tokens={self.max_text_tokens} 超出 4~600")
        if 3 + (self.max_text_tokens + 2) + (self.max_codes + 2) > 2420:
            err("max_codes + max_text_tokens 超过 n_positions=2420")
        if not 0.0 <= float(self.base_dropout) <= 0.5:
            err(f"base_dropout={self.base_dropout} 超出 0~0.5")
        if float(self.beta) < 0.05:
            warn(f"beta={self.beta} 很小：梯度会奖励「无限拉大差距」，"
                 "容易把似然推崩。除非明确要激进对齐，否则 ≥0.1")
        if float(self.sft_weight) == 0.0:
            info("sft_weight=0：纯 DPO。chosen 的绝对似然没有任何东西拉着，"
                 "可能出现「差距拉大但两句都说不出话」的崩塌，观察 acc 时"
                 "同时看 reward 评测台（b6）")
        return n


# ===========================================================================
# 4. 训练器
# ===========================================================================

class DpoTrainer(GL.GptTrainer):
    """偏好对训练器。模型侧（加载/注入/优化器/保险库）与 GPT SFT 完全
    共用，只有数据流和损失是自己的。

    一个 batch 的形状：B 个**对**（不是 B 个样本）。一次反向要跑
    4 组前向：policy×(chosen, rejected) 带梯度，ref×(chosen, rejected)
    在 no_grad + adapters_off 下。所以同 batch_size 下显存约为 SFT 的
    1.5 倍 —— preflight 的估算已按此放大。
    """

    ARCH = ARCH

    def __init__(self, dataset: str, cfg: Optional[GD.LoRAConfig] = None,
                 options: Optional[DpoTrainOptions] = None,
                 run_name: Optional[str] = None, device: Optional[str] = None,
                 model_dir: Optional[str] = None, val_ratio: float = 0.1,
                 resume_from: Optional[str] = None,
                 train_root: Optional[str] = None):
        super().__init__(dataset, cfg=cfg,
                         options=None, run_name=run_name, device=device,
                         model_dir=model_dir, val_ratio=val_ratio,
                         resume_from=resume_from, train_root=train_root)
        self.options = options or DpoTrainOptions()
        # 基类把它建成了 GptTrainOptions；换成我们自己的默认值
        # （否则 options_dict 会把 SFT 的字段写进 run.json）
        if options is None:
            self.options = DpoTrainOptions()
        self.pair_rows: List[PairRow] = []
        self.train_pairs: List[PairRow] = []
        self.val_pairs: List[PairRow] = []
        # 最近一次评估的 {acc, margin}（run.json 与报告用）
        self.eval_metrics: Dict[str, float] = {}
        self.last_train_metrics: Dict[str, float] = {}

    # ---------------- 数据流（覆盖 GPT SFT 的部分） ----------------
    def preflight(self, make_split_if_missing: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ok": False, "errors": [], "warnings": [], "infos": []}

        def _add(x: GD.Notice) -> None:
            out["errors" if x.level == "error" else
                ("warnings" if x.level == "warn" else "infos")].append(x.message)

        if not DS.exists(self.dataset):
            _add(GD.Notice("error", f"数据集 `{self.dataset}` 不存在"))
            return out

        rows = load_pairs(self.dataset)
        if not rows:
            _add(GD.Notice(
                "error", f"`{self.dataset}` 里没有偏好对（{PAIRS_FILE} 为空或不存在）。"
                "到「偏好对齐」页用当前模型构造，或手工导入。"))
            return out
        self.pair_rows = rows

        # ---- 划分：按「对」，一对的两半永远同侧 ----
        sp = load_pair_split(self.dataset)
        if not sp.get("train") and sp.get("train") != []:
            if not make_split_if_missing:
                _add(GD.Notice("error", "还没有偏好对划分，请先点「划分」"))
                return out
            sp = split_pairs(self.dataset, val_ratio=self.val_ratio,
                             seed=self.cfg.seed)
            _add(GD.Notice("info",
                           f"已自动划分：训练 {len(sp['train'])} 对 / "
                           f"验证 {len(sp['val'])} 对"))
        by_id = {r.id: r for r in rows}
        self.train_pairs = [by_id[i] for i in sp.get("train") or [] if i in by_id]
        self.val_pairs = [by_id[i] for i in sp.get("val") or [] if i in by_id]
        # 新增的对没进旧划分：追进 train（不能静默丢掉）
        seen = {r.id for r in self.train_pairs + self.val_pairs}
        extra = [r for r in rows if r.id not in seen]
        if extra:
            self.train_pairs += extra
            _add(GD.Notice("info",
                           f"{len(extra)} 对不在既有划分里，已全部计入训练集"
                           "（重新划分可让它们参与验证）"))

        # ---- 样本池：chosen + rejected 都要能取到 ----
        want_ids: List[str] = []
        for r in self.train_pairs + self.val_pairs:
            want_ids += [r.chosen, r.rejected]
        want_ids = list(dict.fromkeys(want_ids))       # 去重保序
        try:
            pool = GL.SamplePool(self.dataset, want_ids, kind="target")
        except Exception as e:
            _add(GD.Notice("error", f"读偏好对样本失败：{type(e).__name__}: {e}"))
            return out
        if pool.missing:
            _add(GD.Notice("warn",
                           f"{len(pool.missing)} 个偏好对样本缺特征，相关对会被跳过。"
                           "到「数据集」页重新提取即可用。"))
        ok_ids, _bad = pool.usable_ids(self.options.max_codes,
                                       self.options.max_text_tokens)
        pool.restrict(ok_ids)
        self.pools["target"] = pool
        # 只保留「两边都可用」的对
        def _alive(rows_in: List[PairRow]) -> List[PairRow]:
            return [r for r in rows_in
                    if r.chosen in pool.meta and r.rejected in pool.meta]
        n_before = len(self.train_pairs)
        self.train_pairs = _alive(self.train_pairs)
        self.val_pairs = _alive(self.val_pairs)
        dropped = n_before - len(self.train_pairs)
        if dropped:
            _add(GD.Notice("warn", f"{dropped} 对因为缺特征/超长被剔除"))
        if not self.train_pairs:
            _add(GD.Notice("error", "没有一条可用的训练偏好对"))
            return out
        if not self.val_pairs:
            _add(GD.Notice("warn",
                           "**验证偏好对为空**：早停与「相对底座的改善」无法计算"))
        self.train_ids = list(dict.fromkeys(
            [x for r in self.train_pairs for x in (r.chosen, r.rejected)]))
        self.val_ids = list(dict.fromkeys(
            [x for r in self.val_pairs for x in (r.chosen, r.rejected)]))
        self.val_pools = {"target": GL.SamplePool(self.dataset, self.val_ids,
                                                  kind="target")}

        # DPO 不吃回放：偏好对本身就是「当前模型的输出」，混入旧回放
        # 会让 ref/policy 的参照系失去意义。明确告知而不是静默忽略。
        if float(self.cfg.replay_ratio or 0.0) > 0:
            _add(GD.Notice("info",
                           "DPO 不使用回放（偏好对的参照系是当前模型），"
                           "replay_ratio 已忽略"))
            self.cfg.replay_ratio = 0.0
        self.replay_ids, self.replay_src = [], "none"

        # ---- 配置体检 ----
        all_notes = (self.cfg.validate(len(self.train_pairs))
                     + self.options.validate())
        self.report.merge_notes(all_notes)
        for x in all_notes:
            _add(x)

        # ---- 步数与显存 ----
        spe = max(1, math.ceil(len(self.train_pairs) / self.cfg.global_batch))
        epochs_est = max(1, int(self.cfg.epochs))
        self.total_steps = spe * epochs_est
        if self.cfg.max_steps > 0:
            self.total_steps = min(self.total_steps, int(self.cfg.max_steps))
        self.total_steps = max(1, int(self.total_steps))

        est_adapter = GL.EST_ADAPTER_PER_RANK.get(
            self.cfg.target_preset, GL.EST_ADAPTER_PER_RANK["attn"]) * int(self.cfg.rank)
        need = round(GL.estimate_vram_gb(GL.EST_BASE_PARAMS, est_adapter,
                                         self.cfg.bf16, self.cfg.batch_size,
                                         self.options.max_codes) * 1.5, 2)
        vr = GD.vram_headroom(need)

        out.update({
            "ok": not out["errors"],
            "n_pairs": len(rows),
            "n_train": len(self.train_pairs), "n_val": len(self.val_pairs),
            "n_replay_pool": 0, "replay_source": "none", "replay_dataset": "",
            "samples_per_epoch": len(self.train_pairs),
            "steps_per_epoch": spe, "total_steps": self.total_steps,
            "est_adapter_params": est_adapter, "est_vram_gb": need,
            "vram": vr, "device": self.device,
            "avg_margin": (sum(r.margin for r in rows) / len(rows)
                           if rows else 0.0),
        })
        self.report.data.update({
            "n_pairs": len(rows), "n_train": len(self.train_pairs),
            "n_val": len(self.val_pairs), "steps_per_epoch": spe,
            "total_steps": self.total_steps,
        })
        self._pf = out
        return out

    def _samples_per_epoch(self, n_train: int, n_replay_pool: int) -> int:
        return n_train        # DPO：一对就是一条样本，无回放

    def _epoch_batches(self, epoch: int) -> Tuple[List[List[Any]], Dict[str, Any]]:
        opt_seed = int(self.cfg.seed)
        order = [("target", i) for i in range(len(self.train_pairs))]
        # 长度按「一对里较长的那半」计：批内 padding 由 max(c, r) 决定
        def pair_len(item):
            _k, i = item
            if i >= len(self.train_pairs):
                return 0
            r = self.train_pairs[i]
            p = self.pools.get("target")
            return max(p.length_of(r.chosen), p.length_of(r.rejected))
        batches = TB.plan_batches(order, int(self.cfg.batch_size),
                                  bool(self.options.sort_by_length),
                                  seed=opt_seed + epoch, len_of=pair_len)
        info = {"source": "none", "dataset": "", "pool": 0, "ratio": 0.0,
                "n_replay": 0, "achieved": 0.0, "with_replacement": False,
                "note": "DPO：无回放（偏好对的参照系是当前模型）"}
        return batches, info

    # ---------------- DPO 前向 ----------------
    def _dpo_pass(self, samples: Sequence[Dict[str, Any]], refs: bool):
        """一组样本 → 每条的 mel 序列 logprob (B,)。refs=True 时关掉 adapter。"""
        import torch
        b = self._to_device(GL.collate(samples))
        if refs:
            with adapters_off(self.model), torch.no_grad():
                f = FW.gpt_training_forward(
                    self.model, b["style"], b["emo_vec"], b["langs"],
                    b["text_tokens"], b["text_lengths"],
                    b["codes"], b["code_lengths"])
                return mel_seq_logprob(f, bool(self.options.length_normalize))
        f = FW.gpt_training_forward(
            self.model, b["style"], b["emo_vec"], b["langs"],
            b["text_tokens"], b["text_lengths"], b["codes"], b["code_lengths"])
        return f, b

    def _pair_tensors(self, pair_items: Sequence[Tuple[str, int]],
                      pools: Dict[str, GL.SamplePool], grad: bool):
        """取 B 对样本的四个 logprob 张量与 policy 前向（供锚损失复用）。"""
        import torch
        pool = pools.get("target")
        chosen_s, rejected_s = [], []
        for _k, i in pair_items:
            if i >= len(self.train_pairs):
                continue
            r = self.train_pairs[i]
            c = pool.load(r.chosen) if pool else None
            j = pool.load(r.rejected) if pool else None
            if c is not None and j is not None:
                chosen_s.append(c)
                rejected_s.append(j)
        if not chosen_s:
            return None

        pol_c_f, b_c = self._dpo_pass(chosen_s, refs=False)
        pol_c = mel_seq_logprob(pol_c_f, bool(self.options.length_normalize))
        pol_r_f, _b_r = self._dpo_pass(rejected_s, refs=False)
        pol_r = mel_seq_logprob(pol_r_f, bool(self.options.length_normalize))
        ref_c = self._dpo_pass(chosen_s, refs=True)
        ref_r = self._dpo_pass(rejected_s, refs=True)
        return {"pol_c": pol_c, "pol_r": pol_r, "ref_c": ref_c, "ref_r": ref_r,
                "f_c": pol_c_f, "b_c": b_c, "n": len(chosen_s)}

    def _dpo_loss(self, t: Dict[str, Any]):
        """DPO 损失 + 指标。输入是 _pair_tensors 的输出。"""
        import torch
        import torch.nn.functional as F

        beta = float(self.options.beta)
        logits = beta * ((t["pol_c"] - t["ref_c"]) - (t["pol_r"] - t["ref_r"]))
        loss = -F.logsigmoid(logits).mean()
        anchor = torch.zeros((), device=logits.device)
        w = float(self.options.sft_weight or 0.0)
        if w > 0:
            # chosen 的 NLL 锚（按 token 平均，与 SFT 同量纲）：
            # DPO 只关心相对差，chosen 的绝对似然可以塌到没人说得出话；
            # 锚把「说得好」拉回「至少说得出」。
            n_tok = float(t["f_c"].mel_mask.sum())
            anchor = -(t["pol_c"] / max(1.0, n_tok)) if t["f_c"].mel_mask.numel() \
                else torch.zeros_like(t["pol_c"])
            anchor = anchor.mean()
            loss = loss + w * anchor
        return loss, {
            "acc": float((logits > 0).float().mean()),
            "margin": float(logits.mean()),
            "dpo": float(-F.logsigmoid(logits).mean()),
            "anchor": float(anchor),
        }

    def _train_micro_batch(self, items: Sequence[Tuple[str, int]]
                           ) -> Optional[Dict[str, float]]:
        import torch
        if self.model is None:
            return None
        # dropout_off 包住整批：policy 与 ref 必须看到同一个 dropout 世界，
        # 否则 (pol − ref) 里混进两次不同采样的随机差 —— margins 抖成噪声。
        with GL.dropout_off(self.model) if hasattr(GL, "dropout_off") \
                else _noop_ctx():
            pass
        # （GL 没有 dropout_off —— 它在 CFM 那边；这里自己包）
        t = None
        with _dropout_off_ctx(self.model):
            t = self._pair_tensors(items, self.pools, grad=True)
            if t is None:
                return None
            loss, m = self._dpo_loss(t)
            loss.backward()
        self._accum += 1
        self.last_train_metrics = m
        return {"loss": float(loss), "n": t["n"], **m}

    def evaluate(self, max_batches: int = 0) -> Optional[float]:
        """val 上的 DPO loss（float，供早停），acc/margin 记进 eval_metrics。"""
        import torch
        if not self.val_pairs or self.model is None:
            return None
        num = den = 0.0
        accs, margins = [], []
        order = [("target", i) for i in range(len(self.val_pairs))]
        batches = TB.plan_batches(order, int(self.cfg.batch_size), False,
                                  seed=int(self.cfg.seed))
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.no_grad(), _dropout_off_ctx(self.model):
                for k, items in enumerate(batches):
                    if max_batches and k >= int(max_batches):
                        break
                    t = self._pair_tensors_val(items)
                    if t is None:
                        continue
                    loss, m = self._dpo_loss(t)
                    num += float(loss)
                    den += 1
                    accs.append(m["acc"])
                    margins.append(m["margin"])
        finally:
            self.model.train(was_training)
        if den <= 0:
            return None
        self.eval_metrics = {"acc": sum(accs) / len(accs),
                             "margin": sum(margins) / len(margins)}
        self.report.data.update({
            "eval_acc": round(self.eval_metrics["acc"], 4),
            "eval_margin": round(self.eval_metrics["margin"], 4)})
        return num / den

    def _pair_tensors_val(self, items):
        """evaluate 用的 _pair_tensors：对来自 val_pairs，样本取自 val 池。"""
        import torch
        pool = self.val_pools.get("target")
        if pool is None:
            return None
        chosen_s, rejected_s = [], []
        for _k, i in items:
            if i >= len(self.val_pairs):
                continue
            r = self.val_pairs[i]
            c = pool.load(r.chosen)
            j = pool.load(r.rejected)
            if c is not None and j is not None:
                chosen_s.append(c)
                rejected_s.append(j)
        if not chosen_s:
            return None
        pol_c_f, b_c = self._dpo_pass(chosen_s, refs=False)
        pol_c = mel_seq_logprob(pol_c_f, bool(self.options.length_normalize))
        pol_r_f, _ = self._dpo_pass(rejected_s, refs=False)
        pol_r = mel_seq_logprob(pol_r_f, bool(self.options.length_normalize))
        ref_c = self._dpo_pass(chosen_s, refs=True)
        ref_r = self._dpo_pass(rejected_s, refs=True)
        return {"pol_c": pol_c, "pol_r": pol_r, "ref_c": ref_c, "ref_r": ref_r,
                "f_c": pol_c_f, "b_c": b_c, "n": len(chosen_s)}

    def _train_snapshot(self, max_batches: int = 4) -> Optional[float]:
        if not self.train_pairs or self.model is None:
            return None
        order = [("target", i) for i in range(len(self.train_pairs))]
        batches = TB.plan_batches(order, int(self.cfg.batch_size), False,
                                  seed=int(self.cfg.seed))[:max(1, max_batches)]
        import torch
        num = den = 0.0
        with torch.no_grad(), _dropout_off_ctx(self.model):
            for items in batches:
                t = self._pair_tensors(items, self.pools, grad=False)
                if t is None:
                    continue
                loss, _m = self._dpo_loss(t)
                num += float(loss)
                den += 1
        return (num / den) if den > 0 else None

    # ---------------- run.json 的附加字段 ----------------
    def _extra_run_fields(self, pf: Dict[str, Any],
                          extra: Dict[str, Any]) -> Dict[str, Any]:
        d = super()._extra_run_fields(pf, extra)
        d.update({
            "n_pairs": int(pf.get("n_pairs") or 0),
            "beta": float(self.options.beta),
            "sft_weight": float(self.options.sft_weight),
            "length_normalize": bool(self.options.length_normalize),
            "avg_pair_margin": round(float(pf.get("avg_margin") or 0.0), 4),
            # eval_acc / eval_margin 在 evaluate 时写进 report.data，
            # 收尾会随 update_run(data=...) 落进 run.json —— 写在这里只会是
            # prepare 时刻的 None。
        })
        return d


@contextlib.contextmanager
def _noop_ctx():
    yield


@contextlib.contextmanager
def _dropout_off_ctx(model):
    """训练/评估 DPO 时关掉全部 Dropout（与 cfm_lora.dropout_off 同一件事，
    放这里是为了不跨模块引 CFM 的东西）。"""
    import torch.nn as nn
    saved = []
    for m in model.modules():
        if isinstance(m, nn.Dropout) and float(m.p) != 0.0:
            saved.append((m, float(m.p)))
            m.p = 0.0
    try:
        yield len(saved)
    finally:
        for m, p in saved:
            m.p = p


# ===========================================================================
# 5. 偏好对构造器（合成 + 打分 + 入库）
# ===========================================================================

@dataclass
class PairBuildOptions:
    """`build_pairs` 的旋钮。"""
    out_dataset: str = ""           # 偏好对写入哪个数据集（不存在会创建）
    n_candidates: int = 2           # 每条文本合成几个候选；2 = 最优与最差
    min_margin: float = 0.05        # reward 差低于它的对不可信，丢弃
    seed: int = 42
    temperature: float = 0.9        # 候选的采样温度（多样性来源）
    temperature_jitter: float = 0.15    # 第 2+ 个候选在此基础上扰动
    top_p: float = 0.9
    top_k: int = 50
    max_pairs: int = 0              # 0 = 不限
    keep_audio: bool = False        # 把候选 wav 留在数据集里（试听/复核）
    whisper_size: str = RW.DEFAULT_WHISPER
    language: str = "zh"
    wer_weight: float = 0.6
    sim_weight: float = 0.4
    overwrite: bool = False         # True = 清掉旧的偏好对重写

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[GD.Notice]:
        n: List[GD.Notice] = []

        def err(m): n.append(GD.Notice("error", m))
        def warn(m): n.append(GD.Notice("warn", m))

        if not self.out_dataset or not DS.safe_dataset_name(self.out_dataset):
            err("out_dataset 不能为空")
        if not 2 <= int(self.n_candidates) <= 8:
            err(f"n_candidates={self.n_candidates} 超出 2~8")
        if not 0.0 <= float(self.min_margin) <= 1.0:
            err(f"min_margin={self.min_margin} 超出 0~1")
        if float(self.min_margin) < 0.02:
            warn("min_margin 很小：转写噪声就能造成的差距也会被当成偏好，"
                 "这批对学到的多半是噪声")
        if not 0.0 < float(self.temperature) <= 2.0:
            err(f"temperature={self.temperature} 超出 (0, 2]")
        if float(self.temperature) < 0.5:
            warn("候选温度太低：两次合成几乎一样，margin 全被 min_margin 刷掉")
        return n


def build_pairs(engine, dataset: str, opts: PairBuildOptions,
                progress: Optional[Callable[[float, str], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                infer_fn: Optional[Callable[..., str]] = None,
                scorer: Optional[RW.RewardScorer] = None,
                extractor: Optional[FT.FeatureExtractor] = None
                ) -> Dict[str, Any]:
    """用当前模型构造偏好对。

    参数
    ----
    engine      : TTSEngine（已加载、可选已挂 adapter —— 挂着就蒸当前策略，
                  正是 DPO 要对齐的对象）
    dataset     : 源数据集：每条 ready 样本提供 (文本, 参考音频)
    infer_fn / scorer / extractor : 可注入 —— 探针用假实现测编排，
                  真实路径在验收（b9）里端到端跑。

    流程（每条样本）：
        合成 k 个候选（不同种子 + 温度扰动）→ reward 打分 →
        最优/最差 margin ≥ min_margin 才成对 → 导入 out_dataset 并提特征 →
        追加 PairRow。
    """
    out: Dict[str, Any] = {"ok": False, "pairs": 0, "kept": 0, "dropped": 0,
                           "synthesized": 0, "warnings": [], "errors": [],
                           "rows": []}
    notes = opts.validate()
    out["errors"] += [x.message for x in notes if x.level == "error"]
    out["warnings"] += [x.message for x in notes if x.level == "warn"]
    if out["errors"]:
        return out
    if engine is None or getattr(engine, "tts", None) is None:
        out["errors"] = ["引擎未提供或未加载。请先在「系统」页加载模型。"]
        return out
    if not DS.exists(dataset):
        out["errors"] = [f"源数据集 `{dataset}` 不存在"]
        return out

    items = [u for u in DS.load_meta(dataset) if u.status == "ready"
             and (u.text or "").strip()]
    if not items:
        out["errors"] = [f"`{dataset}` 里没有带文本的 ready 样本"]
        return out
    if opts.max_pairs > 0:
        items = items[:int(opts.max_pairs)]

    if not DS.exists(opts.out_dataset):
        DS.create(opts.out_dataset, note=f"DPO 偏好对（源：{dataset}）")
    if opts.overwrite:
        save_pairs(opts.out_dataset, [])
    DS.set_info(opts.out_dataset, dpo_source=dataset,
                dpo_options=opts.to_dict())

    infer = infer_fn or engine.infer
    sc = scorer or RW.RewardScorer(RW.RewardOptions(
        whisper_size=opts.whisper_size, language=opts.language,
        wer_weight=opts.wer_weight, sim_weight=opts.sim_weight))
    ext = extractor or FT.FeatureExtractor(tts=getattr(engine, "tts", None))

    tmp = tempfile.mkdtemp(prefix="dpo_pairs_")
    pairs: List[PairRow] = []
    t0 = time.perf_counter()
    try:
        for i, u in enumerate(items):
            if should_stop and should_stop():
                out["warnings"].append("已手动停止：偏好对保存到已完成为止的部分")
                break
            if progress:
                progress(i / max(1, len(items)),
                         f"偏好对 {i+1}/{len(items)}：{u.text[:18]}…")
            prompt_wav = os.path.join(DS.dir_of(dataset), str(u.audio or ""))
            if not os.path.isfile(prompt_wav):
                out["dropped"] += 1
                continue

            import torch as _torch
            cands = []
            for k in range(int(opts.n_candidates)):
                dst = os.path.join(tmp, f"{u.id}_c{k}.wav")
                try:
                    _torch.manual_seed(int(opts.seed) + i * 97 + k * 13)
                    temp = float(opts.temperature) + (
                        float(opts.temperature_jitter) * k
                        if k > 0 else 0.0)
                    got = infer(spk_audio_prompt=prompt_wav, text=u.text,
                                output_path=dst, verbose=False,
                                lang=(u.lang or "ZH").upper(),
                                do_sample=True,
                                top_p=float(opts.top_p),
                                top_k=(int(opts.top_k) or None),
                                temperature=temp, max_mel_tokens=900)
                    out["synthesized"] += 1
                    if got and os.path.isfile(dst):
                        cands.append(dst)
                except Exception as e:
                    out["warnings"].append(
                        f"{u.id} 候选 {k} 合成失败：{type(e).__name__}: {e}")
            if len(cands) < 2:
                out["dropped"] += 1
                continue

            scored = []
            for cw in cands:
                r = sc.score(cw, u.text, prompt_wav, lang=u.lang)
                if r.get("ok"):
                    scored.append((cw, r))
            if len(scored) < 2:
                out["dropped"] += 1
                continue
            scored.sort(key=lambda x: -float(x[1]["reward"]))
            best_w, best_r = scored[0]
            worst_w, worst_r = scored[-1]
            margin = float(best_r["reward"]) - float(worst_r["reward"])
            if margin < float(opts.min_margin):
                out["dropped"] += 1
                continue

            # 导入两侧 + 提特征
            import torch
            try:
                keep = [best_w, worst_w] + (
                    scored[1:-1] if opts.keep_audio else [])
                imp = DS.import_audio(opts.out_dataset, keep, copy=True,
                                      lang=u.lang)
                if len(imp.get("ids") or []) < 2:
                    out["dropped"] += 1
                    continue
                touched = {}
                out_dir = DS.dir_of(opts.out_dataset)
                # import_audio(copy=True) 会把文件重命名为 <uid>.<ext>
                for uu, (_w, rr) in zip(imp["ids"][:2],
                                        [scored[0], scored[-1]]):
                    wav_p = os.path.join(out_dir, "audio", f"{uu}.wav")
                    try:
                        feat = ext.extract(wav_p, u.text, lang=u.lang or "ZH")
                        p = FT.FeatureExtractor.feature_path(out_dir, uu)
                        torch.save(feat, p)
                        touched[uu] = {
                            "text": u.text, "has_features": True,
                            "features_at": time.time(),
                            "n_codes": int(feat.get("n_codes") or 0),
                            "n_text_tokens": int(feat.get("n_text_tokens") or 0),
                            "mel_len": int(feat.get("mel_len") or 0),
                            "lang_token": int(feat.get("lang_token") or 1),
                        }
                    except Exception as e:
                        out["warnings"].append(
                            f"{uu} 特征提取失败：{type(e).__name__}: {e}")
                if len(touched) < 2:
                    out["dropped"] += 1
                    continue
                FT._apply_meta(opts.out_dataset, touched)
                ids = list(touched.keys())
                row = PairRow(
                    id=f"{u.id}_p{i:03d}",
                    text=u.text, chosen=ids[0], rejected=ids[1],
                    prompt_id=u.id, margin=round(margin, 4),
                    reward_chosen=round(float(best_r["reward"]), 4),
                    reward_rejected=round(float(worst_r["reward"]), 4),
                    source="synth", created_at=time.time())
                pairs.append(row)
                out["kept"] += 1
                out["rows"].append(row.to_dict())
            except Exception as e:
                out["warnings"].append(
                    f"{u.id} 入库失败：{type(e).__name__}: {e}")
                out["dropped"] += 1
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)
        sc.unload()

    if pairs:
        append_pairs(opts.out_dataset, pairs)
        # 新数据集自动划分（对级）
        if not load_pair_split(opts.out_dataset).get("train"):
            split_pairs(opts.out_dataset, val_ratio=0.1, seed=opts.seed)
    DS.refresh_all(opts.out_dataset, require_features=True)
    out["ok"] = not out["errors"]
    out["pairs"] = len(pairs)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


# ===========================================================================
# 6. 报告
# ===========================================================================

def pairs_markdown(ds_name: str, limit: int = 30) -> str:
    rows = load_pairs(ds_name)
    if not rows:
        return f"`{ds_name}` 里还没有偏好对。"
    L = [f"### `{ds_name}` 的偏好对（{len(rows)} 对）", "",
         "| id | margin↑ | reward(选/弃) | 文本 |", "|---|---|---|---|"]
    for r in rows[-int(limit):]:
        L.append(f"| `{r.id}` | {r.margin:.3f} "
                 f"| {r.reward_chosen:.3f} / {r.reward_rejected:.3f} "
                 f"| {r.text[:24]}… |")
    if len(rows) > limit:
        L.append("", f"_只显示最近 {limit} 对（共 {len(rows)}）_")
    margins = [r.margin for r in rows]
    L += ["", f"margin 均值 {sum(margins)/len(margins):.3f} · "
          f"最小 {min(margins):.3f} · 最大 {max(margins):.3f}"]
    return "\n".join(L)
