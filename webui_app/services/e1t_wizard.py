"""e1t_wizard — 「🎵 歌声训练（E1-T）」向导页的服务层。

向导页把 E1-T 的六步训练链（引导复制 → GTSinger 导入 → singing 特征 →
capability 训练 → 合并 → 角色身份 LoRA）接到 UI；本模块提供：

1. `status()`：七步状态检测（文件/数据集/训练 run），渲染成流程总览；
2. 每一步的后台任务 fn 构造器（交给 training.runner 的 TrainRunner 执行，
   与一键三连同一套互斥/进度/停止机制——训练类 require_engine="unloaded"，
   特征提取用自带模型实例同样要求引擎卸载，导入/合并走 CPU）。

**run 分类依据**（run.json 由 CfmTrainer 写入的 singing 字段）：
- capability run：`singing.enabled == true`（公开歌声、条件透传）；
- 歌声身份 run：`singing.enabled == false` 且 `singing.cfg_path` 指向
  config_singing.yaml（在歌声底座上练的角色 LoRA）；
- 其余 = 普通 TTS LoRA（与 E1-T 无关，翻唱页不开歌声开关时用）。
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Dict, List, Optional

from webui_app.config import PROJECT_ROOT
from webui_app.services import gts_fetch as GF
from webui_app.services import gts_import as GI
from webui_app.services import svc_state
from webui_app.training import dataset as DS
from webui_app.training import features as FT
from webui_app.training import merge as MG
from webui_app.training import runs as RN

CHECKPOINTS = os.path.join(PROJECT_ROOT, "checkpoints")
SINGING_CFG = os.path.join(CHECKPOINTS, "config_singing.yaml")
SINGING_PTH = os.path.join(CHECKPOINTS, "s2mel_singing.pth")
BASE_PTH = os.path.join(CHECKPOINTS, "s2mel.pth")


# ---------------------------------------------------------------------------
# 1) 状态检测
# ---------------------------------------------------------------------------

def _run_singing_info(name: str) -> Dict[str, Any]:
    try:
        rj = RN.read_run(name) or {}
    except Exception:
        return {}
    return rj.get("singing") or {}


def capability_runs() -> List[str]:
    """singing.enabled=True 的 CFM run（capability 候选，供合并）。"""
    out = []
    try:
        for r in RN.list_runs(arch="cfm"):
            name = getattr(r, "name", "")
            if not name or not getattr(r, "has_adapter", False):
                continue
            if _run_singing_info(name).get("enabled"):
                out.append(name)
    except Exception:
        pass
    return sorted(out)


def singing_identity_runs() -> List[str]:
    """在歌声底座上练的角色身份 run（cfg_path 指 config_singing 且未开条件）。"""
    out = []
    try:
        for r in RN.list_runs(arch="cfm"):
            name = getattr(r, "name", "")
            if not name or not getattr(r, "has_adapter", False):
                continue
            sg = _run_singing_info(name)
            cfg = str(sg.get("cfg_path") or "")
            if not sg.get("enabled") and "config_singing" in cfg:
                out.append(name)
    except Exception:
        pass
    return sorted(out)


def singing_datasets() -> List[str]:
    """带歌声标签的数据集（导入器会写 technique.json——比逐条读 .pt 便宜）。"""
    out = []
    for name in DS.list_datasets():
        if os.path.isfile(os.path.join(DS.dir_of(name), "technique.json")):
            out.append(name)
    return sorted(out)


def status() -> Dict[str, Any]:
    """七步状态 + 数据源状态。纯文件系统读取，渲染线程可以直接调。"""
    st: Dict[str, Any] = {}
    st["source"] = GF.source_state()
    st["base_pth"] = os.path.isfile(BASE_PTH)
    st["singing_cfg"] = os.path.isfile(SINGING_CFG)
    st["singing_pth"] = os.path.isfile(SINGING_PTH)
    st["singing_pth_bytes"] = (os.path.getsize(SINGING_PTH)
                               if st["singing_pth"] else 0)
    sds = singing_datasets()
    st["singing_datasets"] = sds
    st["singing_features"] = {}
    for d in sds:
        try:
            st["singing_features"][d] = FT.stats(d).get("ready_with_features", 0)
        except Exception:
            st["singing_features"][d] = 0
    st["capability_runs"] = capability_runs()
    st["identity_runs"] = singing_identity_runs()
    # 合并产物是否有内容：比 s2mel.pth 大 = 已并入 LoRA+conditioner（引导
    # 副本初始与 s2mel.pth 等大）；只作提示，不挡操作
    if st["singing_pth"] and st["base_pth"]:
        st["singing_pth_merged_hint"] = (
            st["singing_pth_bytes"] > os.path.getsize(BASE_PTH) + 1024)
    return st


def status_markdown() -> str:
    """流程总览面板（七步 ✅/⬜ + 下一步提示 + 可复用性说明）。"""
    st = status()
    steps = []

    def _mark(ok):
        return "✅" if ok else "⬜"

    src_st = st.get("source") or {}
    if src_st.get("exists"):
        _src_desc = (f"已就绪：{','.join(src_st['langs'])} · "
                     f"{src_st['wavs']} 个 wav"
                     if src_st.get("wavs") else "目录存在但还没有 wav（先拉取）")
    else:
        _src_desc = "未拉取（向导②可镜像下载，也可自行放置）"
    steps.append((_mark(bool(src_st.get("wavs"))), "⓪ 歌声数据源",
                  f"datasets/_sources/GTSinger —— {_src_desc}"))
    cfg_ok = st["singing_cfg"]
    steps.append((_mark(cfg_ok), "前置文件",
                  "checkpoints/config_singing.yaml（随项目分发，只读）"))
    steps.append((_mark(st["singing_pth"]), "① 歌声底座就绪",
                  "checkpoints/s2mel_singing.pth"
                  + ("（已含训练成果）" if st.get("singing_pth_merged_hint")
                     else ("（当前=原底座引导副本）" if st["singing_pth"] else "（未创建）"))))
    n_gts = len(st["singing_datasets"])
    steps.append((_mark(n_gts > 0), "② 公开歌声数据集",
                  f"{n_gts} 个（一次导入全程复用）" if n_gts else "尚未导入 GTSinger"))
    feat_n = sum(st["singing_features"].values())
    steps.append((_mark(feat_n > 0), "③ 歌声特征（F0/技巧）",
                  f"{feat_n} 条已提取" if feat_n else "未提取"))
    n_cap = len(st["capability_runs"])
    steps.append((_mark(n_cap > 0), "④ capability 训练",
                  f"{n_cap} 个 run" if n_cap else "未训练"))
    steps.append((_mark(bool(st.get("singing_pth_merged_hint"))), "⑤ 合并到歌声底座",
                  "s2mel_singing.pth 已含 capability"
                  if st.get("singing_pth_merged_hint") else "待合并"))
    n_id = len(st["identity_runs"])
    steps.append((_mark(n_id > 0), "⑥ 角色歌声身份 LoRA",
                  f"{n_id} 个（{', '.join(st['identity_runs'][:3])}）"
                  if n_id else "未训练"))

    rows = "\n".join(f"| {ic} | **{t}** | {d} |" for ic, t, d in steps)
    nxt = "按顺序完成上面第一个 ⬜ 的步骤即可。"
    if not st["singing_pth"]:
        nxt = "从 **① 引导复制** 开始（10 秒）。"
    elif n_gts == 0:
        nxt = "下一步：**② 导入 GTSinger**（下载好后填路径）。"
    elif feat_n == 0:
        nxt = "下一步：**③ 提取歌声特征**（会自动加载模型，需空闲显存）。"
    elif n_cap == 0:
        nxt = "下一步：**④ capability 训练**（建议先小子集冒烟）。"
    elif not st.get("singing_pth_merged_hint"):
        nxt = "下一步：**⑤ 合并**到 checkpoints/s2mel_singing.pth。"
    elif n_id == 0:
        nxt = "下一步：**⑥ 为角色练歌声身份 LoRA**（现有说话数据集直接复用）。"
    else:
        nxt = "全部就绪 → 翻唱页：LoRA 选身份 run + 勾选「歌声模型（E1-T 实验性）」。"

    return (f"| 状态 | 步骤 | 说明 |\n|---|---|---|\n{rows}\n\n"
            f"> 👉 **{nxt}**\n>\n> **可复用性**：②③ 的数据一次准备、所有角色与后续重训共享；"
            "④⑤ 只在提升歌声能力时重跑；**每个新角色只跑 ⑥**（10–60s 说话素材即可，"
            "现有 TTS 数据集的特征直接复用，不必重新提取）。")


# ---------------------------------------------------------------------------
# 2) 各步后台任务 fn（runner.submit(fn) 形态：fn(progress, should_stop)）
# ---------------------------------------------------------------------------

def bootstrap_fn() -> Callable:
    """① 引导复制：s2mel.pth → s2mel_singing.pth（一次性，绝不反向覆盖）。"""
    def fn(progress, should_stop):
        progress(0.2, "校验文件…")
        if not os.path.isfile(BASE_PTH):
            return {"ok": False, "error": f"找不到 {BASE_PTH}（底座未部署）"}
        if os.path.isfile(SINGING_PTH):
            return {"ok": False,
                    "error": "s2mel_singing.pth 已存在——不覆盖既有产物；"
                             "要重置请先手动删除它（确认里面没有未备份的合并成果）"}
        progress(0.6, "复制底座（约 0.4GB）…")
        shutil.copy2(BASE_PTH, SINGING_PTH)
        progress(1.0, "完成")
        return {"ok": True, "out": SINGING_PTH,
                "note": "引导副本完成。首次 capability 训练/合并都会以它为起点，"
                        "合并后会写入训练成果（原 s2mel.pth 全程不动）。"}
    return fn


def import_fn(src: str, dataset: str, langs: str, pitch_aug: str,
              max_items: int, copy: bool) -> Callable:
    """② GTSinger 导入（CPU/IO 型）。"""
    def fn(progress, should_stop):
        from webui_app.services import gts_import as GI
        lang_list = [x for x in (langs or "").replace("，", ",").split(",") if x]
        return GI.run_import(src.strip(), dataset.strip(), langs=lang_list,
                             pitch_aug=pitch_aug, max_items=int(max_items),
                             copy=bool(copy), progress=progress,
                             should_stop=should_stop)
    return fn


def extract_fn(dataset: str) -> Callable:
    """③ singing 特征提取。

    引擎状态：用「引擎已卸载」要求——FeatureExtractor 自建模型实例
    （约 3GB 显存），与常驻引擎叠加会 OOM；runner 的互斥保证提取期间
    引擎不会被加载（与训练类任务同一档约束）。
    """
    def fn(progress, should_stop):
        r = FT.extract_dataset(dataset.strip(), singing=True,
                               progress=progress, should_stop=should_stop)
        if r.get("stopped"):
            return {"ok": False, "error": "用户停止", **r}
        return {"ok": bool(r.get("ok")), "result": {
            "extracted": r.get("extracted"), "skipped": r.get("skipped"),
            "failed": r.get("failed"),
            "warnings": (r.get("warnings") or [])[:6]}}
    return fn


def _train(dataset: str, run_name: str, preset: str, frames: int, epochs: int,
           singing: bool, cfg_path: Optional[str], progress, should_stop
           ) -> Dict[str, Any]:
    """④/⑥ 共用的训练执行体（capability 与身份只差三个参数）。"""
    from webui_app.training import cfm_lora as CL

    if not cfg_path or not os.path.isfile(cfg_path):
        return {"ok": False, "error": f"找不到 {cfg_path}（config_singing.yaml）"}
    if not os.path.isfile(SINGING_PTH):
        return {"ok": False,
                "error": "未找到 checkpoints/s2mel_singing.pth —— 请先完成第①步引导复制"}

    cfg = CL.default_config(preset)
    if epochs and epochs > 0:
        cfg.epochs = int(epochs)
    opts = CL.CfmTrainOptions(
        singing=bool(singing),
        singing_tech_balance=bool(singing),   # 身份训练无 tech 标签，开关无副作用
        max_total_frames=int(frames) if frames else 1800)
    tr = CL.CfmTrainer(dataset=dataset.strip(), cfg=cfg, options=opts,
                       run_name=(run_name or "").strip() or None,
                       cfg_path=cfg_path)
    progress(0.02, "预检（不加载模型）…")
    pf = tr.preflight()
    if not pf.get("ok"):
        return {"ok": False, "error": "预检未通过",
                "preflight": {"errors": pf.get("errors"),
                              "warnings": (pf.get("warnings") or [])[:6]}}
    rep = tr.run(progress=progress, should_stop=should_stop)
    d = rep.to_dict() if hasattr(rep, "to_dict") else {}
    return {"ok": True, "run": tr.run_name,
            "best_val": d.get("best_val"), "epochs": d.get("epochs"),
            "adapter": RN.adapter_dir(tr.run_name) if tr.run_name else ""}


def capability_train_fn(dataset: str, run_name: str, preset: str,
                        frames: int, epochs: int) -> Callable:
    """④ capability 训练：公开歌声 + 条件透传（singing=True）。"""
    def fn(progress, should_stop):
        return _train(dataset, run_name, preset, frames, epochs,
                      singing=True, cfg_path=SINGING_CFG,
                      progress=progress, should_stop=should_stop)
    return fn


def identity_train_fn(dataset: str, run_name: str, preset: str,
                      frames: int, epochs: int) -> Callable:
    """⑥ 角色歌声身份：singing 底座 + 角色说话数据（条件不参与，
    conditioner 自动冻结——PEFT 只放行 LoRA 参数）。"""
    def fn(progress, should_stop):
        return _train(dataset, run_name, preset, frames, epochs,
                      singing=False, cfg_path=SINGING_CFG,
                      progress=progress, should_stop=should_stop)
    return fn


def merge_fn(run: str) -> Callable:
    """⑤ 合并 capability → checkpoints/s2mel_singing.pth（CPU）。"""
    def fn(progress, should_stop):
        from webui_app.training import merge as MG
        from webui_app.training import guard as GD
        if not (run or "").strip():
            return {"ok": False, "error": "请选择 capability run"}
        d = RN.adapter_dir(run.strip())
        if not os.path.isdir(d):
            return {"ok": False, "error": f"adapter 目录不存在：{d}"}
        progress(0.15, "加载歌声底座 + 合并 ΔW（CPU）…")
        rep = MG.merge_lora_to_checkpoint(MG.MergeOptions(
            adapter_dir=d, arch="cfm_singing", out_path=SINGING_PTH))
        progress(1.0, "完成" if rep.ok else "失败")
        return {"ok": bool(rep.ok), "report": rep.to_dict(),
                "markdown": rep.markdown()}
    return fn

def fetch_fn(langs: List[str], dest: str = "") -> Callable:
    """⓪/② 镜像拉取 GTSinger（CPU/网络型，后台执行）。"""
    def fn(progress, should_stop):
        return GF.fetch(dest or GF.DEFAULT_DEST, list(langs or []),
                        progress=progress, should_stop=should_stop)
    return fn


def _seg(progress, lo: float, hi: float) -> Callable[[float, str], None]:
    """把子阶段进度 [0,1] 映射进全链 [lo,hi]。"""
    def cb(f, msg):
        progress(lo + (hi - lo) * max(0.0, min(1.0, float(f))), msg)
    return cb


def wire_cover_state(identity_run: str) -> Dict[str, Any]:
    """把成果「接线」到翻唱页：写参数记忆（singing 开 + 身份 LoRA）。

    只写 E1-T 相关键，不动用户其它翻唱参数；不写 identity 时也开
    singing（纯参考音色 + 歌声底座是合法用法）。svc_state 的白名单
    已含这些键（阶段47b 修复过静默丢键——那类 bug 的教训）。
    """
    vals: Dict[str, Any] = {"singing_enable": True,
                            "technique_mode": "preserve",
                            "register_mode": "auto",
                            "octave_fix": True,
                            "bon_high": False,
                            "lora_run": identity_run or "",
                            "lora_ckpt": "best"}
    svc_state.save_cover(vals)
    return {"wired": True,
            "lora_run": identity_run or "（纯参考音色）",
            "note": "翻唱页下次进入即载入：歌声模型已开"
                    + (f"，LoRA={identity_run}" if identity_run else "")}


def pipeline_fn(src: str, dataset: str, langs: List[str], pitch_aug: str,
                max_items: int, cap_run: str, cap_preset: str,
                cap_frames: int, cap_epochs: int,
                identity: bool, id_ds: str, id_run: str, id_preset: str,
                id_epochs: int, auto_fetch: bool = True) -> Callable:
    """🚀 一键全链：拉取(可选)→导入→特征→capability→合并→身份(可选)→接线。

    每段失败立即返回（已完成段的成果保留在磁盘/run 里——状态总览
    天然增量，修好问题再点一次会跳过已完成的部分继续走）。
    """
    def fn(progress, should_stop):
        stages: List[str] = []

        def _stop() -> bool:
            return bool(should_stop and should_stop())

        # ---- ⓪ 拉取（数据源缺且允许时）----
        src_st = GF.source_state()
        if not src_st.get("wavs"):
            if not auto_fetch:
                return {"ok": False, "stage": "fetch",
                        "error": "数据源为空且未开启自动拉取；先在②下载"
                                 "或填自备路径重跑导入"}
            progress(0.01, "[⓪] 镜像拉取 GTSinger…")
            r = GF.fetch(GF.DEFAULT_DEST, list(langs or []),
                         progress=_seg(progress, 0.01, 0.12),
                         should_stop=should_stop)
            if not r.get("ok"):
                return {"ok": False, "stage": "fetch", "error": r.get("error", "")}
            stages.append("fetch")

        # ---- ① 前置：底座文件 ----
        if not os.path.isfile(SINGING_PTH):
            progress(0.13, "[①] 引导复制歌声底座…")
            if not os.path.isfile(BASE_PTH):
                return {"ok": False, "stage": "bootstrap",
                        "error": f"找不到 {BASE_PTH}"}
            shutil.copy2(BASE_PTH, SINGING_PTH)
            stages.append("bootstrap")

        # ---- ② 导入 ----
        if _stop():
            return {"ok": False, "stage": "import", "error": "用户停止"}
        use_src = (src or "").strip() or GF.DEFAULT_DEST
        progress(0.15, f"[②] 导入 GTSinger → {dataset}（{use_src}）…")
        r = GI.run_import(use_src, dataset,
                          langs=[l for l in (langs or [])],
                          pitch_aug=pitch_aug, max_items=int(max_items or 0),
                          progress=_seg(progress, 0.15, 0.22),
                          should_stop=should_stop)
        if not r.get("ok"):
            return {"ok": False, "stage": "import", "error": r.get("error", "")}
        stages.append(f"import(+{r.get('imported', 0)}/+{r.get('augmented', 0)})")

        # ---- ③ 特征 ----
        if _stop():
            return {"ok": False, "stage": "extract", "error": "用户停止"}
        progress(0.23, f"[③] singing 特征提取 · {dataset}…")
        r = FT.extract_dataset(dataset, singing=True,
                               progress=_seg(progress, 0.23, 0.38),
                               should_stop=should_stop)
        if r.get("stopped") or not r.get("ok"):
            errs = (r.get("errors") or [])[:3]
            return {"ok": False, "stage": "extract",
                    "error": f"特征提取失败 {r.get('failed', 0)} 条"
                             + (f"：{errs}" if errs else "")}
        stages.append(f"extract(+{r.get('extracted', 0)})")

        # ---- ④ capability 训练 ----
        if _stop():
            return {"ok": False, "stage": "capability", "error": "用户停止"}
        progress(0.39, f"[④] capability 训练 · {cap_run}…")
        r = _train(dataset, cap_run, cap_preset, int(cap_frames),
                   int(cap_epochs or 0), singing=True, cfg_path=SINGING_CFG,
                   progress=_seg(progress, 0.39, 0.80), should_stop=should_stop)
        if not r.get("ok"):
            return {"ok": False, "stage": "capability",
                    "error": r.get("error", ""),
                    "preflight": r.get("preflight")}
        stages.append(f"capability({r.get('run')})")

        # ---- ⑤ 合并 ----
        if _stop():
            return {"ok": False, "stage": "merge", "error": "用户停止"}
        progress(0.81, "[⑤] 合并到歌声底座（CPU）…")
        rep = MG.merge_lora_to_checkpoint(MG.MergeOptions(
            adapter_dir=RN.adapter_dir(r["run"]), arch="cfm_singing",
            out_path=SINGING_PTH))
        if not rep.ok:
            return {"ok": False, "stage": "merge",
                    "error": "；".join(rep.notes[:3])}
        stages.append("merge")

        # ---- ⑥ 身份（可选）----
        id_result = None
        if identity and (id_ds or "").strip():
            if _stop():
                return {"ok": False, "stage": "identity", "error": "用户停止"}
            progress(0.83, f"[⑥] 身份训练 · {id_run}…")
            id_result = _train(id_ds, id_run, id_preset, 0,
                               int(id_epochs or 0), singing=False,
                               cfg_path=SINGING_CFG,
                               progress=_seg(progress, 0.83, 0.98),
                               should_stop=should_stop)
            if not id_result.get("ok"):
                return {"ok": False, "stage": "identity",
                        "error": id_result.get("error", "")}
            stages.append(f"identity({id_result.get('run')})")

        # ---- ⑦ 接线翻唱页 ----
        progress(0.99, "[⑦] 接线翻唱页…")
        wire = wire_cover_state(id_result["run"] if id_result else "")
        progress(1.0, "全链完成")
        return {"ok": True, "stages": stages, "wire": wire,
                "capability_run": r["run"], "merged_to": SINGING_PTH}
    return fn
