"""runner.py —— 训练相关任务的后台执行器。

UI 只调这一个入口：训练（GPT/CFM/DPO）、偏好对构造、评测、合并
全部经 `TrainRunner` 提交到后台线程，UI 用 Timer 轮询 `snapshot()`。

为什么不用 Gradio 自带的 generator/queue：
    · 训练一跑几十分钟，用户要能**切 Tab、关浏览器再回来**——
      状态必须活在本进程的 runner 里，而不是活在某次请求的闭包里；
    · 取消要能穿透 `run(should_stop=...)`，这条线在 BaseTrainer 里
      已经铺好，runner 只需要持有一个标志位；
    · 同一时间只允许一个训练类任务：8GB 卡上两个训练并跑必然
      静默溢出（WDDM 不报 OOM，只降速 20 倍），必须在门口拦。
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from webui_app.training import runs as RN

__all__ = ["TrainRunner", "get_runner"]

JOB_LABELS = {
    "train": "训练", "pairs": "偏好对构造", "eval": "A/B 评测",
    "merge": "合并", "distill": "底座蒸馏", "extract": "特征提取",
}


class TrainRunner:
    """单飞行器（one in flight）：同一时刻最多一个任务。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        # ---- 快照字段（UI 轮询读，无需持锁） ----
        self.kind: str = ""
        self.label: str = ""
        self.running: bool = False
        self.progress: float = 0.0
        self.message: str = ""
        self.started_at: float = 0.0
        self.finished_at: float = 0.0
        self.ok: Optional[bool] = None
        self.error: str = ""
        self.result: Dict[str, Any] = {}
        self.phase: str = ""            # preflight / prepare / run / finalize…
        self._log_lines: List[str] = []
        self._log_run: str = ""
        # 当前任务对引擎的要求（engine 在 load/unload 门口反查它，
        # 见 TTSEngine._busy_runner_req）。任务结束后保留，没人再读。
        self._engine_req: str = "none"

    @property
    def engine_req(self) -> str:
        """running 时的任务对引擎的要求：loaded / unloaded / none。"""
        return self._engine_req

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------
    def submit(self, kind: str, label: str,
               fn: Callable[[Callable[[float, str], None],
                             Callable[[], bool]], Any],
               require_engine: str = "none",
               engine=None,
               holds_engine: bool = False) -> Dict[str, Any]:
        """启动一个后台任务。

        fn(progress_cb, should_stop) -> Any（任意可 JSON 化的结果摘要）。
        require_engine:
            "loaded"  任务需要引擎已加载（合成类：pairs / eval / distill）
            "unloaded" 任务需要引擎已卸载（训练类：显存不够两者共存）
            "none"     无所谓（merge 走 CPU）
        holds_engine: 任务自己会（按需）加载并**全程借用**引擎
            （特征提取就是这种）：等效 require_engine="loaded"，
            但提交时不要求引擎已经加载。占用期间 engine.unload() 会被拒，
            防止「状态条说已卸载、GPU 上却留着第二份模型」。
        """
        with self._lock:
            if self.running:
                return {"ok": False,
                        "message": f"已有任务在跑（{self.label}），等它结束或先停止。"}
            self.kind, self.label = kind, label
            self.running = True
            self.progress = 0.0
            self.message = "启动中…"
            self.started_at = time.time()
            self.finished_at = 0.0
            self.ok = None
            self.error = ""
            self.result = {}
            self.phase = ""
            self._log_lines = []
            self._log_run = ""
            self._engine_req = "loaded" if holds_engine else require_engine
            self._stop_flag.clear()

        # 引擎前置条件在**提交线程**里检查（同步返回给 UI，不用轮询）
        if require_engine == "loaded":
            if engine is None or not getattr(engine, "loaded", False):
                self._finish(False, error="引擎未加载。请先到「系统」页加载模型。")
                return {"ok": False, "message": "引擎未加载。请先到「系统」页加载模型。"}
        elif require_engine == "unloaded":
            if engine is not None and getattr(engine, "loaded", False):
                self._finish(False, error="引擎还占着显存。训练前必须先卸载"
                                          "（8GB 卡放不下两份模型），"
                                          "请到「系统」页卸载后再来。")
                return {"ok": False, "message": "引擎还占着显存，请先卸载（系统页）。"}

        def _progress(frac: float, msg: str):
            self.progress = float(max(0.0, min(1.0, frac)))
            self.message = str(msg)
            self._log_lines.append(f"[{frac:5.0%}] {msg}")
            del self._log_lines[:-400]          # 环形：只留最近 400 行

        def _should_stop() -> bool:
            return self._stop_flag.is_set()

        def _work():
            try:
                res = fn(_progress, _should_stop)
                self.result = res if isinstance(res, dict) else {"value": res}
                # dict 结果里 ok=False 才算失败；没给 ok 的当成功
                self._finish((res.get("ok") is not False)
                             if isinstance(res, dict) else True)
            except Exception as e:
                self._log_lines.append(traceback.format_exc(limit=8))
                self._finish(False, error=f"{type(e).__name__}: {e}")

        self._thread = threading.Thread(target=_work, daemon=True,
                                        name=f"train-{kind}")
        self._thread.start()
        return {"ok": True, "message": f"{JOB_LABELS.get(kind, kind)}已在后台启动"}

    def _finish(self, ok: bool, error: str = "") -> None:
        self.running = False
        self.finished_at = time.time()
        self.ok = ok
        self.error = error or ""
        self.message = "完成" if ok else (error or "失败")

    # ------------------------------------------------------------------
    # 控制
    # ------------------------------------------------------------------
    def cancel(self) -> Dict[str, Any]:
        if not self.running:
            return {"ok": False, "message": "当前没有在跑的任务。"}
        self._stop_flag.set()
        self.message = "已请求停止，等待当前 step 结束…"
        return {"ok": True, "message": "已发出停止信号（当前批跑完即停）"}

    @property
    def stop_requested(self) -> bool:
        return self._stop_flag.is_set()

    # ------------------------------------------------------------------
    # 快照（UI 轮询）
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "kind": self.kind,
            "label": self.label,
            "progress": self.progress,
            "message": self.message,
            "phase": self.phase,
            "seconds": round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at else 0.0,
            "ok": self.ok,
            "error": self.error,
            "result": self.result,
            "stop_requested": self.stop_requested,
        }

    def log_text(self, tail: int = 60) -> str:
        """进度日志 + （训练任务）训练器日志文件的尾部。"""
        lines = list(self._log_lines[-int(tail):])
        if self._log_run:
            try:
                lines.append("")
                lines.append("—— 训练器日志 ——")
                lines.extend(RN.read_log(self._log_run, tail=tail).splitlines())
            except Exception:
                pass
        return "\n".join(lines) if lines else "_还没有日志_"

    def track_run(self, run_name: str) -> None:
        """训练任务把自己的 run 名字登记进来（log_text 会带上它的日志）。"""
        self._log_run = str(run_name)


_RUNNER: Optional[TrainRunner] = None
_RUNNER_LOCK = threading.Lock()


def get_runner() -> TrainRunner:
    """进程级单例。"""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = TrainRunner()
        return _RUNNER


# ===========================================================================
# 任务构造器：把「UI 参数 → 后端调用」的胶水收敛在这里，Tab 只管画界面
# ===========================================================================

def make_train_job(arch: str, dataset: str, cfg_dict: Dict[str, Any],
                   options_dict: Dict[str, Any], run_name: str = "",
                   resume_from: str = "", val_ratio: float = 0.05,
                   tracker: Optional["TrainRunner"] = None):
    """构造一个训练任务 fn(progress, should_stop)。arch: gpt|cfm|dpo。

    tracker 给了的话，prepare 之后把实际 run 名登记进去 ——
    UI 的日志面板就能同时看到 runner 进度与训练器自己的日志。
    """
    from webui_app.training import cfm_lora as CL
    from webui_app.training import dpo as DP
    from webui_app.training import gpt_lora as GL
    from webui_app.training import guard as GD

    def fn(progress, should_stop):
        cfg = GD.LoRAConfig.from_dict(cfg_dict)
        if arch == "gpt":
            opts = GL.GptTrainOptions.from_dict(options_dict)
            tr = GL.GptTrainer(dataset, cfg=cfg, options=opts,
                               run_name=run_name or None, val_ratio=val_ratio,
                               resume_from=resume_from or None)
        elif arch == "cfm":
            opts = CL.CfmTrainOptions.from_dict(options_dict)
            tr = CL.CfmTrainer(dataset, cfg=cfg, options=opts,
                               run_name=run_name or None, val_ratio=val_ratio,
                               resume_from=resume_from or None)
        elif arch == "dpo":
            opts = DP.DpoTrainOptions.from_dict(options_dict)
            tr = DP.DpoTrainer(dataset, cfg=cfg, options=opts,
                               run_name=run_name or None, val_ratio=val_ratio,
                               resume_from=resume_from or None)
        else:
            return {"ok": False, "error": f"未知训练目标 {arch}"}

        pf = tr.preflight()
        if not pf.get("ok"):
            return {"ok": False, "error": "预检未通过：\n· "
                    + "\n· ".join(pf.get("errors") or []),
                    "preflight": pf}
        progress(0.02, f"预检通过：{pf.get('n_train', pf.get('n_pairs', 0))} 条样本"
                       f" · 预估显存 {pf.get('est_vram_gb')} GB")
        try:
            tr.prepare(progress=progress)
        except Exception as e:
            return {"ok": False, "error": f"准备阶段失败：{e}"}
        if tracker is not None:
            tracker.track_run(tr.run_name)
        rep = tr.run(progress=progress, should_stop=should_stop)
        return {"ok": bool(rep.ok), "run": tr.run_name,
                "steps": rep.steps, "best_val": rep.best_val,
                "first_val": rep.first_val,
                "improved": rep.improved,
                "stopped_early": rep.stopped_early,
                "stop_reason": rep.stop_reason,
                "vram_peak_gb": rep.vram_peak_gb,
                "markdown": rep.markdown()}

    return fn
