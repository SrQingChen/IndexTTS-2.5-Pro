"""CPU 并行工具 —— 只给「纯函数 + 逐条独立」的音频处理用。

为什么要单独写一个模块，而不是到处开线程池
==========================================

并行的前提是**不存在共享可变状态**。这里允许并行的只有音频体检与增强：

    · 每条音频读自己的文件、算自己的结果、写自己的文件，互不干扰；
    · 因此**线程数与结果无关** —— 输出与串行逐字节一致（探针里有对账）。

哪些**不许**并行（重要）
========================

    · **特征提取**：要借引擎里的 w2v-BERT / codec / campplus 做前向。
      引擎不是线程安全的，多线程同时进会拿到错的特征甚至崩掉。
    · **择优评测**：要挂 adapter 到引擎上合成。同一时刻引擎上只能有一个
      adapter（`attach_lora` 是替换语义），并行会把候选互相覆盖。
    · **训练**：单卡单进程，显存本来就只够一份。

所以流水线里 GPU 阶段一律留在主线程串行，本模块只服务 CPU 阶段。

为什么线程池而不是进程池
========================

这些活是 librosa / soundfile / scipy 的**解码、重采样与 FFT**，绝大部分时间在
释放 GIL 的 C 代码里；而且 pocketfft / scipy.signal 这些实现本身是**单线程**的，
所以不会出现「N 个 worker × 每个再开 M 个 BLAS 线程」的超订。

进程池反而更差：Windows 上要付 spawn 的代价，每个子进程还要重新 import
torch/librosa（几百 MB、数秒），收益远小于开销。

线程数怎么定
============

    · 硬上限 `MAX_WORKERS = 4` —— 训练与推理本身要用 CPU，抢太狠整体更慢；
    · 默认 `min(4, cpu_count // 4)` —— 给主线程、CUDA、whisper 留出余量；
    · 不超过待处理条数；
    · `workers <= 1` 时**退化为纯串行**，与旧行为完全一致（可用来关掉并行）。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# 硬上限：音频处理是辅助工序，不该把整机的 CPU 抢光。
MAX_WORKERS = 4


def default_workers(n_items: int = 0, requested: int = 0,
                    cap: int = MAX_WORKERS) -> int:
    """决定用几个 worker。

    requested > 0 时按用户的意愿来（仍受 cap 与条数限制）；
    requested <= 0 / 未给时按 cpu_count 自动推一个保守值。
    """
    try:
        want = int(requested or 0)
    except Exception:
        want = 0
    if want > 0:
        w = want
    else:
        w = min(cap, max(1, (os.cpu_count() or 4) // 4))
    w = min(w, max(1, cap))
    if n_items > 0:
        w = min(w, int(n_items))
    return max(1, w)


@dataclass
class ParallelOutcome:
    """并行结果。items 与输入**等长且保序**，失败位置是 None。"""

    items: List[Any] = field(default_factory=list)
    errors: Dict[int, str] = field(default_factory=dict)
    workers: int = 1
    seconds: float = 0.0
    stopped: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def n_done(self) -> int:
        return sum(1 for x in self.items if x is not None)


def map_parallel(fn: Callable[[Any], Any], items: Sequence[Any],
                 workers: int = 1,
                 should_stop: Optional[Callable[[], bool]] = None,
                 on_done: Optional[Callable[[int, Any], None]] = None,
                 ) -> ParallelOutcome:
    """把 fn 逐条作用在 items 上，最多 workers 个并发。

    · 顺序：返回值与输入**逐位对应**，与 workers 无关。
    · 异常：单条失败只记进 errors[i]，不影响其它条目（与串行时的行为一致）。
    · 停止：should_stop() 为真时不再派发**新**任务，已在跑的那几个跑完为止；
      未处理的条目在 items 里是 None，stopped=True。
    · on_done(i, result)：每条完成时回调（用于进度）。若在并行路径上调用，
      它由收集结果的主线程调用 —— 回调里可以安全地碰共享状态与 Gradio。
    """
    seq = list(items)
    n = len(seq)
    out: List[Any] = [None] * n
    errs: Dict[int, str] = {}
    w = max(1, int(workers or 1))
    t0 = time.perf_counter()

    def _call(i: int) -> Any:
        try:
            return fn(seq[i])
        except Exception as e:                      # 单条失败不拖累整批
            errs[i] = f"{type(e).__name__}: {e}"
            return None

    # ---- 串行路径：workers<=1 或只有一条，行为与旧版逐位一致 ----
    if w <= 1 or n <= 1:
        stopped = False
        for i in range(n):
            if should_stop is not None and should_stop():
                stopped = True
                break
            out[i] = _call(i)
            if on_done is not None:
                on_done(i, out[i])
        return ParallelOutcome(out, errs, 1,
                               round(time.perf_counter() - t0, 3), stopped)

    # ---- 并行路径：滚动窗口，避免一次性把上千条都丢进队列 ----
    stopped = False
    nxt = 0
    with ThreadPoolExecutor(max_workers=w,
                            thread_name_prefix="ixcpu") as ex:
        pending: Dict[Any, int] = {}

        def _fill() -> None:
            nonlocal nxt
            while nxt < n and len(pending) < w:
                pending[ex.submit(_call, nxt)] = nxt
                nxt += 1

        _fill()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                i = pending.pop(fut)
                try:
                    out[i] = fut.result()
                except Exception as e:              # 理论上 _call 已兜住
                    errs[i] = f"{type(e).__name__}: {e}"
                    out[i] = None
                if on_done is not None:
                    on_done(i, out[i])
            if not stopped:
                if should_stop is not None and should_stop():
                    stopped = True                  # 不再派发新任务
                else:
                    _fill()

    return ParallelOutcome(out, errs, w,
                           round(time.perf_counter() - t0, 3), stopped)
