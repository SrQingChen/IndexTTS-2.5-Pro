"""奖励打分（reward.py）的回归验证。

两层：
  [1]~[3] 纯逻辑（CPU、永远跑）：文本归一化 / CER / 权重校验 / 渲染；
  [4]~[6] 真模型（本机已缓存 whisper small + campplus 本地权重）：
        转写一致性、声纹自相似=1、score() 端到端。

whisper 首次运行需要联网下载（462 MB，缓存在 checkpoints/hf_cache/whisper）。
下载不了时 [4]~[6] 自动跳过并明确说明，不算失败。

跑法：  .venv\\Scripts\\python.exe tools\\reward_probe.py
"""

import _env  # noqa: F401  路径与 Windows 控制台编码引导，必须在最前

import os
import time

import torch

from webui_app.training import guard as GD                    # noqa: E402
from webui_app.training.reward import (RewardOptions,         # noqa: E402
                                       RewardScorer, asr_language,
                                       cer, normalize_text)

PASS = FAIL = 0
FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def head(t: str):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


EX_DIR = os.path.join(_env.PROJECT_ROOT, "examples")
V3 = os.path.join(EX_DIR, "voice_03.wav")
V1 = os.path.join(EX_DIR, "voice_01.wav")


def whisper_ready() -> bool:
    """whisper small 已缓存（或 30s 内能下下来）才算可用。"""
    cache = os.path.join(_env.PROJECT_ROOT, "checkpoints", "hf_cache", "whisper")
    if any(f.startswith("small") for f in os.listdir(cache) if os.path.isdir(cache)):
        return True
    try:
        import whisper
        whisper.load_model("small", device="cpu", download_root=cache)
        return True
    except Exception as e:
        print(f"  [skip] whisper small 不可用：{type(e).__name__}: {str(e)[:80]}")
        return False


def main() -> int:
    torch.manual_seed(0)

    # =====================================================================
    head("[1] 文本归一化与 CER（纯逻辑）")
    check("标点与空白被去掉",
          normalize_text("你好，世界！ Hello, world.") == "你好世界helloworld",
          normalize_text("你好，世界！ Hello, world."))
    check("全角数字与字母转半角",
          normalize_text("２０２４年ＡＢ") == "2024年ab")
    check("中文引号书名号被去掉",
          normalize_text("「你好」《世界》") == "你好世界")
    check("空串归一化成空串", normalize_text("") == "")
    check("英文统一小写（whisper 输出与目标文本可比）",
          normalize_text("ABC") == normalize_text("abc"))

    check("相同文本 CER=0", cer("今天天气不错", "今天天气不错。") == 0.0)
    check("全错 CER=1", cer("一二三", "四五六") == 1.0)
    check("一字之差 CER=1/4", abs(cer("一二三四", "一二三五") - 0.25) < 1e-9)
    check("多读的字计入错误（分母是 ref）",
          abs(cer("一二三四", "一二三四五六") - 0.5) < 1e-9)
    check("标点差异不影响 CER", cer("你好，世界", "你好世界") == 0.0)
    check("空参考 + 非空转写 = 全错", cer("", "读出了东西") == 1.0)
    check("空参考 + 空转写 = 0", cer("", "") == 0.0)
    check("CER 截到 1（转写比参考长一倍也不会超过 1）",
          cer("一", "一二三四五六七八九十") == 1.0)

    check("语言映射 ZH→zh", asr_language("ZH") == "zh")
    check("语言映射 zhen→zh（数据集里的混拼写法）", asr_language("zhen") == "zh")
    check("语言映射 EN→en", asr_language("EN") == "en")
    check("未知语言回落 zh", asr_language("XX") == "zh")

    # =====================================================================
    head("[2] RewardOptions.validate")
    o = RewardOptions()
    check("默认配置（small / 0.6+0.4）没有 error",
          not [x for x in o.validate() if x.level == "error"])
    errs = lambda vv: [x.message for x in vv.validate() if x.level == "error"]
    warns = lambda vv: [x.message for x in vv.validate() if x.level == "warn"]
    infos = lambda vv: [x.message for x in vv.validate() if x.level == "info"]
    check("whisper 档位非法被拦",
          any("whisper_size" in m for m in errs(RewardOptions(whisper_size="giant"))))
    check("large-v3 给出 8GB 显存 warn（WDDM 静默溢出）",
          any("GB" in m for m in warns(RewardOptions(whisper_size="large-v3"))))
    check("负权重被拦",
          any("权重" in m for m in errs(RewardOptions(wer_weight=-1))))
    check("权重和≠1 给 info（相对比较不受影响）",
          any("权重和" in m for m in infos(RewardOptions(wer_weight=0.5,
                                                          sim_weight=0.2))))
    check("beam_size 越界被拦",
          any("beam" in m for m in errs(RewardOptions(beam_size=99))))
    check("权重归一化：0.5+0.5 合法",
          not errs(RewardOptions(wer_weight=0.5, sim_weight=0.5)))

    # =====================================================================
    head("[3] score 的组合与渲染（不碰模型）")
    sc = RewardScorer(RewardOptions(whisper_size="tiny"))
    # 直接构造 ScoreResult 验证渲染，避免这里触发模型加载
    rows = [
        {"ok": True, "path": "a.wav", "wer": 0.1, "sim": 0.8, "reward": 0.86,
         "asr_text": "今天天气不错"},
        {"ok": True, "path": "b.wav", "wer": 0.3, "sim": 0.6, "reward": 0.68,
         "asr_text": "今天天汽不错"},
        {"ok": False, "path": "c.wav", "error": "FileNotFoundError: x"},
    ]
    md = sc.scores_markdown(rows, "探针")
    check("表格渲染包含两行成功与一行失败",
          "a.wav" in md and "b.wav" in md and "c.wav" in md and "🔴" in md)
    check("均值行按 ok 条目计算",
          "WER 0.200" in md and "reward 0.770" in md, md.splitlines()[-1])
    check("空列表渲染占位符", sc.scores_markdown([]) == "_没有可打分的条目。_")
    check("score 对不存在的音频返回 ok=False 而不是抛异常",
          not sc.score("Z:/不存在.wav", "文本", V1)["ok"])
    check("score_pair 带出具体的错误", "score_pair" is not None and
          not sc.score_pair(V1, V1, "文本", "Z:/不存在.wav")["ok"])

    # reward 公式：w1·(1-WER)+w2·SS，权重和归一
    w1, w2 = 0.6, 0.4
    r_manual = (w1 * (1 - 0.25) + w2 * 0.5) / (w1 + w2)
    check("组合公式手算对得上（0.6·0.75+0.4·0.5）",
          abs(r_manual - 0.65) < 1e-9, f"{r_manual}")

    # =====================================================================
    head("[4] campplus 声纹（本地权重，无网络）")
    try:
        t0 = time.perf_counter()
        e3 = sc.embed(V3)
        dt = time.perf_counter() - t0
        check(f"独立加载 + 首条声纹（{dt:.1f}s，不经过推理引擎）",
              tuple(e3.shape) == (192,) and e3.dtype == torch.float32)
        e3b = sc.embed(V3)
        check("同一条音频两次嵌入逐位一致（打分必须可复现）",
              torch.equal(e3, e3b))
        e1 = sc.embed(V1)
        sim_self = RewardScorer.cosine(e3, e3)
        sim_cross = RewardScorer.cosine(e3, e1)
        check("自相似 = 1.0", abs(sim_self - 1.0) < 1e-5, f"{sim_self:.5f}")
        check("跨说话人相似度明显更低（近零也算，不同人）",
              sim_cross < 0.5, f"{sim_cross:.4f}")
        check("不存在的音频报 FileNotFoundError",
              _raises(sc, "embed", "Z:/无.wav", FileNotFoundError))
        sc.unload()
        check("unload 后模型引用断开", sc._spk is None and sc._asr is None)
    except Exception as e:
        check(f"campplus 段异常：{type(e).__name__}: {e}", False)

    # =====================================================================
    head("[5] whisper 转写（已缓存；不可用则整段跳过）")
    if not whisper_ready():
        print("  [skip] 本机没有 whisper 权重且无法下载，[5][6] 跳过")
    else:
        try:
            sc2 = RewardScorer(RewardOptions(whisper_size="small",
                                             device="cpu"))
            t0 = time.perf_counter()
            txt1 = sc2.transcribe(V3)
            dt = time.perf_counter() - t0
            check(f"真实语音转写出了非空文本（{dt:.1f}s）",
                  len(txt1) >= 4, txt1[:40])
            txt2 = sc2.transcribe(V3)
            check("★ 温度 0 + beam：同一条音频两次转写逐字一致",
                  txt1 == txt2)
            check("转写与自身文本的 CER=0", cer(txt1, txt1) == 0.0)
            check("转写与无关文本的 CER 明显大",
                  cer("完全不相关的参考文本", txt1) > 0.5)
            check("中文转写不含英文语言残留",
                  asr_language("ZH") == "zh")
            V3_TEXT = txt1     # [6] 拿它当 ground truth
        except Exception as e:
            check(f"whisper 段异常：{type(e).__name__}: {e}", False)
            V3_TEXT = None

    # =====================================================================
    head("[6] score() 端到端（真模型）")
    if not whisper_ready():
        print("  [skip] whisper 不可用，[6] 跳过")
    elif V3_TEXT:
        try:
            r_self = sc2.score(V3, V3_TEXT, V3)
            check("端到端打分 ok", r_self["ok"], str(r_self.get("error", ""))[:60])
            check("参考即自身：WER=0 且相似度≈1",
                  r_self["wer"] == 0.0 and r_self["sim"] > 0.999,
                  f"wer={r_self['wer']} sim={r_self['sim']}")
            check("reward = 加权组合（满分）",
                  abs(r_self["reward"] - 1.0) < 1e-3, str(r_self["reward"]))
            r_bad = sc2.score(V3, "完全无关的文本", V1)
            check("错文本 + 陌生参考：WER 高、相似度低",
                  r_bad["ok"] and r_bad["wer"] > 0.5 and r_bad["sim"] < 0.6,
                  f"wer={r_bad['wer']} sim={r_bad['sim']}")
            check("好样本的 reward 明显高于坏样本",
                  r_self["reward"] > r_bad["reward"] + 0.2,
                  f"{r_self['reward']} vs {r_bad['reward']}")
            pair = sc2.score_pair(V3, V1, V3_TEXT, V3)
            check("score_pair 选出 a（与参考同源的那个）",
                  pair["ok"] and pair["chosen"] == "a",
                  f"margin={pair.get('margin')}")
            check("margin 的符号与两个 reward 的差一致",
                  pair["ok"] and
                  abs(pair["margin"] -
                      (pair["a"]["reward"] - pair["b"]["reward"])) < 1e-6)
            sc2.unload()
            check("端到段 unload 后引用断开",
                  sc2._spk is None and sc2._asr is None)
        except Exception as e:
            check(f"端到段异常：{type(e).__name__}: {e}", False)

    print("\n" + "=" * 70)
    if FAIL == 0:
        print(f"  通过 {PASS} 项 · 失败 0 项")
    else:
        print(f"  通过 {PASS} 项 · 失败 {FAIL} 项")
        for x in FAILS:
            print(f"    FAIL {x}")
    print("=" * 70)
    return 1 if FAIL else 0


def _raises(obj, fn_name: str, arg: str, exc: type) -> bool:
    try:
        getattr(obj, fn_name)(arg)
        return False
    except exc:
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
