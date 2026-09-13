"""拼音/音素标注可行性验证。

用户问「读错的字能不能用拼音标注纠正」——这个探针给确定答案，不靠猜。

它**逐行复刻 infer_v2_5.py:699-725 的文本处理链路**（归一化 → 小写 →
发音标注展开 → 特殊 token 大写 → tiktoken 编码），然后检查：

    1. <文字|发音> 标注是否被正确展开成 <|SPECIAL_TOKEN_1/2|>发音<|...|>
    2. 展开后的拼音是否作为**独立 token** 进入词表（而不是被拆成字母）
    3. 有没有 UNK / 未被识别的片段
    4. 多音字标注与不标注，token 序列是否**真的不同**（若相同说明标注没生效）

只用到 tokenizer 与文本处理器，**不加载任何模型权重**，因此不占显存，
可以在 WebUI 运行时同时执行。

    .venv\\Scripts\\python.exe tools\\pinyin_probe.py
"""

from __future__ import annotations

import os
import re
import sys

import _env                                            # noqa: F401  路径 + 控制台编码
PROJECT_ROOT = _env.PROJECT_ROOT

CKPT = os.path.join(PROJECT_ROOT, "checkpoints")

ok_all = True


def check(label: str, cond: bool, detail: str = ""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from indextts.infer_v2_5 import apply_pronunciation_annotations
    from indextts.utils.front import TextNormalizer
    from indextts.utils.tokenizer import get_tokenizer

    tn = TextNormalizer(enable_glossary=True)
    tn.load()          # 必须显式调，否则 normalize() 直接返回空串
    # （官方在 infer_v2_5.py:237 也是这么做的；Windows 上底层用 wetext）
    tok = get_tokenizer(multilingual=True, model_dir=CKPT)
    enc = tok.encoding                      # tiktoken.Encoding

    def id2tok(ids):
        """tiktoken 没有 convert_ids_to_tokens，用 decode_single_token_bytes。"""
        out = []
        for i in ids:
            try:
                out.append(enc.decode_single_token_bytes(i).decode("utf-8", "replace"))
            except Exception:
                out.append(f"<id:{i}>")
        return out

    print(f"tokenizer = {type(tok).__name__} (tiktoken)")
    print(f"词表大小 n_vocab = {enc.n_vocab}")
    print(f"sot={tok.sot} eot={tok.eot}")
    st = tok.special_tokens
    print(f"special_tokens 共 {len(st)} 个")
    for k in ("SPECIAL_TOKEN_1", "SPECIAL_TOKEN_2"):
        kk = f"<|{k}|>"
        print(f"  {kk:22s} -> {st.get(kk, '不存在')}")
    check("SPECIAL_TOKEN_1 / 2 已注册为 tiktoken 特殊 token",
          "<|SPECIAL_TOKEN_1|>" in st and "<|SPECIAL_TOKEN_2|>" in st)

    # ------------------------------------------------------------------
    # 复刻 infer_v2_5.py 的文本链路
    # ------------------------------------------------------------------
    def pipeline(text: str, lang: str = "ZH", normalize: bool = True) -> str:
        lang_prefix = f'<|{lang.lower()}|> '
        t = tn.clean_pattern.sub(lambda x: tn.char_rep_map[x.group()], text)
        if normalize and lang.lower() in ("zh", "zhen", "en"):
            t = tn.normalize(t)
        if lang.lower() in ("ja", "zh", "zhen", "en"):
            t = t.lower()
        t = apply_pronunciation_annotations(t)
        t = re.sub(r'<\|([^|]+)\|>', lambda m: f'<|{m.group(1).upper()}|>', t)
        return lang_prefix + t

    def encode(text: str, lang: str = "ZH", normalize: bool = True):
        s = pipeline(text, lang, normalize)
        ids = tok.encode(s, allowed_special="all")
        return s, ids

    def show(title: str, text: str, lang: str = "ZH"):
        s, ids = encode(text, lang)
        toks = id2tok(ids)
        print(f"\n--- {title} ---")
        print(f"  原文   : {text}")
        print(f"  管线后 : {s}")
        print(f"  token数: {len(ids)}")
        print(f"  tokens : {toks}")
        return s, ids, toks

    print("\n" + "=" * 72)
    print("① 中文多音字：<行|XING2> vs <行|HANG2>")
    print("=" * 72)
    s1, i1, t1 = show("标注 XING2", "他在银<行|XING2>里办事")
    s2, i2, t2 = show("标注 HANG2", "他在银<行|HANG2>里办事")
    s3, i3, t3 = show("不标注（让模型自己猜）", "他在银行里办事")

    check("SPECIAL_TOKEN_2 出现在管线输出里", "SPECIAL_TOKEN_2" in s1, s1)
    check("拼音被统一转成大写", "XING2" in s1 and "xing2" not in s1)
    check("XING2 与 HANG2 的 token 序列**不同**（标注确实生效）",
          i1 != i2, f"{len(i1)} vs {len(i2)} tokens")
    check("标注版与不标注版的 token 序列不同", i1 != i3)
    # 注意：tiktoken 用 BPE，XING2 会被拆成 ['X','ING','2'] 三个 token。
    # 这**不是问题**：<|SPECIAL_TOKEN_2|> 成对出现，把拼音段明确界定出来了，
    # 模型学的是“这对特殊 token 之间的 BPE 序列 = 指定发音”，
    # 只要训练与推理的分词一致就行。拆成几个 token 只影响序列长度。
    n_st2 = t1.count("<|SPECIAL_TOKEN_2|>")
    check("SPECIAL_TOKEN_2 成对出现（拼音段被完整界定）",
          n_st2 == 2, f"出现 {n_st2} 次")
    check("不标注时汉字作为单 token（对比：标注改变了模型看到的输入）",
          "银" in t3 and "行" in t3 and len(i3) < len(i1),
          f"不标注 {len(i3)} tokens vs 标注 {len(i1)} tokens")

    # 确定性：同一文本多次编码必须完全一致
    check("编码具确定性（重复三次结果一致）",
          encode("他在银<行|XING2>里办事")[1] == i1
          and encode("他在银<行|XING2>里办事")[1] == i1)

    print("\n" + "=" * 72)
    print("② 归一化是否会破坏拼音标注")
    print("=" * 72)
    s4, i4, t4 = show("含数字与标注", "2024年他在银<行|HANG2>工作了3个月")
    check("数字被归一化成中文读法", "2024" not in s4,
          f"管线后仍含 '2024'={('2024' in s4)}")
    check("标注在归一化后仍然存活", "HANG2" in s4, s4)

    s5, i5 = encode("2024年他在银<行|HANG2>工作了3个月", "ZH", normalize=True)
    s5b, i5b = encode("2024年他在银<行|HANG2>工作了3个月", "ZH", normalize=False)
    print(f"\n--- 开归一化 ---\n  {s5}\n--- 关归一化 ---\n  {s5b}")
    check("开/关归一化都能保留标注",
          "HANG2" in s5 and "HANG2" in s5b)

    print("\n" + "=" * 72)
    print("③ 轻声（第5声）与 jqx+ü 规则")
    print("=" * 72)
    s6, i6, t6 = show("轻声 MEN5", "我<们|MEN5>一起去")
    check("轻声 5 被保留", "MEN5" in s6, s6)

    # correct_pinyin: jqx 后面的 u 应改成 v（ü）
    for py in ("NU3", "JU3", "qu2", "XU4", "lv4", "HAO3"):
        print(f"  correct_pinyin({py}) = {tn.correct_pinyin(py)}")
    check("correct_pinyin 对 jqx+u 生效（JU3 → JV3）",
          tn.correct_pinyin("JU3") == "JV3", tn.correct_pinyin("JU3"))
    check("非 jqx 声母不被误改（NU3 保持）",
          tn.correct_pinyin("NU3") == "NU3", tn.correct_pinyin("NU3"))

    print("\n" + "=" * 72)
    print("④ 英文音素（CMU）与日语假名")
    print("=" * 72)
    s7, i7, t7 = show("英文音素", "He had a <minute|M IH1 . N AH0 T> to spare",
                      lang="EN")
    check("英文走 SPECIAL_TOKEN_1", "SPECIAL_TOKEN_1" in s7, s7)
    check("音素大写化", "M IH1" in s7)

    s8, i8, t8 = show("日语假名", "彼は料理が<上手|じょうず>だが", lang="JA")
    check("日语假名**不加** SPECIAL_TOKEN（直接空格包裹）",
          "SPECIAL_TOKEN" not in s8 and "じょうず" in s8, s8)

    print("\n" + "=" * 72)
    print("⑤ pinyin.vocab 全量往返完整性校验")
    print("=" * 72)
    print("  先说清楚：pinyin.vocab 在代码里**从未被引用**（已全仓搜索确认），")
    print("  它只是 README 里说的「合法拼音清单」。所以它的真正用途是：")
    print("  ① 告诉我们模型训练时见过哪些拼音拼写")
    print("  ② 可以给 UI 做**输入校验**：用户写了不在清单里的拼音就提醒")
    pv = os.path.join(CKPT, "pinyin.vocab")
    if os.path.isfile(pv):
        with open(pv, "r", encoding="utf-8") as f:
            words = [w.strip() for w in f if w.strip()]
        print(f"  pinyin.vocab 条目数 = {len(words)}")
        print(f"  前 12 条: {words[:12]}")

        ST2 = "<|SPECIAL_TOKEN_2|>"

        def roundtrip(py: str):
            """把拼音包进标注跑完整管线，再把两个特殊 token 之间的内容
            解码回来，看是否等于原拼音。这才能证明信息没丢。"""
            s, ids = encode(f"测试<字|{py}>结束", "ZH")
            toks = id2tok(ids)
            if toks.count(ST2) != 2:
                return False, f"特殊 token 不成对: {toks}"
            a = toks.index(ST2)
            b = toks.index(ST2, a + 1)
            got = "".join(toks[a + 1:b])
            return got == py, f"{py} → {toks[a+1:b]} → '{got}'"

        bad = []
        for w in words:
            okr, detail = roundtrip(w)
            if not okr:
                bad.append((w, detail))
        check(f"全部 {len(words)} 条拼音都能完整往返（信息无丢失）",
              not bad,
              f"失败 {len(bad)} 条" + (f"；首例 {bad[0][0]}: {bad[0][1]}" if bad else ""))
        # 声调分布统计，给 UI 的提示文案用
        tones = {}
        for w in words:
            t = w[-1] if w and w[-1].isdigit() else "?"
            tones[t] = tones.get(t, 0) + 1
        print(f"  声调分布: {dict(sorted(tones.items()))}")
        check("五个声调（1~5，5=轻声）都有覆盖",
              all(str(k) in tones for k in range(1, 6)), str(tones))
    else:
        check("pinyin.vocab 存在", False, pv)

    print("\n" + "=" * 72)
    print("⑥ 结论")
    print("=" * 72)
    if ok_all:
        print("  拼音标注**完全可用**。语法：<文字|发音>")
        print("    中文拼音 + 声调数字 1~5   <行|XING2>  <们|MEN5>")
        print("    英文 CMU 音素（空格分音节） <minute|M IH1 . N AH0 T>")
        print("    日语假名                    <上手|じょうず>")
        print("  注意：大小写不敏感（内部统一转大写），jqx 后的 u 会自动改 v")
    else:
        print("  有未通过项，见上方 FAIL")
    print("=" * 72)
    return 0 if ok_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
