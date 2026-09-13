"""读音纠正服务（services/pronunciation.py）的单元验证。

不加载任何模型权重，只用 tokenizer + TextNormalizer，所以不占显存，
可以在 WebUI 运行时同时跑。

    .venv\\Scripts\\python.exe tools\\pronunciation_test.py
"""

from __future__ import annotations

import os                                              # noqa: F401
import sys

import _env                                            # noqa: F401  路径 + 控制台编码

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    from webui_app.services import pronunciation as PR

    print("\n[1] 词表加载")
    vocab = PR.pinyin_vocab()
    check("pinyin.vocab 加载出 1728 条", len(vocab) == 1728, f"{len(vocab)} 条")
    check("词表已大写化", "XING2" in vocab and "xing2" not in vocab)
    check("五个声调都在", all(any(w.endswith(str(t)) for w in vocab)
                              for t in range(1, 6)))

    print("\n[2] classify：发音体系判定")
    cases = [
        ("行", "XING2", "pinyin"),
        ("行", "xing2", "pinyin"),
        ("行", "MEN5", "pinyin"),
        ("minute", "M IH1 . N AH0 T", "cmu"),
        ("上手", "じょうず", "kana"),
        ("行", "XING", "pinyin"),          # 没声调但格式仍是拼音
        ("行", "???", "suspicious"),
        ("行", "", "suspicious"),
    ]
    for w, p, expect in cases:
        got = PR.classify(w, p)
        check(f"classify({w!r}, {p!r}) == {expect}", got == expect, f"实得 {got}")

    print("\n[3] parse / strip")
    txt = "他在银<行|XING2>里<行|HANG2>走了半天"
    anns = PR.parse(txt)
    check("解析出 2 条标注", len(anns) == 2, f"{len(anns)} 条")
    check("第一条是 XING2", anns and anns[0].pron_upper == "XING2")
    check("第二条是 HANG2", len(anns) > 1 and anns[1].pron_upper == "HANG2")
    check("记录了字符位置", anns and txt[anns[0].start:anns[0].end] == "<行|XING2>",
          txt[anns[0].start:anns[0].end] if anns else "")
    check("声调提取正确", anns and anns[0].tone == "2")
    check("strip_annotations 还原原文",
          PR.strip_annotations(txt) == "他在银行里行走了半天",
          PR.strip_annotations(txt))

    print("\n[4] validate：合法输入不应报警")
    clean = "他在银<行|XING2>里办事，2024年<重|CHONG2>新开始。"
    iss = PR.validate(clean, "ZH")
    bad = [i for i in iss if i.level == "error"]
    check("合法拼音标注无 error", not bad, str([i.message for i in bad]))

    print("\n[5] validate：各类错误都要抓到")
    def has(text, level, keyword, lang="ZH"):
        got = [i for i in PR.validate(text, lang)
               if i.level == level and keyword in i.message]
        check(f"{text[:34]!r} → {level} 含「{keyword}」", bool(got),
              got[0].message[:70] if got else
              str([(i.level, i.message[:40]) for i in PR.validate(text, lang)]))

    has("银<行|XING2", "error", "尖括号不配对")
    has("银<行XING2>", "error", "缺少竖线")
    has("银<行|XING9>", "error", "声调数字")
    has("银<行|XING>", "warn", "没有声调数字")
    has("银<行|????>", "error", "既不是合法拼音")
    has("银<|>", "error", "文字部分是空的")
    has("银<行|>", "error", "发音部分是空的")
    has("银<a|b|c>", "error", "个竖线")

    # 合法的语言前缀与已展开的特殊 token 不该被误报
    for ok_text in ("<|zh|> 普通文本",
                    "粘了 <|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|> 的文本"):
        errs = [i for i in PR.validate(ok_text, "ZH") if i.level == "error"]
        check(f"不误报：{ok_text[:32]!r}", not errs,
              str([i.message[:50] for i in errs]))

    # 不在词表里的拼音
    notinv = PR.validate("银<行|XING9>", "ZH")
    check("非法声调被拦下", any("非法" in i.message for i in notinv))
    notinv2 = PR.validate("银<行|BIANG2>", "ZH")
    check("词表外拼音给出 warn + 相近候选",
          any(i.level == "warn" and "pinyin.vocab" in i.message for i in notinv2),
          str([i.message[:90] for i in notinv2]))

    # jqx + u 提示
    jqx = PR.validate("一<句|JU3>话", "ZH")
    check("jqx+u 给出自动改 V 的提示",
          any(i.level == "info" and "correct_pinyin" in i.message for i in jqx),
          str([(i.level, i.message[:60]) for i in jqx]))
    check("_jqx_fix('JU3') == 'JV3'", PR._jqx_fix("JU3") == "JV3")
    check("_jqx_fix('NU3') == 'NU3'", PR._jqx_fix("NU3") == "NU3")

    # 中文文字配 CMU 音素
    mix = PR.validate("这个<分钟|M IH1 . N AH0 T>很长", "ZH")
    check("中文文字 + CMU 音素 → warn",
          any(i.level == "warn" and "SPECIAL_TOKEN_2" in i.message for i in mix),
          str([(i.level, i.message[:50]) for i in mix]))

    # lang=JA 但写了拼音
    ja = PR.validate("彼は<行|XING2>く", "JA")
    check("lang=JA 写拼音 → warn",
          any(i.level == "warn" and "假名" in i.message for i in ja),
          str([(i.level, i.message[:50]) for i in ja]))

    print("\n[6] preview：与 tools/pinyin_probe.py 的实测结果对齐")
    p = PR.preview("他在银<行|XING2>里办事", "ZH")
    expect_final = "<|zh|> 他在银<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>里办事"
    check("最终文本与探针实测一致", p["final"] == expect_final, p["final"])
    check("token 数 = 13（探针实测值）", p["token_count"] == 13,
          f'{p["token_count"]}')
    check("SPECIAL_TOKEN_2 成对", p["n_special_2"] == 2 and p["paired"])
    check("steps 记录了 6~7 步", len(p["steps"]) >= 6, f'{len(p["steps"])} 步')
    check("识别到 1 条标注", len(p["annotations"]) == 1)

    p2 = PR.preview("2024年他在银<行|HANG2>工作了3个月", "ZH")
    check("数字被归一化且标注存活",
          "二零二四" in p2["final"] and "HANG2" in p2["final"], p2["final"])
    p3 = PR.preview("2024年他在银<行|HANG2>工作了3个月", "ZH", normalize=False)
    check("关归一化时数字保留但标注仍存活",
          "2024" in p3["final"] and "HANG2" in p3["final"], p3["final"])
    p4 = PR.preview("He had a <minute|M IH1 . N AH0 T> to spare", "EN")
    check("英文走 SPECIAL_TOKEN_1",
          p4["n_special_1"] == 2 and p4["n_special_2"] == 0,
          f'st1={p4["n_special_1"]} st2={p4["n_special_2"]}')
    p5 = PR.preview("彼は料理が<上手|じょうず>だが", "JA")
    check("日语假名不加特殊 token",
          p5["n_special_1"] == 0 and p5["n_special_2"] == 0
          and "じょうず" in p5["final"], p5["final"])
    md = PR.preview_markdown("他在银<行|XING2>里办事", "ZH")
    check("preview_markdown 可渲染", len(md) > 200 and "SPECIAL_TOKEN_2" in md,
          f"{len(md)} 字符")

    print("\n[7] heteronyms：多音字候选")
    # 注意两种语义要分开测：
    #   · 词组（「银行」）—— pypinyin 走**词组感知**模式，会直接消歧成 yín háng，
    #     所以候选里**不该**出现 XING2。这正是我们想要的行为：
    #     推荐的是「这个词在这个语境下的合法读法」，而不是一堆无关的多音。
    #   · 单字（「行」）—— 才应该给出全部多音候选，供用户挑一个非默认的。
    c = PR.heteronyms("银行")
    pys = [x["pinyin"] for x in c]
    check("「银行」能查到候选", len(c) > 0, str(pys))
    check("候选里有 HANG2（正确读音）", "HANG2" in pys, str(pys))
    check("词组模式已消歧，不混入 XING2", "XING2" not in pys, str(pys))
    check("候选带 in_vocab 标记", all("in_vocab" in x for x in c))
    check("每个字都有一个默认读音被标出",
          all(any(x["is_default"] for x in c if x["char_index"] == i)
              for i in {x["char_index"] for x in c}),
          str([(x["char"], x["pinyin"], x["is_default"]) for x in c]))
    check("候选都在 pinyin.vocab 里或已标记",
          all(x["in_vocab"] is not False for x in c), str(pys))

    c2 = PR.heteronyms("行")
    pys2 = [x["pinyin"] for x in c2]
    check("单字「行」给出多个多音候选", len(c2) > 2, str(pys2))
    check("单字候选里 XING2 / HANG2 都在",
          "XING2" in pys2 and "HANG2" in pys2, str(pys2))
    check("默认读音排在最前", c2 and c2[0]["is_default"],
          str([(x["pinyin"], x["is_default"]) for x in c2[:4]]))
    c3 = PR.heteronyms("")
    check("空输入返回空列表", c3 == [])
    check("limit 生效", len(PR.heteronyms("行行行行", limit=3)) <= 3)
    hm = PR.heteronym_markdown("银行")
    check("heteronym_markdown 可渲染", "HANG2" in hm and "✅" in hm, f"{len(hm)} 字符")
    hm2 = PR.heteronym_markdown("行")
    check("单字的 markdown 也含 XING2", "XING2" in hm2, f"{len(hm2)} 字符")
    check("查不到时给友好提示",
          "查不到" in PR.heteronym_markdown(""), PR.heteronym_markdown("")[:40])

    print("\n[8] apply_annotation")
    src = "他在银行里办事"
    new, ok = PR.apply_annotation(src, "行", "HANG2")
    check("能插入标注", ok and new == "他在银<行|HANG2>里办事", new)
    new2, ok2 = PR.apply_annotation(new, "行", "HANG2")
    check("重复插入是幂等的", ok2 and new2 == new, new2)
    src3 = "行行出状元"
    new3, ok3 = PR.apply_annotation(src3, "行", "HANG2", which=1)
    check("which 能指定第几个", ok3 and new3 == "行<行|HANG2>出状元", new3)
    new4, ok4 = PR.apply_annotation("没有这个字", "龙", "LONG2")
    check("找不到目标时返回 False", not ok4 and new4 == "没有这个字")
    new5, ok5 = PR.apply_annotation(src, "行", "")
    check("发音为空时不改动", not ok5 and new5 == src)

    print("\n[9] SYNTAX_DOC")
    check("语法文档非空", len(PR.SYNTAX_DOC) > 500, f"{len(PR.SYNTAX_DOC)} 字符")
    check("文档提到 1728 条", "1728" in PR.SYNTAX_DOC)

    print("\n" + "=" * 64)
    print(f"  通过 {len(PASS)} 项 · 失败 {len(FAIL)} 项")
    if FAIL:
        print("  失败项：")
        for f in FAIL:
            print(f"    FAIL {f}")
    print("=" * 64)
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
