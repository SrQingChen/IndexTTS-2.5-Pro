### A/B 评测：adapter(accept_gpt/best@1) vs base(底座)

- 数据集 `accept_stage2` · 3 条 · 种子 42 · 打分 whisper=small（WER×0.6 + SS×0.4）
- 试听目录：`outputs\eval\20260913_163904_adapter_vs_base`

| 指标 | adapter | base | 越好 |
|---|---|---|---|
| WER | 0.1111 | 0.1111 | ↓ |
| 声纹相似 | 0.923 | 0.9207 | ↑ |
| reward | **0.9025** | **0.9016** | ↑ |

**胜负**：🏆 adapter 胜 0 / 0 负 adapter / 3 平 · 平均 Δreward = +0.0009（正 = adapter 好）

<details><summary>逐条明细</summary>

| 样本 | 文本 | adapter reward | base reward | Δ |
|---|---|---|---|---|
| `utt_00001` | 今天天气不错，我们出去走走吧。… | 0.9762 | 0.9694 | 0.0068 |
| `utt_00002` | 这个模型的音色克隆效果相当自然。… | 0.7596 | 0.7611 | -0.0015 |
| `utt_00003` | 请问附近的地铁站应该怎么走？… | 0.9718 | 0.9744 | -0.0026 |

</details>
