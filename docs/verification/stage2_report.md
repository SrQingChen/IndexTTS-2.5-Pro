# 阶段 2 验收报告（真机完整闭环）

- 时间：2026-09-13 16:40:16 · 总耗时 4.0 分钟
- 结论：**✅ 通过**（33 项通过 / 0 项失败）

## 阶段耗时

| 阶段 | 耗时 | 明细 |
|---|---|---|
| A 数据准备 | 1.5 分钟 |  |
| B GPT 训练 | 0.3 分钟 | steps=6 drift=0.000423 |
| C CFM 训练 | 0.1 分钟 | steps=4 drift=0.00319 |
| D DPO 偏好对 | 0.8 分钟 | kept=1 |
| E A/B 评测 | 0.6 分钟 | Δreward=0.0009 |
| F 挂载/合并 | 0.8 分钟 | 漂移 0.000424 |

## 全部断言

- ✅ 引擎加载成功
- ✅ 合成 28 句全部成功
- ✅ 导入 28 条
- ✅ 全部 ready（音频体检过）
- ✅ 特征提取 ok
- ✅ 特征缓存 28 条齐了
- ✅ 划分 train=21 / val=7
- ✅ 引擎已卸载
- ✅ GPT 预检通过
- ✅ GPT 训练 ok
- ✅ GPT 底座复校通过（before+after）
- ✅ GPT 漂移体检有账
- ✅ GPT adapter 已同步
- ✅ GPT 峰值显存有记录
- ✅ CFM 预检通过
- ✅ CFM 训练 ok
- ✅ CFM 底座复校通过
- ✅ CFM 漂移体检有账
- ✅ CFM adapter 已同步
- ✅ adapter 挂载成功
- ✅ 偏好对构造 ok
- ✅ 成对 1 对（合成 8 个候选）
- ✅ pairs.jsonl 落盘且带 reward 双侧
- ✅ A/B 评测 ok
- ✅ 逐条 3 行
- ✅ 胜负表有账
- ✅ 报告与试听文件落盘
- ✅ 评测后引擎上没有残留 adapter
- ✅ 强度旋钮真机生效（0.7）
- ✅ GPT 合并 ok
- ✅ 合并产物存在且 >1GB（真权重量级）
- ✅ 合并校验 missing/unexpected=0
- ✅ CFM 合并 ok 且校验干净

## 产物

- GPT 训练报告：`outputs/acceptance/gpt_report.md`
- CFM 训练报告：`outputs/acceptance/cfm_report.md`
- A/B 评测报告：`outputs/acceptance/ab_report.md`
- 合并权重：`outputs/acceptance/gpt_merged.pth` / `s2mel_merged.pth`