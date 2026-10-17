"""E1-T 歌声训练向导参数登记（params_<域>.py 分文件惯例）。"""

from __future__ import annotations


def build(reg, P):
    _import_step(reg, P)
    _capability_step(reg, P)
    _identity_step(reg, P)


def _import_step(reg, P):
    reg(P(
        key="e1t_import_langs", group="e1t_import", label="语言过滤",
        kind="text", default="zh,en",
        summary="逗号分隔（zh,en,ja,ko…），空=全部语言。",
        info="首跑建议 zh,en（中英流行覆盖面大、数据量可控）",
    ))
    reg(P(
        key="e1t_import_pitch_aug", group="e1t_import", label="音高增广（半音）",
        kind="text", default="-12,+12",
        summary="如 '-12,+12'：导入期为每条样本生成保时长移调副本并继承技巧标签。",
        info="音频域增广（mel 与 F0 标签同源一致）；空=关闭。容量代价 ×3",
        affects="capability 训练的音域覆盖",
        detail_md="""
**为什么在导入期做而不是训练期**：只改 F0 标签不改 mel 的「条件侧增广」
会让同一频谱对应多个音高标签，等于教模型忽略音高条件（调研否决项）。
音频域移调让 (mel, F0) 同源一致，代价是磁盘与一次性提取成本。

SPA-SVC 的训练期随机 6–18 半音增广更省磁盘，但需要在线移调；导入期
固定 ±12 是它的离线近似（两个离散点），首版够用。
""",
    ))
    reg(P(
        key="e1t_import_max_items", group="e1t_import", label="导入条数上限",
        kind="number", default=0, precision=0,
        summary="0=不限。冒烟建议 1500 左右（约几小时音频，训练一夜内可完成）。",
        info="全量 GTSinger 是 80.59h——8GB 卡首跑不建议全量",
    ))
    reg(P(
        key="e1t_import_copy", group="e1t_import", label="复制音频（不硬链接）",
        kind="checkbox", default=False,
        summary="默认硬链接（零额外磁盘）；跨盘/权限问题时勾选复制。",
        info="硬链接失败会自动退回复制",
    ))


def _capability_step(reg, P):
    reg(P(
        key="e1t_cap_preset", group="e1t_cap", label="训练预设",
        kind="dropdown",
        choices=["🟢 保守（首跑推荐）", "⚖️ 均衡", "🎭 表现力", "🔴 激进"],
        default="🟢 保守（首跑推荐）",
        summary="LoRA 注入面/rank/epoch 组合（与训练页同源预设）。",
        info="歌声数据分布与说话差异大，首跑从保守档开始",
    ))
    reg(P(
        key="e1t_cap_frames", group="e1t_cap", label="max_total_frames",
        kind="slider", minimum=800, maximum=2200, step=100, default=1800,
        summary="单条训练样本的总帧数上限（prompt+target）。CFM 注意力 O(T²)，这是显存主旋钮。",
        info="8GB 建议 1800（默认 2200 是 TTS 经验值，歌声底座留些余量）",
        affects="训练显存与可学的乐句长度",
    ))
    reg(P(
        key="e1t_cap_epochs", group="e1t_cap", label="epochs（覆盖预设）",
        kind="slider", minimum=0, maximum=40, step=1, default=0,
        summary="0=用预设自带的 epochs；>0 时覆盖。",
        info="冒烟（小子集）设 2–4；全量建议让早停自己决定",
    ))


def _identity_step(reg, P):
    reg(P(
        key="e1t_id_preset", group="e1t_id", label="训练预设",
        kind="dropdown",
        choices=["🟢 保守（推荐）", "⚖️ 均衡", "🎭 表现力", "🔴 激进"],
        default="🟢 保守（推荐）",
        summary="与 capability 相同的预设体系；身份训练数据少（10–60s），保守档足够。",
    ))
    reg(P(
        key="e1t_id_epochs", group="e1t_id", label="epochs（覆盖预设）",
        kind="slider", minimum=0, maximum=40, step=1, default=0,
        summary="0=用预设自带；角色说话样本通常 <100 条，几轮即可。",
    ))
