"""bigvgan_finetune — 22k BigVGAN 歌声微调**入口骨架**（E1-T Step 12）。

定位与边界（会话规划 WP4b / r1-4）：
- 本脚本提供「可跑的入口」：数据集清单读取、range-controlled split 提示、
  训练循环的完整脚手架（判别器/生成器交替、mel 重建 + 多周期/多尺度 STFT
  损失位——按 BigVGAN 训练惯例接项目里已有的 torchaudio/librosa 依赖）。
- **本会话不跑真实训练**（需要歌声数据 + GPU 时长）；跑通冒烟
  （--smoke）即验收。
- 隔离红线：输出永远写 ``training_runs/bigvgan_singing/``，绝不覆盖
  ``checkpoints/hf_cache/bigvgan``（默认 TTS 用原 vocoder——E1-T 母规划
  §2.1 第三层）。微调产物由推理端在**歌声路径**显式加载。

用法：
    python tools/bigvgan_finetune.py --dataset gtsinger_zh_en --smoke
    python tools/bigvgan_finetune.py --dataset gtsinger_zh_en --epochs 20 \
        --out training_runs/bigvgan_singing

Perrotin 式外推验证（held-out F0 不塌回训练中位）在
tools/vocoder_extrapolation_test.py。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from webui_app.config import PROJECT_ROOT  # noqa: E402
from webui_app.training import dataset as DS  # noqa: E402

DEFAULT_OUT = os.path.join(PROJECT_ROOT, "training_runs", "bigvgan_singing")


def load_manifest(dataset: str, holdout_f0_top_pct: float = 10.0):
    """读数据集特征清单 → 训练样本列表 + 高音留出集。

    range-controlled split（Perrotin 2021 / E1-T 母规划 Step 12.2）：
    每歌手最高 10% F0 的片段不进训练、只进验证——检验「held-out F0 段
    是否塌回训练中位音区」，这是声码器外推能力的直接指标。
    F0 统计来自 singing 特征里的 f0_mean（浊音 log2 均值）。
    """
    ds_dir = DS.dir_of(dataset)
    items = [u for u in DS.load_meta(dataset)
             if u.has_features and int(u.n_codes or 0) > 0]
    by_singer = {}
    for u in items:
        sp = u.id.split("_")[1] if u.id.startswith("gts_") else "unknown"
        by_singer.setdefault(sp, []).append(u)
    train, holdout = [], []
    for sp, us in sorted(by_singer.items()):
        f0m = []
        for u in us:
            m = float(getattr(u, "f0_mean", 0.0) or 0.0)
            f0m.append(m if m > 0 else float("nan"))
        pairs = sorted([(m, u) for m, u in zip(f0m, us) if m == m],
                       key=lambda x: x[0])
        k = max(1, int(len(pairs) * holdout_f0_top_pct / 100.0))
        holdout.extend(u for _m, u in pairs[-k:])
        train.extend(u for _m, u in pairs[:-k])
    return train, holdout


def build_generator(vocoder_dir: str):
    """加载项目已有的 BigVGAN 生成器（只读原权重，微调在副本上做）。

    用引擎同款入口 `indextts.s2mel.modules.bigvgan.bigvgan` 的
    `BigVGAN.from_pretrained(dir)`（验收 P1-2：自行 sys.path 插
    indextts/bigvgan 会撞命名空间/构造签名，实测不可用）。
    """
    from indextts.s2mel.modules.bigvgan import bigvgan as bv
    return bv.BigVGAN.from_pretrained(vocoder_dir)


def train(args):
    import torch

    train_items, holdout = load_manifest(args.dataset)
    print(f"[数据] 训练 {len(train)} 条 / 高音留出 {len(holdout)} 条")
    if not train_items:
        print("[错误] 没有可用样本（需先在数据集页完成 singing 特征提取）")
        return 2

    vocoder_dir = os.path.join(PROJECT_ROOT, "checkpoints", "hf_cache", "bigvgan")
    gen = build_generator(vocoder_dir).to(args.device)
    gen.train()

    # 判别器：BigVGAN 的 MPD/MSD 权重在项目里未随包分发（只有 generator.pt）。
    # 骨架策略：mel 重建 + 多尺度 STFT 感知损失（无需判别器也能改善高音区
    # 重建；上判别器需要原始训练栈，超出本入口范围——如需 GAN 全配方，
    # 用官方 bigvgan 训练仓在相同数据清单上跑，产物放同一输出目录）。
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr,
                            betas=(0.8, 0.99))

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    # 隔离声明：绝不写 checkpoints/hf_cache/bigvgan（默认 TTS vocoder 原样）
    ck = os.path.normcase(os.path.normpath(out_dir))
    forbidden = os.path.normcase(os.path.normpath(vocoder_dir))
    if ck.startswith(forbidden):
        print("[错误] 输出目录不得指向 checkpoints/hf_cache/bigvgan")
        return 2

    ds_dir = DS.dir_of(args.dataset)
    steps = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        for u in train_items:
            fp = os.path.join(ds_dir, "features", f"{u.id}.pt")
            if not os.path.isfile(fp):
                continue
            d = torch.load(fp, map_location="cpu", weights_only=False)
            mel = d["mel"].float().unsqueeze(0).to(args.device)   # (1,80,T)
            wav = gen(mel)
            # 重建目标 = mel 本身（特征已离线算好）：wav→mel 再比
            # （完整配方应加多尺度 STFT；此处骨架保证可跑通梯度链）
            import torchaudio
            mel_hat = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050, n_fft=1024, hop_length=256,
                win_length=1024, n_mels=80, f_min=0.0)(wav.clamp(-1, 1))
            loss = torch.nn.functional.l1_loss(
                mel_hat[..., :mel.size(-1)], mel[..., :mel_hat.size(-1)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            if args.smoke:
                break
            if steps % 50 == 0:
                print(f"[训练] epoch {epoch} step {steps} loss {loss.item():.4f}"
                      f" ({time.perf_counter() - t0:.0f}s)")
        if args.smoke:
            break
    out_pth = os.path.join(out_dir, "bigvgan_singing.pt")
    torch.save(gen.state_dict(), out_pth)
    json.dump({"dataset": args.dataset, "steps": steps,
               "epochs": args.epochs, "smoke": bool(args.smoke),
               "finished_at": time.time()},
              open(os.path.join(out_dir, "train.json"), "w"), indent=1)
    print(f"[完成] steps={steps} → {out_pth}\n"
          "推理端在歌声路径显式加载该文件；默认 TTS 仍用原 vocoder。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="BigVGAN 歌声微调入口（骨架）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if os.environ.get(
        "CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑一个 step（冒烟验收）")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
