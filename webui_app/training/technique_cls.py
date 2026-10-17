"""technique_cls — 歌唱技巧分类器（E1-T Step 3）。

用途：
1. 推理期 auto 模式：对源干声逐帧分类 technique，喂给 CFM 条件
   （懒加载、用完释放——不常驻显存，母规划 Step 3.3）；
2. 评测期：量化「源技巧是否在转换后丢失」（对比源/输出的 tech 分布）。

模型：mel (80,T) → 3 层 CNN + 时间池化 → 7 类（词表与
indextts/s2mel/modules/conditioners.py 一致）。输入是逐**窗**分类
（窗长 win_frames，默认 ~1s），窗内取多数为帧级标签的平滑近似——
GTSinger 的技巧段普遍 ≥1s，更细的粒度对条件化没有增益（调研 Q6：
vibrato 的载体是连续 F0，不需要分类器逐帧跟踪）。

训练数据：singing 特征数据集（features.py 提取的 mel+tech）。
保存：单文件 .pt（结构 dict：state + 配置），推理端 lazy 加载。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_WIN_SEC = 1.0
MEL_FPS = 22050.0 / 256.0
NUM_CLASSES = 7


class TechniqueNet:
    """轻量 CNN 分类头（mel 幅度对数域输入，(B,1,80,W) → (B,7)）。"""

    def __init__(self, num_classes: int = NUM_CLASSES):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()


def mel_windows(mel: np.ndarray, win_frames: int, hop: Optional[int] = None
                ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """mel (80,T) → 窗批 (N,1,80,W) + 每窗的帧区间 [(k0,k1)]。"""
    hop = hop or max(1, win_frames // 2)
    T = mel.shape[1]
    if T <= win_frames:
        return mel[None, None, :, :], [(0, T)]
    outs, spans = [], []
    for k0 in range(0, T - win_frames + 1, hop):
        outs.append(mel[:, k0:k0 + win_frames])
        spans.append((k0, k0 + win_frames))
    return np.stack(outs)[:, None, :, :], spans


def train(dataset: str, out_path: str, epochs: int = 3, lr: float = 1e-3,
          device: str = "cpu", batch: int = 16,
          progress=None) -> Dict:
    """在 singing 特征数据集上训练分类器。

    每窗的标签 = 窗内**非 control 帧的多数类**（无则 control）——
    技巧段短于半窗时会被稀释，这是有意的保守：宁可漏检不可误标。
    """
    import torch
    import torch.nn.functional as F

    from webui_app.training import dataset as DS

    ds_dir = DS.dir_of(dataset)
    feat_dir = os.path.join(ds_dir, "features")
    items = [u for u in DS.load_meta(dataset) if u.has_features]
    X, Y = [], []
    win = int(DEFAULT_WIN_SEC * MEL_FPS)
    for u in items:
        p = os.path.join(feat_dir, f"{u.id}.pt")
        if not os.path.isfile(p):
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "tech" not in d:
            continue
        mel = d["mel"].float().numpy()
        tech = d["tech"].numpy()
        wins, spans = mel_windows(mel, win)
        for w, (k0, k1) in zip(wins, spans):
            seg = tech[k0:k1]
            nz = seg[seg > 0]
            label = int(np.bincount(nz).argmax()) if len(nz) > len(seg) * 0.2 else 0
            X.append(w)
            Y.append(label)
    if not X:
        return {"ok": False, "error": "没有带 tech 标签的样本（需 singing 特征）"}

    X = np.stack(X)
    Y = np.array(Y, dtype=np.int64)
    # 类别均衡权重（falsetto/mixed 少数派）
    counts = np.bincount(Y, minlength=NUM_CLASSES).astype(np.float64)
    w = (1.0 / np.maximum(counts, 1.0))
    w = w / w.sum() * NUM_CLASSES
    cls_w = torch.tensor(w, dtype=torch.float32, device=device)

    net = TechniqueNet().net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = np.random.RandomState(42 + ep).permutation(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.tensor(X[idx], dtype=torch.float32, device=device)
            # log 压缩稳定动态范围（mel 幅度域）
            xb = torch.log(xb.clamp(min=1e-5))
            yb = torch.tensor(Y[idx], device=device)
            loss = F.cross_entropy(net(xb), yb, weight=cls_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        if progress:
            progress((ep + 1) / epochs, f"epoch {ep + 1}/{epochs} loss {tot / max(nb, 1):.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save({"state": net.state_dict(),
                "win_frames": win,
                "num_classes": NUM_CLASSES}, out_path)
    # 简要类别统计
    return {"ok": True, "n_windows": n, "epochs": epochs,
            "class_counts": counts.astype(int).tolist(), "out": out_path}


class TechniqueClassifier:
    """推理端包装：懒加载 → predict(audio 22k mono) → 帧级 tech + 置信度。"""

    _cache: Dict[str, "TechniqueClassifier"] = {}

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        import torch
        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.win = int(ck.get("win_frames") or int(DEFAULT_WIN_SEC * MEL_FPS))
        self.net = TechniqueNet(int(ck.get("num_classes", NUM_CLASSES))).net
        self.net.load_state_dict(ck["state"])
        self.net.eval().to(device)
        self.device = device
        self.path = ckpt_path

    @classmethod
    def get(cls, ckpt_path: str, device: str = "cpu") -> "TechniqueClassifier":
        key = f"{ckpt_path}|{device}"
        if key not in cls._cache:
            cls._cache.clear()          # 只留一份（小模型，但别累积）
            cls._cache[key] = cls(ckpt_path, device)
        return cls._cache[key]

    def release(self):
        TechniqueClassifier._cache.clear()

    # ---- 特征 ----
    def _mel_of(self, wav: np.ndarray) -> np.ndarray:
        import torchaudio
        m = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024,
            n_mels=80, f_min=0.0)
        mel = m(self.torch.tensor(np.asarray(wav, dtype=np.float32))
                .unsqueeze(0)).squeeze(0).numpy()
        return mel

    def predict(self, wav: np.ndarray, total_frames: Optional[int] = None
                ) -> Tuple[np.ndarray, float]:
        """→ (帧级 tech (T,), 平均置信度)。total_frames 与目标 mel 帧对齐。"""
        torch = self.torch
        mel = self._mel_of(wav)
        wins, spans = mel_windows(mel, self.win)
        with torch.no_grad():
            xb = torch.tensor(wins, dtype=torch.float32, device=self.device)
            xb = torch.log(xb.clamp(min=1e-5))
            prob = torch.softmax(self.net(xb), dim=-1).cpu().numpy()  # (N,7)
        T = total_frames if total_frames is not None else mel.shape[1]
        out = np.zeros(T, dtype=np.uint8)
        votes = np.zeros((T, NUM_CLASSES), dtype=np.float32)
        for pr, (k0, k1) in zip(prob, spans):
            k0c, k1c = min(k0, T), min(k1, T)
            if k1c > k0c:
                votes[k0c:k1c] += pr
        out[:] = votes.argmax(axis=1).astype(np.uint8)
        conf = float(np.mean(np.max(votes, axis=1) /
                             np.maximum(votes.sum(axis=1), 1e-6))) if T else 0.0
        return out, conf
