# -*- coding: utf-8 -*-
"""论文图表生成（全部由模型/评测输出真实产生，不含任何硬编码数值或合成热力图）。

用法：
    python training/make_figures.py --ckpt results/ours/ours_best.pth \
        --vqvae-ckpt checkpoints/vqvae_k4096_best.pth

输出（figures/）：
    fig3_2_reconstruction.png       VQ-VAE 重构对比
    fig3_4_codebook_ablation.png    K 值消融曲线（读取 stage1 历史）
    fig4_4_roc.png                  ROC 曲线（读取 predictions.csv）
    fig4_5_attention.png            真实交叉注意力热力图
    fig4_2_training_curves.png      训练/验证曲线
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, get_device                               # noqa: E402
from attention import cross_attention, classifier_attention, tokens_to_grid  # noqa: E402
from dataset import build_tokenizer, IndianaCXRDataset           # noqa: E402
from model import build_variant                                  # noqa: E402
from vqvae import VQVAE                                          # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimSun", "Songti SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300


def _img_transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.5] * 3, [0.5] * 3)])


def fig_reconstruction(vqvae, image_path, out, device):
    img = Image.open(image_path).convert("RGB")
    x = _img_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, recon, _, ppl = vqvae(x)
    orig = (x[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    rec = (recon[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    mse = float(F.mse_loss(recon, x))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(orig); axes[0].set_title("原始胸片")
    axes[1].imshow(rec); axes[1].set_title("VQ-VAE 重构")
    axes[2].imshow(np.abs(orig - rec).mean(-1), cmap="magma")
    axes[2].set_title(f"误差图 (MSE={mse:.4f})")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return mse


def fig_roc(pred_csv, out):
    from sklearn.metrics import roc_curve, auc
    df = pd.read_csv(pred_csv)
    fpr, tpr, _ = roc_curve(df["label"], df["prob"])
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, color="#b71c1c", lw=1.8, label=f"心脏增大 (AUC={a:.4f}, n={len(df)})")
    ax.plot([0, 1], [0, 1], "--", color="#757575", lw=1)
    ax.set_xlabel("假阳性率"); ax.set_ylabel("真阳性率")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(ls=":", alpha=.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return a


def fig_training_curves(hist_csv, out):
    df = pd.read_csv(hist_csv)
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(df["epoch"], df["train_txt"], label="训练文本损失")
    ax1.plot(df["epoch"], df["val_txt"], label="验证文本损失")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("交叉熵损失")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_auc"], ls="--", color="#b71c1c", label="验证 AUC")
    ax2.set_ylabel("AUC")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_codebook(root, out):
    hist = sorted(Path(root).glob("k*_history.csv"))
    if not hist:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for p in hist:
        k = p.stem.split("_")[0]
        df = pd.read_csv(p)
        axes[0].plot(df["epoch"], df["val_mse"], label=f"K={k[1:]}")
        axes[1].plot(df["epoch"], df["train_ppl"], label=f"K={k[1:]}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("验证重构 MSE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("码本困惑度")
    for a in axes:
        a.legend(fontsize=7, frameon=False); a.grid(ls=":", alpha=.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_attention(model, tokenizer, dataset, out, device, n_cases=2):
    """真实交叉注意力：取参考报告中出现 heart/cardiac 的位置，计算预测该词时的注意力。"""
    model.eval()
    words = ["heart", "cardiac", "enlarged"]
    cand = dataset.data[dataset.data["label"] == 1]
    picked = []
    for _, row in cand.head(n_cases).iterrows():
        ids = tokenizer(row["report"], return_tensors="pt")["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids[0])
        for w in words:
            sub = [i for i, t in enumerate(toks) if w in t]
            if sub:
                picked.append((row["filename"], sub[0], w, row["report"]))
                break
    if not picked:
        print("[fig] 未找到含 heart/cardiac 的阳性报告，跳过注意力图")
        return None

    fig, axes = plt.subplots(1, len(picked), figsize=(4 * len(picked), 4))
    if len(picked) == 1:
        axes = [axes]
    for ax, (fname, pos, word, report) in zip(axes, picked):
        img = Image.open(Path(CFG.paths.images) / fname).convert("RGB")
        x = _img_transform()(img).unsqueeze(0).to(device)
        ids = tokenizer(report, return_tensors="pt")["input_ids"].to(device)
        prefix = ids[:, :pos]                      # 因果解码：位置 pos 的 query 预测第 pos+1 个词
        if prefix.size(1) == 0:
            prefix = ids[:, :1]
        attn, prob = cross_attention(model, x, prefix)
        grid = tokens_to_grid(attn[-1], grid=32)
        ax.imshow(img, cmap="gray")
        ax.imshow(grid, cmap="jet", alpha=0.45)
        ax.set_title(f"'{word}' 注意力（p={prob:.3f}）", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(CFG.paths.results_dir) / "ours" / "ours_best.pth"))
    ap.add_argument("--vqvae-ckpt",
                    default=str(Path(CFG.paths.ckpt_dir) / "vqvae_k4096_best.pth"))
    args = ap.parse_args()

    cfg = CFG
    cfg.ensure_dirs()
    device = get_device()
    fig_dir = Path(cfg.paths.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(cfg)

    ds = IndianaCXRDataset(cfg.paths.images, cfg.paths.projections_csv, cfg.paths.reports_csv,
                           tokenizer, max_seq_len=cfg.data.max_seq_len)
    pos = ds.data[ds.data["label"] == 1]
    if len(pos):
        img_path = Path(cfg.paths.images) / pos.iloc[0]["filename"]
    else:
        img_path = Path(cfg.paths.images) / ds.data.iloc[0]["filename"]

    # 1) 重构对比
    if Path(args.vqvae_ckpt).exists():
        ck = torch.load(args.vqvae_ckpt, map_location=device)
        sd = ck.get("state_dict", ck)
        vq = VQVAE(hidden_dim=256, num_embeddings=ck.get("k", 4096)).to(device)
        vq.load_state_dict(sd, strict=False)
        vq.eval()
        mse = fig_reconstruction(vq, img_path, fig_dir / "fig3_2_reconstruction.png", device)
        print(f"[fig] 重构图，MSE={mse:.4f}")

    # 2) ROC
    pred = Path(cfg.paths.results_dir) / "ours" / "predictions.csv"
    if pred.exists():
        a = fig_roc(pred, fig_dir / "fig4_4_roc.png")
        print(f"[fig] ROC，AUC={a:.4f}")

    # 3) 训练曲线
    hist = Path(cfg.paths.results_dir) / "ours" / "ours_history.csv"
    if hist.exists():
        fig_training_curves(hist, fig_dir / "fig4_2_training_curves.png")
        print("[fig] 训练曲线")

    # 4) K 消融
    fig_codebook(Path(cfg.paths.results_dir) / "stage1", fig_dir / "fig3_4_codebook_ablation.png")

    # 5) 真实注意力
    if Path(args.ckpt).exists():
        ck = torch.load(args.ckpt, map_location=device)
        sd = ck.get("state_dict", ck)
        model = build_variant("full", tokenizer.vocab_size).to(device)
        model.load_state_dict(sd, strict=False)
        out = fig_attention(model, tokenizer, ds, fig_dir / "fig4_5_attention.png", device)
        if out:
            print(f"[fig] 注意力图 -> {out}")

    print(f"[fig] 输出目录：{fig_dir}")


if __name__ == "__main__":
    main()
