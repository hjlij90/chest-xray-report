# -*- coding: utf-8 -*-
"""阶段一：VQ-VAE 视觉字典学习。

用法：
    python training/train_vqvae.py --k 4096 --epochs 100

产出：
    checkpoints/vqvae_k{K}_best.pth     最优权重（含配置与验证 MSE）
    results/stage1/k{K}_history.csv     逐轮重构损失、困惑度、码本利用率
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, get_device, set_seed                    # noqa: E402
from data_utils import build_loaders                             # noqa: E402
from dataset import build_tokenizer                              # noqa: E402
from vqvae import VQVAE                                          # noqa: E402


def histogram_ppl(quantizer):
    """按统计窗口（一轮训练或一轮验证）内的码字命中频率估计困惑度（论文式 3-8）。"""
    usage = quantizer.epoch_usage.float()
    if usage.sum() <= 0:
        return 0.0
    p = usage / usage.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


@torch.no_grad()
def eval_recon(model, loader, device, amp):
    model.eval()
    model.quantizer.reset_epoch_usage()
    total, ppl = 0.0, 0.0
    for batch in loader:
        imgs = batch[0].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            _, recon, _, p = model(imgs)
            total += float(F.mse_loss(recon.float(), imgs.float()))
        ppl += float(p)
    n = max(1, len(loader))
    return total / n, ppl / n, histogram_ppl(model.quantizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=CFG.vqvae.num_embeddings)
    ap.add_argument("--epochs", type=int, default=CFG.vqvae.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.vqvae.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.vqvae.lr)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = CFG
    cfg.vqvae.num_embeddings = args.k
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    print(f"[stage1] K={args.k} device={device}")

    tokenizer = build_tokenizer(cfg)
    loaders, _, _, meta = build_loaders(cfg, tokenizer, batch_size=args.batch_size)
    print(f"[stage1] 数据划分：{meta}")

    model = VQVAE(in_channels=3, hidden_dim=cfg.vqvae.hidden_dim,
                  num_embeddings=args.k,
                  commitment_cost=cfg.vqvae.commitment_cost).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=cfg.vqvae.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.vqvae.amp and device.type == "cuda")

    out_dir = Path(args.out_dir or (Path(cfg.paths.results_dir) / "stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = out_dir / f"k{args.k}_history.csv"
    with open(hist, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_mse", "train_vq", "train_ppl_batch",
                                "train_ppl_window", "val_mse", "val_ppl_batch",
                                "val_ppl_window", "dead_ratio", "sec"])

    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.reset_quantizer_usage()
        t0 = time.time()
        run_mse = run_vq = run_ppl = 0.0
        for batch in tqdm(loaders["train"], desc=f"K{args.k} {epoch}/{args.epochs}", leave=False):
            imgs = batch[0].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.vqvae.amp and device.type == "cuda"):
                z_q, recon, vq_loss, ppl = model(imgs)
                mse = F.mse_loss(recon.float(), imgs.float())
                loss = mse + vq_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_mse += float(mse); run_vq += float(vq_loss); run_ppl += float(ppl)

        train_window_ppl = histogram_ppl(model.quantizer)
        revived = model.quantizer.revive_dead_codes()
        dead_ratio = revived / args.k
        val_mse, val_ppl, val_window_ppl = eval_recon(model, loaders["val"], device, cfg.vqvae.amp)
        n_b = max(1, len(loaders["train"]))
        print(f"[stage1] epoch {epoch} | mse {run_mse / n_b:.4f} | ppl {run_ppl / n_b:.1f} "
              f"| ppl(window) {train_window_ppl:.1f} | val mse {val_mse:.4f} "
              f"| val ppl(window) {val_window_ppl:.1f} | 激活死码 {revived} ({dead_ratio:.1%})")
        with open(hist, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, run_mse / n_b, run_vq / n_b, run_ppl / n_b,
                                    train_window_ppl, val_mse, val_ppl, val_window_ppl,
                                    dead_ratio, round(time.time() - t0, 1)])
        if val_mse < best:
            best = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt = Path(cfg.paths.ckpt_dir) / f"vqvae_k{args.k}_best.pth"
    torch.save({"state_dict": best_state, "k": args.k, "val_mse": best,
                "cfg": cfg.to_dict(), "split_meta": meta}, ckpt)
    used, seen = model.codebook_usage(loaders["val"], device)
    util = used / args.k
    print(f"[stage1] 保存 {ckpt} | 验证 MSE {best:.4f} | 码本利用率 {used}/{args.k} = {util:.1%}")
    (out_dir / f"k{args.k}_summary.json").write_text(json.dumps(
        {"K": args.k, "val_mse": best, "used_codes": used, "utilization": util,
         "n_eval_images": seen, "epochs": args.epochs}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
