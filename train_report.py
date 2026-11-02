# -*- coding: utf-8 -*-
"""阶段二：跨模态报告生成模型（Ours）训练。

用法：
    python training/train_report.py --vqvae-ckpt checkpoints/vqvae_k4096_best.pth --tag ours

注意：本脚本是主模型唯一的训练入口。原仓库里的 train_stage2.py（λ=1.0、冻结编码器）、
train_champion.py（λ=5.0、按 uid 分组、20 轮无验证）与 all.py（基线 λ=5.0）
三套超参互不一致，无法说明 best_full_model.pth 究竟由哪一套产生，已废弃。
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, get_device, set_seed                    # noqa: E402
from data_utils import build_loaders                             # noqa: E402
from dataset import build_tokenizer                              # noqa: E402
from evaluate import run_eval                                    # noqa: E402
from model import build_variant                                  # noqa: E402
from trainer import train_one                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae-ckpt", default=None)
    ap.add_argument("--tag", default="ours")
    ap.add_argument("--epochs", type=int, default=CFG.report.epochs)
    ap.add_argument("--cls-weight", type=float, default=CFG.report.cls_weight)
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="复现旧实现中冻结编码器的设置（默认不冻结）")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    cfg = CFG
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    tokenizer = build_tokenizer(cfg)
    loaders, _, _, meta = build_loaders(cfg, tokenizer)
    print(f"[ours] 划分：{meta}")

    model = build_variant("full", tokenizer.vocab_size, pad_idx=tokenizer.pad_token_id)
    if args.vqvae_ckpt:
        ckpt = torch.load(args.vqvae_ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.vq_vae.load_state_dict(sd, strict=False)
        print(f"[ours] 载入视觉字典 {args.vqvae_ckpt} | 未匹配 {missing} | 多余 {unexpected}")

    out_dir = Path(cfg.paths.results_dir) / args.tag
    model, best = train_one(model, loaders, tokenizer, cfg, out_dir, tag=args.tag,
                            epochs=args.epochs, cls_weight=args.cls_weight,
                            freeze_encoder=args.freeze_encoder)
    print(f"[ours] 最优验证文本损失 {best:.4f}")

    if not args.no_eval:
        run_eval(model, loaders["test"], tokenizer, cfg, args.tag, out_dir, device)


if __name__ == "__main__":
    main()
