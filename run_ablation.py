# -*- coding: utf-8 -*-
"""组件消融：每个变体只改变一个因素，其余训练设置完全一致。

用法：
    python training/run_ablation.py --variants full no_pos_embed no_cross_attn no_cls no_vqvae \
        --vqvae-ckpt checkpoints/vqvae_k4096_best.pth
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, get_device, set_seed                    # noqa: E402
from data_utils import build_loaders                             # noqa: E402
from dataset import build_tokenizer                              # noqa: E402
from evaluate import run_eval                                    # noqa: E402
from model import build_variant                                  # noqa: E402
from trainer import train_one                                    # noqa: E402


def load_visual_init(model, ckpt_path, device):
    """所有变体共用同一份视觉字典初始化，确保消融只反映结构差异。"""
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    if not hasattr(model, "vq_vae"):
        return
    missing, unexpected = model.vq_vae.load_state_dict(sd, strict=False)
    print(f"[ablation] 视觉字典初始化：未匹配 {len(missing)} 项，多余 {len(unexpected)} 项")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["full", "no_pos_embed", "no_cross_attn", "no_cls", "no_vqvae"])
    ap.add_argument("--vqvae-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=CFG.report.epochs)
    args = ap.parse_args()

    cfg = CFG
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    tokenizer = build_tokenizer(cfg)
    loaders, _, _, meta = build_loaders(cfg, tokenizer)

    rows = []
    for variant in args.variants:
        set_seed(cfg.seed)
        model = build_variant(variant, tokenizer.vocab_size, pad_idx=tokenizer.pad_token_id)
        load_visual_init(model, args.vqvae_ckpt, device)
        tag = f"ablation_{variant}"
        out_dir = Path(cfg.paths.results_dir) / tag
        _, best = train_one(model, loaders, tokenizer, cfg, out_dir, tag=tag,
                            epochs=args.epochs, cls_weight=cfg.report.cls_weight)
        metrics, _ = run_eval(model, loaders["test"], tokenizer, cfg, tag, out_dir, device)
        # 测量验证集上的码本困惑度（无量化变体记为 NaN）
        ppl = float("nan")
        if variant != "no_vqvae":
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in loaders["val"]:
                    vals.append(float(model.forward(batch[0].to(device))["ppl"]))
            ppl = float(sum(vals) / max(1, len(vals)))
        metrics.update({"variant": variant, "val_ppl": ppl, "best_val_txt": best})
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg.paths.results_dir) / "ablation_metrics.csv", index=False, encoding="utf-8")
    (Path(cfg.paths.results_dir) / "ablation_metrics.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = ["variant", "val_ppl", "AUC", "Recall", "Precision", "BLEU-4", "ROUGE-L", "METEOR"]
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
