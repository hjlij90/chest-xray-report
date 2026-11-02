# -*- coding: utf-8 -*-
"""测试集评测：统一生成、统一指标、逐样本落盘。

用法（单独评估某个权重）：
    python training/evaluate.py --ckpt results/ours/ours_best.pth --variant full --tag ours
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, get_device, set_seed                    # noqa: E402
from data_utils import build_loaders                             # noqa: E402
from dataset import build_tokenizer                              # noqa: E402
from generate import beam_search_decode, decode_batch            # noqa: E402
from metrics import evaluate_predictions                         # noqa: E402
from model import build_variant                                  # noqa: E402


@torch.no_grad()
def run_eval(model, loader, tokenizer, cfg, tag, out_dir, device, beam=True):
    model.eval()
    refs, hyps, labels, probs = [], [], [], []
    for batch in tqdm(loader, desc=f"eval:{tag}"):
        imgs, texts, labs = batch[0].to(device), batch[1], batch[2]
        if beam:
            ids, prob = beam_search_decode(
                model, tokenizer, imgs,
                beam_size=cfg.report.beam_size,
                length_penalty=cfg.report.length_penalty,
                repetition_penalty=cfg.report.repetition_penalty,
                max_new_tokens=cfg.report.max_new_tokens)
        else:
            from generate import greedy_decode
            ids, prob = greedy_decode(model, tokenizer, imgs,
                                      max_new_tokens=cfg.report.max_new_tokens)
        refs.extend(decode_batch(tokenizer, texts))
        hyps.extend(decode_batch(tokenizer, ids))
        labels.extend(labs.cpu().numpy().ravel().tolist())
        probs.extend(prob.detach().cpu().numpy().ravel().tolist())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"reference": refs, "generated": hyps, "label": labels, "prob": probs}) \
        .to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    metrics = evaluate_predictions(refs, hyps, labels, probs, cfg)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"[eval:{tag}] n={len(refs)} 阳性={int(sum(labels))} | "
          f"AUC {metrics.get('AUC', float('nan')):.4f} Rec {metrics['Recall']:.3f} "
          f"Prec {metrics['Precision']:.3f} | BLEU-4 {metrics['BLEU-4']:.4f} "
          f"ROUGE-L {metrics['ROUGE-L']:.4f} METEOR {metrics['METEOR']:.4f}")
    return metrics, pd.DataFrame({"reference": refs, "generated": hyps,
                                  "label": labels, "prob": probs})


def load_checkpoint(path, variant, vocab_size, device):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = build_variant(variant, vocab_size)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[eval] 未加载参数：{missing}")
    if unexpected:
        print(f"[eval] 多余参数：{unexpected}")
    return model.to(device), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="full",
                    choices=["full", "no_pos_embed", "no_cross_attn", "no_vqvae", "no_cls"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    cfg = CFG
    set_seed(cfg.seed)
    device = get_device()
    tokenizer = build_tokenizer(cfg)
    loaders, _, _, _ = build_loaders(cfg, tokenizer)
    model, _ = load_checkpoint(args.ckpt, args.variant, tokenizer.vocab_size, device)
    tag = args.tag or Path(args.ckpt).stem
    run_eval(model, loaders["test"], tokenizer, cfg, tag,
             Path(cfg.paths.results_dir) / tag, device, beam=not args.greedy)


if __name__ == "__main__":
    main()
