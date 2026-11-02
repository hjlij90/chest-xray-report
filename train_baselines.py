# -*- coding: utf-8 -*-
"""基线模型训练与评测（与主模型共用同一患者级划分、同一 lambda、同一解码设置）。

用法：
    python training/train_baselines.py --backbones resnet50_lstm resnet50_transformer vit_transformer
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, get_device, set_seed                    # noqa: E402
from data_utils import build_loaders                             # noqa: E402
from dataset import build_tokenizer                              # noqa: E402
from evaluate import run_eval                                    # noqa: E402
from model import BaselineResNetLSTM, BaselineResNetTransformer, BaselineViTTransformer  # noqa: E402
from trainer import train_one                                    # noqa: E402

REGISTRY = {
    "resnet50_lstm": ("ResNet50+LSTM", BaselineResNetLSTM, {}),
    "resnet50_transformer": ("ResNet50+Transformer", BaselineResNetTransformer, {"backbone": "resnet50"}),
    "resnet101_transformer": ("ResNet101+Transformer", BaselineResNetTransformer, {"backbone": "resnet101"}),
    "vit_transformer": ("ViT-B/16+Transformer", BaselineViTTransformer, {}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(REGISTRY))
    ap.add_argument("--epochs", type=int, default=CFG.report.epochs)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    cfg = CFG
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    tokenizer = build_tokenizer(cfg)
    loaders, _, _, meta = build_loaders(cfg, tokenizer)
    print(f"[baseline] 划分：{meta}")

    rows = []
    for key in args.backbones:
        name, cls, kwargs = REGISTRY[key]
        kwargs = dict(kwargs)
        if "pretrained" in cls.__init__.__code__.co_varnames:
            kwargs["pretrained"] = not args.no_pretrained
        model = cls(vocab_size=tokenizer.vocab_size, num_classes=1,
                    pad_idx=tokenizer.pad_token_id, **kwargs)
        out_dir = Path(cfg.paths.results_dir) / f"baseline_{key}"
        model, best = train_one(model, loaders, tokenizer, cfg, out_dir, tag=f"baseline_{key}",
                                epochs=args.epochs, cls_weight=cfg.report.cls_weight)
        metrics, _ = run_eval(model, loaders["test"], tokenizer, cfg, f"baseline_{key}",
                              out_dir, device)
        metrics["model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    out = Path(cfg.paths.results_dir) / "baseline_metrics.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    (Path(cfg.paths.results_dir) / "baseline_metrics.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(df[["model", "AUC", "Recall", "Precision", "BLEU-4", "ROUGE-L", "METEOR"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
