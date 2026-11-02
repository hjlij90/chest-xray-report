# -*- coding: utf-8 -*-
"""把 results/ 下的评测结果汇总成论文表格（Markdown + CSV），避免手工誊抄。

用法：
    python training/make_tables.py            # 读取 results 下所有 metrics.json
输出：
    results/tables/table_ablation.md|.csv    组件消融
    results/tables/table_comparison.md|.csv  横向对比
    results/tables/table_codebook.md|.csv    码本容量消融
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG                                          # noqa: E402

VARIANT_NAMES = {
    "full": "完整模型（Ours）",
    "no_pos_embed": "去除视觉位置编码",
    "no_cross_attn": "去除跨模态注意力",
    "no_cls": "去除分类约束",
    "no_vqvae": "去除向量量化（连续特征）",
}
BASELINE_NAMES = {
    "baseline_resnet50_lstm": "ResNet50+LSTM",
    "baseline_resnet50_transformer": "ResNet50+Transformer",
    "baseline_resnet101_transformer": "ResNet101+Transformer",
    "baseline_vit_transformer": "ViT-B/16+Transformer",
}


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt_ci(v, ci):
    if v is None or v != v:
        return "—"
    if ci:
        return f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    return f"{v:.4f}"


def main():
    cfg = CFG
    root = Path(cfg.paths.results_dir)
    tdir = root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # ---- 组件消融 ----
    rows = []
    for tag, name in VARIANT_NAMES.items():
        p = root / f"ablation_{tag}" / "metrics.json"
        if not p.exists():
            continue
        m = _load(p)
        rows.append({
            "变体": name,
            "PPL": round(m.get("val_ppl", float("nan")), 2) if m.get("val_ppl") == m.get("val_ppl") else "—",
            "AUC": fmt_ci(m.get("AUC"), m.get("AUC_CI")),
            "Recall": f"{m['Recall']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "BLEU-4": fmt_ci(m.get("BLEU-4"), m.get("BLEU-4_CI")),
            "ROUGE-L": f"{m['ROUGE-L']:.4f}",
            "METEOR": f"{m['METEOR']:.4f}",
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(tdir / "table_ablation.csv", index=False, encoding="utf-8")
        (tdir / "table_ablation.md").write_text(
            df.to_markdown(index=False), encoding="utf-8")
        print(df.to_string(index=False))

    # ---- 横向对比 ----
    rows = []
    ours = root / "ours" / "metrics.json"
    if ours.exists():
        m = _load(ours)
        rows.append({"模型": "Ours（VQ-VAE+Transformer）",
                     "AUC": fmt_ci(m.get("AUC"), m.get("AUC_CI")),
                     "Recall": f"{m['Recall']:.4f}", "Precision": f"{m['Precision']:.4f}",
                     "BLEU-4": fmt_ci(m.get("BLEU-4"), m.get("BLEU-4_CI")),
                     "ROUGE-L": f"{m['ROUGE-L']:.4f}", "METEOR": f"{m['METEOR']:.4f}"})
    for tag, name in BASELINE_NAMES.items():
        p = root / tag / "metrics.json"
        if not p.exists():
            continue
        m = _load(p)
        rows.append({"模型": name,
                     "AUC": fmt_ci(m.get("AUC"), m.get("AUC_CI")),
                     "Recall": f"{m['Recall']:.4f}", "Precision": f"{m['Precision']:.4f}",
                     "BLEU-4": fmt_ci(m.get("BLEU-4"), m.get("BLEU-4_CI")),
                     "ROUGE-L": f"{m['ROUGE-L']:.4f}", "METEOR": f"{m['METEOR']:.4f}"})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(tdir / "table_comparison.csv", index=False, encoding="utf-8")
        (tdir / "table_comparison.md").write_text(df.to_markdown(index=False), encoding="utf-8")
        print(df.to_string(index=False))

    # ---- 码本容量 ----
    rows = []
    for s in sorted((root / "stage1").glob("k*_summary.json")):
        m = _load(s)
        rows.append({"码本容量 K": m["K"], "验证 MSE": round(m["val_mse"], 5),
                     "有效码本数": m["used_codes"],
                     "利用率": f"{m['utilization']:.1%}"})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(tdir / "table_codebook.csv", index=False, encoding="utf-8")
        (tdir / "table_codebook.md").write_text(df.to_markdown(index=False), encoding="utf-8")
        print(df.to_string(index=False))

    print(f"[tables] 输出目录：{tdir}")


if __name__ == "__main__":
    main()
