# -*- coding: utf-8 -*-
"""评测指标：文本生成指标 + 分类指标 + bootstrap 置信区间。

所有指标一次性从预测结果计算并落盘，避免"表里的数字来自不同次运行"。
"""
import numpy as np


def corpus_bleu_scores(refs, hyps):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    ref_tok = [[r.split()] for r in refs]
    hyp_tok = [h.split() for h in hyps]
    out = {}
    for n in (1, 2, 3, 4):
        w = tuple([1.0 / n] * n + [0.0] * (4 - n))
        out[f"BLEU-{n}"] = float(corpus_bleu(ref_tok, hyp_tok, weights=w,
                                             smoothing_function=smooth))
    return out


def rouge_l(refs, hyps):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]))


def meteor(refs, hyps):
    from nltk.translate.meteor_score import meteor_score
    vals = []
    for r, h in zip(refs, hyps):
        try:
            vals.append(meteor_score([r.split()], h.split()))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def bertscore(refs, hyps, device="cpu"):
    from bert_score import BERTScorer
    _, _, f1 = BERTScorer(lang="en", rescale_with_baseline=True, device=device).score(hyps, refs)
    return float(f1.mean())


def cls_metrics(labels, probs, threshold=0.5):
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, \
        accuracy_score, confusion_matrix
    labels = np.asarray(labels).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    preds = (probs >= threshold).astype(int)
    out = {}
    out["AUC"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    out["Precision"] = float(precision_score(labels, preds, zero_division=0))
    out["Recall"] = float(recall_score(labels, preds, zero_division=0))
    out["F1"] = float(f1_score(labels, preds, zero_division=0))
    out["Accuracy"] = float(accuracy_score(labels, preds))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    out["n"] = int(len(labels))
    out["n_pos"] = int(labels.sum())
    out["TP"], out["FP"], out["TN"], out["FN"] = int(tp), int(fp), int(tn), int(fn)
    return out


def bootstrap_ci(labels, probs, refs, hyps, n_boot=1000, seed=42, alpha=0.05):
    """对 AUC 与 BLEU-4 做样本级 bootstrap，给出 95% 置信区间。"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(int).ravel()
    probs = np.asarray(probs).astype(float).ravel()
    n = len(labels)
    auc_samples, bleu_samples = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        auc_samples.append(roc_auc_score(labels[idx], probs[idx]))
        if refs is not None:
            sub_ref = [refs[i] for i in idx]
            sub_hyp = [hyps[i] for i in idx]
            bleu_samples.append(corpus_bleu_scores(sub_ref, sub_hyp)["BLEU-4"])
    out = {}
    if auc_samples:
        out["AUC_CI"] = [float(np.quantile(auc_samples, alpha / 2)),
                         float(np.quantile(auc_samples, 1 - alpha / 2))]
    if bleu_samples:
        out["BLEU-4_CI"] = [float(np.quantile(bleu_samples, alpha / 2)),
                            float(np.quantile(bleu_samples, 1 - alpha / 2))]
    return out


def evaluate_predictions(refs, hyps, labels, probs, cfg=None):
    metrics = {}
    metrics.update(corpus_bleu_scores(refs, hyps))
    metrics["ROUGE-L"] = rouge_l(refs, hyps)
    metrics["METEOR"] = meteor(refs, hyps)
    metrics.update(cls_metrics(labels, probs))
    if cfg is not None and getattr(cfg.evalc, "bertscore", False):
        try:
            metrics["BERTScore"] = bertscore(refs, hyps)
        except Exception as e:
            metrics["BERTScore"] = float("nan")
            print(f"[metrics] BERTScore 跳过：{e}")
    n_boot = cfg.evalc.bootstrap if cfg is not None else 1000
    metrics.update(bootstrap_ci(labels, probs, refs, hyps, n_boot=n_boot))
    return metrics
