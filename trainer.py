# -*- coding: utf-8 -*-
"""训练循环（主模型、基线、消融变体共用）。

统一约定：
- 损失 = 文本交叉熵 + lambda * 加权 BCE（lambda 取自 config.report.cls_weight，默认 5.0）
- 编码器参与联合优化（论文 2.3.1 / 4.3.2 声称分类梯度约束视觉编码器），
  学习率取 lr_encoder，其余参数取 lr_head；
- 每个 epoch 在验证集上计算总损失与分类指标，按总损失保存最优权重并早停。
"""
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import set_seed, get_device


def compute_pos_weight(loader):
    labels = []
    for batch in loader:
        labels.append(batch[2].view(-1))
    y = torch.cat(labels).numpy()
    pos = float((y > 0.5).sum())
    neg = float((y <= 0.5).sum())
    return neg / pos if pos > 0 else 1.0


def _param_groups(model, cfg):
    enc_ids = set()
    if hasattr(model, "vq_vae"):
        enc_ids = {id(p) for p in model.vq_vae.encoder.parameters()}
    head, enc = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (enc if id(p) in enc_ids else head).append(p)
    groups = [{"params": head, "lr": cfg.report.lr_head}]
    if enc:
        groups.append({"params": enc, "lr": cfg.report.lr_encoder})
    return groups


def _cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _eval_epoch(model, loader, tokenizer, cls_crit, device, amp):
    model.eval()
    total, n = 0.0, 0
    labels, probs = [], []
    for imgs, texts, labs in loader:
        imgs, texts, labs = imgs.to(device), texts.to(device), labs.to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(imgs, text_seq=texts[:, :-1], pad_idx=tokenizer.pad_token_id)
            txt = F.cross_entropy(out["report_logits"].reshape(-1, model.vocab_size),
                                  texts[:, 1:].reshape(-1), ignore_index=tokenizer.pad_token_id)
            cls = cls_crit(out["class_logits"], labs)
        total += float(txt) + float(cls) * 0.0     # 验证只看文本损失，分类单列
        n += 1
        labels.extend(labs.detach().cpu().numpy().ravel().tolist())
        probs.extend(torch.sigmoid(out["class_logits"]).detach().cpu().numpy().ravel().tolist())
    from metrics import cls_metrics
    m = cls_metrics(labels, probs)
    return total / max(1, n), m


def train_one(model, loaders, tokenizer, cfg, out_dir, tag="model", epochs=None,
              cls_weight=None, freeze_encoder=False, verbose=True):
    device = get_device()
    set_seed(cfg.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs or cfg.report.epochs
    lam = cfg.report.cls_weight if cls_weight is None else cls_weight

    if freeze_encoder and hasattr(model, "vq_vae"):
        for p in model.vq_vae.parameters():
            p.requires_grad = False

    groups = _param_groups(model, cfg)
    optimizer = AdamW(groups, weight_decay=cfg.report.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total_steps = epochs * steps_per_epoch
    scheduler = _cosine_with_warmup(optimizer, total_steps,
                                    int(total_steps * cfg.report.warmup_ratio))

    pos_weight = compute_pos_weight(loaders["train"])
    cls_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.report.amp and device.type == "cuda")

    hist_path = out_dir / f"{tag}_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_txt", "train_cls", "val_txt",
                    "val_auc", "val_recall", "val_precision", "lr", "sec"])

    best = float("inf"); best_state = None; bad = 0
    print(f"[train:{tag}] 样本 train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} | lambda={lam} | pos_weight={pos_weight:.2f} "
          f"| 编码器冻结={freeze_encoder}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = run_txt = run_cls = 0.0
        pbar = tqdm(loaders["train"], desc=f"{tag} {epoch}/{epochs}", disable=not verbose)
        for imgs, texts, labs in pbar:
            imgs, texts, labs = imgs.to(device), texts.to(device), labs.to(device)
            tgt_in, tgt_out = texts[:, :-1], texts[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.report.amp and device.type == "cuda"):
                out = model(imgs, text_seq=tgt_in, pad_idx=tokenizer.pad_token_id)
                loss_txt = F.cross_entropy(
                    out["report_logits"].reshape(-1, model.vocab_size), tgt_out.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                    label_smoothing=cfg.report.label_smoothing)
                loss_cls = cls_crit(out["class_logits"], labs)
                loss = loss_txt + lam * loss_cls
            scaler.scale(loss).backward()
            if cfg.report.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups
                                                for p in g["params"]], cfg.report.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += float(loss); run_txt += float(loss_txt); run_cls += float(loss_cls)
            pbar.set_postfix({"loss": f"{float(loss):.3f}"})
        n_b = max(1, len(loaders["train"]))
        val_txt, val_cls = _eval_epoch(model, loaders["val"], tokenizer, cls_crit, device,
                                       cfg.report.amp)
        lr_now = optimizer.param_groups[0]["lr"]
        sec = time.time() - t0
        print(f"[{tag}] epoch {epoch} | train {run_loss / n_b:.4f} "
              f"(txt {run_txt / n_b:.4f} + λ·cls {run_cls / n_b:.4f}) | "
              f"val_txt {val_txt:.4f} | val AUC {val_cls['AUC']:.4f} Rec {val_cls['Recall']:.3f}")
        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, run_loss / n_b, run_txt / n_b, run_cls / n_b,
                                    val_txt, val_cls["AUC"], val_cls["Recall"],
                                    val_cls["Precision"], lr_now, round(sec, 1)])

        score = val_txt + lam * 0.0    # 早停依据：验证文本损失
        if score < best:
            best = score; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": epoch, "val_txt": best,
                        "cls_weight": lam, "cfg": cfg.to_dict(), "tag": tag},
                       out_dir / f"{tag}_best.pth")
        else:
            bad += 1
            if bad >= cfg.report.patience:
                print(f"[{tag}] 早停于 epoch {epoch}（验证损失 {best:.4f} 未改善 {bad} 轮）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "tag": tag,
                "cfg": cfg.to_dict()}, out_dir / f"{tag}_last.pth")
    (out_dir / f"{tag}_train_meta.json").write_text(json.dumps(
        {"best_val_txt": best, "epochs_run": epoch, "cls_weight": lam,
         "pos_weight": pos_weight, "freeze_encoder": freeze_encoder,
         "params": sum(p.numel() for p in model.parameters())}, indent=1), encoding="utf-8")
    return model, best
