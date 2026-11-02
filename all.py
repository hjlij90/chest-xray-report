import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import BERTScorer

from dataset import IndianaCXRDataset
from model import (
    MedicalReportSystem,
    BaselineResNetTransformer,
    MedicalReportSystem_NoPosEmbed,
    MedicalReportSystem_NoCrossAttn,
    BaselineResNetLSTM,
    BaselineViTTransformer
)

# ------------------------ 辅助函数 ------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_patient_level_split(tokenizer, save_dir='./splits', seed=42):
    """严格按患者划分数据集 (train/val/test = 80/10/10)"""
    os.makedirs(save_dir, exist_ok=True)
    split_file = os.path.join(save_dir, f'split_seed{seed}.npy')
    if os.path.exists(split_file):
        indices = np.load(split_file, allow_pickle=True).item()
        dataset = IndianaCXRDataset(
            images_dir=r"E:\程序\datasets\images_256x256",
            projections_csv=r"E:\程序\datasets\indiana_projections.csv",
            reports_csv=r"E:\程序\datasets\indiana_reports.csv",
            tokenizer=tokenizer,
            max_seq_len=64,
            use_extreme_aug=False,
            return_patient_id=True
        )
        return dataset, indices['train_idx'], indices['val_idx'], indices['test_idx']
    else:
        dataset = IndianaCXRDataset(
            images_dir=r"E:\程序\datasets\images_256x256",
            projections_csv=r"E:\程序\datasets\indiana_projections.csv",
            reports_csv=r"E:\程序\datasets\indiana_reports.csv",
            tokenizer=tokenizer,
            max_seq_len=64,
            use_extreme_aug=False,
            return_patient_id=True
        )
        patient_ids = dataset.patient_ids
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, temp_idx = next(gss.split(dataset.data, groups=patient_ids))
        temp_patient_ids = patient_ids[temp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_rel, test_rel = next(gss2.split(dataset.data.iloc[temp_idx], groups=temp_patient_ids))
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]
        indices = {'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}
        np.save(split_file, indices)
        return dataset, train_idx, val_idx, test_idx

def compute_pos_weight(train_dataset):
    labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    pos = sum(labels)
    neg = len(labels) - pos
    return neg / pos if pos > 0 else 1.0


@torch.no_grad()
def generate_transformer_baseline(model, images, tokenizer, max_len=64):
    """ResNet101+Transformer 生成函数，修复禁止特殊token的错误"""
    device = images.device
    bs = images.size(0)
    # 编码图像
    cnn_feat = model.cnn_encoder(images)
    proj_feat = model.cnn_proj(cnn_feat)
    B, C, H, W = proj_feat.size()
    vis = proj_feat.view(B, C, -1).permute(0, 2, 1)          # [B, L, D]
    V_len = vis.size(1)
    if V_len != model.visual_pos_embed.size(1):
        pos = F.interpolate(model.visual_pos_embed.permute(0,2,1), size=V_len, mode='linear', align_corners=False).permute(0,2,1)
    else:
        pos = model.visual_pos_embed
    vis = vis + pos
    memory = vis
    # 分类头
    class_logits = model.classifier(memory)
    class_probs = torch.sigmoid(class_logits)
    start = tokenizer.cls_token_id
    end = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id
    generated = torch.full((bs, 1), start, dtype=torch.long, device=device)
    min_len = 5  # 最少生成5个token
    for step in range(max_len - 1):
        tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
        tgt_emb = model.word_embedding(generated)
        tgt_emb = model.text_pos_encoder(tgt_emb)
        out = model.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = model.fc_out(out[:, -1, :])
        # 只禁止 pad token（不再禁止所有特殊token）
        logits[:, pad] = -float('inf')
        # 轻微重复惩罚
        rep_penalty = 1.05
        for b in range(bs):
            for tok in set(generated[b].tolist()):
                if tok not in [start, end, pad]:
                    logits[b, tok] /= rep_penalty
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if step >= min_len and (next_token == end).all():
            break
    return generated, class_probs

@torch.no_grad()
def generate_lstm_baseline(model, images, tokenizer, max_len=64):
    """ResNet50+LSTM 生成函数"""
    device = images.device
    bs = images.size(0)
    img_feat = model.cnn_encoder(images)
    pooled = model.pool(img_feat).view(bs, -1)
    proj = model.cnn_proj(pooled)
    h0 = proj.unsqueeze(0).repeat(2, 1, 1)
    c0 = proj.unsqueeze(0).repeat(2, 1, 1)
    hidden = (h0, c0)
    start = tokenizer.cls_token_id
    end = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id
    generated = torch.full((bs, 1), start, dtype=torch.long, device=device)
    min_len = 5
    for step in range(max_len - 1):
        word_emb = model.word_embedding(generated[:, -1:])
        out, hidden = model.lstm_decoder(word_emb, hidden)
        logits = model.fc_out(out[:, -1, :])
        logits[:, pad] = -float('inf')
        rep_penalty = 1.05
        for b in range(bs):
            for tok in set(generated[b].tolist()):
                if tok not in [start, end, pad]:
                    logits[b, tok] /= rep_penalty
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if step >= min_len and (next_token == end).all():
            break
    class_logits = model.classifier(proj)
    class_probs = torch.sigmoid(class_logits)
    return generated, class_probs

# ------------------------ 评估函数 ------------------------
def evaluate_model(model, test_loader, tokenizer, device, model_name, is_ours=False):
    model.eval()
    all_labels, all_probs = [], []
    all_refs, all_hyps = [], []
    # 选择生成函数
    if is_ours:
        from evaluate import generate_report_beam_search
        generate_func = generate_report_beam_search
    elif isinstance(model, BaselineResNetLSTM):
        generate_func = generate_lstm_baseline
    elif isinstance(model, BaselineResNetTransformer):
        generate_func = generate_transformer_baseline
    else:
        raise ValueError("Unknown model type")
    with torch.no_grad():
        for imgs, texts, labels, _ in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            imgs = imgs.to(device)
            ids, probs = generate_func(model, imgs, tokenizer)
            pred = tokenizer.batch_decode(ids, skip_special_tokens=True)
            true = tokenizer.batch_decode(texts, skip_special_tokens=True)
            pred = [re.sub(r'\[unused\d+\]', '', t).strip() for t in pred]
            true = [re.sub(r'\[unused\d+\]', '', t).strip() for t in true]
            # 防止空字符串导致 BERTScore 异常
            pred = [p if len(p) > 0 else "normal report" for p in pred]
            all_hyps.extend(pred)
            all_refs.extend([[r] for r in true])
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    # 指标计算
    smooth = SmoothingFunction().method1
    ref_tok = [[[w for w in r[0].split()]] for r in all_refs]
    hyp_tok = [[w for w in h.split()] for h in all_hyps]
    bleu1 = corpus_bleu(ref_tok, hyp_tok, weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tok, hyp_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = np.mean([scorer.score(r[0], h)['rougeL'].fmeasure for r,h in zip(all_refs, all_hyps)])
    try:
        meteor = np.mean([meteor_score([r[0].split()], h.split()) for r,h in zip(all_refs, all_hyps)])
    except:
        meteor = 0.0
    bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True, device=device)
    _, _, f1 = bert_scorer.score(all_hyps, [r[0] for r in all_refs])
    bert_f1 = f1.mean().item()
    auc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) > 0.5).astype(int)
    rec = recall_score(all_labels, preds, zero_division=0)
    prec = precision_score(all_labels, preds, zero_division=0)
    f1_score_val = f1_score(all_labels, preds, zero_division=0)
    return {
        'BLEU-1': bleu1, 'BLEU-4': bleu4, 'ROUGE-L': rouge, 'METEOR': meteor, 'BERTScore': bert_f1,
        'AUC': auc, 'Precision': prec, 'Recall': rec, 'F1': f1_score_val
    }

# ------------------------ 训练单个基线模型 ------------------------
def train_baseline(model_class, model_name, train_dataset, val_loader, tokenizer, device, epochs=100, patience=15):
    set_seed(42)
    model = model_class(vocab_size=tokenizer.vocab_size, hidden_dim=256, num_classes=1).to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 与 Ours 一致
    # 动态正样本权重（与 Ours 一致）
    train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    pos_weight = (len(train_labels)-sum(train_labels)) / max(sum(train_labels), 1)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    criterion_txt = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    scaler = GradScaler('cuda')
    pad = tokenizer.pad_token_id

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{epochs}")
        for imgs, texts, labels, _ in pbar:
            imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
            tgt_in, tgt_out = texts[:, :-1], texts[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(imgs, text_seq=tgt_in, pad_idx=pad)
                loss_cls = criterion_cls(outputs['class_logits'], labels)
                logits = outputs['report_logits'].reshape(-1, tokenizer.vocab_size)
                loss_txt = criterion_txt(logits, tgt_out.reshape(-1))
                loss = loss_txt + 5.0 * loss_cls   # 与 Ours 一致
            scaler.scale(loss).backward()
            
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        avg_train = total_loss / len(train_loader)
        # 每 2 轮验证一次（与 Ours 一致）
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, texts, labels, _ in val_loader:
                    imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
                    tgt_in, tgt_out = texts[:, :-1], texts[:, 1:]
                    with autocast('cuda'):
                        outputs = model(imgs, text_seq=tgt_in, pad_idx=pad)
                        loss_cls = criterion_cls(outputs['class_logits'], labels)
                        logits = outputs['report_logits'].reshape(-1, tokenizer.vocab_size)
                        loss_txt = criterion_txt(logits, tgt_out.reshape(-1))
                        loss = loss_txt + 5.0 * loss_cls
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            print(f"{model_name} Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        else:
            print(f"{model_name} Epoch {epoch} | Train Loss: {avg_train:.4f}")
    return model

# ------------------------ 主函数（只跑基线，Ours直接加载已有结果）--------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 对比实验 (Ours 固定 + 基线训练) | 设备: {device}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset, train_idx, val_idx, test_idx = get_patient_level_split(tokenizer, seed=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"训练:{len(train_dataset)}, 验证:{len(val_dataset)}, 测试:{len(test_dataset)}")
    
    results = {}
    # 1. 直接使用 evaluate.py 的结果（手动填入，或自动加载评估）
    print("\n" + "="*60)
    print("注意：Ours 模型的指标请直接使用 evaluate.py 的输出（AUC=0.994, BLEU-4=0.102 等）")
    print("如需自动评估，请取消下面注释并确保 best_full_model.pth 存在")
  
    
    # 2. 训练并评估基线
    print("\n" + "="*60 + "\n训练 ResNet50+LSTM\n" + "="*60)
    model_lstm = train_baseline(BaselineResNetLSTM, "ResNet50+LSTM", train_dataset, val_loader, tokenizer, device, epochs=100, patience=15)
    results["ResNet50+LSTM"] = evaluate_model(model_lstm, test_loader, tokenizer, device, "ResNet50+LSTM", is_ours=False)
    
    print("\n" + "="*60 + "\n训练 ResNet101+Transformer\n" + "="*60)
    model_trans = train_baseline(BaselineResNetTransformer, "ResNet101+Transformer", train_dataset, val_loader, tokenizer, device, epochs=100, patience=15)
    results["ResNet101+Transformer"] = evaluate_model(model_trans, test_loader, tokenizer, device, "ResNet101+Transformer", is_ours=False)
    
    df = pd.DataFrame(results).T
    df.to_csv("baseline_results_fixed.csv")
    print("\n基线结果已保存至 baseline_results_fixed.csv")
    print(df.round(4))

if __name__ == "__main__":
    main()