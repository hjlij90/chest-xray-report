import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupShuffleSplit

# 导入你的模块
from dataset import IndianaCXRDataset
from model import MedicalReportSystem

def get_patient_level_train_loader(tokenizer):
    dataset = IndianaCXRDataset(
        images_dir=r"E:\程序\datasets\images_256x256",
        projections_csv=r"E:\程序\datasets\indiana_projections.csv",
        reports_csv=r"E:\程序\datasets\indiana_reports.csv",
        tokenizer=tokenizer,
        max_seq_len=64,
        use_extreme_aug=True # 训练时开启极限增强
    )
    
    patient_uids = dataset.data['uid'].values 
    print("\n患者级隔离切分 (Patient-level Split)...")
    
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, _ = next(gss1.split(dataset.data, groups=patient_uids))
    
    train_dataset = Subset(dataset, train_idx)
    print(f" 提取纯净训练集: {len(train_dataset)} 条")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    return train_loader

def main():
    print("="*60)
    print("🏆 开始最终模型 (Ours - Full Architecture)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 1. 装载满血版架构
    model = MedicalReportSystem(vocab_size=tokenizer.vocab_size).to(device)
    
    # 2. 挂载已有的 VQ-VAE 视觉底座
    vqvae_weights = r"E:\程序\best_vqvae_weights.pth"
    model.vq_vae.load_state_dict(torch.load(vqvae_weights, map_location=device))
    print(" 成功接驳 VQ-VAE 离散视觉底座！")
    
    # 3. 配置防御武器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.46]).to(device))
    criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    scaler = GradScaler('cuda')
    
    train_loader = get_patient_level_train_loader(tokenizer)
    epochs = 20
    
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"🏆 冠军训练中 [{epoch+1}/{epochs}]")
        for imgs, texts, labels in pbar:
            imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
            tgt_in, tgt_out = texts[:, :-1], texts[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(imgs, text_seq=tgt_in, pad_idx=tokenizer.pad_token_id)
                loss_cls = criterion_cls(outputs['class_logits'], labels)
                logits = outputs['report_logits'].reshape(-1, tokenizer.vocab_size)
                loss_txt = criterion_text(logits, tgt_out.reshape(-1))
                # 文本的 Loss 加上 5倍的分类防漏诊 Loss
                loss = loss_txt + 5.0 * loss_cls
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 
    save_path = "best_full_model.pth"
    torch.save(model.state_dict(), save_path)
    print("\n" + "="*60)
    print(f"最终模型已保存至：【{save_path}】")
    print("运行 evaluate_nlp.py 测试它的写报告能力了！")
    print("="*60)

if __name__ == "__main__":
    main()