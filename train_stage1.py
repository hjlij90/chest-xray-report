import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from dataset import IndianaCXRDataset
from vqvae import VQVAE   # 直接导入，不使用完整模型

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 [阶段一] 运行设备：{device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = IndianaCXRDataset(
        images_dir="E:\\程序\\datasets\\images_256x256",
        projections_csv="E:\\程序\\datasets\\indiana_projections.csv",
        reports_csv="E:\\程序\\datasets\\indiana_reports.csv",
        tokenizer=tokenizer,
        max_seq_len=64
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=128,           
        shuffle=True, 
        drop_last=True,
        num_workers=8,           
        pin_memory=True,         
        prefetch_factor=2,        
        persistent_workers=True
    )

    model = VQVAE(in_channels=3, hidden_dim=256, num_embeddings=4096).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda')

    print("\n" + "="*50)
    print("🚀 开始预训练 VQ-VAE (混合精度加速版)")
    print("="*50)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for imgs, _, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                z_q, x_recon, vq_loss, perplexity = model(imgs)
                recon_loss = F.mse_loss(x_recon, imgs)
                loss = recon_loss + vq_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'PPL': f'{perplexity.item():.1f}'})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | 平均损失：{avg_loss:.4f}")

    torch.save(model.state_dict(), "best_vqvae_weights.pth")
    print(f"\n🎉 阶段一完成！权重已保存至 best_vqvae_weights.pth")

if __name__ == "__main__":
    main()