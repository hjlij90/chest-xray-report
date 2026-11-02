# -*- coding: utf-8 -*-
"""
VQ-VAE 密码本容量 (K值) 消融压测提速版 + 有效码本数监控
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from dataset import IndianaCXRDataset
from vqvae import VQVAE

plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

def compute_effective_codebook_size(model, dataloader, device):
    """统计在整个验证集上被使用过的码本数量"""
    model.eval()
    all_indices = []
    with torch.no_grad():
        for imgs, _, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            z_e = model.encoder(imgs)
            z = model.pre_quant_conv(z_e)
            # 手动计算最近邻索引（不经过quantizer的内部计数器）
            z_perm = z.permute(0, 2, 3, 1).contiguous()
            flat_z = z_perm.view(-1, z.shape[1])
            distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                         + torch.sum(model.quantizer.embedding.weight**2, dim=1)
                         - 2 * torch.matmul(flat_z, model.quantizer.embedding.weight.t()))
            indices = torch.argmin(distances, dim=1)
            all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices)
    unique = torch.unique(all_indices)
    return len(unique)

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" 启动 K 值消融管道（含码本利用率监控）！设备：{device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = IndianaCXRDataset(
        images_dir=r"E:\程序\datasets\images_256x256",
        projections_csv=r"E:\程序\datasets\indiana_projections.csv",
        reports_csv=r"E:\程序\datasets\indiana_reports.csv",
        tokenizer=tokenizer,
        max_seq_len=64
    )
    BATCH_SIZE = 32
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                            num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)

    k_values = [1024, 4096, 8192]
    num_epochs = 15
    all_results = []
    paper_table_data = {k: {} for k in k_values}
    effective_sizes = {k: [] for k in k_values}

    for k in k_values:
        torch.cuda.empty_cache()
        print("="*60)
        print(f"🚀 开始测试密码本容量 K = {k}")
        print("="*60)

        model = VQVAE(in_channels=3, hidden_dim=256, num_embeddings=k).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
        scaler = GradScaler('cuda')

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_recon_loss = 0
            total_ppl = 0

            # 每个 epoch 开始前重置 quantizer 的全局使用计数
            model.quantizer.reset_usage()

            pbar = tqdm(dataloader, desc=f"K={k} Epoch [{epoch}/{num_epochs}]", leave=False)
            for imgs, _, _ in pbar:
                imgs = imgs.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    z_q, x_recon, vq_loss, ppl = model(imgs)
                    recon_loss = F.mse_loss(x_recon, imgs)
                    loss = recon_loss + vq_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_recon_loss += recon_loss.item()
                total_ppl += ppl.item()

            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_ppl = total_ppl / len(dataloader)

            # 每3个epoch统计一次有效码本数（避免耗时过多）
            if epoch % 3 == 0 or epoch == num_epochs:
                # 使用一个小型验证子集，这里直接复用train loader但eval模式
                effective = compute_effective_codebook_size(model, dataloader, device)
                effective_sizes[k].append((epoch, effective))
                print(f"  -> Epoch {epoch}: 有效码本数 = {effective}/{k} (利用率 {effective/k*100:.1f}%)")

            all_results.append({
                "K_Value": k,
                "Epoch": epoch,
                "Recon_Loss": avg_recon_loss,
                "Perplexity": avg_ppl
            })
            print(f"  -> Recon_MSE: {avg_recon_loss:.4f} | Perplexity: {avg_ppl:.2f}")

            if epoch in [1, 10, 15]:
                paper_table_data[k][epoch] = {"PPL": avg_ppl, "MSE": avg_recon_loss}

    # 保存结果并绘图（增加有效码本数子图）
    df = pd.DataFrame(all_results)
    df.to_csv("K值消融实验数据_修正.csv", index=False)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    colors = {1024: '#1f77b4', 4096: '#d62728', 8192: '#2ca02c'}

    for k in k_values:
        sub_df = df[df['K_Value'] == k]
        axes[0].plot(sub_df['Epoch'], sub_df['Recon_Loss'], marker='o', color=colors[k], label=f'K={k}')
        axes[1].plot(sub_df['Epoch'], sub_df['Perplexity'], marker='s', color=colors[k], label=f'K={k}')

        # 有效码本数曲线（插值到epoch轴）
        epochs_eff, sizes = zip(*effective_sizes[k])
        axes[2].plot(epochs_eff, sizes, marker='^', color=colors[k], label=f'K={k}')

    axes[0].set_title('重构损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[1].set_title('密码本困惑度')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].legend()
    axes[2].set_title('有效码本数')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Number of used codes')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('Fig_K值消融对比图_修正.png')
    print("✅ 已保存图像: Fig_K值消融对比图_修正.png")

if __name__ == "__main__":
    main()