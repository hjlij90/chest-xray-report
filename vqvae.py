# -*- coding: utf-8 -*-
"""VQ-VAE 视觉离散化模块。

主要修改（相对原实现，见 docs/实验协议.md 的"代码—论文一致性"一节）：
1. 死码激活改为论文 3.4 节所述的"重构误差最大特征覆写"：
   原实现每个 batch 用 torch.randint 随机采样覆盖零使用码，既不按误差、
   也会因为批内几乎每个码都"零使用"而频繁破坏码本。现改为在每个 epoch
   结束时统计整轮使用情况，把真正的死码替换为 epoch 内重构误差最大的特征。
2. 困惑度按批次内命中概率 p(k) 计算（式 3-8），不再用跨 epoch 的全局累计计数，
   避免"PPL 随训练轮数单调上升"的口径错误。
3. forward 额外返回量化索引，供注意力可视化与码本利用率统计使用。
4. 码本结构保持不变，旧权重（best_vqvae_weights.pth）可直接加载。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    """最近邻向量量化 + 直通估计（STE）。"""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 revive_ratio=1.0, error_pool_size=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.revive_ratio = revive_ratio
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        # 与旧权重兼容：保留 global_usage 缓冲区（持久化，旧 ckpt 里存在）
        self.register_buffer("global_usage", torch.zeros(num_embeddings, dtype=torch.long))
        # 本轮（epoch）使用计数，不写入 state_dict
        self.register_buffer("epoch_usage", torch.zeros(num_embeddings, dtype=torch.long),
                             persistent=False)
        self._error_pool = []          # [(err, feature)]，仅保留误差最大的若干个
        self._pool_limit = error_pool_size
        self._last_batch_ppl = 0.0

    # ---------------- 统计与死码处理 ----------------
    def reset_epoch_usage(self):
        self.epoch_usage.zero_()
        self._error_pool.clear()

    def _record_errors(self, flat_input, quantized, indices):
        """记录本 batch 中重构误差最大的若干特征，供 epoch 末激活死码使用。"""
        with torch.no_grad():
            err = torch.sum((flat_input - quantized) ** 2, dim=1)
            k = min(self._pool_limit, err.numel())
            top = torch.topk(err, k).indices
            for i in top.detach().cpu().tolist():
                self._error_pool.append((float(err[i]), flat_input[i].detach()))
            if len(self._error_pool) > self._pool_limit:
                self._error_pool.sort(key=lambda x: x[0], reverse=True)
                del self._error_pool[self._pool_limit:]

    @torch.no_grad()
    def revive_dead_codes(self):
        """把整轮从未命中的码替换为重构误差最大的特征（论文 3.4 节）。"""
        dead = torch.nonzero(self.epoch_usage == 0).squeeze(1)
        if dead.numel() == 0:
            return 0
        n_revive = max(1, int(dead.numel() * self.revive_ratio))
        dead = dead[:n_revive]
        if not self._error_pool:
            return 0
        self._error_pool.sort(key=lambda x: x[0], reverse=True)
        feats = [f for _, f in self._error_pool]
        for j, code_idx in enumerate(dead.tolist()):
            f = feats[j % len(feats)]
            self.embedding.weight.data[code_idx] = f.to(self.embedding.weight.dtype)
        self.epoch_usage[dead] = 1          # 视为已激活，避免下一轮重复覆写
        return int(dead.numel())

    # ---------------- 前向 ----------------
    def forward(self, inputs):
        # inputs: [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)   # [N, 1]

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,
                                device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view(input_shape)

        # 直通估计：前向用量化值，反向把梯度原样传给编码器输出
        quantized_st = inputs + (quantized - inputs).detach()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)   # 承诺损失
        q_latent_loss = F.mse_loss(quantized_st, inputs.detach())  # 码本损失
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 困惑度：批内命中概率 p(k)（式 3-8）
        probs = encodings.mean(dim=0)
        ppl = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

        if self.training:
            with torch.no_grad():
                batch_usage = torch.bincount(encoding_indices.squeeze(1),
                                             minlength=self.num_embeddings)
                self.epoch_usage += batch_usage
                self.global_usage += batch_usage
            self._record_errors(flat_input, quantized_flat, encoding_indices)

        self._last_batch_ppl = float(ppl)
        return (quantized_st.permute(0, 3, 1, 2).contiguous(), vq_loss, ppl,
                encoding_indices.squeeze(1).view(input_shape[0], input_shape[1], input_shape[2]))


class VQVAE(nn.Module):
    """256x256 胸片 -> 32x32 离散 token 网格。"""

    def __init__(self, in_channels=3, hidden_dim=256, num_embeddings=4096,
                 commitment_cost=0.25, revive_ratio=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=False),
            nn.Conv2d(128, hidden_dim, 4, 2, 1), nn.ReLU(inplace=False),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.pre_quant_conv = nn.Identity()
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim,
                                         commitment_cost=commitment_cost,
                                         revive_ratio=revive_ratio)
        self.decoder = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, 128, 4, 2, 1), nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1), nn.Tanh(),
        )

    def encode(self, x):
        return self.pre_quant_conv(self.encoder(x))

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x, return_indices=False):
        z_e = self.encode(x)
        z_q, vq_loss, ppl, indices = self.quantizer(z_e)
        x_recon = self.decode(z_q)
        if return_indices:
            return z_q, x_recon, vq_loss, ppl, indices
        return z_q, x_recon, vq_loss, ppl

    def reset_quantizer_usage(self):
        self.quantizer.reset_epoch_usage()

    @torch.no_grad()
    def codebook_usage(self, dataloader, device, max_batches=None):
        """统计验证集上被使用的码本数量（论文 3.6.3 的码本利用率）。"""
        was_training = self.training
        self.eval()
        used = torch.zeros(self.quantizer.num_embeddings, dtype=torch.bool)
        seen = 0
        for i, batch in enumerate(dataloader):
            imgs = batch[0].to(device, non_blocking=True)
            _, _, _, idx = self(imgs, return_indices=True)
            used[idx.flatten().cpu()] = True
            seen += imgs.size(0)
            if max_batches and i + 1 >= max_batches:
                break
        self.train(was_training)
        return int(used.sum()), seen
