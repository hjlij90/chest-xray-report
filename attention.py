# -*- coding: utf-8 -*-
"""交叉注意力与分类注意力的真实提取。

论文 4.2 节与图 4-5 声称"生成 heart 一词时的注意力分布"。原代码中的该图是
手写的高斯斑块（app.py / generate_figures.py 里的 mock_attn_heart），与模型无关。
本文件从 nn.TransformerDecoder 各层的 multihead_attn 权重出发，真实计算
query 对 1024 个视觉 token 的注意力并还原到 32x32 网格。
"""
import math
import torch
import torch.nn.functional as F


def _mha_attention(mha, query_in, kv, kv_padding_mask=None):
    """按 nn.MultiheadAttention 的参数复算注意力权重（eval 模式，无 dropout）。"""
    w_q, w_k, _ = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, _ = mha.in_proj_bias.chunk(3, dim=0)
    E = mha.embed_dim
    H = mha.num_heads
    dh = E // H
    B, T, _ = query_in.shape
    S = kv.size(1)
    q = F.linear(query_in, w_q, b_q).view(B, T, H, dh).transpose(1, 2)
    k = F.linear(kv, w_k, b_k).view(B, S, H, dh).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dh)
    if kv_padding_mask is not None:
        scores = scores.masked_fill(kv_padding_mask[:, None, None, :], -1e9)
    weights = F.softmax(scores, dim=-1)          # [B, H, T, S]
    return weights.mean(dim=1)                   # 头平均 -> [B, T, S]


@torch.no_grad()
def cross_attention(model, images, prefix_ids, layer_avg=True):
    """返回解码器各层对视觉 token 的注意力。

    Args:
        images: [1, 3, H, W]
        prefix_ids: [1, T] 已生成的 token 序列（末位为"预测下一个词"的 query 位置）
    Returns:
        attn: [T, 1024] 或 [L, T, 1024]（layer_avg=False）
    """
    model.eval()
    tokens, _, _ = model.visual_tokens(images)
    memory = tokens
    prob = torch.sigmoid(model.classifier(tokens)).detach()
    h = model.text_pos_encoder(model.word_embedding(prefix_ids))
    if getattr(model, "class_cond", None) is not None:
        h = h + model.class_cond(prob)
    mask = model.generate_square_subsequent_mask(prefix_ids.size(1)).to(images.device)

    outs = []
    for layer in model.transformer_decoder.layers:
        q_in = layer.norm2(h)
        w = _mha_attention(layer.multihead_attn, q_in, memory)     # [1, T, S]
        outs.append(w[0].cpu())
        h = layer(h, memory, tgt_mask=mask)
    attn = torch.stack(outs) if not layer_avg else torch.stack(outs).mean(0)
    return attn, prob.item()


@torch.no_grad()
def classifier_attention(model, images):
    """分类头的注意力池化权重，还原为 32x32。"""
    model.eval()
    tokens, _, _ = model.visual_tokens(images)
    logits, w = model.classifier(tokens, return_weights=True)     # [1, 1024]
    return w[0].cpu(), torch.sigmoid(logits).item()


def tokens_to_grid(weights_2d, grid=32, normalize=True):
    """把 [T, S] 或 [S] 的注意力还原为 [grid, grid]。"""
    if weights_2d.dim() == 2:
        weights_2d = weights_2d.mean(0)
    assert weights_2d.numel() == grid * grid, f"token 数 {weights_2d.numel()} 与 {grid}x{grid} 不匹配"
    g = weights_2d.view(grid, grid).float()
    if normalize:
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g.numpy()
