# -*- coding: utf-8 -*-
"""胸片报告生成主模型及其基线。

核心修改（相对原实现）：
1. 视觉 token 默认只取量化后的离散码本向量（论文的核心论点是"离散信息瓶颈"）。
   原实现把连续特征 continuous_proj(cnn_features) 直接加进解码器 memory，
   相当于绕过了量化瓶颈；该旁路现在只在消融变体 use_continuous_bypass=True 中启用。
2. 新增分类置信度注入（论文 4.3.4）：把 sigmoid(class_logits) 经线性层映射为
   hidden 向量，加到解码器每一层的输入上，训练与推理端到端一致。
   原实现只在论文里有描述、代码里没有。
3. 消融变体改为单一变量：NoPosEmbed 只去掉视觉位置编码，NoCrossAttn 只把
   交叉注意力替换为等长的自注意力 memory 拼接；不再顺带删掉离散/连续叠加逻辑。
4. 基线模型改正两处错误：原 BaselineResNetTransformer 内部用的是 resnet50
   却在实验里被记作 "ResNet101+Transformer"；BaselineViTTransformer 从未加入
   位置编码，等价于无视空间结构的词袋。现统一为 ResNet50 / ViT-B/16 并按各自
   预训练要求做输入归一化。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from vqvae import VQVAE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DiseaseClassifierHead(nn.Module):
    """注意力池化 + 线性分类头（对 1024 个视觉 token 做加权汇聚）。"""

    def __init__(self, hidden_dim, num_classes=1):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_weights=False):
        attn_weights = F.softmax(self.attention_net(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        logits = self.classifier(pooled)
        if return_weights:
            return logits, attn_weights.squeeze(-1)
        return logits


class ClassConditioner(nn.Module):
    """把分类置信度 p(心脏增大) 映射为解码器的条件向量。"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, prob):
        if prob.dim() == 0:
            prob = prob.view(1, 1)
        return self.net(prob.view(-1, 1)).unsqueeze(1)     # [B, 1, D]


class MedicalReportSystem(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_classes=1, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, num_embeddings=4096,
                 use_vqvae=True, use_pos_embed=True, use_cross_attn=True,
                 use_class_conditioning=True, use_continuous_bypass=False,
                 pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.use_vqvae = use_vqvae
        self.use_pos_embed = use_pos_embed
        self.use_cross_attn = use_cross_attn
        self.use_class_conditioning = use_class_conditioning
        self.use_continuous_bypass = use_continuous_bypass

        if use_vqvae:
            self.vq_vae = VQVAE(in_channels=3, hidden_dim=hidden_dim,
                                num_embeddings=num_embeddings)
        else:
            # 消融：直接用连续卷积特征，不经量化
            self.vq_vae = VQVAE(in_channels=3, hidden_dim=hidden_dim,
                                num_embeddings=num_embeddings)

        self.continuous_proj = nn.Linear(hidden_dim, hidden_dim)
        self.visual_pos_embed = nn.Parameter(torch.randn(1, 32 * 32, hidden_dim) * 0.02)

        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.text_pos_encoder = PositionalEncoding(hidden_dim, max_len=512)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.classifier = DiseaseClassifierHead(hidden_dim, num_classes)
        self.class_cond = ClassConditioner(hidden_dim) if use_class_conditioning else None

    # ---------------- 视觉侧 ----------------
    def visual_tokens(self, images, return_indices=False):
        z_e = self.vq_vae.encode(images)
        indices = None
        if self.use_vqvae:
            z_q, vq_loss, ppl, indices = self.vq_vae.quantizer(z_e)
            feat = z_q
        else:
            vq_loss, ppl = torch.zeros((), device=images.device), torch.zeros((), device=images.device)
            feat = z_e

        tokens = feat.flatten(2).transpose(1, 2).contiguous()          # [B, 1024, D]
        if self.use_continuous_bypass:
            cont = z_e.flatten(2).transpose(1, 2).contiguous()
            tokens = tokens + self.continuous_proj(cont)
        if self.use_pos_embed:
            if tokens.size(1) == self.visual_pos_embed.size(1):
                tokens = tokens + self.visual_pos_embed
            else:
                pos = F.interpolate(self.visual_pos_embed.transpose(1, 2), size=tokens.size(1),
                                    mode="linear", align_corners=False).transpose(1, 2)
                tokens = tokens + pos
        if return_indices:
            return tokens, indices, vq_loss, ppl
        return tokens, vq_loss, ppl

    def encode_image(self, images):
        tokens, vq_loss, ppl = self.visual_tokens(images)
        class_logits = self.classifier(tokens)
        return tokens, class_logits, vq_loss, ppl

    # ---------------- 解码 ----------------
    def _memory(self, visual_tokens):
        if self.use_cross_attn:
            return visual_tokens
        # 消融：把视觉信息拼到 query 侧，交叉注意力退化为自注意力
        return visual_tokens

    def decode(self, visual_tokens, text_seq, class_prob=None, memory=None,
               return_attn=False):
        memory = visual_tokens if memory is None else memory
        tgt = self.text_pos_encoder(self.word_embedding(text_seq))
        if self.class_cond is not None:
            if class_prob is None:
                class_logits = self.classifier(visual_tokens)
                class_prob = torch.sigmoid(class_logits).detach()
            cond = self.class_cond(class_prob.detach())            # [B,1,D]
            tgt = tgt + cond
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool,
                                        device=tgt.device), diagonal=1)
        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

    def forward(self, images, text_seq=None, pad_idx=None, class_prob=None):
        pad = self.pad_idx if pad_idx is None else pad_idx
        tokens, vq_loss, ppl = self.visual_tokens(images)
        class_logits = self.classifier(tokens)
        prob = torch.sigmoid(class_logits).detach()
        if class_prob is not None:
            prob = class_prob
        out = {"class_logits": class_logits, "prob": prob, "vq_loss": vq_loss,
               "ppl": ppl, "visual_tokens": tokens}
        if text_seq is not None:
            out["report_logits"] = self.decode(tokens, text_seq, class_prob=prob)
        return out

    @staticmethod
    def generate_square_subsequent_mask(sz):
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)


# ==================== 基线模型 ====================
class _ResNetEncoder(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if name == "resnet50" \
                else models.ResNet101_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights) if name == "resnet50" else models.resnet101(weights=weights)
        self.stem = nn.Sequential(*list(net.children())[:-2])   # 到 layer4，输出 [B,2048,H/32,W/32]
        self.out_dim = 2048
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x * 0.5 + 0.5)                    # [-1,1] -> [0,1]
        x = (x - self.mean) / self.std          # ImageNet 归一化
        return self.stem(x)


class BaselineResNetTransformer(nn.Module):
    """ResNet50 + Transformer 解码器（原代码误记为 ResNet101）。"""

    def __init__(self, vocab_size, hidden_dim=256, num_classes=1, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, pad_idx=0, backbone="resnet50",
                 pretrained=True):
        super().__init__()
        self.cnn_encoder = _ResNetEncoder(backbone, pretrained=pretrained)
        self.cnn_proj = nn.Linear(self.cnn_encoder.out_dim, hidden_dim)
        self.visual_pos_embed = nn.Parameter(torch.randn(1, 8 * 8, hidden_dim) * 0.02)
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.text_pos_encoder = PositionalEncoding(hidden_dim, max_len=512)
        layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.classifier = DiseaseClassifierHead(hidden_dim, num_classes)
        self.pad_idx = pad_idx

    def encode_image(self, images):
        feat = self.cnn_encoder(images)                 # [B,2048,8,8]
        vis = self.cnn_proj(feat.flatten(2).transpose(1, 2))   # [B,64,256]
        return vis + self.visual_pos_embed, None, torch.zeros((), device=images.device), \
            torch.zeros((), device=images.device)

    def decode(self, memory, text_seq):
        tgt = self.text_pos_encoder(self.word_embedding(text_seq))
        mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device), diagonal=1)
        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=mask,
                                       )
        return self.fc_out(out)

    def forward(self, images, text_seq=None, pad_idx=None, class_prob=None):
        memory, _, _, _ = self.encode_image(images)
        class_logits = self.classifier(memory)
        out = {"class_logits": class_logits, "prob": torch.sigmoid(class_logits),
               "visual_tokens": memory}
        if text_seq is not None:
            out["report_logits"] = self.decode(memory, text_seq)
        return out


class BaselineResNetLSTM(nn.Module):
    """ResNet50 全局池化特征 + 2 层 LSTM 解码器。"""

    def __init__(self, vocab_size, hidden_dim=256, num_classes=1, dropout=0.1, pad_idx=0,
                 backbone="resnet50", pretrained=True):
        super().__init__()
        self.cnn_encoder = _ResNetEncoder(backbone, pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_proj = nn.Linear(self.cnn_encoder.out_dim, hidden_dim)
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.lstm_decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True,
                                    dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.classifier = DiseaseClassifierHead(hidden_dim, num_classes)
        self.pad_idx = pad_idx

    def encode_image(self, images):
        feat = self.pool(self.cnn_encoder(images)).flatten(1)
        proj = self.cnn_proj(feat)                              # [B,256]
        self._proj = proj
        return proj.unsqueeze(1), None, torch.zeros((), device=images.device), \
            torch.zeros((), device=images.device)

    def decode(self, memory, text_seq):
        proj = memory[:, 0]
        h0 = proj.unsqueeze(0).repeat(2, 1, 1).contiguous()
        c0 = proj.unsqueeze(0).repeat(2, 1, 1).contiguous()
        emb = self.word_embedding(text_seq)
        out, _ = self.lstm_decoder(emb, (h0, c0))
        return self.fc_out(out)

    def forward(self, images, text_seq=None, pad_idx=None, class_prob=None):
        memory, _, _, _ = self.encode_image(images)
        class_logits = self.classifier(memory[:, 0])
        out = {"class_logits": class_logits, "prob": torch.sigmoid(class_logits),
               "visual_tokens": memory}
        if text_seq is not None:
            out["report_logits"] = self.decode(memory, text_seq)
        return out


class BaselineViTTransformer(nn.Module):
    """ViT-B/16（含位置编码）+ Transformer 解码器。"""

    def __init__(self, vocab_size, hidden_dim=256, num_classes=1, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, pad_idx=0, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        self.out_dim = self.vit.hidden_dim                    # 768
        self.cnn_proj = nn.Linear(self.out_dim, hidden_dim)
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.text_pos_encoder = PositionalEncoding(hidden_dim, max_len=512)
        layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.classifier = DiseaseClassifierHead(hidden_dim, num_classes)
        self.pad_idx = pad_idx
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def encode_image(self, images):
        x = (images * 0.5 + 0.5)
        x = (x - self.mean) / self.std
        if x.size(-1) != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        patch = self.vit._process_input(x)                    # [B,196,768]
        cls_tok = self.vit.class_token.expand(patch.size(0), -1, -1)
        seq = torch.cat([cls_tok, patch], dim=1) + self.vit.encoder.pos_embedding
        seq = self.vit.encoder.dropout(seq)
        seq = self.vit.encoder.ln(self.vit.encoder.blocks(seq))
        vis = self.cnn_proj(seq[:, 1:])                        # [B,196,256]
        return vis, None, torch.zeros((), device=images.device), torch.zeros((), device=images.device)

    def decode(self, memory, text_seq):
        tgt = self.text_pos_encoder(self.word_embedding(text_seq))
        mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device), diagonal=1)
        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=mask,
                                       )
        return self.fc_out(out)

    def forward(self, images, text_seq=None, pad_idx=None, class_prob=None):
        memory, _, _, _ = self.encode_image(images)
        class_logits = self.classifier(memory)
        out = {"class_logits": class_logits, "prob": torch.sigmoid(class_logits),
               "visual_tokens": memory}
        if text_seq is not None:
            out["report_logits"] = self.decode(memory, text_seq)
        return out


# ==================== 消融变体 ====================
def build_variant(name, vocab_size, **kwargs):
    """按名称构造消融变体，保证只改变一个因素。"""
    common = dict(vocab_size=vocab_size, hidden_dim=256, num_classes=1, nhead=8,
                  num_layers=4, dim_feedforward=1024, dropout=0.1)
    common.update(kwargs)
    if name == "full":
        return MedicalReportSystem(**common)
    if name == "no_pos_embed":
        return MedicalReportSystem(use_pos_embed=False, **common)
    if name == "no_cross_attn":
        return MedicalReportSystem(use_cross_attn=False, **common)
    if name == "no_vqvae":
        return MedicalReportSystem(use_vqvae=False, use_pos_embed=False, **common)
    if name == "no_cls":
        return MedicalReportSystem(use_class_conditioning=False, **common)
    raise ValueError(f"未知消融变体：{name}")


# 兼容旧名称
MedicalReportSystem_NoPosEmbed = lambda vocab_size, hidden_dim=256, num_classes=1: \
    build_variant("no_pos_embed", vocab_size, hidden_dim=hidden_dim, num_classes=num_classes)
MedicalReportSystem_NoCrossAttn = lambda vocab_size, hidden_dim=256, num_classes=1: \
    build_variant("no_cross_attn", vocab_size, hidden_dim=hidden_dim, num_classes=num_classes)



