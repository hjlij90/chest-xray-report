# -*- coding: utf-8 -*-
"""CPU 冒烟测试：不依赖 GPU，验证关键实现正确性。

运行：
    python training/tests_smoke.py
（也可用 pytest -q training/tests_smoke.py）
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CFG, set_seed                              # noqa: E402
from vqvae import VQVAE, VectorQuantizer                      # noqa: E402
from model import MedicalReportSystem, build_variant          # noqa: E402


def test_vqvae_shapes_and_gradient():
    set_seed(0)
    vq = VQVAE(hidden_dim=64, num_embeddings=64)
    x = torch.randn(2, 3, 64, 64) * 0.5
    z_q, recon, vq_loss, ppl, idx = vq(x, return_indices=True)
    assert z_q.shape == (2, 64, 8, 8), z_q.shape
    assert recon.shape == x.shape
    assert idx.shape == (2, 8, 8)
    assert 1 <= float(ppl) <= 64 + 1e-6
    # 直通估计：梯度必须能回到编码器
    vq.zero_grad()
    loss = ((recon - x) ** 2).mean() + vq_loss
    loss.backward()
    grads = [p.grad for p in vq.encoder.parameters() if p.grad is not None]
    assert grads and torch.isfinite(torch.stack([g.abs().sum() for g in grads])).all()
    print("  ✓ VQ-VAE 形状 / STE 梯度 / 困惑度范围")


def test_dead_code_revival_uses_max_error():
    set_seed(0)
    q = VectorQuantizer(num_embeddings=16, embedding_dim=8)
    # 人为制造 2 个死码
    q.epoch_usage[:] = 1
    q.epoch_usage[:2] = 0
    feats = torch.randn(32, 8)
    # 记录一个已知的"高误差特征"
    high = torch.full((8,), 10.0)
    q._error_pool = [(0.1, torch.zeros(8)), (99.0, high)]
    n = q.revive_dead_codes()
    assert n >= 1
    # 死码应被高误差特征覆写，而不是随机向量
    assert torch.allclose(q.embedding.weight.data[0], high, atol=1e-5)
    print(f"  ✓ 死码激活使用最大重构误差特征（覆写 {n} 个）")


def test_model_forward_and_variants():
    set_seed(0)
    vocab = 200
    hidden = 64
    model = MedicalReportSystem(vocab_size=vocab, hidden_dim=hidden, nhead=4,
                                num_layers=2, dim_feedforward=128, num_embeddings=32,
                                pad_idx=0)
    x = torch.randn(2, 3, 64, 64)
    txt = torch.randint(1, 20, (2, 8))
    out = model(x, text_seq=txt, pad_idx=0)
    assert out["report_logits"].shape == (2, 8, vocab)
    assert out["class_logits"].shape == (2, 1)

    # 分类条件应影响输出：把概率置 0 / 1，解码 logits 应有差异
    prob0 = torch.zeros(2, 1); prob1 = torch.ones(2, 1)
    a = model.decode(out["visual_tokens"], txt, class_prob=prob0)
    b = model.decode(out["visual_tokens"], txt, class_prob=prob1)
    assert not torch.allclose(a, b), "分类置信度注入未生效"
    print("  ✓ 主模型前向 / 分类置信度确实影响解码输出")

    for v in ["no_pos_embed", "no_cross_attn", "no_cls", "no_vqvae"]:
        m = build_variant(v, vocab, hidden_dim=hidden, nhead=4, num_layers=2,
                          dim_feedforward=128, num_embeddings=32)
        o = m(x, text_seq=txt, pad_idx=0)
        assert o["report_logits"].shape == (2, 8, vocab), f"{v} 输出形状异常"
    print("  ✓ 四个消融变体均能前向")


def test_beam_search_forbidden_tokens():
    set_seed(0)
    from generate import beam_search_decode

    class Tok:
        pad_token_id = 0
        cls_token_id = 101
        sep_token_id = 102
        vocab_size = 120

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["x"] * ids.size(0)

        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

    model = MedicalReportSystem(vocab_size=120, hidden_dim=64, nhead=4, num_layers=2,
                                dim_feedforward=128, num_embeddings=32, pad_idx=0)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    ids, prob = beam_search_decode(model, Tok(), x, beam_size=3, max_new_tokens=12)
    banned = set(range(1, 100)) | {0, 101}
    produced = set(ids[0, 1:].flatten().tolist())   # 首 token 是作为提示的 [CLS]
    assert not (produced & banned), f"生成了被禁 token：{produced & banned}"
    assert ids[0, 0] == 101
    print("  ✓ 束搜索不产生 [PAD]/[unused*]/[CLS]")


def test_metrics():
    from metrics import evaluate_predictions
    refs = ["the heart is enlarged", "no acute cardiopulmonary findings"]
    hyps = ["the heart is enlarged", "no acute cardiopulmonary finding"]
    labels = [1, 0]
    probs = [0.9, 0.1]
    m = evaluate_predictions(refs, hyps, labels, probs, None)
    assert m["AUC"] == 1.0 and m["Recall"] == 1.0
    assert 0 <= m["BLEU-4"] <= 1
    print(f"  ✓ 指标计算正常（BLEU-4={m['BLEU-4']:.3f}, AUC={m['AUC']:.1f}）")


def test_split_no_patient_leakage():
    from data_utils import get_patient_split
    from dataset import build_tokenizer
    cfg = CFG
    tok = build_tokenizer(cfg)
    tr, va, te, meta = get_patient_split(cfg, tok)
    assert len(tr) + len(va) + len(te) == meta["n"]
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert meta["train_patients"] + meta["val_patients"] + meta["test_patients"] == meta["patients"]
    print(f"  ✓ 患者级划分无泄漏：train {meta['train']} / val {meta['val']} / test {meta['test']}，"
          f"阳性 {meta['train_pos']}/{meta['val_pos']}/{meta['test_pos']}")


if __name__ == "__main__":
    print("[smoke] 开始 CPU 冒烟测试")
    test_vqvae_shapes_and_gradient()
    test_dead_code_revival_uses_max_error()
    test_model_forward_and_variants()
    test_beam_search_forbidden_tokens()
    test_metrics()
    test_split_no_patient_leakage()
    print("[smoke] 全部通过")
