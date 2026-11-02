# -*- coding: utf-8 -*-
"""报告解码：贪心与束搜索。

原 evaluate.py 的束搜索有两个实质缺陷：
1. 先累加 log_probs 得到 next_scores，之后才把特殊 token 置 -inf 并"加回" log_probs，
   导致被禁 token 的分数其实已经进入了排序；
2. 重复惩罚按 CTRL 式的"正负分路径"实现（logits/1.05 或 logits*1.05），
   而论文只交代了重复惩罚系数 1.05，未说明该实现细节。
本文件按标准流程重写：先屏蔽特殊 token，再施加重复惩罚，最后才累加分数；
长度惩罚采用 score / len^alpha（论文式 4-3）。
"""
import torch
import torch.nn.functional as F


def special_token_ids(tokenizer):
    """返回需要禁止生成的 token id 集合。

    BERT 词表 0 为 [PAD]、1..99 为 [unused*]，二者在语料中从不出现，
    生成时必须屏蔽；[CLS]/[SEP] 由流程控制，不作为普通候选。
    """
    ids = {tokenizer.pad_token_id, tokenizer.cls_token_id}
    ids.update(range(1, 100))
    return ids


@torch.no_grad()
def _prepare(model, images):
    tokens, vq_loss, ppl = model.visual_tokens(images)
    class_logits = model.classifier(tokens)
    prob = torch.sigmoid(class_logits)
    memory = tokens
    cond = None
    if getattr(model, "class_cond", None) is not None:
        cond = model.class_cond(prob.detach())
    return memory, prob, cond


@torch.no_grad()
def greedy_decode(model, tokenizer, images, max_new_tokens=None, min_len=5):
    device = images.device
    max_new = max_new_tokens or 96
    memory, prob, cond = _prepare(model, images)
    bsz = images.size(0)
    forbid = torch.tensor(sorted(special_token_ids(tokenizer)), device=device)
    gen = torch.full((bsz, 1), tokenizer.cls_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    for step in range(max_new):
        tgt = model.text_pos_encoder(model.word_embedding(gen))
        if cond is not None:
            tgt = tgt + cond
        mask = nn_mask(gen.size(1), device)
        out = model.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=mask)
        logits = model.fc_out(out[:, -1, :])
        logits[:, forbid] = -float("inf")
        if step < min_len:
            logits[:, tokenizer.sep_token_id] = -float("inf")
        nxt = logits.argmax(-1, keepdim=True)
        nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, tokenizer.pad_token_id), nxt)
        gen = torch.cat([gen, nxt], dim=1)
        finished |= (nxt.squeeze(1) == tokenizer.sep_token_id)
        if finished.all():
            break
    return gen, prob


def nn_mask(sz, device):
    import torch.nn as nn
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


@torch.no_grad()
def beam_search_decode(model, tokenizer, images, beam_size=3, length_penalty=0.8,
                       repetition_penalty=1.05, max_new_tokens=96, min_len=5):
    """逐样本束搜索。返回 (生成序列 [B, L], 分类概率 [B])。"""
    device = images.device
    memory, prob, cond = _prepare(model, images)
    forbid = torch.tensor(sorted(special_token_ids(tokenizer)), device=device)
    end_id = tokenizer.sep_token_id

    all_seqs = []
    for b in range(images.size(0)):
        mem_b = memory[b:b + 1]
        cond_b = cond[b:b + 1] if cond is not None else None
        # beams: list of (tokens list, logprob sum)
        beams = [([tokenizer.cls_token_id], 0.0)]
        finished = []
        for step in range(max_new_tokens):
            candidates = []
            for seq, lp in beams:
                gen = torch.tensor([seq], dtype=torch.long, device=device)
                tgt = model.text_pos_encoder(model.word_embedding(gen))
                if cond_b is not None:
                    tgt = tgt + cond_b
                mask = nn_mask(gen.size(1), device)
                out = model.transformer_decoder(tgt=tgt, memory=mem_b, tgt_mask=mask)
                logits = model.fc_out(out[:, -1, :])            # [1, V]
                logits = logits.clone()
                logits[:, forbid] = -float("inf")
                if step < min_len:
                    logits[:, end_id] = -float("inf")
                # 重复惩罚：对已出现的 token 降低概率（按 logit 符号分别处理）
                if repetition_penalty and repetition_penalty != 1.0 and len(seq) > 1:
                    for tok in set(seq[1:]):
                        if tok in (end_id, tokenizer.pad_token_id):
                            continue
                        if logits[0, tok] > 0:
                            logits[0, tok] /= repetition_penalty
                        else:
                            logits[0, tok] *= repetition_penalty
                logp = F.log_softmax(logits.float(), dim=-1)[0]
                topv, topi = torch.topk(logp, beam_size)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    candidates.append((seq + [i], lp + v))
            # 长度惩罚后排序，保留 beam_size 个
            def score(c):
                return c[1] / (max(len(c[0]), 1) ** length_penalty)

            candidates.sort(key=score, reverse=True)
            beams = []
            for seq, lp in candidates:
                if seq[-1] == end_id:
                    finished.append((seq, lp))
                else:
                    beams.append((seq, lp))
                if len(beams) >= beam_size:
                    break
            if not beams:
                break
        if finished:
            def final_score(c):
                return c[1] / (max(len(c[0]) - 1, 1) ** length_penalty)
            best = max(finished, key=final_score)[0]
        else:
            best = beams[0][0]
        all_seqs.append(best)

    pad_len = max(len(s) for s in all_seqs)
    out_ids = torch.full((len(all_seqs), pad_len), tokenizer.pad_token_id, dtype=torch.long)
    for i, s in enumerate(all_seqs):
        out_ids[i, :len(s)] = torch.tensor(s)
    return out_ids.to(device), prob.squeeze(-1)


def decode_batch(tokenizer, ids):
    """解码并清理 BERT 特殊 token 与 [unused*] 残留。"""
    import re
    texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
    return [re.sub(r"\[unused\d+\]", "", t).strip() for t in texts]

