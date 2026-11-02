# -*- coding: utf-8 -*-
"""放射影像辅助诊断演示系统（Gradio）。

相对原实现：
1. 热力图改为真实交叉注意力（attention.py），删除 mock_attn_heart 合成斑块；
   原实现把固定高斯斑块当作"模型注意力"展示，属于编造可视化结果。
2. 默认 share=False，避免把含患者影像的界面暴露到公网。
3. 只加载一份权重（原实现先加载 vq_vae 再加载 full model，覆盖关系混乱）。
4. 记录端到端推理延迟，供论文表 5-2 使用。

启动：
    python training/app.py --ckpt results/ours/ours_best.pth
"""
import argparse
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention import cross_attention, classifier_attention, tokens_to_grid  # noqa: E402
from config import CFG, get_device                              # noqa: E402
from dataset import build_tokenizer                             # noqa: E402
from generate import beam_search_decode, decode_batch           # noqa: E402
from model import build_variant                                 # noqa: E402

_TOKEN_OF_INTEREST = ("heart", "cardiac", "silhouette", "enlarged")


def _transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.5] * 3, [0.5] * 3)])


class ReportService:
    def __init__(self, ckpt_path, device):
        self.device = device
        self.tokenizer = build_tokenizer(CFG)
        ck = torch.load(ckpt_path, map_location=device)
        sd = ck.get("state_dict", ck)
        self.model = build_variant("full", self.tokenizer.vocab_size,
                                   pad_idx=self.tokenizer.pad_token_id).to(device)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[app] 权重差异：missing={missing} unexpected={unexpected}")
        self.model.eval()
        self.transform = _transform()

    @torch.no_grad()
    def analyze(self, image: Image.Image, target_word: str):
        t0 = time.perf_counter()
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        ids, prob = beam_search_decode(
            self.model, self.tokenizer, x,
            beam_size=CFG.report.beam_size,
            length_penalty=CFG.report.length_penalty,
            repetition_penalty=CFG.report.repetition_penalty,
            max_new_tokens=CFG.report.max_new_tokens)
        report = decode_batch(self.tokenizer, ids)[0]
        p = float(prob.reshape(-1)[0])

        # 真实注意力：先编码出报告前缀，再对目标词的 query 位置取交叉注意力
        prefix_ids = ids[:, :max(1, ids.size(1) - 1)]
        words = [target_word] if target_word else list(_TOKEN_OF_INTEREST)
        attn_src, used_word = None, None
        toks = self.tokenizer.convert_ids_to_tokens(ids[0])
        for w in words:
            hits = [i for i, t in enumerate(toks) if w and w.lower() in t.lower()]
            if hits:
                attn_src = prefix_ids[:, :max(1, hits[0])]
                used_word = w
                break
        if attn_src is not None:
            attn, _ = cross_attention(self.model, x, attn_src)
            grid = tokens_to_grid(attn[-1], grid=32)
            src_name = f"生成 “{used_word}” 时的交叉注意力"
        else:
            w_map, _ = classifier_attention(self.model, x)
            grid = tokens_to_grid(w_map, grid=32)
            src_name = "分类头注意力（未在报告中找到目标词）"

        heat = _overlay(image.convert("RGB").resize((256, 256)), grid)
        latency_ms = (time.perf_counter() - t0) * 1000
        detail = (f"心脏增大概率：{p:.4f}\n判定阈值：0.5（阳性偏保守，可下调）\n"
                  f"端到端延迟：{latency_ms:.0f} ms\n注意力来源：{src_name}")
        return report, heat, detail


def _overlay(image, grid):
    import matplotlib.cm as cm
    base = np.asarray(image).astype(np.float32) / 255.0
    g = np.asarray(grid, dtype=np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    cmap = cm.get_cmap("jet")(g)[..., :3]
    blended = 0.6 * base + 0.4 * cmap
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(CFG.paths.results_dir) / "ours" / "ours_best.pth"))
    ap.add_argument("--share", action="store_true", help="生成公网链接（含医学影像，慎用）")
    args = ap.parse_args()

    device = get_device()
    service = ReportService(args.ckpt, device)

    with gr.Blocks(title="胸部X光片辅助诊断原型") as demo:
        gr.Markdown("## 胸部 X 光片心脏增大辅助诊断与报告生成\n"
                    "本系统为科研原型，输出仅供研究参考，不能替代放射科医师诊断。")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="上传正位胸片")
                word = gr.Dropdown(choices=list(_TOKEN_OF_INTEREST) + [""],
                                   value="heart", label="注意力目标词")
                btn = gr.Button("生成报告", variant="primary")
            with gr.Column():
                out_text = gr.Textbox(label="生成报告（英文）", lines=6)
                out_heat = gr.Image(label="注意力叠加图")
                out_info = gr.Textbox(label="诊断与性能信息", lines=4)
        btn.click(service.analyze, [img, word], [out_text, out_heat, out_info])
        gr.Markdown("### 使用说明\n"
                    "1. 输入 256×256 正位胸片（PNG/JPG）；\n"
                    "2. 模型输出为英文报告，与训练语料（IU-CXR）一致；\n"
                    "3. 注意力图由解码器交叉注意力权重真实计算，非人工合成。")
    demo.launch(share=args.share, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
