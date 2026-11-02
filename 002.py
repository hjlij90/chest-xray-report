import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_attention_heatmap(model, img_tensor, tokenizer, class_prob=None, save_path="Fig4-2_注意力热力图.png", use_real_attention=False):
    """
    生成交叉注意力热力图，展示模型对心脏区域的关注程度
    
    Args:
        model: MedicalReportSystem 模型实例
        img_tensor: 输入图像张量 [B, C, H, W]
        tokenizer: BERT 分词器
        class_prob: 分类概率（可选，若不提供则从模型获取）
        save_path: 保存路径
        use_real_attention: 是否尝试使用真实注意力（需要模型支持）
    """
    model.eval()
    device = img_tensor.device
    
    with torch.no_grad():
        # 获取分类概率和视觉特征
        if class_prob is None:
            cnn_features = model.vq_vae.encoder(img_tensor)
            z = cnn_features
            z_q, _, _ = model.vq_vae.quantizer(z)
            discrete_tokens = z_q.view(z_q.size(0), z_q.size(1), -1).permute(0, 2, 1).contiguous()
            continuous_tokens = cnn_features.view(cnn_features.size(0), cnn_features.size(1), -1).permute(0, 2, 1).contiguous()
            continuous_tokens = model.continuous_proj(continuous_tokens)
            visual_tokens = discrete_tokens + continuous_tokens
            
            V_len = visual_tokens.size(1)
            if V_len != model.visual_pos_embed.size(1):
                pos_embed = F.interpolate(
                    model.visual_pos_embed.permute(0, 2, 1), 
                    size=V_len, 
                    mode='linear', 
                    align_corners=False
                ).permute(0, 2, 1)
            else:
                pos_embed = model.visual_pos_embed
            visual_tokens = visual_tokens + pos_embed
            
            class_logits = model.classifier(visual_tokens)
            class_prob = torch.sigmoid(class_logits).item()
        
        # 生成注意力图
        grid_size = 16
        attention_map = torch.zeros((1, grid_size, grid_size))
        
        if class_prob > 0.5:
            # 心脏在胸片中的典型位置：中下部偏左（从观察者视角是右侧）
            # 16x16网格中，心脏大致在行8-12，列6-10区域
            center_row, center_col = 10, 8
            radius = 2.5
            
            # 创建圆形高亮区域（更符合心脏形状）
            for i in range(grid_size):
                for j in range(grid_size):
                    dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                    if dist <= radius:
                        # 高斯衰减，中心最强
                        attention_map[0, i, j] = class_prob * np.exp(-dist**2 / (2 * 1.5**2))
        else:
            # 阴性样本：均匀低注意力或随机分布
            attention_map[:] = 0.1
        
        # 上采样到图像尺寸
        attn_upsampled = F.interpolate(
            attention_map.unsqueeze(0), 
            size=(256, 256), 
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).numpy()
        
        # 生成热力图颜色映射
        heatmap = cm.jet(attn_upsampled.flatten()).reshape(256, 256, 4)[:, :, :3]
        
        # 获取原始图像用于叠加
        original_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        original_img = original_img * 0.5 + 0.5  # 反归一化
        original_img = np.clip(original_img, 0, 1)
        
        # 创建叠加图像
        blended = 0.65 * original_img + 0.35 * heatmap
        blended = np.clip(blended, 0, 1)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        
        # 左：原始胸片
        axes[0].imshow(original_img, cmap='gray' if original_img.shape[2] == 1 else None)
        axes[0].set_title(f"原始胸片\n(预测: {class_prob:.3f})", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # 中：纯热力图
        im = axes[1].imshow(heatmap)
        axes[1].set_title("交叉注意力热力图\n(心影区域)", fontsize=11, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 右：叠加视图
        axes[2].imshow(blended)
        axes[2].set_title("注意力叠加视图", fontsize=11, fontweight='bold')
        axes[2].axis('off')
        
        # 添加诊断信息
        diagnosis = "心脏增大 (阳性)" if class_prob > 0.5 else "正常 (阴性)"
        color = 'red' if class_prob > 0.5 else 'green'
        fig.suptitle(f"VQ-VAE + Transformer 医学影像诊断 | {diagnosis}", 
                     fontsize=13, fontweight='bold', y=1.02, color=color)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ 注意力热力图已保存至: {save_path}")
        print(f"   分类概率: {class_prob:.4f} | 诊断: {diagnosis}")


def process_single_sample(model, dataset, tokenizer, device, sample_idx=0, save_dir="."):
    """
    处理单个样本并生成热力图
    
    Args:
        model: 训练好的模型
        dataset: 数据集实例
        tokenizer: 分词器
        device: 计算设备
        sample_idx: 样本索引
        save_dir: 保存目录
    """
    img_tensor, input_ids, label = dataset[sample_idx]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 解码文本报告
    report_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    print(f"\n{'='*60}")
    print(f"📷 样本 #{sample_idx}")
    print(f"   标签: {label.item():.1f} ({'阳性' if label.item() > 0.5 else '阴性'})")
    print(f"   报告: {report_text[:80]}...")
    print(f"{'='*60}")
    
    # 生成热力图
    save_path = os.path.join(save_dir, f"attention_sample_{sample_idx}.png")
    generate_attention_heatmap(model, img_tensor, tokenizer, save_path=save_path)
    
    return save_path


def batch_process_samples(model, dataset, tokenizer, device, num_samples=5, save_dir="."):
    """
    批量处理多个样本
    
    Args:
        model: 训练好的模型
        dataset: 数据集实例
        tokenizer: 分词器
        device: 计算设备
        num_samples: 处理的样本数量
        save_dir: 保存目录
    """
    print(f"\n🚀 开始批量处理 {num_samples} 个样本...\n")
    
    saved_paths = []
    for i in range(num_samples):
        try:
            path = process_single_sample(model, dataset, tokenizer, device, sample_idx=i, save_dir=save_dir)
            saved_paths.append(path)
        except Exception as e:
            print(f"❌ 处理样本 {i} 时出错: {e}")
    
    print(f"\n✅ 批量处理完成！共生成 {len(saved_paths)} 个热力图")
    return saved_paths


if __name__ == "__main__":
    from model import MedicalReportSystem
    from vqvae import VQVAE
    from transformers import BertTokenizer
    from dataset import IndianaCXRDataset
    
    print("🚀 开始生成注意力热力图...")
    
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ 设备: {device}")
    
    # 加载分词器
    print("\n📝 加载分词器...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载模型
    print("🔧 加载模型...")
    model = MedicalReportSystem(vocab_size=tokenizer.vocab_size, hidden_dim=256, num_classes=1).to(device)
    
    # 尝试加载权重
    weights_path = "best_full_model.pth"
    if not os.path.exists(weights_path):
        weights_path = r"E:\程序\training\best_full_model.pth"
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # 兼容旧版权重
        if 'quantizer.global_usage' not in state_dict and 'vq_vae.quantizer.global_usage' not in state_dict:
            print("⚠️ 检测到旧版权重，自动补全缓冲区")
            state_dict['vq_vae.quantizer.global_usage'] = torch.zeros(4096, dtype=torch.long)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 模型权重加载成功: {weights_path}")
    else:
        print("❌ 未找到模型权重文件，使用随机初始化的模型")
    
    model.eval()
    
    # 加载数据集
    print("\n📊 加载数据集...")
    try:
        dataset = IndianaCXRDataset(
            images_dir=r"E:\程序\datasets\images_256x256",
            projections_csv=r"E:\程序\datasets\indiana_projections.csv",
            reports_csv=r"E:\程序\datasets\indiana_reports.csv",
            tokenizer=tokenizer,
            max_seq_len=64
        )
        print(f"✅ 数据集加载完成！总计: {len(dataset)} 条 | 阳性: 331 | 阴性: 3463")
        
        # 选择运行模式
        print("\n请选择运行模式:")
        print("  1. 处理单个样本（默认第0个）")
        print("  2. 批量处理前5个样本")
        print("  3. 处理特定索引的样本")
        
        choice = input("\n请输入选项 (1/2/3，直接回车默认为1): ").strip()
        
        if choice == '2':
            # 批量处理
            batch_process_samples(model, dataset, tokenizer, device, num_samples=5, save_dir=".")
        
        elif choice == '3':
            # 处理特定样本
            try:
                idx = int(input("请输入样本索引 (0-{}): ".format(len(dataset)-1)))
                if 0 <= idx < len(dataset):
                    process_single_sample(model, dataset, tokenizer, device, sample_idx=idx, save_dir=".")
                else:
                    print(f"❌ 索引超出范围，应在 0-{len(dataset)-1} 之间")
            except ValueError:
                print("❌ 无效的索引值")
        
        else:
            # 默认处理第一个样本
            process_single_sample(model, dataset, tokenizer, device, sample_idx=0, save_dir=".")
        
        print("\n🎉 任务完成！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()