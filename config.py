# -*- coding: utf-8 -*-
"""全局配置。

所有实验超参数集中在此，脚本不得再出现硬编码路径与魔法数字。
路径默认相对于项目根目录（training/ 的上一级），可用环境变量 CXR_ROOT 覆盖。
"""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(os.environ.get("CXR_ROOT", Path(__file__).resolve().parents[1]))


@dataclass
class Paths:
    root: Path = PROJECT_ROOT
    images: Path = PROJECT_ROOT / "datasets" / "images_256x256"
    projections_csv: Path = PROJECT_ROOT / "datasets" / "indiana_projections.csv"
    reports_csv: Path = PROJECT_ROOT / "datasets" / "indiana_reports.csv"
    splits_dir: Path = PROJECT_ROOT / "splits"
    ckpt_dir: Path = PROJECT_ROOT / "checkpoints"
    results_dir: Path = PROJECT_ROOT / "results"
    figures_dir: Path = PROJECT_ROOT / "figures"


@dataclass
class DataCfg:
    tokenizer: str = "bert-base-uncased"
    max_seq_len: int = 128          # 原为 64，会截断 40% 以上的报告，见 README 说明
    projection: str = "Frontal"     # 只使用正位片，与论文 3.2 节一致
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # 训练期数据增强：仅对阳性样本施加几何/亮度扰动（论文 3.2 节）
    aug_rotation: int = 10
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class VQVAECfg:
    in_channels: int = 3
    hidden_dim: int = 256
    num_embeddings: int = 4096
    commitment_cost: float = 0.25   # 论文 3.5 节的 beta
    # 死码激活：每 revive_every 步执行一次，用批内重构误差最大的特征覆写死码
    revive_every: int = 50
    revive_ratio: float = 1.0       # 每次刷新多少比例的死码（1.0 = 全部死码）
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    amp: bool = True
    grad_clip: float = 0.0


@dataclass
class ReportCfg:
    hidden_dim: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # 多任务联合优化
    cls_weight: float = 5.0         # 论文式 (2-9) 的 lambda
    use_class_conditioning: bool = True   # 分类置信度注入解码器每一层（论文 4.3.4）
    label_smoothing: float = 0.0
    # 优化
    batch_size: int = 64
    epochs: int = 60
    lr_head: float = 1e-4
    lr_encoder: float = 5e-5        # 分类梯度直接作用于编码器，需更小学习率
    encoder_frozen: bool = False    # 论文声称分类梯度约束编码器，故默认不冻结
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    patience: int = 10
    amp: bool = True
    # 推理
    beam_size: int = 3
    length_penalty: float = 0.8
    repetition_penalty: float = 1.05
    max_new_tokens: int = 96


@dataclass
class EvalCfg:
    batch_size: int = 16
    bootstrap: int = 1000
    bertscore: bool = False         # 需要联网下载模型，默认关闭
    attention_tokens: tuple = ("heart", "cardiac", "enlarged")


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataCfg = field(default_factory=DataCfg)
    vqvae: VQVAECfg = field(default_factory=VQVAECfg)
    report: ReportCfg = field(default_factory=ReportCfg)
    evalc: EvalCfg = field(default_factory=EvalCfg)
    seed: int = 42

    def to_dict(self):
        d = asdict(self)
        for k, v in list(d["paths"].items()):
            d["paths"][k] = str(v)
        return d

    def ensure_dirs(self):
        for p in (self.paths.splits_dir, self.paths.ckpt_dir,
                  self.paths.results_dir, self.paths.figures_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42, deterministic: bool = True):
    """固定所有随机源，保证同一 seed 下结果可复现。"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = Config()
