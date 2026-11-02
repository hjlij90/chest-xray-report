# -*- coding: utf-8 -*-
"""Indiana University Chest X-ray 数据集封装。

相对原实现的关键修改：
1. patient_id 由文件名前缀（"1_IM-0001-4001.dcm.png" -> "1"）解析；
   原实现用 re.extract(r"(\\d+)") 取首个数字段，恰好也能取到患者号，
   但缺少校验，一旦文件名格式变化就会静默把不同患者并成一组。
   这里改为显式前缀匹配并在失败时直接报错。
2. 文本拼接后做统一清洗（空白归一、重复标点合并）。
3. 数据增强只在训练集生效（is_train），验证/测试保持确定性。
4. 报告长度默认 128 个 subword（原 64 会截断约 40% 的报告，见 docs/实验协议.md）。
"""
import os
import re
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_PATIENT_RE = re.compile(r"^(\d+)_")


class IndianaCXRDataset(Dataset):
    def __init__(self, images_dir, projections_csv, reports_csv, tokenizer,
                 max_seq_len=128, projection="Frontal", is_train=False,
                 aug=None, return_patient_id=False):
        self.images_dir = str(images_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.projection = projection
        self.is_train = is_train
        self.return_patient_id = return_patient_id
        self.aug = aug or dict(rotation=10, brightness=0.2, contrast=0.2)

        proj_df = pd.read_csv(projections_csv)
        reports_df = pd.read_csv(reports_csv)

        reports_df = reports_df.dropna(subset=["findings", "impression"], how="all").copy()
        reports_df["report"] = (reports_df["findings"].fillna("") + " "
                                + reports_df["impression"].fillna(""))

        if projection:
            proj_df = proj_df[proj_df["projection"] == projection]
        self.data = pd.merge(proj_df, reports_df, on="uid", how="inner").reset_index(drop=True)

        # 患者标识（用于患者级划分，防止同一患者跨集合泄漏）
        pid = self.data["filename"].astype(str).str.extract(_PATIENT_RE)[0]
        if pid.isna().any():
            bad = self.data.loc[pid.isna(), "filename"].head(5).tolist()
            raise ValueError(f"无法从文件名解析 patient_id：{bad}")
        self.data["patient_id"] = pid.astype(str)

        # 标签：Problems 或 MeSH 中出现 cardiomegaly 即为阳性
        problems = self.data["Problems"].fillna("").str.lower()
        mesh = (self.data["MeSH"].fillna("").str.lower()
                if "MeSH" in self.data.columns else pd.Series([""] * len(self.data)))
        self.data["label"] = ((problems.str.contains("cardiomegaly"))
                              | (mesh.str.contains("cardiomegaly"))).astype(int)
        self.data["report"] = self.data["report"].map(_clean_report)

        self.patient_ids = self.data["patient_id"].values
        pos = int(self.data["label"].sum())
        print(f"[data] 样本 {len(self.data)} 条 | 阳性 {pos} | 阴性 {len(self.data) - pos} "
              f"| 患者 {self.data['patient_id'].nunique()} 人")

        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.base_transform = transforms.Compose([transforms.ToTensor(), norm])
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(degrees=self.aug["rotation"]),
            transforms.ColorJitter(brightness=self.aug["brightness"],
                                   contrast=self.aug["contrast"]),
            transforms.ToTensor(), norm,
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片文件：{img_path}")
        image = Image.open(img_path).convert("RGB")

        label = int(row["label"])
        use_aug = self.is_train and label == 1
        image = self.aug_transform(image) if use_aug else self.base_transform(image)

        enc = self.tokenizer(row["report"], max_length=self.max_seq_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.return_patient_id:
            return image, input_ids, label_tensor, str(row["patient_id"])
        return image, input_ids, label_tensor


def _clean_report(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"(\s?[.,;:])+\s?[.,;:]", ".", text)
    text = re.sub(r"\s+\.", ".", text)
    return text.strip()


def build_tokenizer(cfg):
    from transformers import BertTokenizer
    try:
        return BertTokenizer.from_pretrained(cfg.data.tokenizer, local_files_only=True)
    except Exception:
        return BertTokenizer.from_pretrained(cfg.data.tokenizer)

