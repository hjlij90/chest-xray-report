# -*- coding: utf-8 -*-
"""数据划分与 DataLoader 构建。

统一协议：患者级（patient-level）GroupShuffleSplit，80/10/10，seed=42。
划分结果落盘到 splits/patient_split_seed42.json，所有脚本共用同一份，
避免出现"主模型用随机划分、基线用患者划分"这种不可比的实验。
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from dataset import IndianaCXRDataset, build_tokenizer


def _dataset(cfg, tokenizer, is_train=False, with_pid=False):
    return IndianaCXRDataset(
        images_dir=cfg.paths.images,
        projections_csv=cfg.paths.projections_csv,
        reports_csv=cfg.paths.reports_csv,
        tokenizer=tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        projection=cfg.data.projection,
        is_train=is_train,
        aug=dict(rotation=cfg.data.aug_rotation, brightness=cfg.data.aug_brightness,
                 contrast=cfg.data.aug_contrast),
        return_patient_id=with_pid,
    )


def get_patient_split(cfg, tokenizer=None):
    """返回 (train_idx, val_idx, test_idx)，并在磁盘上复用同一份划分。"""
    tokenizer = tokenizer or build_tokenizer(cfg)
    cfg.ensure_dirs()
    cache = Path(cfg.paths.splits_dir) / f"patient_split_seed{cfg.data.split_seed}.json"

    if cache.exists():
        obj = json.loads(cache.read_text(encoding="utf-8"))
        meta = obj["meta"]
        return obj["train"], obj["val"], obj["test"], meta

    ds = _dataset(cfg, tokenizer, with_pid=True)
    groups = ds.patient_ids
    n = len(ds)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=cfg.data.val_ratio + (1 - cfg.data.train_ratio - cfg.data.val_ratio),
                             random_state=cfg.data.split_seed)
    train_idx, temp_idx = next(gss1.split(np.zeros(n), groups=groups))
    groups2 = groups[temp_idx]
    # 在剩余 20% 里再按患者二分成 val / test（各 10%）
    rel = cfg.data.val_ratio / (cfg.data.val_ratio + (1 - cfg.data.train_ratio - cfg.data.val_ratio))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - rel, random_state=cfg.data.split_seed)
    val_rel, test_rel = next(gss2.split(np.zeros(len(temp_idx)), groups=groups2))

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(temp_idx)[val_rel]
    test_idx = np.asarray(temp_idx)[test_rel]

    meta = dict(
        n=n, seed=cfg.data.split_seed,
        train=len(train_idx), val=len(val_idx), test=len(test_idx),
        train_pos=int(ds.data.iloc[train_idx]["label"].sum()),
        val_pos=int(ds.data.iloc[val_idx]["label"].sum()),
        test_pos=int(ds.data.iloc[test_idx]["label"].sum()),
        train_patients=int(ds.data.iloc[train_idx]["patient_id"].nunique()),
        val_patients=int(ds.data.iloc[val_idx]["patient_id"].nunique()),
        test_patients=int(ds.data.iloc[test_idx]["patient_id"].nunique()),
        patients=ds.data["patient_id"].nunique(),
        max_seq_len=cfg.data.max_seq_len,
        projection=cfg.data.projection,
    )
    # 断言三集合患者不重叠
    p_tr = set(ds.data.iloc[train_idx]["patient_id"])
    p_va = set(ds.data.iloc[val_idx]["patient_id"])
    p_te = set(ds.data.iloc[test_idx]["patient_id"])
    assert not (p_tr & p_va) and not (p_tr & p_te) and not (p_va & p_te), "患者级划分出现交集"

    cache.write_text(json.dumps(dict(train=train_idx.tolist(), val=val_idx.tolist(),
                                     test=test_idx.tolist(), meta=meta), ensure_ascii=False,
                                indent=1), encoding="utf-8")
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), meta


def build_loaders(cfg, tokenizer=None, batch_size=None):
    tokenizer = tokenizer or build_tokenizer(cfg)
    tr, va, te, meta = get_patient_split(cfg, tokenizer)

    full = _dataset(cfg, tokenizer, is_train=False, with_pid=True)
    train_ds = _dataset(cfg, tokenizer, is_train=True)
    assert len(full) == len(train_ds), "训练集与基础数据集顺序不一致，索引将失效"

    bs = batch_size or cfg.report.batch_size
    train_loader = DataLoader(Subset(train_ds, tr), batch_size=bs, shuffle=True, drop_last=True,
                              num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                              persistent_workers=cfg.data.num_workers > 0)
    val_loader = DataLoader(Subset(full, va), batch_size=bs, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                            persistent_workers=cfg.data.num_workers > 0)
    test_loader = DataLoader(Subset(full, te), batch_size=cfg.evalc.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                             persistent_workers=cfg.data.num_workers > 0)
    return dict(train=train_loader, val=val_loader, test=test_loader), full, (tr, va, te), meta
