import pandas as pd
import numpy as np
import os

def extract_cardiomegaly_dataset(reports_path: str, projections_path: str, output_csv: str = "clean_cardiomegaly.csv") -> None:
    print("🚀 开始执行数据清洗与特征提纯链路...")
    
    # 1. 加载数据
    df_reports = pd.read_csv(reports_path)
    df_projections = pd.read_csv(projections_path)
    
    # 2. 内连接 (Inner Join): 确保有图的才有报告，有报告的才有图
    df_merged = pd.merge(df_projections, df_reports, on='uid', how='inner')
    df_merged = df_merged[df_merged['projection'].str.contains('Frontal', case=False, na=False)].copy()
    # 3. 文本拼接与清洗
    # 填充 NaN 为空字符串，使用向量化操作拼接
    df_merged['findings'] = df_merged['findings'].fillna('')
    df_merged['impression'] = df_merged['impression'].fillna('')
    df_merged['report_text'] = df_merged['findings'] + " " + df_merged['impression']
    
    # 过滤掉完全没有报告内容的废数据 (长度 < 10)
    df_merged = df_merged[df_merged['report_text'].str.strip().str.len() > 10].copy()
    
    # 4. 二分类打标 (Labeling)
    df_merged['MeSH'] = df_merged['MeSH'].fillna('')
    # 正则匹配，忽略大小写
    df_merged['label'] = np.where(df_merged['MeSH'].str.contains('cardiomegaly', case=False, na=False, regex=True), 1, 0)
    
    # 5. 组装并去重
    # 选取必需列: 图片名, 投影方向, 文本, 标签
    final_cols = ['filename', 'projection', 'report_text', 'label']
    df_final = df_merged[final_cols].drop_duplicates(subset=['filename']).reset_index(drop=True)
    
    # 🛡️ 强制断言检查 (防数据穿透)
    assert df_final['label'].isin([0, 1]).all(), "标签生成异常，包含非 0/1 值！"
    assert df_final['report_text'].isnull().sum() == 0, "清洗后依然存在空文本！"
    
    # 保存结果
    df_final.to_csv(output_csv, index=False)
    
    pos_count = df_final['label'].sum()
    neg_count = len(df_final) - pos_count
    
    print(f"✅ 清洗完成！已保存至 {output_csv}")
    print(f"📊 最终可用图像总数: {len(df_final)}")
    print(f"📈 正样本(1): {pos_count} | 📉 负样本(0): {neg_count}")
    print(f"⚖️ 建议在分类损失函数中设置正样本权重 (pos_weight) 约为: {neg_count/pos_count:.2f}")

if __name__ == "__main__":
    # 替换为你实际的文件路径
    extract_cardiomegaly_dataset('E:\程序\datasets\indiana_projections.csv', 'E:\程序\datasets\indiana_reports.csv')