# -*- coding: utf-8 -*-

"""
数据预处理
"""

import pandas as pd
from sklearn.model_selection import train_test_split

class DataPre:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
    def split(self):
        # 读取CSV文件
        df = pd.read_csv(self.data_path, encoding='utf-8')
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        # 可选：保存划分后的数据集
        train_df.to_csv(self.config["train_data_path"], index=False)
        val_df.to_csv(self.config["valid_data_path"], index=False)
    def analysis(self):
        # 读取CSV文件
        df = pd.read_csv(self.data_path, encoding='utf-8')
        # 计算label为0和1的数量
        count_label_0 = (df['label'] == 0).sum()
        count_label_1 = (df['label'] == 1).sum()
        # 计算review的平均长度
        average_length = df['review'].str.len().mean()
        # 计算小于平均长度的review数量
        count_less_than_average = (df['review'].str.len() < average_length).sum()
        # 计算大于平均长度的review数量
        count_greater_than_average = (df['review'].str.len() > average_length).sum()
        # 打印结果
        print("Label为0的数量:", count_label_0)
        print("Label为1的数量:", count_label_1)
        print("Review的平均长度:", average_length)
        print("小于平均长度的Review数量:", count_less_than_average)
        print("大于平均长度的Review数量:", count_greater_than_average)
        result = [count_label_0, count_label_1, average_length, count_less_than_average, count_greater_than_average]
        return result
    
if __name__ == "__main__":
    from config import Config
    dp = DataPre(Config["data_path"], Config)
    dp.analysis()

