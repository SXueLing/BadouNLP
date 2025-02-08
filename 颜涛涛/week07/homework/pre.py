import csv
import random
import pandas as pd
import numpy as np
import torch

from config import Config

# 设置随机值 确保可复现
# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


# 数据预处理  切分训练数据和预测数据
provide_csv_path = Config['provide_csv_path']
# data = pd.read_csv(provide_csv_path, header=1)
data = pd.read_csv(provide_csv_path, skiprows=1)

# 设置训练数据和预测数据的比例
train_ratio = Config['train_ratio']
# shuffle 随机打乱提供的数据 获取索引
shuffle_indices = np.random.permutation(data.index)
# 计算训练数据的大小
train_size = int(len(data) * train_ratio)
# 训练数据的索引
train_indices = shuffle_indices[:train_size]
# 测试数据的索引
evaluate_indices = shuffle_indices[train_size:]

# 根据索引加载训练和测试数据
train_data = data.iloc[train_indices]
evaluate_data = data.iloc[evaluate_indices]

train_data_csv_path = Config['train_data_csv_path']
evaluate_data_csv_path = Config['evaluate_data_csv_path']

train_data.to_csv(train_data_csv_path, index=False)
evaluate_data.to_csv(evaluate_data_csv_path, index=False)


total_count = 0
total_length = 0
total_sentence_size = []
with open(provide_csv_path,'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    # 跳过第一行
    next(reader)
    for line in reader:
        line = line[1:]
        row_text = ''.join(line)
        total_length += len(row_text)
        total_count += 1
sentence_avg_length = total_length / total_count
print(sentence_avg_length)
