import numpy as np
import torch

# 生成一个样本
def build_sample():
    x = np.random.random(5) * 100
    # 找出最大值所在的索引
    max_index = np.argmax(x)   # 输出最大值的索引
    # y = [0,0,0,0,0]
    # y[max_index] = 1
    # print(f'样本:{x}，类别：{y}')

    return x, max_index   # 返回一个样本，和它的分类

# 生成样本数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # print(y)
        Y.append(y)
    # return X, Y
    return torch.FloatTensor(X), torch.tensor(Y)


build_dataset(10)

