"""
用于生成样例数据
"""
import numpy as np
import torch

def data_builder():
    #生成随机五维向量
    data = np.random.random(5)
    #找到最大的向量序列
    max_data = np.max(data)
    max_value_indices = float(np.where(data == max_data)[0][0])
    #返回五维向量及其分类
    # print(data,max_value_indices)
    return data,max_value_indices

def dataset_builder(num):
    #初始化x，y
    x = []
    y = []
    #循环装入对应数据
    for i in range(num):
        m,n = data_builder()
        x.append(m)
        y.append(n)
    return torch.FloatTensor(np.array(x)),torch.LongTensor(np.array(y))
if __name__ == "__main__":
    print(dataset_builder(3))
