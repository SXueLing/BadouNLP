# -*- coding: utf-8 -*-
"""
test_fun.py
描述: 
作者: TomLuo 21429503@qq.com
日期: 11/26/2024
版本: 1.0
"""
import numpy as np

def convert_to_one_hot(vector):
    """
    将一个5维向量转换为只有最大值位置为1，其余位置为0的向量。

    参数:
    vector (list or np.array): 输入的5维向量

    返回:
    np.array: 转换后的one-hot向量
    """
    max_index = np.argmax(vector)  # 找到最大值的索引
    one_hot_vector = np.zeros_like(vector)  # 创建一个全0的向量
    one_hot_vector[max_index] = 1  # 将最大值位置设置为1
    return one_hot_vector

# 示例
vector = [0.86889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843]
one_hot_vector = convert_to_one_hot(vector)
print(one_hot_vector)
