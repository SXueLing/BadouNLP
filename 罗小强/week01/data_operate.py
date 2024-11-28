# -*- coding: utf-8 -*-
import torch

# Initialize a tensor containing numbers from 0 to 11
x = torch.arange(12)

print(f"x is {x}")

# 打印变量x的形状信息
print(f'{x.shape}')

# Print the number of elements in the tensor x
print(f'{x.numel()}')

# Reshape the variable x into a two-dimensional array of shape 3x4
X = x.reshape(3, 4)

print(f'{X}')

# 初始化一个形状为(2, 3, 4)的张量，用于后续的深度学习模型训练或数据处理
# 选择全零张量是为了在开始训练或处理前，提供一个没有先前信息的初始状态
X = torch.zeros((2, 3, 4))

print(f'{X}')

# 初始化一个形状为(2, 3, 4)的张量X，所有元素值为1
# torch.ones 函数用于创建一个所有元素都为 1 的张量。关于第一个参数的括号使用，有以下区别：
# 有括号：当第一个参数被括号包围时，括号内的参数会被视为一个元组，表示张量的形状
# 无括号：当第一个参数没有被括号包围时，参数会被直接作为张量的形状
# 总结：
# 有括号：适用于明确指定一个元组作为形状参数。
# 没有括号：适用于直接传入多个参数，每个参数代表一个维度。
# 这两种方式在功能上是等价的，选择哪种方式主要取决于个人偏好和代码的可读性。

X = torch.ones((2, 3, 4))

print(f'{X}')

# 初始化一个形状为3x4的张量X，其元素从标准正态分布（均值为0，方差为1）中随机采样
X = torch.randn(3, 4)

print(f'{X}')

# 初始化一个2维张量X，用于后续的神经网络训练或数据处理
# 这个张量代表了一个4x4的矩阵，包含了需要被处理的数据
X = torch.tensor([[1, 2, 3, 5],
                  [7, 4, 5, 7],
                  [12, 24, 45, 78],
                  [44, 55, 33, 77]])
print(f'{X}')

x = torch.tensor([1, 2, 3, 4, 6])
y = torch.tensor([2, 3, 4, 5, 6])
print(f'{x + y}')
print(f'{x * y}')
print(f'{x - y}')
print(f'{x / y}')
print(f'{x ** y}')
# 对张量中的所有元素进行求和，会产生一个单元素张量
print(f'{x.to(torch.float32).sum()}')
print(f'{x.to(torch.float32).mean()}')
print(f'{x.to(torch.float32).std()}')
print(f'{x.argmax()}')
# 计算并打印变量x的指数值，使用torch库的exp方法
print(f'{torch.exp(x.to(torch.float32))}')
'''
逻辑运算符构建二元张量。 
以X == Y为例： 
对于每个位置，如果X和Y在该位置相等，则新张量中相应项的值为1。 
这意味着逻辑语句X == Y在该位置处为真，否则该位置为0。
'''
print(f'{x == y}')

# 创建一个包含12个元素的张量，并将其重塑为3x4的形状
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# 直接创建一个指定值的张量y
y = torch.tensor([[1, 2, 3, 4], [5, 6, 6, 7], [7, 8, 9, 10]])
'''
多个张量连结（concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量。 
我们只需要提供张量列表，并给出沿哪个轴连结。
在PyTorch中，可以使用 torch.cat 函数将多个张量沿着指定的维度进行拼接。具体来说：
torch.cat(tensors, dim=0)：沿着指定的维度 dim 将张量列表 tensors 拼接在一起。
dim 参数决定了拼接的方向：
dim=0：沿着行方向拼接。维度0：沿着行方向拼接，结果是一个新的张量，行数增加，列数不变。
dim=1：沿着列方向拼接 维度1：沿着列方向拼接，结果是一个新的张量，列数增加，行数不变。
'''
# 在维度0上拼接张量x和y，并打印结果
print(f'{torch.cat((x, y), dim=0)}')
# 在维度1上拼接张量x和y，并打印结果
print(f'{torch.cat((x, y), dim=1)}')
