import torch
import random
import numpy as np


# 为每个字生成一个标号
# {"a": 1, "b": 2, "c": 3...}
# abc -> [1,2,3]

# 构建一个字典 vocab（字符到数字的映射表，为每一个字符分配唯一的数字编号，便于深度学习训练）
def build_vocab():
    chars = "你bcdefghijklmnopqrstuvwxyz"  # 自定义字符集
    vocab = {"pad": 0}  # 填充字符（Padding），用于在序列处理时对长度不足的序列进行补齐
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字符对应自己的索引值+1
    vocab['unk'] = len(vocab)  # 用于表示 未知字符 的特殊标记（unknown）
    return vocab


# 随机生成一个样本
# 特定字符 “你” 在字符串的第几个位置，则为第几个分类

# 从所有字中选取 sentence_length 个字（共 sentence_length 个类别）
def build_sample(vocab, sentence_length):
    # 随机从字表选取 sentence_length 个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 设置标签
    y = np.zeros(sentence_length, dtype=int)  # 默认所有标签为 0

    # 如果字符 "你" 存在，设置其位置为 1（表示“你”的位置）
    if '你' in x:
        index = x.index('你')  # 获取 "你" 的位置
        y[index] = 1  # 设置为 1，表示在该位置有字符 "你"

    # 将自定义字符集之外的其它字符编号；为了做 embedding
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换为对应的数字索引

    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)  # 构造一组训练样本
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)  # y 应该是 LongTensor 类型


if __name__ == "__main__":
    vocab = build_vocab()  # 构建 vocab 字典
    print(f"Vocabulary: {vocab}")

    # 构建数据集，生成 10 个样本，每个样本有 6 个字符
    dataset_x, dataset_y = build_dataset(10, vocab, 6)

    # 输出数据
    print(f"Dataset X: {dataset_x}")
    print(f"Dataset Y: {dataset_y}")