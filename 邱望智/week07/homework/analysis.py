# -*- coding: utf-8 -*-

import csv
import random
from config import Config

"""
数据分析：正负样本数，文本平均长度等
训练集/验证集划分
"""


def data_analysis():
    with open(Config["data_path"], "r", encoding="utf8") as f:
        reader = csv.reader(f)
        list_csv = list(reader)
    pop_row = list_csv.pop(0)  # 去除表头数据
    review_len_dict = {}  # 统计评论长度数据
    for row in list_csv:
        review = row[1]
        review_len_dict[len(review)] = review_len_dict.get(len(review), 0) + 1
    max_length = max_length_by_data(review_len_dict, list_csv)  # 根据统计的评论长度分布，选取合适的文本长度
    print("max_length:", max_length)
    Config["max_length"] = max_length  # 设置文本最大长度
    train_data, test_data = divide_train_test_data(list_csv)  # 训练集，测试集划分
    train_data.insert(0, pop_row)
    test_data.insert(0, pop_row)
    with open(Config["train_data_path"], "w", newline='', encoding="utf8") as f:  # 训练集.csv
        writer = csv.writer(f)
        for row in train_data:
            writer.writerow(row)
    with open(Config["valid_data_path"], "w", newline='', encoding="utf8") as f:  # 测试集.csv
        writer = csv.writer(f)
        for row in test_data:
            writer.writerow(row)


def divide_train_test_data(list_csv):
    positive_review = []
    negative_review = []
    for row in list_csv:
        label = int(row[0])
        if label == 1:
            positive_review.append(row)
        else:
            negative_review.append(row)
    train_data = train_data_set(positive_review, negative_review)
    test_data = [x for x in list_csv if x not in train_data]
    return train_data, test_data


def train_data_set(positive_review, negative_review, train_rate=0.8):
    train_data = []
    positive_sample = random.sample(positive_review, int(len(positive_review) * train_rate))
    negative_sample = random.sample(negative_review, int(len(negative_review) * train_rate))
    train_data.extend(positive_sample)
    train_data.extend(negative_sample)
    return train_data


def max_length_by_data(review_len_dict, list_csv, cover_rate=0.95):
    review_len_dict = sorted(review_len_dict.items(), key=lambda x: x[0], reverse=True)
    remove_num = int(len(list_csv) * (1 - cover_rate))
    max_length = 0
    for key, value in review_len_dict:
        remove_num -= value
        if remove_num <= 0:
            max_length = key
            break
    return max_length


if __name__ == "__main__":
    data_analysis()
