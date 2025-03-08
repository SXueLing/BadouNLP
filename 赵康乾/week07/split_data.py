# -*- coding: utf-8 -*-

import csv
import random
from config import Config


seed = Config["seed"] 
random.seed(seed)

def split_data(input_path, train_path, valid_path, split_ratio = 0.8):
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        label = reader[0]
        data = reader[1:]
    
    #打乱数据并分割成训练集和验证集，比例4：1
    random.shuffle(data)
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    valid_data = data[train_size:]

    #添加第一行label后分别保存训练数据和验证数据
    with open(train_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(label)
        writer.writerows(train_data)

    with open(valid_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(label)
        writer.writerows(valid_data)

if __name__ == "__main__":
    input_path = './data/reviews.csv'
    train_path = './data/train_reviews.csv'
    valid_path = './data/valid_reviews.csv'
    split_data(input_path, train_path, valid_path)
