# -*- coding: utf-8 -*-
import random
from config import Config

"""
切割训练集和验证集
"""

def split_file(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
    random.shuffle(lines)
    total_lines_num = len(lines)
    train_lines_num = int(0.8 * total_lines_num)

    train_lines = lines[:train_lines_num]
    valid_lines = lines[train_lines_num:]

    with open(Config['train_data_path'], 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)

    with open(Config['valid_data_path'], 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)


split_file(Config['original_data_path'])
