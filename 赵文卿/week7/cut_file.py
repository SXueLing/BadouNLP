'''
Author: Zhao
Date: 2025-01-08 19:24:57
LastEditTime: 2025-01-09 15:49:03
FilePath: cut_file.py
Description: 

'''
import os
import json
import random
import shutil

#将源目录下文件80%用于训练，20%用于测试
def split_data(source_dir, train_dir, test_dir, train_radio= 0.8):
    # 获取所有文件名
    all_file = os.listdir(source_dir)
    # 打乱文件列表
    random.shuffle(all_file)
    # 计算训练数据的数量
    train_size = int(len(all_file) * train_radio)

    # 分割训练数据和测试数据
    train_files = all_file[:train_size]
    test_files = all_file[train_size:]

    # 确保目标目录存在
    os.makedirs(train_dir, exist_ok=True) 
    os.makedirs(test_dir, exist_ok=True)

    # 复制文件到相应目录
    # for file in train_files:
    #     print(f"复制训练文件到相应目录: {train_dir}")
    #     shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    # for file in test_files: 
    #     print(f"复制测试文件到相应目录: {test_dir}")
    #     shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

def split_json_data(source_file, train_file, test_file, train_ratio=0.8):
    # 读取源数据文件
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 打乱数据顺序 
    random.shuffle(lines)

    # 计算训练数据的数量 
    train_size = int(len(lines) * train_ratio)
                     
    # 分割数据 
    train_lines = lines[:train_size] 
    test_lines = lines[train_size:]

    # 保存训练数据到文件
    with open(train_file, 'w', encoding='utf-8') as f: 
        print("保存训练数据到文件")
        f.writelines(train_lines)
    # 保存测试数据到文件
    with open(test_file, 'w', encoding='utf-8') as f:
        print("保存测试数据到文件")
        f.writelines(test_lines)

            
# if __name__ == "__main__":
#     # 测试
#     # source_directory = "week7/test/source_directory" 
#     # train_directory = "week7/test/train_directory" 
#     # test_directory = "week7/test/test_directory"
#     # split_data(source_directory, train_directory, test_directory)

#     source_file = "week7/data/tag_news.json" 
#     train_file = "week7/data/train_file.json" 
#     test_file = "week7/data/test_file.json" 
#     split_json_data(source_file, train_file, test_file)
#     pass
