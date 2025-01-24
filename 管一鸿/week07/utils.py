import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
import os
import re
def split_data(data_path, train_data_path, val_data_path, test_size=0.3, random_state=76):
    """
    将CSV数据分割为训练集和测试集，并保存为新的CSV文件。
    
    :param data_path: 原始数据文件路径 (CSV)
    :param train_data_path: 训练集数据保存路径 (CSV)
    :param val_data_path: 测试集数据保存路径 (CSV)
    :param test_size: 测试集的比例，默认是0.3
    :param random_state: 随机种子，保证每次分割结果相同
    """
    # 读取原始CSV数据
    df = pd.read_csv(data_path)
    
    # 使用train_test_split将数据划分为训练集和测试集
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    
    # 保存训练集和测试集为新的CSV文件
    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)
    
    print(f"数据已成功分割：")
    print(f"训练集保存到: {train_data_path}")
    print(f"测试集保存到: {val_data_path}")



def get_model_path_from_folder(folder_path):
    """
    从文件夹中加载所有的 .pth 模型文件，并提取 config["model_type"] 和 hidden_size。
    :param folder_path: 模型文件所在的文件夹路径
    :return: 模型路径列表，模型类型列表，以及每个模型的 hidden_size
    """
    model_paths = []
    model_types = []
    hidden_sizes = []
    
    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pth"):
            model_path = os.path.join(folder_path, file_name)  # 拼接文件路径
            # 使用正则表达式提取 model_type 和 hidden_size
            match = re.match(r"^(.*?)_lr_(\d+\.\d+)_pooling_style_(.*?)_hidden_size_(\d+)_epoch_(\d+)\.pth$", file_name)
            if match:
                model_type = match.group(1)  # 获取 "_lr_" 之前的部分，即模型类型
                hidden_size = int(match.group(4))  # 获取 hidden_size 后面的值
                model_paths.append(model_path)
                model_types.append(model_type)
                hidden_sizes.append(hidden_size)
            else:
                print(f"无法从文件名 {file_name} 提取模型类型和 hidden_size。")
    
    return model_paths, model_types, hidden_sizes

import csv

def write_to_csv( model, learning_rate, hidden_size, pooling_style, acc, time=0,file_path="./output/result.csv"):
    """
    将传入的模型信息写入到 CSV 文件中。
    :param file_path: CSV 文件的路径
    :param model: 模型名称
    :param learning_rate: 学习率
    :param hidden_size: 隐藏层大小
    :param pooling_style: 池化方式
    :param acc: 准确度
    :param time: 时间，默认为 0,测试后手动填写
    """
    # 检查文件是否存在
    file_exists = os.path.exists(file_path)
    
    # 打开 CSV 文件，如果文件不存在则创建文件
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(['Model', 'Learning_Rate', 'Hidden_Size', 'pooling_style', 'acc', 'time'])
        
        # 写入数据行
        writer.writerow([model, learning_rate, hidden_size, pooling_style, acc, time])

    print(f"数据已写入 {file_path}")


if __name__ == '__main__':
    split_data("./data/data.csv",Config["train_data_path"],Config["valid_data_path"])