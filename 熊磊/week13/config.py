# -*- coding: utf-8 -*-

"""
配置参数信息
"""

import os, sys
# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)

# 获取脚本所在目录
script_dir = os.path.dirname(script_path)

# 使用相对路径（假设数据文件在脚本所在目录的 data 子目录下）
data_path = os.path.join(script_dir, "data", "your_file.csv")

model_path = os.path.join(script_dir,  "model_output")
schema_path = os.path.join(script_dir, "ner_data", "schema.json")
train_data_path = os.path.join(script_dir, "ner_data", "train")
valid_data_path = os.path.join(script_dir, "ner_data", "test")
vocab_path = os.path.join(script_dir, "chars.txt")



Config = {
    "model_path": model_path,
    "schema_path": schema_path,
    "train_data_path": train_data_path,
    "valid_data_path": valid_data_path,
    "vocab_path": vocab_path,
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 50,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"F:\motrix\新建文件夹\深度学习\第六周语言模型\bert-base-chinese",
    "model_type": "bert",
    "vocab_size": 4622,
}

if __name__ == "__main__":
    # import os, sys
    # # 获取当前脚本的绝对路径
    # script_path = os.path.abspath(__file__)

    # # 获取脚本所在目录
    # script_dir = os.path.dirname(script_path)
    # print(script_dir)
    # # 使用相对路径（假设数据文件在脚本所在目录的 data 子目录下）
    # data_path = os.path.join(script_dir, "data", "your_file.csv")
    # print(data_path)
    pass
