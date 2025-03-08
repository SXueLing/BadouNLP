# -*- coding:utf-8 -*-
"""
配置参数信息
"""
import time

workspace_path = rf"D:\PycharmProjects\pythonProject\test_ai\homework\week07"

Config = {
    "model_path": "output",
    "test_data_path": rf"{workspace_path}\data\raw\文本分类练习.csv",
    "train_data_path": rf"{workspace_path}\data\processed\train_data.csv",
    "valid_data_path": rf"{workspace_path}\data\processed\valid_data.csv",
    "result_data_path": rf"{workspace_path}\output\result_data_{time.time()}.csv",
    
    "vocab_path": "../chars.txt",

    "model_type": "bert_lstm",
    "max_length": 30,
    "hidden_size": 256,
    "num_layers": 2,
    # "epoch": 15,
    "epoch": 1,
    "batch_size": 128,
    "pooling_style": "max",

    "optimizer": "adam",
    "learning_rate": 1e-3,

    "pretrain_model_path": rf"{workspace_path}\..\..\pretrain_models\bert-base-chinese",
    "seed": 987
}
