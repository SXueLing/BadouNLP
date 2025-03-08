'''
Author: Zhao
Date: 2025-01-08 19:32:54
LastEditTime: 2025-01-09 15:48:48
FilePath: config.py
Description: 

'''
# 配置参数信息

Config = {
    # 模型存放路径
    "model_path": "week7/nn_pipline/model",
    #训练语料
    "train_data_path": "week7/data/train_tag_news.json",    # 训练集
    "valid_data_path": "week7/data/valid_tag_news.json",    # 验证集 
    "test_path": "week7/data/test_file.json",          # 测试集
    "vocab_path":"week7/data/chars.txt",                    # 词表
    #"class_num" : 3,
    #"vocab_size": 20,
    "model_type":"bert",
    "max_length": 30,
    "hidden_size" : 256,                                    # lstm隐藏层
    "in_channels" : 3,
    "out_channels" : 4,
    "kernel_size" : 3,                                      # 卷积核（过滤器）的大小
    "num_layers" : 2,                                       # lstm层数
    "conv_type" : "1d",                                     # 1D卷积或者2D卷积
    "epoch": 15,                                            # epoch数
    "batch_size": 128,
    "pooling_style":"max",                                  #Pooling类型
    "optimizer": "adam",                                    #优化器
    "learning_rate": 1e-3,                                  #学习率，大模型一般调低
    #bert 路径
    "pretrain_model_path":r"E:\第六周 语言模型\bert-base-chinese",
    "output_hidden_states": False,
    "seed": 987
}