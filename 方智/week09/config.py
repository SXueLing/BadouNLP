# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",# 模型保存路径
    "schema_path": "ner_data/schema.json",# 标签schema文件路径
    "train_data_path": "ner_data/train",# 训练数据集路径
    "valid_data_path": "ner_data/test",# 验证数据集路径
    "vocab_path":"chars.txt",# 词表文件路径
    "max_length": 100,# 最大序列长度
    "hidden_size": 256,# 隐藏层大小
    "num_layers": 2,# 隐藏层层数
    "epoch": 20,# 训练轮数
    "batch_size": 16,# 批处理大小
    "optimizer": "adam",# 优化器
    "learning_rate": 1e-3,# 学习率
    "use_crf": False,# 是否使用CRF层
    "class_num": 9,# 类别数
    "bert_path": r"../../week6 语言模型和预训练/bert-base-chinese"# bert预训练模型路径
}

