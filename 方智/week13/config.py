# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../../week9 序列标准问题/ner/ner_data/schema.json",
    "train_data_path": "../../week9 序列标准问题/ner/ner_data/train",
    "valid_data_path": "../../week9 序列标准问题/ner/ner_data/test",
    "vocab_path":"chars.txt",
    # "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    # "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 16,
    # "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    # "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf":False,
    "class_num": 9,
    "bert_path":r"..\..\week6 语言模型和预训练\bert-base-chinese",
    # "seed": 987
}