# -*- coding: utf-8 -*-

"""
配置参数信息
"""

workspace_path = rf"D:\PycharmProjects\pythonProject\test_ai\homework\week09"

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "pretrain_model_path": rf"{workspace_path}\..\..\pretrain_models\bert-base-chinese",
}

