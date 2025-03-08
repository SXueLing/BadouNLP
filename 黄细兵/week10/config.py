# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "corpus_path": "corpus.txt",
    "valid_data_path": "ner_data/test",
    "vocab_path":"vocab.txt",
    "max_length": 100,
    "input_dim": 256,  # 每个字的维度
    "num_layers": 1,
    "epoch_num": 20,  # 训练轮数
    "batch_size": 64,   # 每次训练样本个数
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\huang\AI\bert-base-chinese",
    "train_sample": 50000,  # 每轮训练总共训练的样本总数
    "window_size": 10  #窗口大小
}

