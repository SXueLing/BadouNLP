# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"D:\projects\nlp\week9\model_output",
    "schema_path": r"D:\projects\nlp\week9\ner_data\schema.json",
    "train_data_path": r"D:\projects\nlp\week9\ner_data/train",
    "valid_data_path": r"D:\projects\nlp\week9\ner_data/test",
    "vocab_path":r"D:\projects\nlp\week9\chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 100,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\projects\nlp\week6\bert-base-chinese\bert-base-chinese"
}

