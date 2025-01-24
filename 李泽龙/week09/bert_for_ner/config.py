# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "E:/nlp_learn/practice/week09/ner/ner_data/schema.json",
    "train_data_path": "E:/nlp_learn/practice/week09/ner/ner_data/train",
    "valid_data_path": "E:/nlp_learn/practice/week09/ner/ner_data/test",
    "vocab_path":"E:/nlp_learn/practice/week09/bert-base-chinese/vocab.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"E:\nlp_learn\practice\week09\bert-base-chinese",
    "use_bert": True,  #用于切换模型
}

