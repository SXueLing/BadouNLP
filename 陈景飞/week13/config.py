# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "schema_path": "E:/nlp_learn/practice/week09/ner/ner_data/schema.json",
    "train_data_path": "E:/nlp_learn/practice/week09/ner/ner_data/train",
    "valid_data_path": "E:/nlp_learn/practice/week09/ner/ner_data/test",
    "vocab_path":"E:/nlp_learn/practice/week09/bert-base-chinese/vocab.txt",
    "model_type":"bert",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "class_num": 9,
    "use_crf": False,
    "bert_path": r"E:\nlp_learn\practice\week09\bert-base-chinese",
    "seed": 987
}

