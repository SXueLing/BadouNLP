# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "pretrain_model_path": r"F:\NLP\资料\week6 语言模型和预训练\bert-base-chinese",
    "model_type": "bert",
    "schema_path": "../ner_data/schema.json",
    "train_data_path": "../ner_data/train",
    "valid_data_path": "../ner_data/test",
    "vocab_path": "../chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"F:\Desktop\work_space\pretrain_models\bert-base-chinese"
}
