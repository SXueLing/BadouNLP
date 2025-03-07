# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 3,
    "epoch": 20,
    "batch_size": 32,
    "tuning_tactics": "lora_tuning",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\NLP\pretrain_model\bert-base-chinese",
    "seed": 987
}

