# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "/Users/jessicachan/nlp20/ner/ner_data/train",
    "valid_data_path": "/Users/jessicachan/nlp20/ner/ner_data/test",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 20,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 64,
    "use_crf": False,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "class_num": 9,
    "pretrain_model_path":"/Users/jessicachan/nlp20/bert-base-chinese",
    "seed": 987
}
