# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 512,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"E:\BaiduNetdiskDownload\八斗精品课nlp\第六周 语言模型/bert-base-chinese",

    # 模型微调参数
    "hidden_dropout_prob": 0.2,  # 原始默认值0.1
    "attention_probs_dropout_prob": 0.1
}

