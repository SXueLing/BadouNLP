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
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 3,
    "epoch": 10,
    "batch_size": 30,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "pretrain_model_path":r"D:\learning\week6\xiawu\bert-base-chinese",
    "model_type": "bert",
    "tuning_tactics": "lora_tuning",
    "pooling_style": "max",
    "seed": 987,
    "lora_rank": 8,  # LoRA的秩
    "lora_alpha": 16,  # LoRA的alpha
    "lora_dropout": 0.1,  # LoRA的dropout
}
