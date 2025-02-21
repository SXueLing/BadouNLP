 # -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "/Users/jessicachan/nlp20/lstm语言模型生成文本/corpus.txt",
   
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/Users/jessicachan/nlp20/bert-base-chinese",
    "bert_path": r"/Users/jessicachan/nlp20/bert-base-chinese",
    "seed": 987,
    "vocab_size": 21128
}

