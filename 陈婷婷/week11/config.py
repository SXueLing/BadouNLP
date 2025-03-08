# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "data_path": r"/Users/jessicachan/nlp20/transformers-生成文章标题/sample_data.json",
   
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 1,
    "epoch": 20,
    "batch_size": 40,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/Users/jessicachan/nlp20/bert-base-chinese",
    "bert_path": r"/Users/jessicachan/nlp20/bert-base-chinese",
    "seed": 987,
    "vocab_size": 21128
}

