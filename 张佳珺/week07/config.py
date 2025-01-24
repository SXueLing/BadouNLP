# -*- coding: utf-8 -*-

"""
配置参数信息
"""
# D:\八斗课程\AI人工智能培训\第七周 文本分类\week7 文本分类问题
Config = {
    "model_path": "output",
    # "train_data_path": "../data/train_tag_news.json",
    # "valid_data_path": "../data/valid_tag_news.json",
    # "vocab_path":"chars.txt",
    "train_data_path": r"F:/八斗课程/AI人工智能培训/第七周 文本分类/week7 文本分类问题/train_data.json",
    "valid_data_path": r"F:/八斗课程/AI人工智能培训/第七周 文本分类/week7 文本分类问题/test_data.json",
    "vocab_path":r"F:/八斗课程/AI人工智能培训/第七周 文本分类/week7 文本分类问题/nn_pipline/chars.txt",
    "model_type":"fast_text",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"F:\八斗课程\AI人工智能培训\第六周 语言模型\bert-base-chinese",
    "seed": 987
}

