# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/output",
    "train_data_path": "D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/data/train_tag_news.json",
    "valid_data_path": "D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/data/valid_tag_news.json",
    "vocab_path":"D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 768, # 256
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 30,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\python_project_git\ai_study\week6 语言模型和预训练\week6 语言模型和预训练\bert-base-chinese\bert-base-chinese",
    "seed": 987
}

