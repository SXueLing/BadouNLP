# # -*- coding: utf-8 -*-

# """
# 配置参数信息
# """

# Config = {
#     "model_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/ner/model_output",
#     "schema_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/ner/ner_data/schema.json",
#     "train_data_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/ner/ner_data/train",
#     "valid_data_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/ner/ner_data/test",
#     "vocab_path":"D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/ner/chars.txt",
#     "max_length": 100,
#     "hidden_size": 256,
#     "num_layers": 2,
#     "epoch": 20,
#     "batch_size": 16,
#     "optimizer": "adam",
#     "learning_rate": 1e-3,
#     "use_crf": False,
#     "class_num": 9,
#     "bert_path": r"D:\python_project_git\ai_study\week6 语言模型和预训练\week6 语言模型和预训练\bert-base-chinese\bert-base-chinese"
# }

# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/week13/model_output",
    "schema_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/week13/ner_data/schema.json",
    "train_data_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/week13/ner_data/train",
    "valid_data_path": "D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/week13/ner_data/test",
    "vocab_path":"D:/python_project_git/ai_study/week13 大语言模型相关第三讲/week13 大语言模型相关第三讲/week13/chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "seed": 987,
    "bert_path": r"D:\python_project_git\ai_study\week6 语言模型和预训练\week6 语言模型和预训练\bert-base-chinese\bert-base-chinese"
}

