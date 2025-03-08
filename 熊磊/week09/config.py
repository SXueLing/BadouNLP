# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": r"F:\motrix\新建文件夹\深度学习\第九周序列标注\week9 序列标注问题\ner_bert\ner_data\schema.json",
    "train_data_path": r"F:\motrix\新建文件夹\深度学习\第九周序列标注\week9 序列标注问题\ner_bert\ner_data\train",
    "valid_data_path": r"F:\motrix\新建文件夹\深度学习\第九周序列标注\week9 序列标注问题\ner_bert\ner_data\test",
    "vocab_path":r"F:\motrix\新建文件夹\深度学习\第九周序列标注\week9 序列标注问题\ner_bert\chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"F:\motrix\新建文件夹\深度学习\第六周语言模型\bert-base-chinese",
    "model_type": "bert",
    "vocab_size": 4622,
}

