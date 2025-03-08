'''
Author: Zhao
Date: 2025-01-21 15:31:03
LastEditors: 
LastEditTime: 2025-01-22 11:42:00
FilePath: config.py
Description: 

'''
Config = {
    "model_path": "week9/homework/model_output",
    "schema_path": "week9/data/schema.json",
    "train_data_path": "week9/data/train",
    "valid_data_path": "week9/data/test",
    "vocab_path":"week9/data/chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers" : 2,
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"E:\bert-base-chinese"
}