"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/valid",
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "max_length": 30,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": "../bert-base-chinese"
}
