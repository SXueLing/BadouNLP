# -*- coding: utf-8 -*-

CONFIG = {
    "max_length": 128,
    "num_layers": 2,
    "n_epoches": 50,
    "batch_size": 16,
    "hidden_size": 256,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "n_classes": 9,
    "model_path": "model_output",
    "schema_path": "datas/schema.json",
    "train_data_path": "datas/train",
    "test_data_path": "datas/test",
    "vocab_path": "datas/chars.txt",
    "bert_path": "datas/bert-base-chinese",
}
