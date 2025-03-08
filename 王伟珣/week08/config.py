# -*- coding: utf-8 -*-

CONFIG = {
    "hidden_size": 128,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "n_epoches": 20,
    "batch_size": 32,
    "epoch_data_size": 200,
    "max_sentence_len": 20,
    "pretrain_bert_model_path": "../datas/bert-base-chinese",
    "schema_path": "datas/schema.json",
    "train_data_path": "datas/train.json",
    "valid_data_path": "datas/valid.json",
    "vocab_path": "datas/chars.txt",
    "model_path": "model_output",
}
