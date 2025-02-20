import os
import torch

# Config = {
#     "model_path": "output",
#     "input_max_length": 120,
#     "output_max_length": 30,
#     "epoch": 200,
#     "batch_size": 32,
#     "optimizer": "adam",
#     "learning_rate":1e-3,
#     "seed":42,
#     "vocab_size":6219,
#     "vocab_path":"vocab.txt",
#     "train_data_path": r"sample_data.json",
#     "valid_data_path": r"sample_data.json",
#     "beam_size":5
#     }
Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "vocab_path":"./chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 200,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": True,
    "class_num": 9,
    "bert_path": "./pretrain_models/bert-base-chinese",
}