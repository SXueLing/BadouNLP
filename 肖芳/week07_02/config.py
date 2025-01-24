# -*- coding: utf-8 -*-
import fancy_utils as utils
from enum import Enum

"""
配置参数信息
"""
# 所有可用的模型
class ModelType(Enum):
    fast_text = "fast_text"
    lstm = "lstm"
    gru = "gru"
    rnn = "rnn"
    cnn = "cnn"
    gated_cnn = "gated_cnn"
    stack_gated_cnn = "stack_gated_cnn"
    rcnn = "rcnn"
    bert = "bert"
    bert_lstm = "bert_lstm"
    bert_cnn = "bert_cnn"
    bert_mid_layer = "bert_mid_layer"

# Pooling 类型
class PoolingType(Enum):
    max = "max"
    avg = "avg"

class OptimizerType(Enum):
    adam = "adam"
    sgd = "sgd"

Config = {
    "model_path": "output",
    "train_data_path": "data/train_datas.csv",
    "valid_data_path": "data/val_datas.csv",
    "vocab_path": "data/vocab.txt",
    "model_type": ModelType.lstm.value,
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":PoolingType.avg.value,
    "optimizer": OptimizerType.adam.value,
    "learning_rate": 1e-3,
    "pretrain_model_path": utils.BERT_PATH,
    "seed": 987
}

