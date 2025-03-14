import torch.nn as nn
from config import Config
from transformers import BertForTokenClassification
from torch.optim import Adam, SGD

TorchModel = BertForTokenClassification.from_pretrained(
    Config["pretrain_model_path"], num_labels=9)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)