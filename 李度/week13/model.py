import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from torch.optim import Adam, SGD

# NER用的类不一样
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["pretrain_model_path"],
    num_labels=9
)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
