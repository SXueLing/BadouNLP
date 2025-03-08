import torch.nn as nn
import torch.optim as op
from transformers import BertTokenizer, AutoTokenizer

CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "model": "bert",
    "use_bert_pooler": False,
    "optimizer": "adamw",
    "loss": "ce",
    "tokenizer": "bert",
    "pooling": "max",
    "seq_max_length": 256,
    "encoder_layers": 5,
}

LOSS_FUNC = {
    "ce": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bcel": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
}

OPTIMIZER = {
    "sgd": op.SGD,
    "adam": op.Adam,
    "adamw": op.AdamW,
    "rmsp": op.RMSprop,
}

TOKENIZER = {
    "bert": BertTokenizer,
    "auto": AutoTokenizer,
}


def load_config():
    result = CONFIG.copy()
    result["optimizer"] = OPTIMIZER[result["optimizer"]]
    result["loss"] = LOSS_FUNC[result["loss"]]
    result["tokenizer"] = TOKENIZER[result["tokenizer"]]
    return result
