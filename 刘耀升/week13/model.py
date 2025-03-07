import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, PreTrainedModel, BertConfig
import torch
#TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"])
class TorchModel(PreTrainedModel):
    def __init__(self, config):
        bertConfig = BertConfig.from_pretrained(config["pretrain_model_path"], num_labels=config["class_num"])
        super().__init__(bertConfig)
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        x = self.bert(x)[0]
        x = self.dropout(x)
        predict = self.classifier(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
        if target is not None:
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            return predict

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
