import torch.nn as nn
import torch
from torchcrf import CRF
from transformers import BertModel
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        # hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        # num_layers = config["num_layers"]
        class_num = config["class_num"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx = 0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True, bidirectional = True, num_layers = num_layers)
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict = False)
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, class_num)
        # self.linear = nn.Linear(2 * hidden_size, class_num)
        self.crf = CRF(class_num, batch_first = True)
        self.usr_crf = config["use_crf"]
        self.loss = nn.CrossEntropyLoss(ignore_index = -1)

    def forward(self, x, target = None):
        # x = self.embedding(x)
        # x, _ = self.lstm(x)
        x, _ = self.bert(x)
        predict = self.linear(x)
        if target is not None:
            if self.usr_crf:
                mask = target.gt(-1)
                return - self.crf(predict, target, mask, reduction = "mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.usr_crf:
                return self.crf.decode(predict)
            else:
                return predict

def choose_optimizer(config, model):
    optim = config["optimizer"]
    lr = config["learning_rate"]
    if optim == "SGD":
        return torch.optim.SGD(model.parameters(), lr)
    else:
        return torch.optim.Adam(model.parameters(), lr)

