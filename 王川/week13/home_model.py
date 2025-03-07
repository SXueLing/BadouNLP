import torch.nn as nn
import torch
from torchcrf import CRF
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # vocab_size = config["vocab_size"] + 1
        # hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hidden_size, class_num)
        # nn.init.xavier_normal_(self.linear.weight)
        # nn.init.zeros_(self.linear.bias)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)        # 对于padding后值为-1的label不做loss计算

    def forward(self, x, y=None):
        # x = self.embedding(x)
        x, _ = self.bert(x)
        x, _ = self.lstm(x)
        y_pred = self.linear(x)     # output shape: bsz * s_len * class_num
        if y is not None:
            if self.use_crf:
                mask = y.gt(-1)     # 值为-1的y不做loss计算
                return -self.crf_layer(y_pred, y, mask, reduction="mean")
            else:
                return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(y_pred)
            else:
                return y_pred

def choose_optimizer(config, model):
    lr = config["learning_rate"]
    optim = config["optimizer"]
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr)
