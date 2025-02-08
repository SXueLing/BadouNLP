
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        # 用bert的参数
        self.layer = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify_bert = nn.Linear(self.bert.config.hidden_size, class_num)
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True) # 第一维是batch_size
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1) #loss采用交叉熵损失，padding为-1不用计算

    def forward(self, x, target=None):
        if self.config["model_type"] == "bert":
            x, _ = self.bert(x) # batch_size, sen_len -> batch_size, sen_len, hidden_size
            predict = self.classify_bert(x) # batch_size, sen_len, class_num
        else:
            x = self.embedding(x)
            predict = self.classify(x) # batch_size, sen_len, class_num
        # x, _ = self.layer(x) # batch_size, sen_len, hidden_size
        # print(target, 'x.shape3')
        # torch.Size([16, 100, 9])

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) # labels用-1补齐的，所以忽略-1的预测结果
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict) # 结果是batch_size * sentence_len
            else:
                return predict # 如果没有用crf，返回的结构是batch_size, sen_len, class_num，最后一维代表分类的权重，每一类别都有权重，
                                # 值最大的位置就是预测的那个分类
            
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    
if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 100
    model = TorchModel(Config)
    x = model(torch.LongTensor([[1,2,3], [4,5,6]]))
    print(111)
    print(x) # torch.Size([2, 3, 9])