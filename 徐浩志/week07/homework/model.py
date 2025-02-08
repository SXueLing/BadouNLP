import torch
import torch.nn as nn
from transformers import BertModel
from config import config

class TorchModel(nn.Module):

    def __init__(self):
        super(TorchModel,self).__init__()
        hidden_size = config['hidden_size']
        self.if_embedding = config['if_embedding']
        class_num = config['class_num']
        encoder_type = config['encoder_type']
        self.embedding = nn.Embedding(5000,hidden_size)

        if encoder_type == 'LSTM':
            self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif encoder_type == 'Bert':
            self.encoder = BertModel.from_pretrained(r"/Users/xuhaozhi/PycharmProjects/pythonProject/AI/week6/bert-base-chinese",return_dict=False)
        elif encoder_type == 'GRU':
            self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size,  class_num)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, x, y=None):
        # embedding层
        if self.if_embedding :
            x = self.embedding(x)

        # encoder层
        out, _ = self.encoder(x)
        x = out[:, -1, :]

        # 线性分类层
        y_pred = self.linear(x)
        y_pred = self.sigmoid(y_pred)

        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y)

