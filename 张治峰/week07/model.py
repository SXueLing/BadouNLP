import torch
import torch.nn as nn
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self,config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        model_type = config["model_type"]
        num_embeddings =config["vocab_size"]+1
        self.pooling_style = config["pooling_style"]
        self.use_bert = False
        self.embedding = nn.Embedding(num_embeddings,hidden_size,padding_idx=0)
        if model_type =='lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type =='bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, 2) # 输出二分类
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    def forward(self,x,target=None):
        if self.use_bert:
            x = self.encoder(x)   
        else:
           x =  self.embedding(x)
           x =  self.encoder(x)   

        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]        
        if self.pooling_style=='max':
            self.pooling_layer  = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1,2)).squeeze()    
        predict = self.classify(x)  
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x