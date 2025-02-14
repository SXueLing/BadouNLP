# -*- coding: utf-8 -*-
# @Date    :2025-02-12 21:24:52
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam, SGD


class TorchModel(nn.Module):
    """docstring for TorchModel"""

    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.class_num = config["class_num"]
        self.model_type = config["model_type"]
        self.num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.hidden_size,
                                      padding_idx=0)
        if self.model_type == "fast_text":
            self.encoder = lambda x: x
        elif self.model_type == "lstm":
            self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        elif self.model_type == "gru":
            self.encoder = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        elif self.model_type == "rnn":
            self.encoder = nn.RNN(self.hidden_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        elif self.model_type == "cnn":
            self.encoder = CNN(config)
        elif self.model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif self.model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif self.model_type == "rcnn":
            self.encoder = RCNN(config)
        elif self.model_type == "bert":
            # 以bert的hidden_size位置
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif self.model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif self.model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif self.model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(self.hidden_size,self.class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

    def forward(self,x,target=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x,tuple):
            x = x[0] # [batch,lenght,embedding]
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])   # 将length->1
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])

        x = self.pooling_layer(x.transpose(-1,-2)).squeeze(-1) 
        # x:batch,embedding
        predict = self.classify(x)  # batch,class_num

        if target is not None:
            return self.loss(predict,target.squeeze(-1))
        else:
            return predict



class CNN(nn.Module):
    """docstring for CNN
    x:[batch,lenght,embedding]
    out:[batch,lenght,embedding]
    """

    def __init__(self, config):
        super(CNN, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.kernel_size = config["kernel_size"]
        pad = int((self.kernel_size-1)/2)  # 保证序列长度不变，通过扩充边缘实现，原尺寸的数据输出
        self.cnn = nn.Conv1d(in_channels=self.hidden_size,
                             out_channels=self.hidden_size,
                             kernel_size=self.kernel_size,
                             bias=False,
                             padding=pad)

    def forward(self, x: torch.Tensor):
        # x [batch,max_len,embedding_size]
        x = x.transpose(1, 2)  # 符合卷积数据
        out = self.cnn(x).transpose(1, 2)  # 符合序列数据
        return out


class GatedCNN(nn.Module):
    """docstring for GateCNN"""
    # 数据尺寸并不发生改变

    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config=config)
        self.gate = CNN(config=config)

    def forward(self, x: torch.Tensor):
        a = self.cnn(x)  # [batch,lenght,embedding]
        b = self.gate(x)
        b = torch.sigmoid(b)  # 进行一个门化
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    """docstring for StackGatedCNN"""
    # 数据尺寸并不发生改变

    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers))
        self.ff_liner_layer1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers))
        self.ff_liner_layer2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers))
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers))
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers))

    def forward(self, x: torch.Tensor):
        # 模仿Transformer多层结构，堆叠多层使用
        for i in range(self.num_layers):
            # 进行gateCNN
            gcnn_x = self.gcnn_layers[i](x)
            # 进入残差归一化层
            x = gcnn_x + x
            x = self.bn_after_gcnn[i](x)
            # 进入前馈神经网络
            l1 = self.ff_liner_layer1[i](x)
            l1 = torch.relu(l1)
            l2 = self.ff_liner_layer2[i](x)
            # 再进入残差归一化
            x = self.bn_after_ff[i](x+l2)
        return x


class RCNN(nn.Module):
    """docstring for R_GateCNN"""
    # 数据格式保持不变

    def __init__(self, config):
        super(RCNN, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(input_size=self.hidden_size,
                          hidden_size=self.hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x: torch.tensor):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    """docstring for ClassName"""
    # 自带embedding,传入的数据也有所不同

    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(
            config["pretrain_model_path"], return_dict=False)
        # print(self.bert)
        # print(self.bert)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.bert.config.hidden_size,
                            batch_first=True)

    def forward(self, x: torch.Tensor):
        x = self.bert(x)[0]
        x, _ = self.lstm(x)
        return x


class BertCNN(nn.Module):
    """docstring for BertCNN"""

    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(
            config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size  # 保持与bert数据格式一致
        self.cnn = CNN(config)

    def forward(self, x: torch.Tensor):
        # x : LongTensor ,[batch,sequence]
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    """docstring for BertMidLayer"""

    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(
            config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True  # 中间数据调整为也进行输出

    def forward(self, x: torch.Tensor):
        layer_states = self.bert(x)[2]
        print(len(layer_states))
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    config = {
        "hidden_size": 256,
        "kernel_size": 3,
        "num_layers": 4,
        "pretrain_model_path": r"J:\\八斗课堂学习\\第六周 语言模型\\bert-base-chinese\\bert-base-chinese",
    }

    # 针对普通的embedding后的序列数据
    x = torch.randn(12, 6, 12)
    # embedd前的索引数据
    x = torch.randint(1, 2000, (12, 6))
    print("序列的数据格式要求： ", x.dtype)
    bertmidlayer = BertMidLayer(config=config)
    out = bertmidlayer(x)
    print(out.shape)
