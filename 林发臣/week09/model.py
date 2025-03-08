# 所有模型的集合


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchcrf import CRF


class dic_model(nn.Module):

    def __init__(self, config):
        super(dic_model, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        self.user_bert = False
        self.double_lstm = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        encoder_result = load_model(self, config)
        self.encoder = encoder_result[0]
        self.hidden_size = encoder_result[1]
        if self.double_lstm:
            self.classifier = nn.Linear(self.hidden_size * 2, class_num)
        else:
            self.classifier = nn.Linear(self.hidden_size, class_num)
        self.crf_layer = CRF(class_num, True)
        self.use_crf = config["use_crf"]
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def trans_input_to_v(self, x):
        if not self.user_bert:  # x = (bs, l)
            x = self.embedding(x)  # x = (bs, l) ->(bs, l,h)
        x = self.encoder(x)  # (bs, l,h) -> ((bs, l,h),(l,h))
        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        x = self.classifier(x)  # (bs, l,h) -> (bs, l,c)
        return x

    def forward(self, x, target=None):  # x ->bs,l,h
        predict = self.trans_input_to_v(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # (number, class_num), (number)
                predict = predict.view(-1, predict.shape[-1])
                target = target.view(-1)
                return self.loss(predict, target)
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def load_model(self, config):
    result_encoder = ''
    hidden_size = config['hidden_size']
    model_type = config['model_type']
    num_layers = config['num_layers']
    if model_type == 'bert':
        self.user_bert = True
        result_encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        hidden_size = result_encoder.config.hidden_size
    if model_type == 'rnn':
        result_encoder = nn.RNN(hidden_size, hidden_size, bias=True, num_layers=num_layers, batch_first=True)
    if model_type == 'cnn':
        result_encoder = CNN(config)
    if model_type == 'lstm-d':
        result_encoder = nn.LSTM(hidden_size, hidden_size, bidirectional=True, bias=True, num_layers=num_layers,
                                 batch_first=True)
        self.double_lstm = True
    if model_type == 'lstm':
        result_encoder = nn.LSTM(hidden_size, hidden_size,  bias=True, num_layers=num_layers,
                                 batch_first=True)
    if model_type == 'gru':
        result_encoder = nn.GRU(hidden_size, hidden_size, bias=True, num_layers=num_layers, batch_first=True)
    if model_type == 'fast_text':
        result_encoder = lambda x: x
    if model_type == 'text_rnn':
        result_encoder = TextRNN(config)
    if model_type == 'text_cnn':
        result_encoder = TextCNN(config)
    if model_type == 'gated_cnn':
        result_encoder = GatedCNN(config)
    if model_type == 'text_rcnn':
        result_encoder = TextRCNN(config)
    if model_type == "stack_gated_cnn":
        result_encoder = StackGatedCNN(config)
    if model_type == "bert_lstm":
        self.user_bert = True
        result_encoder = BertLstm(config)
        hidden_size = result_encoder.bert.config.hidden_size
    if model_type == "bert_rnn":
        self.user_bert = True
        result_encoder = BertRNN(config)
        hidden_size = result_encoder.bert.config.hidden_size
    if model_type == "bert_cnn":
        self.user_bert = True
        result_encoder = BertCNN(config)
        hidden_size = result_encoder.bert.config.hidden_size
    if model_type == "bert_gcnn":
        self.user_bert = True
        result_encoder = BertGCNN(config)
        hidden_size = result_encoder.bert.config.hidden_size
    if model_type == "bert_mid_layer":
        self.user_bert = True
        result_encoder = BertMidLayer(config)
        hidden_size = result_encoder.bert.config.hidden_size
    return result_encoder, hidden_size


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.lstm = nn.LSTM(config['hidden_size'], config['hidden_size'], num_layers=config['num_layers'], bias=True,
                            batch_first=True)
        self.droupout = nn.Dropout(config['droup_out_pro'])

    def forward(self, x):
        x, _ = self.lstm(x)  # (bs*l*h),l*h
        x = self.droupout(x)  # bs,l,h
        return x


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.cnn(x)
        return x


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn1 = CNN(config)
        self.cnn2 = CNN(config)

    def forward(self, x):
        x = torch.mul(self.cnn2(x), nn.functional.sigmoid(self.cnn1(x)))
        return x


class TextRCNN(nn.Module):
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.rnn = nn.LSTM(config['hidden_size'], config['hidden_size'], num_layers=config['num_layers'], bias=True,
                           batch_first=True)
        self.cnn = GatedCNN(config)

    def forward(self, x):  # bs,l,h
        x, _ = self.rnn(x)  # 第一个参数x = bs,l,h    第二个参数_=bs,h
        x = self.cnn(x)  # bs,l,h
        return x


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        # ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        # 仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  # 通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  # 之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  # 一层线性
            l1 = torch.relu(l1)  # 在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1)  # 二层线性
            x = self.bn_after_ff[i](x + l2)  # 残差后过bn
        return x


class BertLstm(nn.Module):
    def __init__(self, config):
        super(BertLstm, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.rnn = nn.LSTM(config['hidden_size'], config['hidden_size'], bias=True, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.rnn(x)
        return x


class BertRNN(nn.Module):
    def __init__(self, config):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.rnn = nn.RNN(config['hidden_size'], config['hidden_size'], num_layers=config['num_layers'], bias=True,
                          batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.rnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertGCNN(nn.Module):
    def __init__(self, config):
        super(BertGCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]  # (13, batch, len, hidden)
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states
