# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF  # pip install pytorch-crf
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        # num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # bidirectional表示是否使用双向LSTM，为True时，返回的hidden_size为hidden_size * 2，这就是为什么Linear层的输入是hidden_size * 2
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        
        # 参数return_dict=False表示返回的不是字典，而是元组
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        
        # self.classify = nn.Linear(hidden_size * 2, class_num)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        # CRF.forward()的参数：
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)，即真实标签
        # mask: (seq_length, batch_size)，即标签是否有效的mask
        # reduction: 'none' | 'sum' | 'mean' | 'token_mean'，即返回值的处理方式
        # 其中的emissions参数指的是发射得分，比如我们可以在lstm层后输出每个token的特征，
        # 并将其出入一个全连接层（输出维度为标签数量），这样每个token对应一个关于各标签的得分分布，
        # 所有token构成的这个张量就是emissions了，主要也就将lstm和crf衔接了起来。
        # CRF中发射矩阵是由LSTM的输出通过全连接层得到的，转移矩阵会在训练时自动学习，不需要人为指定。
        # CRF.forward()的返回值是最大对数似然，需要取负号转化为loss
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        x, _ = self.bert(x)  # input shape:(batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)  bert包含了embedding层
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        # 这里区分了使用CRF和不使用CRF的情况
        # 如果使用CRF，就正常过CRF层
        # 如果不使用CRF，就直接计算交叉熵损失
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                # CRF的返回值是最大对数似然，需要取负号转化为loss
                # 如果使用的是其它第三方CRF实现，可能用法不一样
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)