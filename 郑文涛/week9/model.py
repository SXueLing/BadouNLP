# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
"""
建立网络模型结构
"""


class TorchModel(nn.Module):

    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)

        # BERT模型
        self.bert = BertModel.from_pretrained(
            r"C:\Users\ada\Desktop\学习\八斗AI课\第六周 语言模型\bert-base-chinese",
            return_dict=False)  # 使用预训练的BERT模型
        # 分类层（将BERT的输出转换为标签类别）
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        # CRF层（可选，用于序列标注）
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        # 损失函数
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    def forward(self, x, target=None):
        """
        当输入真实标签时，返回loss值；无真实标签时，返回预测结果
        """
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)

        # BERT模型的输出: [batch_size, seq_len, bert_hidden_size] 和 [batch_size, bert_hidden_size]
        # sequence_output包含了每个token（字、词）对应的特征，形状是[batch_size, seq_len, hidden_size]。
        # pooler_output是BERT的[CLS]标记的输出，形状是[batch_size, hidden_size]，通常用于分类任务。
        sequence_output, pooler_output = self.bert(x)  # x为一个表述输入文本的张量
        # 分类层输出: (batch_size, seq_len, bert_hidden_size) -> (batch_size, class_num)
        predict = self.classify(sequence_output)
        # pooler_output是针对整个句子的表示，而sequence_output才是每个token的表示。
        # 通常在NER任务中，我们会使用sequence_output来预测每个token的标签。

        if target is not None:
            # 如果有真实标签，计算损失
            if self.use_crf:
                # CRF层需要使用mask来排除填充部分
                mask = target.gt(-1)  # gt(-1) 即排除标签为-1的部分作为padding
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 如果不使用CRF，使用交叉熵损失
                #(number, class_num), (number)
                predict = predict.view(
                    -1,
                    predict.shape[-1])  # 展平: (batch_size * seq_len, num_tags)
                target = target.view(-1)  # 展平: (batch_size * seq_len)
                return self.loss(predict, target)  # 计算交叉熵损失
                # ???Expected input batch_size (16) to match target batch_size (1600)
        else:
            # 如果没有真实标签，返回预测值
            if self.use_crf:
                # 如果使用CRF，返回CRF解码后的标签序列
                return self.crf_layer.decode(predict)
            else:
                # 否则直接返回分类结果
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
