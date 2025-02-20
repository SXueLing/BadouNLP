import logging

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os

from transformers import BertModel
from loader import load_data, load_vocab
from evaluate import Evaluator
import config


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        self.classify = nn.Linear(768, len(vocab))
        self.bert = BertModel.from_pretrained("/Users/duli/Desktop/UT Austin/自学内容/八斗课件/NLP/six-lm/bert-base-chinese")
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, causal_mask=None):
        y_pred = self.classify(
            self.bert(x, attention_mask=causal_mask).last_hidden_state
        )   #output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(save_weight=True):
    epoch_num = 60        #训练轮数
    train_data = load_data("sample_data.json", config.Config, logger)
    vocab = load_vocab("vocab.txt")
    model = LanguageModel(vocab)    #建立模型
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)   #建立优化器
    # 加载效果测试类
    evaluator = Evaluator(config.Config, model, logger)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            input_seq, target_seq = batch_data # BERT是纯编码器，不需要额外一份label来做loss
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            causal_mask = get_causal_mask(input_seq, vocab["[SEP]"])
            loss = model(input_seq, target_seq, causal_mask=causal_mask)
            train_loss.append(float(loss))
            loss.backward()
            optim.step()
            optim.zero_grad()
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(train_loss)))
        evaluator.eval(epoch)
    # if not save_weight:
    #     return
    # else:
    #     base_name = os.path.basename(corpus_path).replace("txt", "pth")
    #     model_path = os.path.join("model", base_name)
    #     torch.save(model.state_dict(), model_path)
    #     return


def get_causal_mask(seq, sep_index):
    seq_len = seq.size(1)
    sep_positions = (seq == sep_index).nonzero()
    attention_mask = torch.ones((seq.size(0), seq.size(1), seq.size(1)), device="mps")
    for i in range(seq.size(0)):
        sep_pos = sep_positions[sep_positions[:, 0] == i, 1]
        if len(sep_pos) > 0:  # 确保找到了SEP
            sep_pos = sep_pos[0]  # 取第一个SEP位置

            # 创建SEP之后序列的causal mask
            # 生成上三角矩阵(包含对角线)
            causal_mask = torch.tril(
                torch.ones(seq_len - sep_pos - 1, seq_len - sep_pos - 1, device="mps")
            )

            # 将causal mask放入正确的位置
            attention_mask[i, :seq_len - sep_pos - 1, sep_pos + 1:] = causal_mask
    return attention_mask



if __name__ == "__main__":
    train(save_weight=False)