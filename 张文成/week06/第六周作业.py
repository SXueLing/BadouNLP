#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel
"""

总参数量
= token_embeddings + segment_embeddings + position_embeddings + (self_attention + Layer_Normalization + feed_forward + Layer_Normalization) * transformers_num
= 768*vocab_size   + 2*768              + 512*768             + (Wq+b + Wk+b + Wv+b + liner + 2*hidden_size + (fc1 + fc2) * 2 + 2*hidden_size) * transformers_num
= 768*21128   + 2*768              + 512*768                  + (768*768+1*768 + 768*768+1*768 + 768*768+1*768 + 768*768 + 2*768 + (768*3072+3072*768)*2 + 2*768) * 12

下面以bert-base-chinese为例
"""



def count_bert_parameters(bert_model):
    return sum(p.numel() for p in bert_model.parameters() if p.requires_grad)





if __name__ == "__main__":
    # main()
    # test_strings = ["juvaee", "yrwfrg", "rtweqg", "nlhdww"]
    # predict("model.pth", "vocab.json", test_strings)


    # 加载BERT模型
    bert_model = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\八斗精品课nlp\第六周 语言模型\bert-base-chinese", return_dict=False)

    # 计算并打印BERT模型的参数量
    print(bert_model)
    print(f'The BERT model has {count_bert_parameters(bert_model):,} trainable parameters.')
    print(768*21128   + 2*768              + 512*768             + (768*768+1*768 + 768*768+1*768 + 768*768+1*768 + 768*768+768 + 2*768 + (768*3072+3072+3072*768+768)*1 + 2*768) * 1 + 768*768+768)
