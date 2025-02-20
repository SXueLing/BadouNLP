# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import json
from config import Config
import random
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

'''
输入：‘今天天气怎么样’，‘天气不错’
输出：[[CLS] 今 天 天 气 怎 么 样 [SEP] 天 气 不  错  [SEP][PAD]]
      [  *   *  *  *  * *  *  *   天   气 不 错[SEP]  *    *]
同时生成一个掩码矩阵，把回答部分做成下三角以实现自回归输入
    |---question---|---answer---|---pad---|
  --
  q  1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 
  u  1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 
  e  1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 
  s  1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 
  -- 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 
  a  1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 
  n  1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 
  s  1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 
  -- 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 
  p  1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 
  a  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 
  d  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
  |  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
  |  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
  --
'''

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.corpus_path = config['corpus_path']
        self.bert_path = config['bert_path']
        self.data = []
        self.max_length = config['input_max_length']
        self.build_dataset(config)

    def build_dataset(self, config):
        with open(self.corpus_path, encoding='utf8') as f:
            for index, line in enumerate(f):
                line = json.loads(line)
                title = line['title']
                context = line['content']
                
                question_and_answer, answer, mask = self.encode_sentence(config, title, context)

                self.data.append([torch.LongTensor(question_and_answer),
                                  torch.LongTensor(answer),
                                  mask])
        return
    
    def encode_sentence(self, config, title, context):
        tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # [CLS] 今 天 天 气 怎 么 样 [SEP] 天 气 不  错  [SEP]
        title_and_context_id = tokenizer.encode(title,context,
                                    padding = False,
                                    max_length = self.max_length,
                                    truncation = True,
                                    return_overflowing_tokens=False)
        title_length = len(title)
        # 今 天 天 气 怎 么 样 [SEP] 天 气 不  错  [SEP]
        context_id = title_and_context_id[1:] # 错位
        #  *   *  *  *  * *  *  *   天   气 不 错[SEP]
        context_id[:(title_length+1)] = [-100] * (title_length+1) # 替换成 * （不参与loss计算)
        title_and_context_id = self.padding_id(title_and_context_id, 0) # 加 0 padding
        context_id = self.padding_id(context_id, -100) # 加 -100 padding

        # 构造mask
        # [CLS] 今 天 天 气 怎 么 样 这部分全 1
        # [SEP] 天 气 不  错 [SEP] [PAD] 这部分下三角 1
        l_mask = torch.ones(self.max_length, (title_length + 1))
        r_mask = torch.zeros(self.max_length, (self.max_length - title_length - 1))
        tril_mask = torch.tril(torch.ones((self.max_length - title_length - 1), (self.max_length - title_length - 1)))
        r_mask[(title_length + 1):, :(self.max_length - title_length - 1)] = tril_mask
        mask = torch.cat([l_mask, r_mask], dim=1)
        return (title_and_context_id, context_id, mask)
    
    def padding_id(self, input, pad_id):
        input = input[:self.max_length]
        input += [pad_id] * (self.max_length - len(input))
        return input
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]


def load_data(config, shuffle = True):
    dg = DataGenerator(config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    tokenizer = BertTokenizer.from_pretrained(Config['bert_path'])
    dg = DataGenerator(Config)
    test_id = dg[0]
    print(test_id)
    print(tokenizer.decode(test_id[0]))
    print(test_id[0].shape)
    print(test_id[1].shape)
    print(test_id[2].shape)
    '''
    dl = load_data(Config)
    for batch in dl:
        print(batch)
        break
    '''
