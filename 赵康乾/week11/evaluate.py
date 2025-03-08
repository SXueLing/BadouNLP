# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertTokenizer, BertModel
from config import Config
from model import TorchModel
from collections import defaultdict
import numpy as np
import math
import logging


'''
每轮测试生成一个回答
'''

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.vocab = self.tokenizer.get_vocab() # dict{word:index}
        self.vocab_size = self.tokenizer.vocab_size
        self.model = model
        self.logger = logger
        self.window_size = config['window_size']
        self.question = '阿根廷歹徒抢服装尺码不对拿回店里换'
        
    def eval(self, epoch):
        self.logger.info(f'开始第{epoch}轮测试：')
        self.model.eval()
        self.model.cpu()
        test_sentence = self.generate_sentence(self.question, 
                                               self.model, 
                                               self.vocab)
        self.logger.info(f'测试生成的句子：{test_sentence}')
        return
    
    def generate_sentence(self, opening, model, vocab):
        # 取最新的window_size预测新字符，直到生成换行符或总字数30个字
        reverse_vocab = dict((y,x) for x,y in vocab.items())
        with torch.no_grad():
            pred_char = ''
            while pred_char != '/n' and len(opening) < 30:
                opening += pred_char
                input_ids = self.tokenizer.encode(opening,
                                                  padding = False,
                                                  return_tensors="pt",
                                                  add_special_tokens = True)
                #print(input_ids.shape)
                if torch.cuda.is_available():
                    x = x.cuda()
                id_output = model(mask = None, question_and_answer = input_ids)
                #print(id_output)
                #print(id_output.shape)
                id_output = id_output[0][-1]
                #print(id_output)
                index = self.sample(id_output)
                # print(index)
                pred_char = self.tokenizer.decode(index)
        return opening

    def sample(self, id_output):
        # 10%概率随机返回一个词表序号，90%概率返回概率最大的序号
        if random.random() > 0.1:
            sampling_strategy = 'greedy'
        else:
            sampling_strategy = 'sample'
        
        if sampling_strategy == 'greedy':
            return int(torch.argmax(id_output))
        elif sampling_strategy == 'sample':
            id_output = id_output.cpu().numpy()
            return np.random.choice(list(range(len(id_output))), p=id_output)
                                            

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    from config import Config
    from model import TorchModel

    model = TorchModel(Config)
    evaluator = Evaluator(Config, model, logger)
    evaluator.eval(1)
