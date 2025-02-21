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
每轮测试生成一个句子，并计算困惑度
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
        self.opening = '李慕楞了一下，以为'
        
    def eval(self, epoch):
        self.logger.info(f'开始第{epoch}轮测试：')
        self.model.eval()
        self.model.cpu()
        test_sentence = self.generate_sentence(self.opening, 
                                               self.model, 
                                               self.vocab, 
                                               self.window_size)
        ppl = self.calc_perplexity(test_sentence, 
                                   self.model, 
                                   self.vocab, 
                                   self.window_size)
        self.logger.info(f'测试生成的句子：{test_sentence}')
        self.logger.info(f'困惑度：{ppl}')
        return
    
    def generate_sentence(self, opening, model, vocab, window_size):
        # 取最新的window_size预测新字符，直到生成换行符或总字数30个字
        reverse_vocab = dict((y,x) for x,y in vocab.items())
        with torch.no_grad():
            pred_char = ''
            while pred_char != '/n' and len(opening) < 30:
                opening += pred_char
                sentence_input = opening[:window_size]
                input_ids = self.tokenizer.encode(sentence_input,
                                                  padding = False,
                                                  return_tensors="pt",
                                                  add_special_tokens = False)
                mask_size = min(len(opening), window_size)
                attention_mask = torch.tril(torch.ones(mask_size, mask_size))
                if torch.cuda.is_available():
                    x = x.cuda()
                id_output = self.model(input_ids, attention_mask)[0][-1]
                index = self.sample(id_output)
                # pred_char = reverse_vocab.get(index, '[UNK]')
                pred_char = self.tokenizer.decode([index])
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
            return np.random.choice(list(range(len(id_output))))
        
    def calc_perplexity(self, sentence, model, vocab, window_size):
        prob = 0
        with torch.no_grad():
            for i in range(1, len(sentence)):
                start = max(0, i - window_size)
                window = sentence[start:i]
                x = self.tokenizer.encode(window,
                                          padding = False,
                                          return_tensors="pt",
                                          add_special_tokens = False)
                mask_size = min(len(x), window_size)
                attention_mask = torch.tril(torch.ones(mask_size, mask_size))

                target = sentence[i]
                target_index = vocab.get(target)
                if torch.cuda.is_available():
                    x = x.cuda()
                pred_prob_distribute = model(x, attention_mask)[0][-1]
                softmax = nn.Softmax(dim = -1)
                pred_prob_distribute = softmax(pred_prob_distribute)
                target_prob = pred_prob_distribute[target_index]
                prob += math.log(target_prob, 10)
        return 2 ** (prob * ( -1 / len(sentence)))
                                    

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    from config import Config
    from model import TorchModel

    model = TorchModel(Config)
    evaluator = Evaluator(Config, model, logger)
    evaluator.eval(1)
