import random

import torch.nn as nn
import torch
from transformers import BertTokenizer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageBertToSftEvaluate:

    def __init__(self, model: nn.Module, config):
        self.config = config
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.cuda_enable = torch.cuda.is_available()

    def eval(self, sen, model):
        print(self.generate_sentence(sen, model, self.tokenizer))
        # model.eval()
        # with torch.no_grad():
        #     pred_str = ''
        #     while pred_str != '\n' and len(sen) <= 150:
        #         sen += pred_str
        #         sen_encode = self.tokenizer.encode(sen)
        #         sen_v = torch.LongTensor(sen_encode)
        #         if self.cuda_enable:
        #             sen_v = sen_v.cuda()
        #         output_d = self.model(sen_v.unsqueeze(0))
        #         x = output_d[0][-1]
        #         index = sampling_strategy(x)
        #         pred_str = self.tokenizer.decode(index)
        #     print(sen)

    def generate_sentence(self, openings, model, tokenizer):
        model.eval()
        openings = tokenizer.encode(openings)
        with torch.no_grad():
            # 生成文本超过30字则终止迭代
            while len(openings) <= 50:
                x = torch.LongTensor([openings])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = model(x)[0][-1]
                index = sampling_strategy(y)
                openings.append(index)
        return tokenizer.decode(openings)


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
