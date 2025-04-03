# -*- coding: utf-8 -*-
# @Date    :2025-03-02 00:36:15
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text

import torch
import random
import numpy as np
from loader import load_vocab, MyDataSet
from model import LanguageModel
from config import Config
from transformers import BertTokenizer


def generate_sentence(openings, model, tokenizer, window_size):
    # reverse_vocab = dict((idx, char)for char, idx in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ''
        while len(openings) <= 200:
            
            openings += pred_char
            x = tokenizer.encode(openings,
                                 # return_tensors='pt',
                                 # max_length=10,
                                 # padding='max_length',
                                 # truncation=True,
                                 add_special_tokens=False)  # 本身自带一个维度
            # print(x)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()

            y_distribution = model(x)[0][-1]

            idx = sampling_strategy(y_distribution)
            pred_char = ''.join(tokenizer.decode([idx]))
            # print(pred_char)
        print(openings)
        return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.2:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


if __name__ == '__main__':
    dataset = MyDataSet(Config)
    # torch.manual_seed(42)
    # random.seed(42)
    # np.random.seed(42)
    # torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    model = LanguageModel(Config)
    model.cuda()
    model.load_state_dict(torch.load("./output/epoch_bert05_20.pth"))
    openings = "举头望明月，低头思故乡"
    # vocab = load_vocab(Config['vocab_path'])
    tokenizer = BertTokenizer.from_pretrained(Config['bert_path'])
    window_size = Config["window_size"]
    generate_sentence(openings, model, tokenizer, window_size)
