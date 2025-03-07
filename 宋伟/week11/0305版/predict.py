# -*- coding: utf-8 -*-
# @Date    :2025-03-05 17:15:17
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text

import torch
from model import LanguageModel
from transformers import BertTokenizer
import random
import numpy as np

def generate_sentence(openings,model:LanguageModel,tokenizer:BertTokenizer):
    model.eval()
    openings = tokenizer.encode(openings) # 这里记录了特殊符号

    with torch.no_grad():
        while len(openings) <= 100:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x.cuda()
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

if __name__ == '__main__':
    from config import Config
    tokenizer = BertTokenizer.from_pretrained(Config['bert_path'])
    model = LanguageModel(Config)
    model.load_state_dict(torch.load("./output/epoch_bert_sft_20.pth"))
    openings = "北京明年拟推工作日半价观看电影"
    x = generate_sentence(openings,model,tokenizer)
    print(x)
