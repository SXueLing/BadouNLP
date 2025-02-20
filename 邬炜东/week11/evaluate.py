import random

import torch
from transformers import BertTokenizer
import numpy as np


class Evaluation:
    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.vocab, self.vocab_reverse = self.load_vocab_reverse(self.config["vocab_path"])

    def evaluator(self, start):
        start += self.config["bridge_token"]
        one_id = -1
        preds = []
        input_id = self.tokenizer.encode(start)
        input_id.pop()
        with torch.no_grad():
            while one_id != 102 and len(preds) <= 30:
                input_id_copy = torch.LongTensor(input_id).unsqueeze(0).cuda()
                one_id = self.pred_next(input_id_copy)
                input_id.append(one_id)
                preds.append(one_id)
        answer = self.tokenizer.decode(input_id)
        return answer

    def pred_next(self, input_id):
        y = self.model(input_id)[0][-1].cpu().numpy()
        y_pred = self.sampling_strategy(y)
        return y_pred

    def sampling_strategy(self, prob_distribution):
        if random.random() > self.config["sampling"]:
            y_pred = np.argmax(prob_distribution)
        else:
            y_pred = np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
        return y_pred

    def load_vocab_reverse(self, vocab_path):
        vocab = {}
        vocab_reverse = {}
        vocab["padding"] = 0
        vocab_reverse[0] = "padding"
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                vocab[line.strip()] = index + 1
                vocab_reverse[index + 1] = line.strip()

        return vocab, vocab_reverse
