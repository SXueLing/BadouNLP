# -*- coding: utf-8 -*-
import re
from collections import defaultdict
import logging
import jieba
import torch
from loader import load_vocab
from config import Config
from model import TorchModel
from evaluate import Evaluator
from transformers import BertTokenizer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.vocab = load_vocab(config["vocab_path"])
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()


    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    def predict(self, sentence):
        if self.config["model_type"] == "bert":
            input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True)
        else:
            input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            # 如果模型没用到crf，输出维度：(sen_len, num_tags)
            # 如果模型使用crf，输出维度：(sen_len,)
            pred_result = self.model(input_id)
        pred_entities = self.show_entities(pred_result, sentence)
        for key, value in pred_entities.items():
            print(key, value)

    def show_entities(self, pred_result, sentence):
        if not self.config["use_crf"]:
            pred_result = torch.argmax(pred_result, dim=-1)
            pred_result = pred_result.cpu().detach().tolist()
        evaluator = Evaluator(self.config, self.model, logger)
        pred_result = pred_result[0]
        pred_entities = evaluator.decode(sentence, pred_result)
        return pred_entities


if __name__ == "__main__":
    vocab = load_vocab(Config["vocab_path"])
    Config["vocab_size"] = len(vocab)
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model_output/bert_crf.pth"))
    pd = Predictor(Config, model)
    pd.predict('李嘉琪今天下午开车前往杭州公安局上班')

    # while True:
    #     sentence = input("请输入问题：")
    #     pd.predict(sentence)

