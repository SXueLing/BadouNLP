# -*- coding: utf-8 -*-
import re
from collections import defaultdict
import logging
import jieba
import torch
from peft import LoraConfig, get_peft_model

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
            input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], truncation=True, padding="max_length")
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
        pred_result = torch.argmax(pred_result.logits, dim=-1)
        pred_result = pred_result.cpu().detach().tolist()
        evaluator = Evaluator(self.config, self.model, logger)
        pred_result = pred_result[0]
        pred_entities = evaluator.decode(sentence, pred_result)
        return pred_entities


if __name__ == "__main__":
    from config import Config
    tuning_tactics = Config["tuning_tactics"]

    print("正在使用 %s"%tuning_tactics)

    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    #重建模型
    model = TorchModel
    model = get_peft_model(model, peft_config)
    state_dict = model.state_dict()

    #将微调部分权重加载
    if tuning_tactics == "lora_tuning":
        loaded_weight = torch.load('output/lora_tuning.pth')
    state_dict.update(loaded_weight)

    #权重更新后重新加载到模型
    model.load_state_dict(state_dict)
    predictor = Predictor(Config, model)
    sen = '王小明今天下午在南京市政府开会'
    predictor.predict(sen)



