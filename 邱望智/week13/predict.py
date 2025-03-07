# -*- coding: utf-8 -*-
import torch
from loader import load_data
from config import Config
from model import TorchModel, choose_optimizer
import re
from transformers import BertTokenizer
from collections import defaultdict
from peft import get_peft_model, LoraConfig

"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = self.load_vocab(config["bert_path"])
        model = TorchModel(config)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
        model = get_peft_model(model, peft_config)
        state_dict = model.state_dict()
        state_dict.update(torch.load(model_path))
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        print("模型加载完毕!")

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        return BertTokenizer.from_pretrained(vocab_path)

    def encode_sentence(self, text, padding=True):
        return self.tokenizer.encode(text, 
                                     padding="max_length",
                                     max_length=self.config["max_length"],
                                     truncation=True)

    def decode(self, sentence, labels):
        sentence = "$" + sentence
        labels = "".join([str(int(x)) for x in labels[:len(sentence)+1]])
        results = defaultdict(list)
        
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results


    def predict(self, sentence):
        input_ids = self.encode_sentence(sentence)
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_ids]))[0]
            labels = torch.argmax(res, dim=-1)
        results = self.decode(sentence, labels)
        return results

if __name__ == "__main__":
    lora = Predictor(Config, "model_output/lora.pth")
    sentence = "东方红太阳升，中国出了个毛泽东，中华人民共和国万岁，世界人民大团结万岁"
    res = lora.predict(sentence)
    print(res)
    while True:
        sentence = input("请输入问题：")
        res = lora.predict(sentence)
        print(res)
