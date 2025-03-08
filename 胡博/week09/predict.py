@@ -0,0 +1,62 @@
# -*- coding: utf-8 -*-
import torch
from loader import load_data
from config import Config
from model import TorchModel, choose_optimizer
import re
from transformers import BertTokenizer
from collections import defaultdict
"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()

    def predict(self, sentence):
        input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True)
        input_id = torch.LongTensor([input_ids])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            pred = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            pred_result = torch.argmax(pred.squeeze(), dim=-1)
            pred_result = pred_result.cpu().detach().tolist()
            entities = self.decode(sentence, pred_result)
        return entities

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[1:len(sentence) + 1]])
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

if __name__ == "__main__":
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model_output/epoch_20.pth"))
    pd = Predictor(Config, model)

    while True:
        sentence = input("请输入问题：")
        res = pd.predict(sentence)
        print(res)




