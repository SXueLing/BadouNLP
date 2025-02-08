# -*- coding: utf-8 -*-
import os.path

import torch
import json
from config import Config
from model import dic_model
from loader import get_loader
import logging
from collections import defaultdict
import re
import train_util
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
模型效果测试
"""


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.use_bert = config['use_bert']
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = dic_model(config)
        if torch.cuda.is_available():
            logger.info("gpu可以使用，迁移模型至gpu")
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.old_data_list = None
        self.compare_tensor = None
        self.loader_handler, self.loader_train_data = get_loader(config["train_data_path"], config, shuffle=False,
                                                                 load_flag=False)
        self.model.eval()
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        sentences = split_sentences(sentence)
        s_encoders = []
        results = defaultdict(set)
        for sentence in sentences:
            s_encoder = self.loader_handler.encode_sentence_pre(sentence)
            s_encoders.append(s_encoder)
        with torch.no_grad():
            s_c = self.model(torch.tensor(s_encoders).cuda())
            for s_item, cal_result in zip(sentences, s_c):
                s_c_index = cal_result
                if torch.is_tensor(cal_result):
                    s_c_value, s_c_index = torch.max(cal_result, dim=-1)
                    s_c_index = s_c_index.tolist()
                if self.use_bert:
                    s_c_index.pop(0)
                result = self.decode(s_item, s_c_index)
                for ket, value in result.items():
                    for v_item in value:
                        results[ket].add(v_item)
            return results

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
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


def split_sentences(text):
    # 使用正则表达式匹配句号、逗号，并进行分割
    # 保留标点符号，并避免分割出空白项
    sentences = re.split(r'(，|。)', text)

    # 合并分割的内容，确保标点符号在句子末尾
    result = []
    buffer = ""
    for part in sentences:
        if part in {'，', '。'}:
            buffer += part  # 将标点添加到当前句子
            result.append(buffer.strip())  # 完整句子添加到结果中
            buffer = ""  # 清空缓存
        else:
            buffer += part  # 添加非标点部分

    # 处理没有标点结束的部分
    if buffer.strip():
        result.append(buffer.strip())

    return result


if __name__ == "__main__":
    path = r'D:\badou\code\nlp20\第九周作业\ner\model\bert\5663@bert@768@128@max@0.0001@20@0.0.bin\5663@bert@768@128@max@0.0001@13@0.6926.bin'
    model_name = os.path.basename(path)
    Config = train_util.get_config_from_model_name(model_name, Config)
    train_util.do_train_pre(Config)
    sl = SentenceLabel(Config, path)
    while True:
        sen = input()
        if sen:
            print(f'预测类别【{sl.predict(sen)}】')
