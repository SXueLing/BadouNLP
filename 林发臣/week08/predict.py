# -*- coding: utf-8 -*-
import os.path

import torch
import json
from config import Config
from model import dic_model
import nlp_util as nlpu
from loader import get_loader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
模型效果测试
"""


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
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
        self.prepare_all_sentence()
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    def prepare_all_sentence(self):
        self.old_data_list = self.loader_train_data.dataset.cal_data
        cal_all_sen_tuple = [(self.model(torch.tensor(item[0]).unsqueeze(0).cuda()), item[1]) for item in
                             self.old_data_list]
        self.compare_tensor = torch.stack([item[0] for item in cal_all_sen_tuple], dim=0)

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
        s_encoder = self.loader_handler.encode_sentence_pre(sentence)
        with torch.no_grad():
            s_c = self.model(torch.tensor(s_encoder).unsqueeze(0).cuda())
            s_c = s_c.unsqueeze(0)
            result_list = torch.matmul(s_c, self.compare_tensor.T)
            max_vals_dim1, indices_dim1 = torch.max(result_list, dim=1)
            list_of_tuples = {index: value for index, value in enumerate(self.old_data_list)}
            v = [list_of_tuples[int(i)][1] for i in indices_dim1]
            v_c = torch.tensor(v).cuda()
            return self.index_to_sign[int(v_c)]


if __name__ == "__main__":
    path = 'D:\\badou\\code\\nlp20\\第八周作业\\model\\5913@fast_text@768@256@max@5e-05@3@0.8793.bin'
    model_name = os.path.basename(path)
    Config = nlpu.get_config_from_model_name(model_name, Config)
    sl = SentenceLabel(Config, path)
    while True:
        sen = input()
        print(f'预测类别【{sl.predict(sen)}】')
