# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from collections import defaultdict
from dataloader import data_loader


class Evaluator:
    def __init__(self, config, model, logger):
        self.model = model
        self.logger = logger
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.use_crf = config['use_crf']
        self.datas = data_loader(
            config, config['train_data_path'], False
        )
        self.keys = ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]
        return
    
    def eval(self):
        self.stats_dict = {}
        for key in self.keys:
            self.stats_dict[key] = defaultdict(int)

        self.model.eval()
        for index, batch_data in enumerate(self.datas):
            sentences = self.datas.dataset.sentences[index * self.batch_size: (index+1) * self.batch_size]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return
    
    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
    
        if not self.use_crf:
            pred_results = torch.argmax(pred_results, dim=-1)
    
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.use_crf:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            sentence_len = len(sentence) if len(sentence) < self.max_length-2 else self.max_length-2
            true_entities = self.decode(sentence, true_label[1:sentence_len])
            pred_entities = self.decode(sentence, pred_label[1:sentence_len])
            for key in self.keys:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in self.keys:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in self.keys])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in self.keys])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in self.keys])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return
    
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


if __name__ == "__main__":
    from config import CONFIG
    from model import NerModel
    import logging

    logging.basicConfig(
        level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    model = NerModel(CONFIG)
    model.load_state_dict(torch.load("model_output/ner_epoch_20.pth"))
    if torch.cuda.is_available():
        model = model.cuda()

    evalutor = Evaluator(CONFIG, model, logger)
    evalutor.eval()

