'''
Author: Zhao
Date: 2025-01-20 20:36:44s
LastEditTime: 2025-01-22 13:07:33
FilePath: evaluate.py
Description: 模型效果测试

'''
import torch
import numpy as np
from collections import defaultdict
import re
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # 加载验证数据
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        # 打印开始测试的消息
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        # 初始化统计字典
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)
                           }
        self.model.eval()  # 设置模型为评估模式
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data

            # 获取当前批次的句子
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: index * self.config["batch_size"] + len(labels)]
            with torch.no_grad():
                pred_results = self.model(input_ids)  
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return
    
    def write_stats(self, labels, pred_results, sentences):
        # 确保标签、预测结果和句子的数量一致
        assert len(labels) == len(pred_results) == len(sentences)
        #self.logger.info(pred_results.shape)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
            #self.logger.info("Predicted Labels: %s" % pred_results[:5])
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            # 调试打印
            # self.logger.info("True Labels: %s" % true_label[:10]) 
            #self.logger.info("Predicted Labels: %s" % pred_label[:10])
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # self.logger.info("True Labels: %s", true_entities)
            # self.logger.info("Predicted Labels: %s", pred_entities)
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([k for k in pred_entities[key] if k in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return
    
    def decode(self, sentence, labels):
        labels_str = "".join([str(x) for x in labels])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels_str):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels_str):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels_str):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels_str):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
    


