# -*- coding: utf-8 -*-
import torch
from loader import load_data
import os
import json
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("%s模型开始测试第%d轮模型效果：" % (self.config["model_type"], epoch))
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        if epoch == self.config["epoch"]:
            self.save_results(acc)    #在最后一轮记录最终结果
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
    
    def save_results(self, acc):
        results = {
            "model_tytpe" : self.config["model_type"],
            "hidden_size" : self.config["hidden_size"],
            "batch_size" : self.config["batch_size"],
            "pooling_style" : self.config["pooling_style"],
            "lr" : self.config["learning_rate"],
            "acc" : acc
        }
        results_path = self.config["results_path"]
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        else:
            existing_results = []

        existing_results.append(results)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=4)
            self.logger.info("测试结果已保存到文件%s" % results_path)
