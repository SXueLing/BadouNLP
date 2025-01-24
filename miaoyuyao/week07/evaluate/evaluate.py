# -*- coding: utf-8 -*-
"""
模型效果测试
"""
import os
import time

import pandas as pd
import torch.cuda

from test_ai.homework.week07.config.logging_config import logger
from test_ai.homework.week07.data.loader import load_data
from test_ai.homework.week07.config.running_config import Config
from test_ai.homework.week07.models.torch_model import TorchModel


class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        start_time = time.time()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)  # 不输入labels，使用模型当前参数进行预测
                self.write_stats(labels, pred_results)
        end_time = time.time()
        avg_100_time = (100 * (end_time - start_time)) / len(self.valid_data)
        acc = self.show_stats()
        return acc, avg_100_time

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
        logger.info("预测集合条目总量：%d" % (correct + wrong))
        logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        logger.info("--------------------")
        return correct / (correct + wrong)
