# -*- coding: utf-8 -*-
import csv
import time

import torch
from utils.loader import load_data

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
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        start_time = time.time()
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果


        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)

        acc = self.show_stats(start_time)
        self.output_csv(start_time)
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

    def show_stats(self,start_time):
        end_time = time.time()
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        # 计算并打印耗时
        elapsed_time = end_time - start_time
        self.logger.info("平均百条耗时：%f 秒" % (elapsed_time / ((correct + wrong)/100)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
    def output_csv(self,start_time):
        end_time = time.time()
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        # 计算并打印耗时
        elapsed_time = end_time - start_time
        # 数据，可以是一个列表的列表，其中每个内部列表代表CSV中的一行
        data = [
            [self.config["model_type"], self.config["learning_rate"], self.config["hidden_size"], (correct +wrong), correct, (correct / (correct + wrong)), elapsed_time/ ((correct + wrong)/100)],
        ]
        # 打开（或创建）一个CSV文件用于写入
        with open(self.config["output_csv_path"], mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入数据到CSV文件
            writer.writerows(data)
        return
