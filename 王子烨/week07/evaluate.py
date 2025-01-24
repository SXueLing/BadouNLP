# -*- coding: utf-8 -*-
# @Time    : 2025/1/8 16:10
# @Author  : yeye
# @File    : evaluate.py
# @Software: PyCharm
# @Desc    :
import torch.cuda

from loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=True)
        self.stats_dict = {"wrong": 0, "correct": 0}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果:" % epoch)
        self.model.eval()
        self.stats_dict = {"wrong": 0, "correct": 0}  # 清空上一轮结果
        for index, batch_size in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_size]
            input_id, label = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id)
            self.write_stats(label, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, label, pred_results):
        assert len(label) == len(pred_results)
        for true_label, pred_label in zip(label, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d 预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测正确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------------")
        return correct / (correct + wrong)
