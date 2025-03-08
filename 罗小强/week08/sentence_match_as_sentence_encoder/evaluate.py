# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"triplet_loss": 0.0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"triplet_loss": 0.0}  # 清空前一轮的测试结果
        self.model.eval()
        total_loss = 0.0
        count = 0
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            anchor, positive, negative = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                anchor_vector = self.model(anchor)
                positive_vector = self.model(positive)
                negative_vector = self.model(negative)
                triplet_loss = self.model.cosine_triplet_loss(anchor_vector, positive_vector, negative_vector)
                total_loss += triplet_loss.item()
                count += 1
        self.stats_dict["triplet_loss"] = total_loss / count
        self.show_stats()
        return

    def show_stats(self):
        triplet_loss = self.stats_dict["triplet_loss"]
        self.logger.info("验证集上的Triplet Loss: %f" % triplet_loss)
        self.logger.info("--------------------")
        return
