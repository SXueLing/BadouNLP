'''
Author: Zhao
Date: 2025-01-08 20:38:05
LastEditTime: 2025-01-09 15:48:24
FilePath: evaluate.py
Description: 

'''
import torch
from loader import load_data

class Evaluator:
    def __init__(self, config,model,logger):
        """ 
        初始化 Evaluator 类 
        :param config: 配置字典，包含各种超参数和路径 
        :param model: 训练好的模型 
        :param logger: 日志记录器对象 
        """
        self.config = config
        self.model= model
        self.logger = logger
        # 加载验证数据
        self.valid_data = load_data(config["valid_data_path"],config, shuffle=True)
        # 初始化统计字典，用于存储测试结果
        self.stats_dict = {"correct":0,"wrong":0}

    def eval(self,epoch):
        """ 
        评估模型在验证集上的表现 
        :param epoch: 当前训练的轮数 
        :return: 验证集上的准确率 
        """
        #self.logger.info(f"开始测试第{epoch}轮模型效果：")
        self.model.eval()   # 切换模型到评估模式
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                # 模型预测
                pred_results = self.model(input_ids)
            # 记录预测结果
            self.write_state(labels, pred_results)
        # 计算并显示统计数据
        accuracy = self.show_stats()
        return accuracy
    
    def write_state(self, labels, pred_results):
        """ 
        记录模型的预测结果 
        :param labels: 真实标签 
        :param pred_results: 预测结果 
        """
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)   # 获取预测结果的最大值索引
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        """ 
        显示并记录模型的评估统计数据 
        :return: 模型在验证集上的准确率 
        """
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        accuracy = correct / total

        # self.logger.info(f"预测集合条目总量：{total}") 
        # self.logger.info(f"预测正确条目：{correct}，预测错误条目：{wrong}") 
        # self.logger.info(f"预测准确率：{accuracy:.6f}") 
        # self.logger.info("--------------------")
        return correct / (correct + wrong)