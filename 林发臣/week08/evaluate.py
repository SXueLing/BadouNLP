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
        self.valid_data = load_data(config["valid_data_path"], config, 'test', shuffle=False)
        self.train_data = load_data(config["train_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        self.cal_old_data = self.do_evaluate_install(self.model)
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return round(acc, 4)

    def do_evaluate_install(self, model) -> list:
        old_data_list = self.train_data.dataset.cal_data
        with torch.no_grad():
            return [(model(torch.tensor(item[0]).unsqueeze(0).cuda()), item[1]) for item in old_data_list]

    def write_stats(self, labels, pred_results):
        compare_tensor = torch.stack([item[0] for item in self.cal_old_data], dim=0)
        result_list = torch.matmul(pred_results, compare_tensor.T)
        max_vals_dim1, indices_dim1 = torch.max(result_list, dim=1)
        list_of_tuples = {index: value for index, value in enumerate(self.cal_old_data)}
        v = [list_of_tuples[int(i)][1] for i in indices_dim1]
        v_t = torch.tensor(v).cuda()
        same_count = (v_t == labels).sum()  # 返回一个张量，值为相同元素的数量
        # 如果需要 Python 的 int
        same_count_int = same_count.item()
        # 计算不同位置个数
        diff_count = (v_t != labels).sum()  # 同理，返回一个张量
        diff_count_int = diff_count.item()
        self.stats_dict["correct"] += same_count_int
        self.stats_dict["wrong"] += diff_count_int
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
