import torch
from loader import load_data

"""
模型测试效果
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dic = {"correct": 0, "wrong": 0} # 统计预测正确和错误的数量

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dic = {"correct": 0, "wrong": 0} # 清空上一轮测试结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids)    # 不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dic["correct"] += 1
            else:
                self.stats_dic["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dic["correct"]
        wrong = self.stats_dic["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目数：%d, 错误条目数：%d" %(correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        return correct / (correct + wrong)