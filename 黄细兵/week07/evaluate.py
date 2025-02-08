from safetensors import torch
import torch
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_date = load_data(config['valid_data_path'], config, shuffle=True)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def evaluate(self, epoch):
        self.logger.info("开始测试第%d轮模型效果" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}

        for index, batch_date in enumerate(self.valid_date):
            if torch.cuda.is_available():
                batch_date = [d.cuda() for d in batch_date]
            input_id, labels = batch_date
            with torch.no_grad():
                pre = self.model(input_id)
            self.write_stats(labels, pre)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pre):
        assert len(labels) == len(pre)
        for label, pre_label in zip(labels, pre):
            model_test = torch.argmax(pre_label)
            if int(label) == int(model_test):
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
