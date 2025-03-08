"""
@Project ：cgNLPproject 
@File    ：evaluate_my.py
@Date    ：2025/1/8 11:05 
"""
import torch
from loader_my import load_data
import time

class evaluate:
    def __init__(self, config, model, log):
        self.data = load_data(config, config['valid_data_path'])
        self.model = model
        self.config = config
        self.log = log

    def eval(self, epoch):
        self.log.info(f'开始第{epoch}轮的预测：---------------')
        self.model.eval()
        self.pred_dict = {'correct':0,'wrong':0}
        for x, y in self.data:
            with torch.no_grad():
                y_pred = self.model(x)
            self.write_state(y, y_pred)
        acc = self.show_state(self.pred_dict)
        return acc

    def write_state(self, y, y_pred):
        assert len(y) == len(y_pred), '预测数量与标签数量不一致'
        for true_label, pred_label in zip(y, y_pred):
            pred_label = torch.argmax(pred_label)
            if true_label == pred_label:
                self.pred_dict['correct'] += 1
            else:
                self.pred_dict['wrong'] += 1
        return

    def show_state(self, pred_dict):
        correct = pred_dict['correct']
        wrong = pred_dict['wrong']
        log = self.log
        log.info(f'预测集合条目总量：{correct+wrong}')
        log.info(f'预测正确条目：{correct},预测错误条目：{wrong}')
        log.info(f'预测准确率：{correct / (correct + wrong)}')
        log.info("--------------------")
        return correct / (correct + wrong)




