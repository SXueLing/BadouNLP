"""
@Project ：cgNLPproject 
@File    ：eval_08.py
@Date    ：2025/1/14 14:47 
"""
import torch
import torch.nn as nn
import logging
from loader_08 import load_dataset

class Evaluator:
    def __init__(self, config, model_, logger):
        self.res = {'correct':0, 'wrong':0}
        self.eval_data = load_dataset(config, 'evaluate')
        self.train_data = load_dataset(config, 'train')
        self.model = model_
        self.logger = logger

    def know_to_vector(self):
        self.know_questions = []
        self.know_labels = {}
        for label,know_questions_list in self.train_data.dataset.train_dataset.items():    #batch_size, sentence_length
            for question in know_questions_list:
                self.know_labels[len(self.know_questions)] = label
                self.know_questions.append(question)    #batch_size, sentence_length
        self.know_questions = torch.stack(self.know_questions, dim=0)
        self.know_questions_vector = self.model(self.know_questions)    # questions_length, hidden_size
        self.know_questions_vector = nn.functional.normalize(self.know_questions_vector, dim=-1)
        return


    def eval(self, epoch):
        self.logger.info(f'开始第{epoch}轮的预测：---------------')
        eval_data = self.eval_data
        self.model.eval()
        self.know_to_vector()
        self.res = {'correct':0, 'wrong':0}
        for test_questions, labels in eval_data:
            test_questions_vector = self.model(test_questions)    # batch_size, hidden_size
            for test_question, label in zip(test_questions_vector, labels):
                cos_distances = torch.mm(test_question.unsqueeze(0), self.know_questions_vector.T)    # s1: hidden_size,  s2: questions_length, hidden_size
                label_index = int(torch.argmax(cos_distances.squeeze()))
                if self.know_labels[label_index] == int(label):
                    self.res['correct'] += 1
                else:
                    self.res['wrong'] += 1
        self.show_stats()
        return

    def show_stats(self):
        correct = self.res["correct"]
        wrong = self.res["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return


if __name__ == "__main__":
    from config_08 import Config
    from model_08 import SimNetwork

    train_data = load_dataset(Config, 'train')
    model = SimNetwork(Config)
    evaluator = Evaluator(Config, model)
    evaluator.eval()