# -*- coding: utf-8 -*-
import torch
from loader import load_data
import numpy as np

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.tokenizer = self.train_data.dataset.tokenizer
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.questions = []
        for standard_question_index, questions in self.train_data.dataset.knwb.items():
            for question in questions:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.questions)] = standard_question_index
                self.questions.append(question)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            test_questions, labels = batch_data
            predicts = []
            for test_question in test_questions:
                input_ids = []
                positive_ids = []
                negative_ids = []
                for question in self.questions:
                    # 获取三元组的编码
                    anchor_id, pos_id, neg_id = self.train_data.dataset.encode_sentence(test_question, question, "")
                    input_ids.append(anchor_id)
                    positive_ids.append(pos_id)
                    negative_ids.append(neg_id)

                with torch.no_grad():
                    input_ids = torch.LongTensor(input_ids)
                    positive_ids = torch.LongTensor(positive_ids)
                    negative_ids = torch.LongTensor(negative_ids)
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        positive_ids = positive_ids.cuda()
                        negative_ids = negative_ids.cuda()
                    scores = self.model(input_ids, positive_ids, negative_ids).detach().cpu().tolist()
                hit_index = np.argmax(scores)
                # print(hit_index)
                predicts.append(hit_index)
            self.write_stats(predicts, labels)
        self.show_stats()
        return

    def write_stats(self, predicts, labels):
        assert len(labels) == len(predicts)
        for hit_index, label in zip(predicts, labels):
            hit_index = self.question_index_to_standard_question_index[hit_index] #转化成标准问编号
            if int(hit_index) == int(label):
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
        return