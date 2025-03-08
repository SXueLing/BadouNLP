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
        # 验证集
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0} #用于存储测试结果

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        # 标准答案的index，对应下面的所有问题的input_id(onehot值)
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id) #把标准答案的index和下面的很多input_id分别放到不同的数组里，一一对应，方便获取
        with torch.no_grad():
            # 堆叠在一起，两个2*3的，变成(1+1) * 2 * 3
            question_matrixs = torch.stack(self.question_ids, dim=0)
            # tensor([[1, 2, 3],
            # [4, 5, 6]])
            # 转化成所有类别累积的答案长度*句子长度，例如一个类别下面有10个答案，每个答案句子长度为100，就是10*100，
            # 然后每个类别的答案都加起来
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            #将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return
    
    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data # input_id, label_index
            with torch.no_grad():
                test_question_vectors = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return
    
    def write_stats(self, test_question_vectors, labels):
        # 32, 32 * 128。batch_size是32。labels是32，表示每一个句向量的类别。
        # print(test_question_vectors.shape, labels, len(labels), 'labelslabelslabelslabelslabels')
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            #通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            #test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            # knwb_vectors是标准答案计算转成的词向量。test_question_vector是验证集计算的词向量
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze())) #命中问题标号
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
                        