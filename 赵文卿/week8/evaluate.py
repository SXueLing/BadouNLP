'''
Author: Zhao
Date: 2025-01-14 20:16:03
LastEditTime: 2025-01-15 18:27:57
FilePath: evaluate.py
Description: Evaluator 类用于评估模型在验证数据集上的性能
        初始化：加载配置、模型、日志记录器，以及验证和训练数据。
        知识库向量化：将训练数据中的问题编码为向量。
        模型评估：对验证数据进行评估，并计算测试问句向量与知识库向量的相似度。
        统计结果记录：记录和显示正确与错误的预测统计结果。
'''
import torch
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        # 初始化 Evaluator 类，加载配置、模型和日志记录器
        self.config = config
        self.model = model
        self.logger = logger
        # 加载验证数据和训练数据
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        # 初始化统计字典，记录正确和错误的数量
        self.stats_dict = {"correct": 0, "wrong": 0}
    
    def knwb_to_vector(self):
        # 将知识库中的问题编码为向量
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        # 遍历训练数据中的标准问题和对应的问句ID
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 将问句ID与标准问题索引相关联，并添加到问句ID列表中
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            # 将问句ID列表转换为张量
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            # 使用模型将问句ID编码为向量，并进行归一化
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return
    
    def eval(self, epoch):
        # 评估模型在验证数据上的效果
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        # 将知识库中的问题编码为向量
        self.knwb_to_vector()
        # 遍历验证数据
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            # 假设验证数据为二元组：输入问句和标签
            input_id, labels = batch_data
            with torch.no_grad():
                # 使用模型将验证数据中的问句编码为向量
                test_question_vectors = self.model(input_id)
            # 记录模型在验证数据上的统计结果
            self.write_stats(test_question_vectors, labels)
        # 显示统计结果
        self.show_stats()
        return

    
    def write_stats(self, test_question_vectors, labels):
        # 将测试问句向量和标签进行匹配，更新统计字典
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 计算测试问句向量与知识库向量的相似度
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            # 找到相似度最高的索引
            hit_index = int(torch.argmax(res.squeeze()))
            # 将索引转换为标准问题索引
            hit_index = self.question_index_to_standard_question_index[hit_index]
            # 根据匹配结果更新统计字典
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return



    
    def show_stats(self): # 显示统计结果 
        total = self.stats_dict["correct"] + self.stats_dict["wrong"] 
        accuracy = self.stats_dict["correct"] / total if total > 0 else 0 
        self.logger.info(f"Accuracy: {accuracy:.4f}, Correct: {self.stats_dict['correct']}, Wrong: {self.stats_dict['wrong']}") 
        return
