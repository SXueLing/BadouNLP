'''
Author: Zhao
Date: 2025-01-14 20:00:51
LastEditTime: 2025-01-15 18:22:04
FilePath: predict.py
Description: 

'''
import torch
from loader import load_data
from config import Config
from model import SiameseNetwork, choose_optimizer

class Predictor:
    def __init__(self, config, model, knwb_data):
        # 初始化 Predictor 类，加载配置、模型和知识库数据
        self.config = config
        self.model = model
        self.train_data = knwb_data
        
        # 根据是否有可用的GPU，将模型迁移到GPU或CPU
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 将知识库中的问题向量化
        self.knwb_to_vector()

    # 将知识库中的问题向量化，为匹配做准备
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        # 创建标准问题索引到问题编号的映射字典
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())
        
        # 遍历训练数据中的标准问题和对应的问句ID
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
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

    # 将输入的句子编码为ID序列
    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    # 预测输入句子的标准问题
    def predict(self, sentence):
        # 编码输入句子为ID序列
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        
        # 将输入句子ID序列迁移到GPU（如果可用）
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        
        with torch.no_grad():
            # 使用模型将输入句子编码为向量进行预测
            test_question_vector = self.model(input_id)
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze())) # 获取最相似的问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index] # 转化成标准问编号
        
        # 返回标准问题
        return self.index_to_standard_question[hit_index]


if __name__ == "__main__":
    knwb_data = load_data(Config["train_data_path"], Config)
    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load("week8/model_output/epoch_10.pth"))
    pd = Predictor(Config, model, knwb_data)

    sentence = "这个卡的消费"
    res = pd.predict(sentence)
    print(res)# 话费查询
    # while True:
    #     sentence = input("请输入问题：")
    #     res = pd.predict(sentence)
    #     print(res)

