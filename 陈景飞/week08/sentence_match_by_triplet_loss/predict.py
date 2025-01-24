# -*- coding: utf-8 -*-
import torch
from loader import load_data, load_vocab
from config import Config
from model import SiameseNetwork, choose_optimizer
import jieba

"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.vocab = load_vocab(config["vocab_path"]) #字表
        self.config["vocab_size"] = len(self.vocab) #字表大小
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        # self.knwb_to_vector()

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            #将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id

    def predict(self, a, p, n):
        a = torch.LongTensor([self.encode_sentence(a)])
        p = torch.LongTensor([self.encode_sentence(p)])
        n = torch.LongTensor([self.encode_sentence(n)])
        if torch.cuda.is_available():
            a,p,n = a.cuda(),p.cuda(),n.cuda()
        with torch.no_grad():
            test_question_vector = self.model(a,p,n) #不输入labels，使用模型当前参数进行预测
            # print(test_question_vector)

        return  test_question_vector

if __name__ == "__main__":
    a = "查话费"
    p = "查一下我的电话费"
    n = "今晚吃什么"

    Config["vocab"] = load_vocab(Config["vocab_path"])
    Config["vocab_size"] = len(Config["vocab"])
    model = SiameseNetwork(Config)
    '''
    PyTorch 计划在未来的版本中改变 weights_only 的默认值，将其设置为 True。当 weights_only=True 时，只有模型的权重会被加载，这限制了可以在解包过程中执行的函数，从而提高了安全性。
    因为当 weights_only=False 时，torch.load 会使用默认的 pickle 模块，而恶意的 pickle 数据可能在解包时执行任意代码。
    '''
    model.load_state_dict(torch.load("model_output/epoch_" + str(Config["epoch"]) + ".pth", weights_only=True))
    pd = Predictor(Config, model)
    res = pd.predict(a, p, n)

    if res[0].item() == 1:
        print("a,p,n 是正样本")
    else:
        print("a,p,n 是负样本")









