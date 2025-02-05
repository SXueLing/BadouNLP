# -*- coding : UTF-8 -*- #
import torch
from model import TorchModel
from config import Config
import logging
from main import  logger
from transformers import BertTokenizer
from loader import  load_vocab
from numpy import argmax



def Predict(model_path,input_strings):
    x=[]
    for input_string in input_strings:
        if Config["model_type"] == "bert":
            tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
            input_id =tokenizer.encode(input_string, truncation=True, max_length=Config["max_length"],
                                             padding='max_length')
        else:
            input_id = encode_sentence(input_string)

        x.append(input_id)
        # print(input_id)
        # print(len(input_id))


    #加载模型
    model = TorchModel(Config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重



    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测

    index_to_label = {0: '差评', 1: '好评'}
    for i, input_string in enumerate(input_strings):
        print(input_string,index_to_label[argmax(result[i].numpy())])
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果


def encode_sentence(text):
    input_id = []
    vocab = load_vocab(Config["vocab_path"])
    Config["vocab_size"] = len(vocab)
    for char in text:
        input_id.append(vocab.get(char, vocab["[UNK]"]))
    input_id = padding(input_id)
    return input_id

#补齐或截断输入的序列，使其可以在一个batch内运算
def padding(self, input_id):
    input_id = input_id[:self.config["max_length"]]
    input_id += [0] * (self.config["max_length"] - len(input_id))
    return input_id




if __name__ == "__main__":
    vocab = load_vocab(Config["vocab_path"])
    Config["vocab_size"] = len(vocab)
    Config["class_num"] = 2
    test_strings = ["非常好吃，家人非常喜欢", "有点凉了,口感不好", "送货很快", "口味不好，下次不买了"]
    Predict("output/epoch_15bert.pth",test_strings)
