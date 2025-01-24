# coding:utf8
# auther: 王良顺

import json

import torch
from torch.nn import DataParallel

import TorchModel as tmodel


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = tmodel.TorchModel(char_dim, vocab)    #建立模型
    # 将模型包装在DataParallel中
    model = DataParallel(model)
    # 将模型移动到GPU上
    model = model.cuda()
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        for i, input_string in enumerate(input_strings):
            print(f"输入：{input_string}, 预测位置：第{result[i].argmax() + 1 if result[i].argmax() < sentence_length else -1}位")
            # print(f"预测位置：第{result[i].argmax() + 1 if result[i].argmax() < sentence_length else -1}位")

if __name__ == '__main__':
    test_strings = ["fnvfeeimvl", "wz你dfguytj", "rqwdegopq你", "n我kwwwzyqh"]
    predict("model.pt", "vocab.json", test_strings)


