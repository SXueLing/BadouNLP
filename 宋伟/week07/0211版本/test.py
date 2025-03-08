# -*- coding: utf-8 -*-
# @Date    :2025-02-16 18:47:47
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
from loader import MyDataSet,load_data
from evaluate import Evaluator
from model import TorchModel
from config import Config

'''
进行一些实例数据的（评论的预测）的分析

[description]
'''

def prediction(text_data,model):
    index_to_label = {1: "积极评论", 0: "消极评论"}
    model.eval()
    with torch.no_grad():
        print(model(text_data))
        pre_index = torch.argmax(model(text_data)).item()
    pre_label = index_to_label.get(pre_index,"dea")
    print(text_data,"属于",pre_label)

if __name__ == '__main__':
    model_path = r"./output/epoch_lstm_10"
    # 拿一条数据进行分析
    train_data,val_data = load_data(Config)
    text_data = "真是饕餮盛宴"
    dataset = MyDataSet(Config)
    text_Ids = torch.tensor(dataset.encode_sentence(text_data),dtype=torch.long)
    # 实例化模型结构
    model = TorchModel(Config)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    # 将数据传入是，数据格式是[batch,sequence],结果是[batch,class_num]的数据格式
    text_Ids = text_Ids.unsqueeze(0)
    prediction(text_Ids,model)