# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from loader import load_data
from transformers import BertTokenizer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def train(config):
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.backends.mps.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.to("mps")
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.to("mps") for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
    torch.save(model.state_dict(), "model.pth")

    # 训练结束后使用模型预测
    predict(model, "李慕")

    return model, train_data

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = torch.nn.functional.softmax(prob_distribution, dim=-1)
        prob_distribution = prob_distribution.detach().to('cpu').numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
    
def getNextWord(text, model):
    tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])
    input_ids = tokenizer.encode(text, max_length=len(text), pad_to_max_length=True, truncation=True, add_special_tokens=False)
    input_ids = torch.LongTensor([input_ids]) # (1 * 2)
    cuda_flag = torch.backends.mps.is_available()
    if cuda_flag:
        input_ids = input_ids.to("mps")
    y_predict = model(input_ids) # (1, 2, vocab_size)
    # 拿到最后一个字的vocab_size的概率分布
    y_predict = y_predict[0, -1, :] # (vocab_size,)
    y_predict_index = sampling_strategy(y_predict)
    # 找到下标对应的词
    predic_char = tokenizer.decode(y_predict_index) # (1,)
    return predic_char

def predict(model, start_text):
    text = start_text
    model.eval()
    
    for i in range(30):
        next = getNextWord(text, model)
        text += next
    print(text)
    


if __name__ == "__main__":
    # model, train_data = train(Config)

    # 根据路径加载模型
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model.pth"))
    cuda_flag = torch.backends.mps.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.to("mps")

    predict(model, "让他在半年之前，就不能做出")
    predict(model, "李慕站在山路上，深深的呼吸")

