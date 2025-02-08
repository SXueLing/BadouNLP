# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import torch
import os
import random
import time
import numpy as np
import pandas as pd
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from preprocess import DataPre
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
# logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志记录器的级别

# 创建一个FileHandler，用于将日志写入文件
file_handler = logging.FileHandler('app1_9_15.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 设置FileHandler的级别

# 创建一个日志格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将FileHandler添加到日志记录器
logger.addHandler(file_handler)

import warnings
warnings.filterwarnings('ignore')

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #数据预处理
    if not os.path.exists(config["train_data_path"]) or not os.path.exists(config["valid_data_path"]):
        data_pre = DataPre(config["data_path"], config)
        data_pre.split()
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 记录训练开始时间
    train_start_time = time.time()

    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        
        # 记录训练结束时间
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        # 记录预测开始时间
        predict_start_time = time.time()
        acc = evaluator.eval(epoch)
        # 记录预测结束时间
        predict_end_time = time.time()
        # 计算预测时间
        predict_time = predict_end_time - predict_start_time
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, train_time, predict_time

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    result =  []
    for model in ["fast_text","rnn","cnn","gated_cnn","gru","lstm"]:
        Config["model_type"] = model
        logger.info("============模型 %s 开始===============" % model)
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ['avg']:
                        Config["pooling_style"] = pooling_style
                        logger.info("当前配置：%s" % Config)
                        accuracy, train_time, predict_time = main(Config)
                        print("最后一轮准确率：", accuracy, "当前配置：", Config)
                        result.append([model, accuracy, lr, hidden_size, batch_size, pooling_style, train_time, predict_time])
                        logger.info("-------------下一次执行---------------")
    # result =  []
    # for model in ["bert"]:
    #     Config["model_type"] = model
    #     logger.info("============模型 %s 开始===============" % model)
    #     for lr in [2e-5, 3e-5]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [768]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ['avg']:
    #                     Config["pooling_style"] = pooling_style
    #                     logger.info("当前配置：%s" % Config)
    #                     accuracy, train_time, predict_time = main(Config)
    #                     print("最后一轮准确率：", accuracy, "当前配置：", Config)
    #                     result.append([model, accuracy, lr, hidden_size, batch_size, pooling_style, train_time, predict_time])
    #                     logger.info("-------------下一次执行---------------")
    
    # 创建 DataFrame
    df = pd.DataFrame(result, columns=['model', 'acc', 'lr', 'hidden_size', 'batch_size', 'pooling_style', 'train_time', 'predict_time'])
    # 打印表格
    print(df)
    
    # for model in ["gated_cnn", 'bert', 'lstm']:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)


