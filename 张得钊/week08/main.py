# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import pandas as pd

# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志记录器的级别

# 创建一个FileHandler，用于将日志写入文件
file_handler = logging.FileHandler('app2.log', encoding='utf-8')
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

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_a, input_p, input_n = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_a, input_p, input_n, config["margin"])
            logger.info("loss: %f" % loss.item())
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        accuracy = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "2_epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return accuracy

if __name__ == "__main__":
    # main(Config)

    result =  []
    logger.info("==================开始===================")
    # 遍历不同的超参数组合
    for model in ["dnn"]:
        Config["model_type"] = model
        logger.info("--------------------------------------")
        logger.info("-------------当前模型：%s--------------" % model)
        for lr in [5e-3, 1e-3, 5e-4, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [64, 128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [16, 32, 64]:
                    Config["batch_size"] = batch_size
                    for margin in [0.1, 0.5, 1.0, 1.5, 2.0]:
                        Config["margin"] = margin
                        logger.info("当前配置 -- 模型: %s, 学习率: %f, 隐藏层大小: %d, 批大小: %d, margin: %f" 
                                    % (model, lr, hidden_size, batch_size, margin))
                        accuracy = main(Config)
                        result.append([accuracy, model, lr, hidden_size, batch_size, margin])
                        logger.info("-------------下一次执行---------------")
    # 创建 DataFrame
    df = pd.DataFrame(result, columns=['acc', 'model', 'lr', 'hidden_size', 'batch_size', 'margin'])
    # 创建一个ExcelWriter对象，指定文件名和模式为追加（'a'）
    with pd.ExcelWriter("result.xlsx", mode='a', if_sheet_exists='replace') as writer:
        # 将DataFrame写入Sheet2
        df.to_excel(writer, sheet_name='Sheet4', index=False)
    # 打印表格
    print(df)
    