# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import DataGenerator, load_data
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    # 创建模型保存目录
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    
    # 读取数据
    train_data = load_data(config)    

    # 加载预训练模型
    model = TorchModel(config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 创建优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    epoch = 0
    for i in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info(f'epoch {epoch} start')
        train_loss = [] # 记录本轮的loss

        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            q_a, answer, mask = batch_data   
            loss = model(mask = mask, question_and_answer = q_a, answer = answer)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            logger.info(f'batch {index} finish')
        
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)

            
if __name__ == "__main__":

    main(Config)        


            
