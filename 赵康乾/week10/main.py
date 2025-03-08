# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import DataGenerator
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
    train_dg = DataGenerator(config)

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

    # 训练模型，循环epoch轮，每轮取batch_times次样品，进行训练
    epoch = 0
    for i in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info(f'epoch {epoch} start')
        train_loss = [] # 记录本轮的loss

        for batch in range(config['batch_times']):
            train_dg.build_dataset(config) # 避免重复读取语料，随机建立数据集
            input_id = train_dg.data_input
            target_id = train_dg.data_output
            attention_mask = train_dg.data_mask

            if torch.cuda.is_available():
                input_id, target_id, attention_mask = input_id.cuda(), target_id.cuda(), attention_mask.cuda()
            
            optimizer.zero_grad()
            loss = model(input_id, attention_mask, target_id)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            logger.info(f'batch {batch} end, loss is {loss.item()}')
        
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)

            
if __name__ == "__main__":

    main(Config)      
