# -*- coding: utf-8 -*-
# @Date    :2025-02-11 22:13:16
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# 打印日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# 随机种子，保证可以复现
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建记录训练后的模型放置目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载数据
    train_data,val_data = load_data(config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否是否GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型到gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config,model)
    # 加载效果测试类
    # evaluator = Evaluator(config,model,logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" %epoch)
        train_loss = []
        for index,batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            token_IDs,labels = batch_data
            loss = model(token_IDs,labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data)/2)==0:
                logger.info("batch loss %f" %loss)
        logger.info("epoch average loss: %f" %np.mean(train_loss))

if __name__ == '__main__':
    main(Config)