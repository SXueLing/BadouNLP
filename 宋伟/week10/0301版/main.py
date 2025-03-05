# -*- coding: utf-8 -*-
# @Date    :2025-03-02 00:09:16
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
from config import Config
from loader import data_load
from model import LanguageModel,choose_optimizer
import logging
import random
import os
import numpy as np

# 日志打印
logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 随机种子，保证复现
seed = Config['seed']
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 保存路径
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    # 加载数据
    train_data = data_load(config)
    # 加载模型
    model = LanguageModel(config)
    # gpu使用
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型到GPU")
        model.cuda()
    # 优化器
    optimizer = choose_optimizer(model,config)

    # 训练
    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} start")
        train_loss = []
        for idx,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = {key:data.cuda() for key,data in batch_data.items()}
                pass
            inputs = batch_data['inputs']
            labels = batch_data['labels']
            

            # break
            loss = model(inputs,labels)
            # break
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if idx%int(len(train_data)/2) == 0:
                logger.info(f"batch loss: {loss}")
        logger.info(f"epoch average loss:{np.mean(train_loss)}")
    model_path = os.path.join(config['model_path'],f"epoch_bert04_{epoch}.pth")
    torch.save(model.state_dict(),model_path)
    return None

if __name__ == '__main__':
    main(Config)

