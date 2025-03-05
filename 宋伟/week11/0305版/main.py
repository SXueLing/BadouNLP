# -*- coding: utf-8 -*-
# @Date    :2025-03-05 16:53:29
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text


import torch
from config import Config
from loader import data_load
from model import LanguageModel,choose_optimizer
import logging
import os
import numpy as np

# 日志打印
logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 种子设置,保证可以复现
seed=Config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 检测gpu是否可用
    cuda_flag = torch.cuda.is_available()
    # 模型路径
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    # 加载数据
    train_data = data_load(config)
    # 加载模型
    model = LanguageModel(config)
    if cuda_flag:
        logger.info('gpu可以使用，模型参数迁移到GPU')
        model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config,model=model)

    # 训练
    for epoch in range(config['epoch_num']):
        epoch += 1
        model.train()
        logger.info(f'epoch {epoch} start:')
        train_loss = []
        for idx,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = {key:data.cuda() for key,data in batch_data.items()}

            loss = model(**batch_data)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if idx% int(len(train_data)/2) == 0:
                logger.info(f'batch loss:{loss.item()}')
        logger.info(f"epoch:{epoch} average loss:{np.mean(train_loss)}")
    model_path = os.path.join(config['model_path'],f"epoch_bert_sft_{epoch}.pth")
    torch.save(model.state_dict(),model_path)

    return


if __name__ == '__main__':
    main(Config)