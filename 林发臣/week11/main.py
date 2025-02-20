import time

import torch.nn as nn
import torch
import json
import train_util
import nlp_util as nlpu
from model import LanguageBertToSftModel
from evaluate import LanguageBertToSftEvaluate
from loader import load_data
import logging
import numpy as np
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def choose_optimizer(model: nn.Module, config):
    optimizer = config['optimizer']
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        return torch.optim.SGD(model.parameters(), lr=config['learning_rate'])


def train(config):
    start_time = time.time()
    cuda_enable = torch.cuda.is_available()
    if cuda_enable:
        logger.info('gpu可以使用')
    # 加载模型
    my_model = LanguageBertToSftModel(config)
    if cuda_enable:
        my_model = my_model.cuda()
    # 加载训练数据
    torch.autograd.set_detect_anomaly(True)
    dl = load_data(config, config['train_data_path'])
    # 加载测试类
    evaluate = LanguageBertToSftEvaluate(my_model, config)
    # 记载优化器
    optimizer = choose_optimizer(my_model, config)
    # 开始训练
    for epoch in range(config['epoch']):
        my_model.train()
        loss = []
        epoch += 1
        for index, batch_data in enumerate(dl):
            optimizer.zero_grad()
            if cuda_enable:
                batch_data = [data.cuda() for data in batch_data]
            x, x_mask, label = batch_data
            loss_item = my_model(x, x_mask, label)
            loss_item.backward()
            optimizer.step()
            loss.append(loss_item.item())
            # if index % (len(batch_data) // 2) == 0:
            #     logger.info(f'current_epoch={epoch},current_loss={np.mean(loss)}')
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(loss)))
        evaluate.eval('为什么今天上海这么热', my_model)
        evaluate.eval('北京明年拟推工作日半价观看电影', my_model)
        path_name = train_util.trans_save_path(config, epoch=epoch)
        torch.save(my_model.state_dict(), path_name)


if __name__ == '__main__':
    from config import Config
    train_util.mutil_train(Config, train, '第十一周作业')
