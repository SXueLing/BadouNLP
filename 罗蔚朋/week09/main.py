# -*- coding: utf-8 -*-
import logging
import os.path

import numpy as np
import torch.cuda

import config
from evaluate import Evaluator
from loader import load_data
from model import TorchModel, choose_optimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    # 加载训练数据
    train_data = load_data(config['train_data_path'], config)
    # 加载模型
    model = TorchModel(config)
    # gpu是否可用
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} begin")
        train_loss = []
        for index, batch_data in enumerate(train_data):
            # 梯度归零
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            loss = model(input_id, labels)
            # 反向传播，计算梯度
            loss.backward()
            # 优化器更新参数
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info(f"batch loss {loss}")
        logger.info(f"epoch average loss: {np.mean(train_loss)}")
        evaluator.eval(epoch)
        model_path = os.path.join(config['model_path'], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main(config.Config)
