# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    # 加载训练数据和模型
    train_data = load_data(config['train_data_path'], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 选择优化器
    optimizer = choose_optimizer(config, model)
    # 训练
    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info('epoch %d start' % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids, attention_mask, label_ids = batch_data # 词序号，掩码，真实标签
            loss = model(input_ids, attention_mask, label_ids)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        logger.info(f"保存模型到路径: {model_path}")
        torch.save(model.state_dict(), model_path)
        logger.info("模型保存成功")
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)   
