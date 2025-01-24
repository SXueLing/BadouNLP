'''
Author: Zhao
Date: 2025-01-14 20:16:03
LastEditTime: 2025-01-15 18:24:14
FilePath: main.py
Description: 初始化：检查并创建模型存储目录，加载训练数据。
        模型准备：初始化孪生网络模型，选择优化器，并检查是否使用 GPU 进行训练。
        训练循环：在每个训练周期内：
            遍历训练数据，计算三元组损失，并更新模型参数。
            记录并输出每个周期的平均损失。
        模型评估：每个周期结束后，对模型进行评估，并记录性能指标。
        模型保存：将训练后的模型参数保存到指定路径。
'''
import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    # 检查模型路径是否存在，如果不存在则创建目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 初始化模型
    model = SiameseNetwork(config)

    # 检查是否有可用的GPU，如果有则将模型迁移至GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 选择优化器
    optimizer = choose_optimizer(config, model)

    # 初始化评估器
    evaluator = Evaluator(config, model, logger)

    # 训练模型，遍历每个训练周期
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        # 遍历训练数据的每个批次
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            # 如果使用GPU，则将数据迁移至GPU
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            anchor, positive, negative = batch_data

            # 计算模型的损失
            loss = model(anchor, positive, negative)
            train_loss.append(loss.item())

            # 反向传播梯度并更新模型参数
            loss.backward()
            optimizer.step()

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        
        # 每个周期结束后进行模型评估
        evaluator.eval(epoch)

    # 保存训练后的模型参数
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    # 运行主函数
    main(Config)
