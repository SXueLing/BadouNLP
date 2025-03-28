import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
"""
模拟训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    # 创建保存模型的目录
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("使用GPU进行训练")
        model = model.cuda()
    # 选择优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 开始训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("开始第%d轮训练：" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            optimizer.zero_grad()   # 梯度归零
            input_ids, labels = batch_data
            loss = model(input_ids, labels) # 计算loss
            loss.backward() # 反向传播
            optimizer.step()    # 更新权重

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        
        logger.info("第%d轮训练结束，平均loss %f" % (epoch, np.mean(train_loss)))
        acc = evaluator.eval(epoch)

    return acc


if __name__ == "__main__":
    main(Config)