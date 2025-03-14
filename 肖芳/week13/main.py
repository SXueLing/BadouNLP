# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

from peft import LoraConfig, get_peft_model

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)

    # 标识是否使用gpu
    cuda_flag = torch.backends.mps.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.to("mps")
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.to("mps") for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        evaluator.eval(epoch)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)


# 训练结果 6层bert
#  - 开始测试第10轮模型效果：
#  - INFO - PERSON类实体，准确率：0.922680, 召回率: 0.927461, F1: 0.925060
#  - INFO - LOCATION类实体，准确率：0.870445, 召回率: 0.899582, F1: 0.884769
#  - INFO - TIME类实体，准确率：0.902857, 召回率: 0.887640, F1: 0.895179
#  - INFO - ORGANIZATION类实体，准确率：0.728155, 召回率: 0.789474, F1: 0.757571
#  - INFO - Macro-F1: 0.865644
#  - INFO - Micro-F1 0.880613
#  - INFO - --------------------
#  - INFO - epoch average loss: 0.013914

# 2025-03-08 20:16:47,197 - __main__ - INFO - 开始测试第10轮模型效果：
# 2025-03-08 20:16:48,171 - __main__ - INFO - PERSON类实体，准确率：0.920213, 召回率: 0.891753, F1: 0.905754
# 2025-03-08 20:16:48,171 - __main__ - INFO - LOCATION类实体，准确率：0.897872, 召回率: 0.882845, F1: 0.890290
# 2025-03-08 20:16:48,171 - __main__ - INFO - TIME类实体，准确率：0.895425, 召回率: 0.769663, F1: 0.827790
# 2025-03-08 20:16:48,171 - __main__ - INFO - ORGANIZATION类实体，准确率：0.776596, 召回率: 0.768421, F1: 0.772482
# 2025-03-08 20:16:48,171 - __main__ - INFO - Macro-F1: 0.849079
# 2025-03-08 20:16:48,171 - __main__ - INFO - Micro-F1 0.863367
# 2025-03-08 20:16:48,171 - __main__ - INFO - --------------------
# 2025-03-08 20:16:48,171 - __main__ - INFO - epoch average loss: 0.020810