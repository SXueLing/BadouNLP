# -*- coding: utf-8 -*-
# @Time    : 2025/1/8 16:10
# @Author  : yeye
# @File    : main.py
# @Software: PyCharm
# @Desc    :
import torch
import os
import random
import os
import pandas as pd
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])

    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optimiezer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimiezer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimiezer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss:%f" % loss)
        logger.info("epoch average loss:%f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    return acc

if __name__ == '__main__':
    result = []
    for model in ['bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-4, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        accuracy = main(Config)  # 假设 main 函数返回准确率

                        # 打印日志信息
                        print(f"最后一轮准确率：{accuracy} 当前配置：{Config}")

                        # 将配置和准确率保存到results列表中
                        result.append({
                            "model_type": model,
                            "learning_rate": lr,
                            "hidden_size": hidden_size,
                            "batch_size": batch_size,
                            "pooling_style": pooling_style,
                            "accuracy": accuracy
                        })

        # 将所有的结果保存为一个DataFrame
    df_results = pd.DataFrame(result)

    # 保存为Excel文件
    df_results.to_excel("model_comparison_results.xlsx", index=False, engine='openpyxl')

    print("所有结果已保存到 model_comparison_results.xlsx")

