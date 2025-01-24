# -*- coding: utf-8 -*-

import logging
import os
import random

import numpy as np
import torch

from config import Config
from evaluate import Evaluator
from loader import load_data
from model import TorchModel, choose_optimizer
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc


if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    show_data=[]
    png = "./result.png"
    result_path = "./result.txt"
    if os.path.exists(png):
        os.remove(png)
    if os.path.exists(result_path):
        os.remove(result_path)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("模型\t学习率\t隐藏层大小\t批大小\t池化方式\t准确率\n")
        # 超参数的网格搜索
        for model in ["gated_cnn", 'bert', 'lstm']:
            Config["model_type"] = model
            for lr in [1e-3, 1e-4]:
                Config["learning_rate"] = lr
                for hidden_size in [128]:
                    Config["hidden_size"] = hidden_size
                    for batch_size in [64, 128]:
                        Config["batch_size"] = batch_size
                        for pooling_style in ["avg", 'max']:
                            Config["pooling_style"] = pooling_style
                            print("最后一轮准确率：", main(Config), "当前配置：", Config)
                            Format = "%s\t%f\t%d\t%d\t%s\t%f\n" % (
                            Config["model_type"], Config["learning_rate"], Config["hidden_size"], Config["batch_size"],
                            Config["pooling_style"], main(Config))
                            f.write(Format)
                            show_data.append(Format)
    print("所有模型的结果：")
    plt.figure(figsize=(10, 5))
    plt.title("模型效果对比")
    plt.xlabel("模型")
    plt.ylabel("准确率")
    plt.bar(range(len(show_data)), [float(data.split("\t")[-1]) for data in show_data], tick_label=[data.split("\t")[0] for data in show_data])
    plt.show()

    plt.savefig(png)
    # print(show_data)


