# Placeholder for main training script
import csv
import os

import numpy as np
import torch
import random


from utils.loader import load_data,ParseCsv
from utils import logger as log
from config import parseConfig
from model import TorchModel,choose_optimizer
from evaluate import Evaluator


def main():
    # 解析配置
    conf = parseConfig()
    # 初始化日志
    logger = log.getLogger(__name__,conf)
    # 配置随机种子
    seed = conf['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #拆分数据样本
    ParseCsv(conf['raw_data_path'],conf)

    #创建保存模型的目录
    if not os.path.isdir(conf["model_path"]):
        os.mkdir(conf["model_path"])
    # 初始化csv
    if not os.path.exists(conf["output_csv_path"]):
        # 数据，可以是一个列表的列表，其中每个内部列表代表CSV中的一行
        headers = [
            ['Model','Learning_Rate','Hidden_Size','预测集合条目总量','预测正确条目','预测准确率','Time(s/100)'],
        ]
        # 打开（或创建）一个CSV文件用于写入
        with open(conf["output_csv_path"], mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入数据到CSV文件
            writer.writerows(headers)

    # 加载训练数据
    train_data = load_data(conf["train_data_path"], conf)
    # 加载模型
    model = TorchModel(conf)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(conf, model)
    # 加载效果测试类
    evaluator = Evaluator(conf, model, logger)
    # 训练
    for epoch in range(conf["epoch"]):
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

    model_path = os.path.join(conf["model_path"], "epoch_%d_%s.pth" % (epoch, conf["model_type"]))
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


if __name__ == '__main__':
    main()
