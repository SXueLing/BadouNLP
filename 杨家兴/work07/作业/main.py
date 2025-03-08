
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
import pandas as pd

df = pd.DataFrame(columns=["model_type", "learning_rate", "hidden_size", "batch_size", "pooling_style", "acc"])

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data #输入变化时这里需要修改，比如多输入，多输出的情况
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
    for model in ["cnn", "fast_text", "lstm"]:
        Config["model_type"] = model
        acc = main(Config)
        print("最后一轮准确率：", acc, "当前配置：", Config["model_type"])
        df = df._append({"model_type": model, "learning_rate": Config["learning_rate"],
                          "hidden_size": Config["hidden_size"], 
                         "batch_size": Config["batch_size"], "pooling_style": Config["pooling_style"],
                           "acc": acc}, ignore_index=True)
    df.to_excel("result.xlsx", index=False)

    # 主要修改点: 将文本分成训练集和验证集。config路径修改。
    # loader路径进行测试，分类改成好评和差评。
    # 输入和实际输出修改掉，然后训练。
    # class_num会loader里统计，用于转成几分类任务：
    # self.classify = nn.Linear(hidden_size, class_num)


    # main(Config)
    # print("最后一轮准确率：", main(Config), "当前配置：", Config)
    # 最后一轮准确率： 0.7722222222222223 当前配置： {'model_path': 'output', 
    #     'train_data_path': '../data/train_tag_news.json',
    #     'valid_data_path': '../data/valid_tag_news.json',
    #     'vocab_path': 'chars.txt', 'model_type': 'bert',
    #     'max_length': 30, 'hidden_size': 256,
    #     'kernel_size': 3, 'num_layers': 2, 'epoch': 5,
    #     'batch_size': 128, 'pooling_style': 'max',
    #     'optimizer': 'adam', 'learning_rate': 0.001,
    #     'pretrain_model_path': '/Users/mac/Documents/bert-base-chinese',
    #     'seed': 987, 'class_num': 18, 'vocab_size': 4622}


    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])
 
    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn", "bert", "lstm"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", "max"]:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)

                        