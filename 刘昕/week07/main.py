# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from use_model import predict
import time
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
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
    evaluator = Evaluator(config, model, logger, "valid_data_path")
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
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        
    model_path = os.path.join(config["model_path"], "%s_%f_%d_%d_%s" % (config["model_type"], config["learning_rate"], config["hidden_size"], config["batch_size"], config["pooling_style"]))
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)
    # print("最后一轮准确率：", main(Config), "当前配置：", Config)
    test_strings = ["这个东西真的很好，很好用，很方便", "这个东西真的很差，很难用，很不方便"]
    predict(test_strings, Config["vocab_path"], Config["model_path"] + "/gated_cnn_0.000100_128_64_avg")


    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ['bert']:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     accuracy = main(Config)
    #                     print("最后一轮准确率：", accuracy, "当前配置：", "模型类型：", Config["model_type"], "学习率：", Config["learning_rate"], "隐藏层大小：", Config["hidden_size"], "批次大小：", Config["batch_size"], "池化方式：", Config["pooling_style"])
    #                     predict_data_path = Config["predict_data_path"]
    #                     vocab_path = Config["vocab_path"]
    #                     model_path = Config["model_path"] + "/%s_%f_%d_%d_%s" % (Config["model_type"], Config["learning_rate"], Config["hidden_size"], Config["batch_size"], Config["pooling_style"])
    #                     # 计算预测100条时间
    #                     load_model = TorchModel(Config)
    #                     load_model.load_state_dict(torch.load(model_path))
    #                     evaluator = Evaluator(Config, load_model, logger, "predict_data_path")
    #                     start = time.time()
    #                     acc = evaluator.eval(1)
    #                     end = time.time()
    #                     print("100条预测时间：%.8f秒，准确率为 %f" % (end - start, acc))