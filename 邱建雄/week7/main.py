# -*- coding: utf-8 -*-
import json

import pandas as pd
import torch
import os
import random
import os
import numpy as np
import logging
import time
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
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
    #创建保存模型的目录
    t0 = time.time()
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
    # 记录模型最后一轮损失
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        # logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
        # logger.info("epoch average loss: %f" % np.mean(train_loss))
        model_end_loss = np.mean(train_loss)
        acc = evaluator.eval(epoch)
    # logger.info("model average loss: %f" % model_end_loss)
    t1 = time.time()
    # print("model average loss: %f" % model_end_loss)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, t1 - t0, model_end_loss


if __name__ == "__main__":
    logger.setLevel(logging.WARNING)
    # print(main(Config))
    # cuda_flag = torch.cuda.is_available()
    # print(cuda_flag)

    #
    # # for model in ["cnn"]:
    # #     Config["model_type"] = model
    # #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])
    model_results = pd.DataFrame(
        columns=['model', 'max_length', 'hidden_size', 'kernel_size', 'num_layers', 'epoch', 'batch_size',
                 'pooling_style', 'optimizer', 'learning_rate', 'class_num', 'vocab_size', 'Final_accuracy',
                 'final_loss'
                 ])
    # # 对比所有模型
    # # 中间日志可以关掉，避免输出过多信息
    # # 超参数的网格搜索
    for model in ["rnn", 'cnn', 'fast_text', 'gru', "gated_cnn", "stack_gated_cnn", "rcnn", "bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        final_accuracy, times, final_loss = main(Config)
                        results = ([{
                            'model': Config["model_type"],
                            'max_length': Config["max_length"],
                            'hidden_size': Config["hidden_size"],
                            'kernel_size': Config["kernel_size"],
                            'num_layers': Config["num_layers"],
                            'epoch': Config["epoch"],
                            'batch_size': Config["batch_size"],
                            'pooling_style': Config["pooling_style"],
                            'optimizer': Config["optimizer"],
                            'learning_rate': Config["learning_rate"],
                            'class_num': Config["class_num"],
                            'vocab_size': Config["vocab_size"],
                            'final_accuracy': final_accuracy,
                            'final_loss': final_loss,
                            'time': times
                        }])
                        df_results = pd.DataFrame(results)
                        model_results = pd.concat([model_results, df_results], ignore_index=True)
        print(model_results)
    model_results.to_csv("model_results_1.csv", index=False)


    #                     print("最后一轮准确率：", main(Config), "当前配置：", json.dumps(Config, indent=4))
