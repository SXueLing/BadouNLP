# -*- coding: utf-8 -*-
import time
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
import pandas as pd

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


def test(config):
    model = TorchModel(config)
    model.load_state_dict(torch.load(config["model_save_path"]))
    model.eval()
    start = time.time()
    with torch.no_grad():
        test_data = load_data(config["valid_data_path"], config)
        for index in range(100):
            input_ids, labels = next(iter(test_data))
            pred = model(input_ids)
    end_time = time.time()
    return end_time - start


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
    model_save_name = "model_type_%s_learning_rate_%s_epoch_%d_batch_size_%d_pooling_style_%s.pth" % (
        config['model_type'], config['learning_rate'], config["epoch"], config['batch_size'],
        config['pooling_style'])
    model_path = os.path.join(config["model_path"],
                              model_save_name)
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    config['model_save_path'] = model_path
    return acc


if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    data = {
        "Model": [],
        "Learning_Rate": [],
        "Hidden_Size": [],
        "Acc": [],
        "Time": []
    }
    for model in ["gated_cnn", 'bert', 'lstm', 'bert_lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        acc = main(Config)
                        print("最后一轮准确率：", acc, "当前配置：", Config)
                        data["Model"].append(model)
                        data["Learning_Rate"].append(lr)
                        data["Hidden_Size"].append(hidden_size)
                        data["Acc"].append(acc)
                        #                         预测100条
                        cost_time = test(config=Config)
                        data["Time"].append(cost_time)
        df = pd.DataFrame(data)
        file_path = '模型对比.xlsx'  # 输出文件的路径和名称
        sheet_name = 'Sheet1'  # Excel 中的工作表名称

        # 使用 openpyxl 引擎写入 Excel 文件
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
