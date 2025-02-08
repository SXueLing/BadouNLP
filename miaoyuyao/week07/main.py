# -*- coding: utf-8 -*-
"""
模型训练主程序
"""
import os.path
import random

import numpy as np
import pandas as pd
import torch

from test_ai.homework.week07.config.logging_config import logger
from test_ai.homework.week07.config.running_config import Config
from test_ai.homework.week07.data.loader import load_data
from test_ai.homework.week07.evaluate.evaluate import Evaluator
from test_ai.homework.week07.models.optimizer import choose_optimizer
from test_ai.homework.week07.models.torch_model import TorchModel

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    global batch_data, acc, avg_100_time
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    else:
        logger.info("gpu不可以使用")
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model)

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
        acc, avg_100_time = evaluator.eval(epoch)

    return acc, avg_100_time


if __name__ == "__main__":
    csv_file_path = Config["result_data_path"]

    for model in ["bert_lstm", 'bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        res = main(Config)
                        additional_data = {
                            "Model": [Config["model_type"]],
                            "Learning_Rate": Config["learning_rate"],
                            "Hidden_Size": Config["hidden_size"],
                            "batch_size": Config["batch_size"],
                            "pooling_style": Config["pooling_style"],
                            "acc":  res[0],
                            "time(预测100条耗时）": res[1]
                        }
                        df_additional = pd.DataFrame(additional_data)
                        file_exists = os.path.isfile(csv_file_path)
                        df_additional.to_csv(csv_file_path, mode='a', index=False, header=not file_exists,
                                             encoding='utf-8')
                        logger.info(f"本次数据已成功记录到 {csv_file_path}")
