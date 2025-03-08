# -*- coding: utf-8 -*-

import torch
import pandas as pd
import random
import os
import numpy as np
from datetime import datetime
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
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
    evaluator = Evaluator(config, model, logger)
    #训练
    time_list = []
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("model %s epoch %d begin" % (config["model_type"], epoch))
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
        acc, time_tmp = evaluator.eval(epoch)
        time_list.append(time_tmp)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, sum(time_list) / len(time_list) if time_list else 0

class ModelComparison:
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_comparison(self, model_configs):
        """运行模型比较实验"""
        for model_config in model_configs:
            # 更新配置
            current_config = self.base_config.copy()
            current_config.update(model_config)

            # 训练模型并记录结果
            acc, inference_time = main(current_config)

            # 保存结果
            result = {
                'Model': current_config['model_type'],
                'Learning_Rate': current_config['learning_rate'],
                'Hidden_Size': current_config['hidden_size'],
                'Batch_Size': current_config['batch_size'],
                'Pooling_Style': current_config['pooling_style'],
                'Accuracy': round(acc, 4),
                'Inference_Time_ms': round(inference_time, 2)
            }
            self.results.append(result)

        # 保存结果到CSV
        self.save_results()

    def save_results(self):
        """保存实验结果到CSV文件"""
        results_df = pd.DataFrame(self.results)

        # 创建结果目录
        results_dir = 'model_comparison_results'
        os.makedirs(results_dir, exist_ok=True)

        # 保存结果
        csv_path = os.path.join(results_dir, f'model_comparison_{self.timestamp}.csv')
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # 打印结果表格
        print("\nModel Comparison Results:")
        print(results_df.to_string(index=False))

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn", 'bert', 'lstm']:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg", 'max']:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)

    model_configs = [
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'max'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'max'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'max'
        },
        {
            'model_type': 'gated_cnn',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'max'
        },

        {
            'model_type': 'bert',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'max'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'max'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'max'
        },
        {
            'model_type': 'bert',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'max'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'max'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-3,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'max'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 64,
            'pooling_style': 'max'
        },
        {
            'model_type': 'lstm',
            'learning_rate': 1e-4,
            'hidden_size': 128,
            'batch_size': 128,
            'pooling_style': 'max'
        }
        # 可以继续添加更多模型配置
    ]
    comparison = ModelComparison(Config)
    comparison.run_comparison(model_configs)
## 对比结果
# Model Comparison Results:
#     Model  Learning_Rate  Hidden_Size  Batch_Size Pooling_Style  Accuracy  Inference_Time_ms
# gated_cnn         0.0010          128          64           avg    0.6694               1.48
# gated_cnn         0.0010          128         128           avg    0.6583               2.82
# gated_cnn         0.0001          128          64           avg    0.3750               2.06
# gated_cnn         0.0001          128         128           avg    0.3722               4.12
# gated_cnn         0.0010          128          64           max    0.6806               1.46
# gated_cnn         0.0010          128         128           max    0.6889               3.14
# gated_cnn         0.0001          128          64           max    0.4722               1.76
# gated_cnn         0.0001          128         128           max    0.3611               2.96
#      bert         0.0010          128          64           avg    0.7722              13.73
#      bert         0.0010          128         128           avg    0.7833              30.52
#      bert         0.0001          128          64           avg    0.7861              16.72
#      bert         0.0001          128         128           avg    0.7889              31.05
#      bert         0.0010          128          64           max    0.7806              16.44
#      bert         0.0010          128         128           max    0.7917              29.12
#      bert         0.0001          128          64           max    0.8056              24.07
#      bert         0.0001          128         128           max    0.8083              30.43
#      lstm         0.0010          128          64           avg    0.5694              18.77
#      lstm         0.0010          128         128           avg    0.5528              21.26
#      lstm         0.0001          128          64           avg    0.3528              16.99
#      lstm         0.0001          128         128           avg    0.1667              19.44
#      lstm         0.0010          128          64           max    0.5722              16.77
#      lstm         0.0010          128         128           max    0.5611              17.75
#      lstm         0.0001          128          64           max    0.3667              17.02
#      lstm         0.0001          128         128           max    0.1944              21.30