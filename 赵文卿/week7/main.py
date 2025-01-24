'''
Author: Zhao
Date: 2025-01-08 18:00:17
LastEditTime: 2025-01-08 15:42:14
FilePath: main.py
Description: 模型训练主程序

'''

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
import itertools

# 设置日志记录器配置
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以确保结果的可复现性
seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录（如果不存在）
    if not os.path.isdir(config["model_path"]):
        logger.info("路径不存在，开始创建路径")
        os.mkdir(config["model_path"])

    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    
    #加载模型
    model = TorchModel(config)
    
    # 检查是否有可用的 GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    
    #加载优化器
    optimizer = choose_optimizer(config, model)
    
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 开始训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []
        #logger.info(f"epoch {epoch} begin")
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # if index % int(len(train_data) // 2) == 0:
            #     logger.info(f"Batch loss: {loss:.6f}")
        
        avg_loss = np.mean(train_loss)       
        logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.6f}")
        
        # 在每个epoch后评估模型效果
        acc = evaluator.eval(epoch)
        logger.info(f"Epoch {epoch} 准确率: {acc:.6f}")

    # model_path = os.path.join(config["model_path"], f"week7_epoch_{epoch}.pth" )
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc


if __name__ == "__main__":
    # 运行默认配置
    main(Config)

    # 配置参数 
    model_types = ["gated_cnn", "bert", "lstm","fast_text","bert_mid_layer"] 
    learning_rates = [1e-3, 1e-4] 
    hidden_sizes = [128] 
    batch_sizes = [64, 128] 
    pooling_styles = ["avg", 'max'] 
    # 生成所有配置组合 
    for model_type, lr, hidden_size, batch_size, pooling_style in itertools.product(model_types, learning_rates, hidden_sizes, batch_sizes, pooling_styles): 
        Config["model_type"] = model_type 
        Config["learning_rate"] = lr 
        Config["in_channels"] = hidden_size 
        Config["out_channels"] = hidden_size 
        Config["hidden_size"] = hidden_size 
        Config["batch_size"] = batch_size 
        Config["pooling_style"] = pooling_style 
        final_acc = main(Config) 
        print("最后一轮准确率：", main(Config), "当前配置：", Config)
