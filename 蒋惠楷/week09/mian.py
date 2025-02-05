# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, BertModelWithCRF, BERTLSTMCRF,choose_optimizer
from evaluate import Evaluator
from loader import load_data
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def train_single_model(config):
    """ 单模型训练函数（原main函数内容） """
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    train_data = load_data(config["train_data_path"], config)
    
    # 模型选择
    if config["model_type"] == "Bert-CRF":
        model = BertModelWithCRF(config)
    elif config["model_type"] == "LSTM-CRF":
        model = TorchModel(config)
    elif config["model_type"] == "Bert-LSTM-CRF":
        model = BERTLSTMCRF(config)
    else:
        raise ValueError("不支持的模型类型")

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"正在训练模型 {config['model_type']}，epoch {epoch} 开始")
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            if config["attention_mask"] and config["model_type"] != "LSTM-CRF":
                input_ids, attention_mask, labels = batch_data
                loss = model(input_ids, attention_mask, labels)
            else:
                input_id, labels = batch_data
                loss = model(input_id, labels)
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info(f"{config['model_type']} 模型 batch loss: {loss:.4f}")
        logger.info(f"{config['model_type']} 模型 epoch 平均损失: {np.mean(train_loss):.4f}")
        evaluator.eval(epoch)
        scheduler.step()

    model_path = os.path.join(config["model_path"], f"Epoch_{config['epoch']}_{config['model_type']}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"{config['model_type']} 模型已保存至 {model_path}")

def main(config):
    """ 主控制函数 """
    if isinstance(config["model_type"], list):  # 多模型训练模式
        for model_type in config["model_type"]:
            current_config = config.copy()
            current_config["model_type"] = model_type
            train_single_model(current_config)
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:  # 单模型训练模式（保持兼容）
        train_single_model(config)

if __name__ == "__main__":
    main(Config)
