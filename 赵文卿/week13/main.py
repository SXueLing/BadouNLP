# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig 

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子 
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)

    # 选择微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        # 设置Lora的配置
        peft_config = LoraConfig(
            r = 8,
            lora_alpha = 32,
            lora_dropout = 0.1,
            target_modules = ["query", "key", "value"]
            )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type = "SEQ_CLS",num_virtual_tokens = 10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type = "SEQ_CLS",num_virtual_tokens = 10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    
    model = get_peft_model(model, peft_config)

    if tuning_tactics == "lora_tuning":
        # 设置需要微调的模块 
        for param in model.get_submodule("model").get_submodule("classify").parameters():
            param.requires_grad = True


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
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)
    return acc

def save_tunable_parameters(model, model_path):
    saved_params = {
        k: v.to("cpu")
        for k,v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, model_path)

if __name__ == "__main__":
    main(Config)
