# -*- coding: utf-8 -*-

from sklearn.multiclass import OutputCodeClassifier
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

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    train_data = load_data(config["train_data_path"],
                           config)  #返回的是一个DataLoader对象
    #加载模型
    model = TorchModel()

    #大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(r=8,
                                 lora_alpha=32,
                                 lora_dropout=0.1,
                                 target_modules=["query", "key", "value"])
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS",
                                          num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS",
                                         num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS",
                                         num_virtual_tokens=10)

    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())

    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule(
                "classifier").parameters():
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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            output = model(input_ids=input_ids,
                           labels=labels)  # 模型输出是 TokenClassifierOutput 对象
            # 提取损失值
            loss = output.loss

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model,
                            model_path)  #只保存模型中可训练的参数，适用于大模型微调场景，可减少保存文件的大小。
    # torch.save(model.state_dict(), model_path) #会保存模型的所有可学习参数，适用于保存完整的模型状态。
    # return evaluator.eval(epoch)
    # 文本分类任务：返回 evaluator.eval(epoch) 是为了直接获取模型的评估结果，方便对模型的性能进行评估和比较。
    return model, train_data
    # NER 任务：返回 model 和 train_data 是为了便于后续对模型进行进一步的操作和对训练数据进行分析。


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)


if __name__ == "__main__":
    model, train_data = main(Config)
