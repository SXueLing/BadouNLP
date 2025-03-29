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
from peft import get_peft_model, LoraConfig, TaskType


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def peft_wrapper(model):
    peft_wrapper = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query","value"]
    )
    return get_peft_model(model, peft_wrapper)

def main(config):
    # 创建保存模型的目录
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = TorchModel(config)
    model = peft_wrapper(model)
    if torch.cuda.is_available():
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 训练模型
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("开始训练第%d轮模型" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_data = [x.cuda() for x in batch_data]
            input_id, labels = batch_data
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)

    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
