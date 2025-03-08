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

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 模型保存路径
    if os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载测试数据
    train_data = load_data(config['train_data_path'], config, shuffle=True)
    # 模型
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 优化器
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        logger.info("epoch--%d, 开始训练" % epoch)
        model.train()
        train_loss = []
        for index, batch in enumerate(train_data):
            if cuda_flag:
                batch = [d.cuda() for d in batch]
            inputs, labels = batch
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.evaluate(epoch)
    return acc

if __name__ == '__main__':
    # main(Config)
    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        print("最后一轮准确率：", main(Config), "当前配置：", Config)
