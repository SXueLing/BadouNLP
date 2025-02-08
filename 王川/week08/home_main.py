import logging
import torch
import numpy as np
from home_config import Config
from home_model import SiameseNetwork, choose_optimizer
from home_loader import load_data
from home_evaluate import Evaluator

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    train_data = load_data(Config["train_data_path"], Config)
    model = SiameseNetwork(Config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，已将模型迁移到GPU上训练")
        model = model.cuda()
    optim = choose_optimizer(Config, model)
    evaluator = Evaluator(Config, model, logger)
    for epoch in range(Config["epoch"]):
        train_loss = []
        model.train()
        logger.info("开始第%d轮epoch" %(epoch + 1))
        for idx, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            a, p, n = batch_data
            optim.zero_grad()
            loss = model(a, p, n)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            if idx % (len(train_data) // 2) == 0:
                logger.info("batch loss：%f" %(loss.item()))
        logger.info("epoch average loss：%f" %(np.mean(train_loss)))
        acc =evaluator.eval(epoch + 1)

if __name__ == "__main__":
    main()
