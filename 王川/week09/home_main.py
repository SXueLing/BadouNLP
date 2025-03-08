import logging
import torch
import numpy as np
from home_model import TorchModel, choose_optimizer
from home_loader import load_data
from home_config import Config
from home_evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger= logging.getLogger(__name__)

def main():
    train_data = load_data(Config["train_data_path"], Config)
    model = TorchModel(Config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    optim = choose_optimizer(Config, model)
    evaluator = Evaluator(Config, model, logger)
    for epoch in range(Config["epoch"]):
        model.train()
        train_loss = []
        logger.info("开始第%d轮Epoch" %(epoch +1))
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            inputs, labels = batch_data
            optim.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            if index % (len(train_data) // 2) == 0:
                logger.info("Batch Loss：%f" %(loss.item()))
        logger.info("Epoch Average Loss：%f" %(np.mean(train_loss)))
        acc = evaluator.eval(epoch + 1)

if __name__ == "__main__":
    main()
