import logging
import numpy as np
import random
import torch
import os
from test_config import Config
from test_loader import load_data
from test_model import TorchModel, choose_optimizer
from test_evaluate import Evaluator

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main():
    if not os.path.isdir(Config["model_path"]):
        os.mkdir(Config["model_path"])
    train_data = load_data(Config["train_data_path"], Config) #只能放在model前面,不然就会报错，因为load_data涉及到了self.config["vocab_size] = len(self.vocab)
    model = TorchModel(Config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，将模型迁移到GPU上训练")
        model = model.cuda()
    evaluator = Evaluator(Config, model, logger)
    optim = choose_optimizer(Config, model)
    for epoch in range(Config["epoch"]):
        logger.info("第%d轮Epoch开始" %(epoch + 1))
        train_loss = []
        model.train()
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input, target = batch_data
            optim.zero_grad()
            loss = model(input, target)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            if index % (len(train_data)//2) == 0:
                logger.info("batch loss：%f" %(loss.item()))
        logger.info("epoch average loss：%f" %(np.mean(train_loss)))
        acc = evaluator.eval(epoch + 1)

if __name__ == "__main__":
    main()
