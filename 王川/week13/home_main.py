import logging
import torch
import numpy as np
import os
import time
import random
from home_loader import load_data
from home_model import TorchModel, choose_optimizer
from home_evaluate import Evaluator
from home_config import Config
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    train_data = load_data(Config["train_data_path"], Config)
    model = TorchModel(Config)

    tuning_tactics = Config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],  # 精确匹配模块路径
            modules_to_save=["linear", "lstm"]  # 确保自定义层可训练
        )
    model = get_peft_model(model, peft_config)
    # print("Trainable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，将模型迁移到GPU上训练")
        model = model.cuda()

    optim = choose_optimizer(Config, model)
    evaluator = Evaluator(Config, model, logger)
    start_time = time.time()
    for epoch in range(Config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []
        logger.info("开始%d轮模型训练" % epoch)
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            inputs, labels = batch_data
            optim.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            if index % (len(train_data)//2) == 0:
                logger.info("batch loss: %f" % loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    end_time = time.time()
    logger.info("训练时间为：%fs" % (end_time - start_time))
    # if not os.path.isdir(Config["model_path"]):
    #     os.mkdir(Config["model_path"])
    # model_path = os.path.join(Config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
