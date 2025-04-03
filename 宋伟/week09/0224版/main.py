
import torch
import numpy as np
import logging
from config import Config
from loader import data_load
from model import TorchModel,choose_optimizer
from evaluate import Evaluator
import random
import os


logging.basicConfig(level=logging.INFO,format= '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = data_load(config["train_data_path"],config)

    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型到gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config,model)

    evaluate = Evaluator(config,model,logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch_{epoch}_begin")
        train_loss = []
        for index,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data[0]['input_ids'] = batch_data[0]['input_ids'].cuda()
                batch_data[0]['attention_mask'] = batch_data[0]['attention_mask'].cuda()
                batch_data[1] = batch_data[1].cuda()

            input_id,labels = batch_data
            loss = model(input_id,labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data)/2) == 0:
                logger.info(f"batch loss {loss}")
        logger.info(f"epoh average loss:{np.mean(train_loss)}")
        evaluate.eval(epoch)
    model_path = os.path.join(config["model_path"],f"epoch_{epoch}_bert_{Config['use_bert']}.pth")
    torch.save(model.state_dict(),model_path)
    return model,train_data

if __name__ == '__main__':
    model,train_data = main(Config)



