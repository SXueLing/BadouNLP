import os
from loader import load_data
from model import SiameseNetwork,choose_optimizer
import logging
import torch
from evaluate import Evaluator
import numpy as np
from config import Config


logging.basicConfig(level = logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"],config)
    model = SiameseNetwork(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("cuda is available")
        model = model.cuda()
    optimizer = choose_optimizer(config,model)
    evaluator = Evaluator(config,model,logger)
    for epoch in range(config["epoch"]):
        epoch +=1
        model.train()
        logger.info("epoch{}begin".format(epoch))
        train_loss =[]
        for index,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1,input_id2,input_id3 = batch_data
            loss = model(input_id1,input_id2,input_id3)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss:{:.8f}".format(np.mean(train_loss)))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"],"epoch{}.pth".format(epoch))
    torch.save(model.state_dict(),model_path)
    return

if __name__ == "__main__":
    main(Config)

