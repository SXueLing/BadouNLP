
import numpy as np
import logging

import torch

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    train_datas = load_dataset(config, 'train')
    # model = SimNetwork(config)
    # optim = choose_optim(config, model)
    model = TripletNetwork(config)
    optim = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    epoch_num = config['epoch_num']
    for epoch in range(epoch_num):
        model.train()
        losses = []
        for train_data in train_datas:
            optim.zero_grad()
            qus_1, qus_2, label = train_data
            loss = model(qus_1, qus_2, label.squeeze())
            losses.append(loss.item())
            loss.backward()
            optim.step()
        logger.info(f"epoch average loss: {np.mean(losses)}")
        evaluator.eval(epoch)



if __name__ == "__main__":
    from loader_triplet import load_dataset
    from model_triplet import TripletNetwork, choose_optimizer
    from config import Config
    from eval import Evaluator
    main(Config)
