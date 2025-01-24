# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
from config import CONFIG
from dataloader import load_data
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator


def train(config):
    logger = get_logger()

    train_data = load_data(config, mode='train')

    model = SiameseNetwork(config)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config['n_epoches']):
        model.train()
        logger.info("epoch %d begin" % (epoch+1))
        train_loss = []
        for _, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            a, p, n = batch_data
            loss = model(a, p, n)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch+1)

    if not os.path.exists(config['model_path']):
        os.mkdir(config['model_path'])
    model_path = os.path.join(config['model_path'], "epoch_%d.pth" % (epoch+1))
    torch.save(model.state_dict(), model_path)
    return


def get_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)
    

if __name__ == "__main__":
    train(CONFIG)
