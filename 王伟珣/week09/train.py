# -*- coding: utf-8 -*-

import torch
import os
import logging
import numpy as np
from config import CONFIG
from dataloader import data_loader
from model import NerModel, choose_optimizer
from evaluate import Evaluator
from tqdm import tqdm


def train(config):
    logging.basicConfig(
        level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    train_data = data_loader(config, config['train_data_path'])

    model = NerModel(config)
    if torch.cuda.is_available():
        logger.info("Use GPU.")
        model = model.cuda()
    
    optimizer = choose_optimizer(config, model)

    evaluator = Evaluator(config, model, logger)

    n_epoches = config['n_epoches']
    for epoch in range(n_epoches):
        model.train()
        logger.info("Epoch %d/%d begin" % (epoch+1, config['n_epoches']))
        train_loss = []
        for _, batch_data in tqdm(enumerate(train_data), total=len(train_data)):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            inputs, labels = batch_data
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("Epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval()

    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    model_path = os.path.join(config['model_path'], "ner_epoch_%d.pth" % (epoch+1))
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    train(CONFIG)
