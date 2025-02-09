# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Model training main program
"""

def main(config):
    # Create a directory to save the model
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # Load training data
    train_data = load_data(config["train_data_path"], config)

    # Load the model
    model = SiameseNetwork(config)

    # Check if GPU is available
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU is available, migrating the model to GPU")
        model = model.cuda()

    # Load the optimizer
    optimizer = choose_optimizer(config, model)

    # Load the evaluator
    evaluator = Evaluator(config, model, logger)

    # Training loop
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch} begin")

        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            # Move data to GPU if available
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            # Unpack the triplet data
            anchor, positive, negative = batch_data

            # Compute the loss
            loss = model(anchor, positive, negative)
            train_loss.append(loss.item())

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Log batch loss (optional)
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info(f"Batch loss: {loss.item()}")

        # Log average loss for the epoch
        logger.info(f"Epoch {epoch} average loss: {np.mean(train_loss)}")

        # Evaluate the model
        evaluator.eval(epoch)

    # Save the final model
    model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    return


if __name__ == "__main__":
    main(Config)