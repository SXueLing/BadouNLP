# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import pandas as pd
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_config(config):
    required_keys = ["model_type", "learning_rate", "hidden_size", "batch_size", "pooling_style", "epoch", "model_path"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")


def main(config):
    validate_config(config)

    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("Using GPU for training")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        model.train()
        logger.info(f"Epoch {epoch + 1}/{config['epoch']} starting...")
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() if isinstance(d, torch.Tensor) else d for d in batch_data]

            optimizer.zero_grad()

            try:
                if len(batch_data) == 3:  # BERT models
                    input_ids, attention_mask, labels = batch_data
                elif len(batch_data) == 2:  # Non-BERT models
                    input_ids, labels = batch_data
                    attention_mask = None  # Set attention_mask to None for non-BERT models
                else:
                    raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
            except ValueError as e:
                logger.error(f"Error in batch data format: {batch_data}. Error: {e}")
                continue

            loss = model(input_ids, attention_mask, labels)
            if not isinstance(loss, torch.Tensor):
                logger.error(f"Invalid loss value: {loss}")
                continue

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if index % max(1, int(len(train_data) / 10)) == 0:
                logger.info(f"Batch {index}/{len(train_data)} loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch + 1} average loss: {np.mean(train_loss):.4f}")
        acc = evaluator.eval(epoch)

        model_path = os.path.join(config["model_path"], f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)

    return acc


if __name__ == "__main__":
    results_df = pd.DataFrame(
        columns=["model_type", "learning_rate", "hidden_size", "batch_size", "pooling_style", "acc"])
    iteration = 0
    max_iterations = 12  # Set a higher threshold for maximum iterations
    save_interval = 5
    results_path = "results.xlsx"
    best_acc = 0

    for model_type in ["gated_cnn", "fast_text", "lstm"]:
        Config["model_type"] = model_type
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 256]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style

                        logger.info(f"Starting training with config: {Config}")
                        try:
                            acc = main(Config)
                        except Exception as e:
                            logger.error(f"Error during training: {e}")
                            acc = 0

                        results_df = results_df._append({
                            "model_type": model_type,
                            "learning_rate": lr,
                            "hidden_size": hidden_size,
                            "batch_size": batch_size,
                            "pooling_style": pooling_style,
                            "acc": acc
                        }, ignore_index=True)

                        if acc > best_acc:
                            best_acc = acc
                            logger.info(f"New best accuracy: {best_acc:.4f} for model {model_type}")

                        iteration += 1
                        if iteration >= max_iterations:
                            logger.info("Reached max_iterations. Stopping further training.")
                            break

                        if iteration % save_interval == 0:
                            results_df.to_excel(results_path, index=False)
                            logger.info(f"Intermediate results saved to {results_path}")

                    if iteration >= max_iterations:
                        break
                if iteration >= max_iterations:
                    break
            if iteration >= max_iterations:
                break

    results_df.to_excel(results_path, index=False)
    logger.info(f"Final results saved to {results_path}")