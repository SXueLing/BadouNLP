import torch 
import os
import random
import numpy as np
import pandas as pd
import logging
from week06_config import config
from week06_load import load_data
from week06_split import split_file
from week06_model import TorchModel, choose_optimizer
from week06_evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    data_path = config["data_path"]
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch"]):
        epoch +=1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            optimizer.zero_grad()
            reviews, labels = batch_data
            loss = model(reviews, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(batch_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    return acc

if __name__ == "__main__":
    main(config)
    #for model in ["fast_text", "lstm", "bert"]:
    models = {}
    models_train_result = {}
    for model in ["gated_cnn", 'bert', 'lstm']:
        config["model_type"] = model
        lr = config["learning_rate"] 
        if model == "bert":
            config["learning_rate"] = 1e-5
        for hidden_size in [128]:
            config["hidden_size"] = hidden_size
            for batch_size in [64]:
                config["batch_size"] = batch_size
                acc = main(config)
                models["model_type"].append(model)
                models["learning_rate"].append(lr)
                models["hidden_size"].append(hidden_size)
                models["acc"].append(acc)
    df_models = pd.DataFrame(models)
    df_models.to_excel("compare.xlsx")
    print(models)


