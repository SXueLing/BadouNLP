import torch
import numpy as np
import random
from config import Config
from model import clsposModel, choose_optimizer
from loader import load_data
from evalate import Evaluate

seed = Config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def main(config):
    data_train, data_val = load_data(config)
    
    model = clsposModel(config)
    optimizer = choose_optimizer(Config, model)

    evaluator = Evaluate(config, model, data_val)

    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        train_loss = []
        for index, batch_data in enumerate(data_train):
            optimizer.zero_grad()
            x, y = batch_data
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        acc = evaluator.eval()
        # print(f"Epoch:{epoch} , Loss:{np.mean(train_loss)} , Acc:{acc}")
    return acc

if __name__ =='__main__':
    # main(Config)

    for model in ['lstm', 'cnn', 'bert']:
        Config['model_type'] = model
        for lr in [1e-3, 1e-4]:
            Config['learning_rate'] = lr
            for batch_size in [64, 128]:
                Config['batch_size'] = batch_size
                for pooling_style in ['avg', 'max']:
                    Config['pooling_style'] = pooling_style
                    for optimizer in ['sgd', 'adam']:
                        Config['optimizer'] = optimizer
                        print(f"最后一轮准确率：{main(Config)}，当前配置：{Config}")
