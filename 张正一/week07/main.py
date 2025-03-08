import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    # 保存模型的目录
    if not os.path.exists(config['model_path']):
        os.makedirs(config['model_path'])
        
    print(27, torch.cuda.is_available())
    # 训练数据
    train_data = load_data(config['train_data_path'], config)
    
    print(train_data)
    
    model = TorchModel(config)
    
    #加载优化器
    optimizer = choose_optimizer(config, model)
    # #加载效果测试类
    # evaluator = Evaluator(config, model, logger)
    print(39, len(train_data))
     #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"{epoch} begin")
        train_loss = []
        for index, batch_data in enumerate(train_data):

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info(f"batch loss {loss}")
        logger.info(f"epoch average loss: {np.mean(train_loss)}")
        # acc = evaluator.eval(epoch)
        # return acc
    torch.save(model.state_dict(), config['model_path'] + '/gru_model.pth')  #保存模型权重
    # torch.save(model.state_dict(), config['model_path'] + '/lstm_model.pth')  #保存模型权重
    # torch.save(model.state_dict(), config['model_path'] + '/rnn_model.pth')  #保存模型权重

if __name__ == '__main__':
    main(Config)