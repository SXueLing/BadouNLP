import torch
import torch.nn as nn
from loader import load_data
from config import Config
from model import TorchModel
import numpy as np
from evaluate import Evaluator
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    train_data = load_data(Config, Config['data_path'])
    model = TorchModel(Config)
     # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        print("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    model.train()
    # 加载优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'])
    # 加载模型测试
    evaluate = Evaluator(Config, model, logger)
    for epoch in range(Config['epochs']):
        watch_loss = []
        for index, data in enumerate(train_data):
            if cuda_flag:
                data = [d.cuda() for d in data]
            inputs, labels = data
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f"Epoch {epoch+1} loss: {np.mean(watch_loss)}")
        evaluate.eval(epoch+1)
    
if __name__ == '__main__':
    main()