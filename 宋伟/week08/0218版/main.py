'''
模型训练的主要步骤

[description]
'''

import torch
import os
from config import Config
from loader import load_data
from model import SiameseNetwork, choose_optimizer
import logging
import numpy as np
import random
from eavluate import Evaluate

# 训练记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 随机种子,保证数据的可复现性
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config['model_path']):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"],config)
    print(config['vocab_size'])
    # 加载模型
    model = SiameseNetwork(config)
    # 标识是否调用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以调用，将模型迁居到gpu中")
        model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config,model)
    # 加载测试
    evaluator = Evaluate(config,model,logger)
    # 进行训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch:{epoch} begin")
        train_loss = [] # 记录每个epoch中，每个batch 中的平均损失
        for index,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                # 方便后面进行解包
                batch_data = [data.cuda() for data in batch_data] 
            input_id1,input_id2, label_or_input_id3 = batch_data
            loss = model(input_id1,input_id2,label_or_input_id3)
            train_loss.append(loss.item()) # 将每个批次损失存储
            # 每半个批次输出,0,中间，最后
            if index % int(len(train_data)/2)==0:
                logger.info(f"bath loss {loss}")
            loss.backward()
            optimizer.step()
        logger.info(f"epoch average loss :{np.mean(train_loss)}")
        evaluator.eval(epoch)
        print(f'{"separation":=^50}')
        
    # 创建文件
    model_path = os.path.join(config["model_path"],f"epoch_{config['loss']}_{config['epoch']}.pth")
    # 储存到文件内
    torch.save(model.state_dict(),model_path)
    return 

if __name__ == '__main__':
    main(Config)







