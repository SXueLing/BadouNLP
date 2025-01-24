import os.path
import time

from model import dic_model
import torch
import logging
from loader import load_data
from loader import install_data
from evaluate import Evaluator
import numpy as np
import csv
import importlib

config_module = "Config"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    # 重新加载模块
    module = importlib.import_module(config_module)
    importlib.reload(module)
    return module.Config  # 假设 config.py 中有一个字典变量 config


def get_optimizer(my_model, config):
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(my_model.parameters(), lr=config['learning_rate'])
    else:
        return torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])


def train(config):
    start_time = time.time()  # 开始计时
    model_name = config['model_type'] + '_' + str(config['hidden_size']) + '_' + str(config['batch_size']) + '_' + \
                 config['pooling_style'] + '_' + str(config['learning_rate']).replace('\\.', '') + '.bin'
    model_name = config['model_path'] + model_name
    my_model = dic_model(config)
    model_optimizer = get_optimizer(my_model, config)
    # 拿到训练数据
    train_data = load_data(config["train_data_path"], config)
    evaluator = Evaluator(config, my_model, logger)
    epoch_num = config['epoch']
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        my_model = my_model.cuda()
    acc = ''
    for epoch in range(epoch_num):
        my_model.train()
        epoch += 1
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [i.cuda() for i in batch_data]
            model_optimizer.zero_grad()
            loss = my_model(batch_data[0], batch_data[1])
            watch_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(watch_loss))
        acc = evaluator.eval(epoch)
    torch.save(my_model.state_dict(), model_name)
    return acc, (time.time() - start_time)


if __name__ == '__main__':
    from config import Config
    from collections import defaultdict

    # 初始化处理数据
    install_data(Config["train_data_all_path"], Config)
    with open(Config['out_csv_path'], 'w', encoding='utf-8') as file:
        fieldnames = ["Model", "Learning_Rate", "Hidden_Size", "Batch_Size", "Pooling_Style", "Acc", "Time_Cos",
                      "Epoch"]
        # 创建 CSV 写入器
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        config_path = Config['config_path']
        for model_type in Config['model_type_list']:
            data = defaultdict(str)
            Config['model_type'] = model_type
            data['Model'] = model_type
            for batch_size in Config['batch_size_list']:
                Config['batch_size'] = batch_size
                hidden_size_list = Config['hidden_size_list']
                data['Batch_Size'] = batch_size
                if 'bert' in model_type:
                    hidden_size_list = [768]
                for hidden_size in hidden_size_list:
                    Config['hidden_size'] = hidden_size
                    data['Hidden_Size'] = hidden_size
                    for pooling_style in Config['pooling_style_list']:
                        Config['pooling_style'] = pooling_style
                        data['Pooling_Style'] = pooling_style
                        for learning_rate in Config['learning_rate_list']:
                            Config['learning_rate'] = learning_rate
                            data['Learning_Rate'] = learning_rate
                            logger.info('模型：%s，学习率：%s，每批样本数量：%s，隐藏层：%s，池化方式：%s' % (
                                Config['model_type'], Config['learning_rate'], Config['batch_size'],
                                Config['hidden_size'],
                                Config['pooling_style']
                            ))
                            Acc, Cost_Time = train(Config)
                            data['Acc'] = Acc
                            data['Time_Cos'] = str(round(Cost_Time, 2))
                            data['Epoch'] = Config['epoch']
                            writer.writerow(data)
        file.close()

    # train(Config)
