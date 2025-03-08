"""
@Project ：cgNLPproject 
@File    ：main_my.py
@Date    ：2025/1/6 15:48 
"""
import numpy

from config_my import Config
from loader_my import load_data
from model_my import MyTorchModule, choose_optimizer
from evaluate_my import evaluate
import logging
import random
import numpy as np
import torch
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def main(config):
    # 加载数据
    train_data = load_data(config, config["train_data_path"])
    # 加载模型
    model = MyTorchModule(config)
    # 加载优化器
    optim = choose_optimizer(config, model)
    # 加载预测模块
    eval = evaluate(config, model, logger)

    epoch_num = config['epoch_num']
    cost_times_ = []
    for i in range(epoch_num):
        start_time = time.time()
        model.train()
        losses = []
        epoch = i + 1
        for x, y in train_data:
            optim.zero_grad()
            loss = model(x, y)
            losses.append(loss.item())
            loss.backward()
            optim.step()
            if (len(losses)+1) % int(len(train_data) / 2) == 0:
                logger.info(f'第{epoch}轮的第{len(losses)}个batch的loss：{loss}')
        end_time = time.time()
        cost_time = end_time - start_time
        cost_times_.append(cost_time)
        logger.info(f'第{epoch}轮的平均loss：{numpy.mean(losses)}')
        logger.info(f'第{epoch}轮的训练耗时：{cost_time}s')
        acc_ = eval.eval(epoch)
    return acc_, numpy.mean(cost_times_)

def export_excel(config_, acc_, cost_time_):
    columns = {'模型类型':[],'隐藏单元数':[],'每个batch的样本数量':[],'学习率':[],'池化方式':[],'准确率':[],'平均训练耗时':[]}
    datas = pd.DataFrame(columns)
    for accu, cost_time, config in zip(acc_, cost_time_, config_):
        data = {'模型类型':[config['model_type']],'隐藏单元数':[config['hidden_size']],'每个batch的样本数量':[config['batch_size']],'学习率':[config['lr']],'池化方式':[config['pooling_style']],'准确率':[accu],'平均训练耗时':[cost_time]}
        datas_pd = pd.DataFrame(data)
        datas = pd.concat([datas, datas_pd],ignore_index=True)
    datas.to_excel('output.xlsx')


if __name__ == '__main__':
    accs = []
    cost_times = []
    cons = []
    for model_type in ['bert','lstm','bert_lstm','bert_cnn']:
        Config["model_type"] = model_type
        for lr in [1e-3, 1e-4]:
            Config["lr"] = lr
            for batch_size in [64, 128]:
                Config["batch_size"] = batch_size
                for pooling_style in ["avg", 'max']:
                    Config["pooling_style"] = pooling_style
                    acc, mean_cost_time = main(Config)
                    cons.append({"model_type":model_type,"lr":lr,"batch_size":batch_size,"hidden_size":Config['hidden_size'],"pooling_style":Config['pooling_style']})
                    accs.append(acc)
                    cost_times.append(mean_cost_time)
    export_excel(cons,accs,cost_times)