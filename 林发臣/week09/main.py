import time

from model import dic_model
import torch
import logging
from loader import load_data
from evaluate import Evaluator
import numpy as np
import train_util as train_util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimizer(my_model, config):
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(my_model.parameters(), lr=config['learning_rate'])
    else:
        return torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])


def train(config):
    start_time = time.time()  # 开始计时
    train_util.do_train_pre(config)
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
                batch_data = [i.cuda() for i in batch_data if torch.is_tensor(i)]
            model_optimizer.zero_grad()
            loss = my_model(batch_data[0], batch_data[1])
            watch_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(watch_loss))
        acc = evaluator.eval(epoch)
        if acc >= 0.5:
            model_name = train_util.trans_save_path(config, acc, epoch)
            torch.save(my_model.state_dict(), model_name)
    return acc, (time.time() - start_time)


if __name__ == '__main__':
    from config import Config

    train_util.mutil_train(Config, train, '第九周作业')
