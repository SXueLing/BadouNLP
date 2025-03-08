import time

from model import dic_model
import torch
import logging
from loader import load_data
from evaluate import Evaluator
import numpy as np
import nlp_util as nlpu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimizer(my_model, config):
    if config['optimizer'] == 'sgd':
        return torch.optim.SGD(my_model.parameters(), lr=config['learning_rate'])
    else:
        return torch.optim.Adam(my_model.parameters(), lr=config['learning_rate'])


def train(config):
    start_time = time.time()  # 开始计时
    nlpu.do_train_start(config)
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
    old_weight = my_model.classifier.weight.detach().clone()
    for epoch in range(epoch_num):
        my_model.train()
        epoch += 1
        watch_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [i.cuda() for i in batch_data if torch.is_tensor(i)]
            model_optimizer.zero_grad()
            loss = my_model(batch_data[0], batch_data[1], batch_data[2])
            watch_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(watch_loss))
        acc = evaluator.eval(epoch)
        new_weight = my_model.classifier.weight.detach().clone()
        weight_diff = torch.sum(torch.abs(old_weight - new_weight))
        print(f"Epoch {epoch}, weight diff = {weight_diff.item():.6f}")
        old_weight = new_weight
        model_name = config['model_path'] + nlpu.trans_model_name(config, acc, epoch)
        torch.save(my_model.state_dict(), model_name)
    return acc, (time.time() - start_time)


if __name__ == '__main__':
    from config import Config

    # 训练使用
    train(Config)
    # 本地玩玩
    # Config['model_type'] = 'bert_mid_layer'
    # Config['hidden_size'] = 768
    # Config['batch_size'] = 256
    # Config['pooling_style'] = 'avg'
    # Config['learning_rate'] = 0.0001
    # my_model = dic_model(Config)
    # my_model.load_state_dict(torch.load('./model/bert_mid_layer_768_256_avg_0.0001.bin'))
    # sen = '商家不把顾客留言放心上，强调不要辣椒，商家还是放了很了很多，弄的孩子无法吃，其他菜品味道不错，快递小哥很不错都送院里来了很棒！'
    # my_model.eval()
    # tokenize_encode = ''
    # if 'bert' not in Config['model_type']:
    #     dg = DataGenerator(Config['valid_data_path'], Config, False)
    #     tokenize_encode = dg.encode_sentence(sen)
    # else:
    #     tokenize = BertTokenizer.from_pretrained(Config['pretrain_model_path'], return_dict=True)
    #     tokenize_encode = tokenize.encode(sen, max_length=Config['max_length'], pad_to_max_length=True)
    # tokenize_encode = torch.LongTensor(tokenize_encode)
    # tokenize_encode = tokenize_encode.unsqueeze(0)
    # result = my_model(tokenize_encode)
    # print(result)
