import numpy as np
from config import Config
from corpus import data
from model import TorchModel, optim_type_get
from evaluate import epoch_eval
import torch

def train_model():
    # 获取数据对象
    train_data_path = Config["train_data_path"]
    train_data = data(train_data_path)
    train_data.load()
    Config["vocab_size"] = train_data.vocab_size()
    Config["class_num"] = train_data.label_num()
    test_data_path = Config["test_data_path"]
    test_data = data(test_data_path)
    test_data.load()
    epoch_train_num = Config["epoch_train_num"]
    batch_size = Config["batch_size"]
    # 获取模型
    epoch = Config["epoch"]
    model = TorchModel(Config)
    optimizer = optim_type_get(model, Config)
    log = [] # 记录每轮训练效果
    # 训练过程
    for i in range(epoch):
        model.train()
        loss_item = []
        train_x, train_y = train_data.get_data(epoch_train_num)
        for batch in range(int(epoch_train_num / batch_size)):
            if Config["model_type"] == "bert":
                # BERT 输入需要是字典
                x = {
                    "input_ids": train_x["input_ids"][batch * batch_size: (batch + 1) * batch_size],
                    "attention_mask": train_x["attention_mask"][batch * batch_size: (batch + 1) * batch_size]
                }
            else:
                x = train_x[batch * batch_size: (batch + 1) * batch_size]
            y = train_y[batch * batch_size: (batch + 1) * batch_size]
            loss = model(x, y) # 计算loss
            loss.backward() # 计算梯度
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 清空梯度
            loss_item.append(loss.item())
        print("第%d轮的loss为：%f" % (i+1, np.mean(loss_item)))
        log.append(np.mean(loss_item))
        epoch_eval(model, test_data)
    # 保存模型
    path = Config["model_path"]
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    train_model()
