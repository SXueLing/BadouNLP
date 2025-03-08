import torch
from loader import loader
from evaluate import Evaluation
from model import All_Models, choose_optim
import random
from config import Config
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os

# 随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    train_data = loader(config["train_data_path"], config)
    model = All_Models(config)
    optim = choose_optim(model, config)
    eval = Evaluation(config, model)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    acc_epoch = []
    los_epoch = []
    for epoch in range(config["epochs"]):
        model.train()
        los = []
        print(f"===========================epoch:{epoch + 1}========================")
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            train_x, train_y = batch_data
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            los.append(loss.item())
        acc = eval.evaluator()
        acc_epoch.append(acc)
        los_epoch.append(np.mean(los))
        print(f"第{epoch + 1}个epoch的loss为{np.mean(los)}")
    plt.plot(range(len(acc_epoch)), acc_epoch, label="acc")
    plt.plot(range(len(los_epoch)), los_epoch, label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), config["model_path"])
    time_for_100 = get_predict_time(model, config)
    return acc_epoch[-1], time_for_100


# 获取预测100个样本的时间
def get_predict_time(model, config):
    data = loader(config["eval_data_path"], config, ispredict=True)
    model.eval()
    with torch.no_grad():
        times = 0
        for index, eval_data in enumerate(data):
            if torch.cuda.is_available():
                eval_data = [d.cuda() for d in eval_data]
            x, _ = eval_data
            start = time.time()
            _ = model(x)
            end = time.time()
            times += end - start
            if index >= 50:
                break
    return times


# 把结果做成csv
def append_dict_to_csv(data_dict, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()
        writer.writerow(data_dict)


# 运行整个框架
if __name__ == "__main__":
    # main(Config)
    for lr in [1e-2, 1e-3, 1e-4]:
        Config["learning_rate"] = lr
        for hs in [768]:
            Config["hidden_size"] = hs
            for nl in [1]:
                Config["num_layers"] = nl
                for bs in [16, 256]:
                    Config["batch_size"] = bs
                    for model in ["bert"]:
                        Config["model_type"] = model
                        result = {}
                        result["learning_rate"] = lr
                        result["hidden_size"] = hs
                        result["num_layers"] = nl
                        result["batch_size"] = bs
                        result["model"] = model
                        print("------------------Trying:--------------------")
                        print(result)
                        acc, time_for_100 = main(Config)
                        result["acc"] = acc
                        result["time_for_100"] = time_for_100
                        append_dict_to_csv(result, "result.csv")
