import torch
from loader import loader
from evaluate import Evaluation
from model import All_Models, choose_optim
import random
from config import Config
import numpy as np
import matplotlib.pyplot as plt

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
    micro_precision_epoch = []
    micro_recall_epoch = []
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
        micro_precision, micro_recall = eval.evaluator()
        micro_precision_epoch.append(micro_precision)
        micro_recall_epoch.append(micro_recall)
        los_epoch.append(np.mean(los))
        print(f"第{epoch + 1}个epoch的loss为{np.mean(los)}")
    plt.plot(range(len(los_epoch)), los_epoch, label="loss")
    plt.plot(range(len(micro_precision_epoch)), micro_precision_epoch, label="micro_precision")
    plt.plot(range(len(micro_recall_epoch)), micro_recall_epoch, label="micro_recall")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), config["model_path"])


# 运行整个框架
if __name__ == "__main__":
    main(Config)
    # for lr in [1e-2, 1e-3, 1e-4]:
    #     Config["learning_rate"] = lr
    #     for hs in [768]:
    #         Config["hidden_size"] = hs
    #         for nl in [1]:
    #             Config["num_layers"] = nl
    #             for bs in [16, 256]:
    #                 Config["batch_size"] = bs
    #                 for model in ["bert"]:
    #                     Config["model_type"] = model
    #                     result = {}
    #                     result["learning_rate"] = lr
    #                     result["hidden_size"] = hs
    #                     result["num_layers"] = nl
    #                     result["batch_size"] = bs
    #                     result["model"] = model
    #                     print("------------------Trying:--------------------")
    #                     print(result)
    #                     acc, time_for_100 = main(Config)
    #                     result["acc"] = acc
    #                     result["time_for_100"] = time_for_100
    #                     append_dict_to_csv(result, "result.csv")
