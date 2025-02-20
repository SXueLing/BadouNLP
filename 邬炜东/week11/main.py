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
        los_epoch.append(np.mean(los))
        print("loss:", np.mean(los))
        print(eval.evaluator("""
        新华网喀布尔7月18日电 驻阿富汗美军18日说，美军一架F－15E战斗机当天在阿富汗东部坠毁，造成两名机组人员死亡。美军发表声明说，这架飞机当天凌晨在阿富汗东部执行作战任务时坠毁，但“并非被地面炮火击落”。美军说，尾随的另一架战机“并未发现地面上有任何火力”。美军将对战机坠毁原因展开调查。本月14日，北约一架运送军用物资的直升机在阿富汗南部赫尔曼德省被地面炮火击落，塔利班武装宣称对这一事件负责。
        """))
    plt.plot(range(len(los_epoch)), los_epoch, label="loss")
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
