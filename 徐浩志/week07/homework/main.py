from model import TorchModel
from data_builder import Dataset
from torch.utils.data import DataLoader
from eval import eval
import torch
import numpy as np
# from config import config

def main():
    model = TorchModel()
    data = Dataset(0.9)
    data.change_data('train')
    train_data = DataLoader(data, 500, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    for each in range(20):
        model.train()
        data.change_data('train')
        loss_list = []
        # if acc >0.8:
        #     opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        # else:
        #     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # 开始每轮训练
        for x, y in train_data:
            loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_list.append(loss.detach().numpy())
        print(np.mean(loss_list))

        # 测试结果
        data.change_data('test')
        test_data = DataLoader(data, 1)
        eval(model, test_data)
if __name__ == '__main__':
    main()