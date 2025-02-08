import torch
import torch.nn as nn
import numpy as np

from build_dataset import build_dataset

class Model(nn.Module):
    def __init__(self, in_features=5):
        super(Model, self).__init__()

        # 线性层
        self.hidden = nn.Linear(in_features=in_features,out_features=5)
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward_pred(self, x):
        y_pred = self.hidden(x)
        return y_pred

    def forward_calc_loss(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


def main():
    epoch_num = 1000
    batch_size = 20
    learning_rate = 0.001

    model = Model()
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_X, train_Y = build_dataset(10000)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(len(train_X) // batch_size):
            x = train_X[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_Y[batch_index * batch_size: (batch_index + 1) * batch_size]

            y_pred = model.forward_pred(x)

            loss = model.forward_calc_loss(y_pred, y)

            loss.backward()  # 计算梯度
            optim.step()  # 更新权
            optim.zero_grad()  # 梯度归零

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        if loss < 0.00001:
            break
    # 保存模型
    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()





