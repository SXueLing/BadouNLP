import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 构建模型结构
class Mymodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Mymodel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.BN1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 构建训练数据
train_x = torch.randn(100, 5)
train_y = train_x.argmax(dim=-1)

# 构建训练参数
epochs = 100
batch_size = 8
learning_rate = 0.001

# 构建模型
input_dim = 5
hidden_dim = 128
output_dim = 5
cross_entropy_loss = nn.CrossEntropyLoss()
model = Mymodel(input_dim, hidden_dim, output_dim)
optim = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
loss_epoch = []
for epoch in range(epochs):
    loss_sum = 0
    for i in range(0, train_x.shape[0], batch_size):
        x_batch = train_x[i:i + batch_size]
        y_batch = train_y[i:i + batch_size]
        pred = model(x_batch)
        loss = cross_entropy_loss(pred, y_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_sum += loss.item() * batch_size
    loss_epoch.append(loss_sum / train_x.shape[0])
    print('第{}次迭代,损失为{}'.format(epoch+1, loss_sum / train_x.shape[0]))

# 画图
plt.plot(loss_epoch)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.savefig('loss_plot.png')

# 预测
X = torch.randn(10, 5)
Y = X.argmax(dim=-1)
pred = model(X)
result = pred.argmax(dim=-1)
acc = result.eq(Y).sum().item() / X.shape[0]

print('预测结果为:{}'.format(result))
print("该模型的准确率为:{}".format(acc))