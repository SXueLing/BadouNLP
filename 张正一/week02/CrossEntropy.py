import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
  def __init__(self, inputSize, num_Class):
    super(TorchModel, self).__init__()
    # 定义线性层
    self.linaer = nn.Linear(inputSize, num_Class)
    # 定义交叉熵
    self.createEntropyLoss = nn.CrossEntropyLoss()

  def forward(self, X, y_true = None):
    y = self.linaer(X)
    if y_true is not None:
      # 计算损失
      return self.createEntropyLoss(y, y_true)
    else:
      return y
    

# 创建单个样本
def build_sample():
  x = np.random.random(5)
  # 最大值索引就是该类
  y = np.argmax(x)
  return x, y

# 创建数据集
def build_dataset(sample_num):
  X = []
  Y = []
  for i in range(sample_num):
    x, y = build_sample()
    X.append(x)
    Y.append(y)
  return torch.FloatTensor(X), torch.LongTensor(Y)
    

def main():
  # 训练轮数
  epoch_num = 20
  # 每次训练样本个数
  batch_size = 20
  # 训练样本数量
  train_sample_num = 5000
  # 输入向量维度
  input_size = 5
  # 学习率
  learning_rate = 0.001
  # 创建模型
  model = TorchModel(input_size, 5)
  # 选择优化器
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
  # 生成训练集
  train_x, train_y = build_dataset(train_sample_num)
  print(59, train_x, train_y)
  for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    for batch_index in range(train_sample_num // batch_size):
      x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
      y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
      loss = model(x, y) #计算loss
      loss.backward() # 计算梯度
      optim.step() #更新权重
      optim.zero_grad() #梯度置为0
      watch_loss.append(loss.item())
    
    print(f'第{epoch+1}轮loss：{np.mean(watch_loss)}')

  # 随机生成5个样本进行测试
  test_data_x, test_data_y = build_dataset(5)
  with torch.no_grad():  # 不计算梯度
    result = model.forward(torch.FloatTensor(test_data_x))  # 模型预测
    _, predict = torch.max(result, 1)
  for test_x, pred, test_y in zip(test_data_x, predict, test_data_y):
    print(f'测试输入：{test_x}, 测试输出：{pred}, {test_y}')

if __name__ == '__main__':
  main()

