import torch.nn as nn
import torch
import numpy
from create_dataset import *


class MyModel(nn.Module):
  def __init__(self, input_size, output_size) -> None:
    super().__init__()
    self.linear = nn.Linear(input_size, input_size)
    self.linear2 = nn.Linear(output_size, output_size)
    # self.activation = nn.Softmax(dim=-1)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x, y=None):
    x = self.linear(x)
    y_pred = self.linear2(x)
    if y == None:
      return y_pred
    else:
      return self.loss(y_pred, y)
    

def main():
  # 初始化变量
  batch_size = 20
  lr = 0.01

  # 初始化训练集 
  X,Y = build_dataset(5000)
  # 初始化模型
  model = MyModel(5, 5)
  # 绑定优化器
  optim = torch.optim.SGD(model.parameters(), lr=lr)

  for round in range(1000):
    model.train()
    round_loss = []
    for bi in range(len(X) // batch_size):
      start = bi * batch_size
      end = (bi + 1) * batch_size
      batch_x = X[start:end]
      batch_y = Y[start:end]
      # 计算损失
      loss = model.forward(batch_x, batch_y) 
      # 计算梯度
      loss.backward()
      # 更新权重
      optim.step()
      # 重置梯度
      optim.zero_grad()
      round_loss.append(loss.item())
    print("round", round,"loss", numpy.mean(round_loss))
  
  torch.save(model.state_dict(), "classification.bin")
      

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = MyModel(5, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("x", vec, "res",np.argmax(res)+1)


if __name__ == "__main__":
    # main()
    test_vec = [
       [0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
       [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
       [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
       [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894],
       [0.6,0.5,0.1,0.4,0.1],
    ]
    predict("classification.bin", test_vec)

