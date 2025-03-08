
import torch.nn as nn
import torch
import torch.optim as optim

## 数据准备
#生成五个个随机的五维向量
x=[torch.randn(5) for i in range(5)]
#获取真实的五维向量中最大的值的索引
y=[ torch.argmax(i) for i in x]
# 将列表维度降维为5*5的矩阵
x=torch.stack(x)
# 将预测值 降维 为1*5的矩阵
y=torch.stack(y)
print(x)
print(y)




## 创建模型

class MyTorchModel(nn.Module):
    # input_size : 代表输入的样本数量
    # middle_size : 代表中间的隐藏网络层维度大小
    # output_size : 代表输出的结果约束到几维中
    def __init__(self,input_size,middle_size,output_size):
        ## 初始化构造方法
        super(MyTorchModel, self).__init__()
        # 代表创建 样本数量*隐藏维度的 网络模型
        self.layer1=nn.Linear(input_size,middle_size)
        # 代表创建 隐藏维度*结果维度的 网络模型
        self.layer2=nn.Linear(middle_size,output_size)

    ## 该方法代表实际执行x过网络层预测y
    def forward(self,x):
        x=self.layer1(x)
        y_pred=self.layer2(x)
        return y_pred

## 建立torch模型
my_model=MyTorchModel(5,10,5)


## 使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
# 选择优化器 设置学习率
optimizer = optim.SGD(my_model.parameters(), lr=0.01)

# 开始循环训练该模型
# 训练计数器
counter=0
for i in range(1000):
    counter+=1
    # 计算预测值
    y_pred=my_model.forward(x)
    # 计算交叉熵 损失值
    loss=ce_loss(y_pred,y)
    # 梯度归0
    optimizer.zero_grad()
    # 反向传播,利用交叉熵反应的梯度反向修改模型参数
    loss.backward()
    # 更新模型参数
    optimizer.step()
    # 每10 轮打印 一次训练标签
    if i % 10 == 0:
        print(f"第 {counter} 轮，训练标签: {torch.argmax(y_pred, dim=1)}")
    # 每100 轮打印 一次损失函数
    if i % 100 == 0:
        print(i, loss.item())
        #print(f"真实标签: {y}")
    ## 如果已经得出预测结果 则停止训练
    if torch.equal(torch.argmax(y_pred,dim=1),y):
        print(f"一共训练：{counter} 轮")
        break

#训练结束后 进行对比
print(f"真实标签: {y}")
print(f"预测标签: {torch.argmax(y_pred,dim=1)}")
