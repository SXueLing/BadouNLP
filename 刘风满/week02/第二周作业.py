##导入需要的包
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#创建
class ModelTarget(nn.Module):
    def __init__(self,input_size):
        super(ModelTarget,self).__init__()
        self.fc1=nn.Linear(input_size,5)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(5,5)
        #激活函数，sigmoid归一化到0-1之间
        #self.activation=nn.Sigmoid()
        self.loss=nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        y_pred=self.fc1(x)
        if y is not None:
            loss=self.loss(y_pred,y)
            return loss
        else:
            return y_pred



#创建数据集,传入行数和列数，返回输入数据集与标签集
def create_randomdata(train_rowCount,train_colCount):
    #调用numpy的random模块生成随机数据生成指定维度的数据集
    x=np.random.random((train_rowCount,train_colCount))
    # print("随机数据集：",x)
    #生成标签集，标签集的维度与输入数据集的行数相同,计算每一行的最大值索引。
    #argmax()函数返回指定轴上的数据最大值的索引
    max_index=np.argmax(x,axis=1)
    # print("最大值索引：",max_index)
    #创建一个标签机与x的行数相同，列数相同的矩阵
    y=np.zeros_like(x)

    y[np.arange(x.shape[0]),max_index]=1
    # print("标签集：",y)
    return torch.FloatTensor(x),torch.FloatTensor(y)

train_rowCount=5000
train_colCount=5
def main():
    train_batch_size=20
    #创建模型
    model=ModelTarget(5)
    #打印模型结构
    # print(model)
    X,Y= create_randomdata(train_rowCount, train_colCount)
    #定义优化器
    optimizer=optim.SGD(model.parameters(),lr=0.01)

    log=[]
    #训练模型
    for epoch in range(1000):
        # 开始训练
        model.train()
        watch_loss=[]
        #根据训练的每次训练batch获取对应的批次训练数据,计算训练数据总数整除每次训练的batch数，取整避免最后的不足batch数
        for batch_idx in range(train_rowCount//train_batch_size):
            #获取当前批次的训练数据
            train_x=X[batch_idx*train_batch_size:(batch_idx+1)*train_batch_size]
            #获取当前批次的训练标签
            train_y_true=Y[batch_idx*train_batch_size:(batch_idx+1)*train_batch_size]
            # print(train_x)
            # print(train_y_true)

            #计算损失函数
            loss=model(train_x,train_y_true)
            #反向传播,计算梯度
            loss.backward()
            #更新参数
            optimizer.step()
            #清空梯度
            optimizer.zero_grad()
            # print(loss.item())
            watch_loss.append(loss.item())
        #np.mean()函数计算数组的均值
        print("第%d次训练的平均损失为：%f"%(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model)
        # 记录训练结果,acc为准确率,np.mean(watch_loss)为平均损失
        log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(),"model.pth")
    # 绘制训练结果
    import matplotlib.pyplot as plt
    plt.plot([i[0] for i in log], label='acc')
    plt.plot([i[1] for i in log], label='loss')
    plt.legend()     # 显示图例
    plt.show()


#评估模型的准确率
def evaluate(model):
    # 开始测试
    model.eval()
    # 测试数据集
    eval_rowCount = 5
    eval_x, eval_y = create_randomdata(eval_rowCount, train_colCount)
    # 计算测试集的预测值
    with torch.no_grad():
        eval_y_pred = model(eval_x)
        # 计算测试集的准确率,argmax()函数返回指定轴上的数据最大值的索引,float()函数将结果转化为浮点数,mean()函数计算数组的均值,item()函数将结果转化为python的数值类型
        acc = (eval_y_pred.argmax(dim=1) == eval_y.argmax(dim=1)).float().mean().item()
        print("测试集的准确率为：%f" % acc)
        return acc

def predict():
    # 加载模型
    model = ModelTarget(5)
    model.load_state_dict(torch.load("model.pth"))
    # 预测数据集
    predict_rowCount = 10
    predict_x, _predict_y = create_randomdata(predict_rowCount, train_colCount)
    # 计算预测集的预测值
    with torch.no_grad():
        predict_y_pred = model(predict_x)
    print("输入数据集：", predict_x)
    print("预测集的标签：", _predict_y.argmax(dim=1))
    print("预测集的预测值：", predict_y_pred.argmax(dim=1))


if __name__ == '__main__':
    main()
    # 测试数据集
    #predict()
