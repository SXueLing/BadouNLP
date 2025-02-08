# coding:utf8
# auther: 王良顺
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import os

DATASET_TEST_FOLDER = 'Test_Dataset'
# DATASET_FOLDER = './Dataset/'
DATASET_FILE_X = 'Dataset_X'
DATASET_FILE_Y = 'Dataset_Y'
EXECUTE_FOLDER = 'Execute_Predict'

class TorchModel(nn.Module,):
    def __init__(self,loss):
        super(TorchModel,self).__init__()
        self.liner = nn.Linear(5,5) #一个线性层,输出1个五维向量
        self.activation =  nn.functional.softmax # 激活函数使用softmax
        self.loss = loss # 损失函数loss采用 交叉熵;reduction='mean':计算的是每个样本的平均损失，已经做了归一化处理
    # 当输入了真实标签值y,返回loss值;没有时返回预测结果
    def forward(self,x,y=None):
        x = self.liner(x) #x传入第一个线性层
        x = self.liner(x)  # x传入第一个线性层
        y_pred = self.activation(x) #依次将线性层的结果传入激活函数
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred


#生成一份训练样本
#生成一个5维向量,五维随机向量最大的数字在哪维就属于哪一类
def build_data():
    data = np.random.random(5)

    #哪类大就哪类为1: [10,20,30,40,50]=>[0,0,0,0,1]
    data_y = []
    #记录标志保证只存在一个最大值
    exit_max = False
    for d in data:
        if d == max(data) and exit_max==False:
            data_y.append(1)
            exit_max = True
        else:
            data_y.append(0)
    # return data, np.where(data == max(data))[0]
    return data,data_y

#随机生成一份训练样本
def build_dataset(num):
    X = []
    Y = []
    for i in range(num):
        x,y = build_data()
        X.append(x)
        Y.append(y)
    # 借用torch.FloatTensor 将向量转为矩阵
    return torch.FloatTensor(X),torch.FloatTensor(Y)

#将一份训练数据保存在一个文件中
def save_dataset(dataset,name,n):
    fileName = createFileName(name,n)
    # print(fileName)
    torch.save(dataset,fileName)

# 生成一份训练数据,保存到文件里,然后读取此文件
def get_datasetFromFile(name,n):
    fileName = createFileName(name,n+1)
    # print(fileName)
    return torch.load(fileName)

def createFileName(name,n):
    return name+"_"+str(n)+".pt"

# 生成train_num个训练数据,分别保存在num个文件中
def createDatasetFile(train_num,num,folder):
    # 校验文件夹是否存在
    if not os.path.exists(folder):
        os.mkdir(folder)

    for n in range(train_num):
        train_x, train_y = build_dataset(num)
        save_dataset(train_x, folder+'/'+DATASET_FILE_X,n+1)
        save_dataset(train_y, folder+'/'+DATASET_FILE_Y, n+1)

#根据当前训练次数获取训练文件
def getDatasetFile(train_num,folder):
    x = get_datasetFromFile('./'+folder+'/'+DATASET_FILE_X, train_num)
    y = get_datasetFromFile('./'+folder+'/'+DATASET_FILE_Y, train_num)
    return torch.FloatTensor(x), torch.FloatTensor(y)

# 测试每轮的准确率
def evaluate(model,epoch):
    #调整为执行模式
    model.eval()
    # 测试100个数据
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    # 由于y的都是 1和0,所以和的值就是正样本的个数,100-正样本个数就是负样本的个数
    # print("第%d次训练的模型预测数据有%d个正样本,%d个负样本"%(epoch+1,sum(y),test_sample_num-sum(y)))

    correct,wrong = 0,0
    # 模型不计算梯度
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y): # 与真实数据进行对比
            if max_tensor(y_p,y_t):
                correct += 1 # 正样本正确
            else:
                wrong +=1
    corr = correct/(correct+wrong)
    print("正确预测个数:%d,错误预测个数:%d,正确率:%f"%(correct,wrong,corr))
    return corr

#判断是最大值
def max_tensor(y_p,y_t):
    data_y = torch.zeros(5)
    # 记录标志保证只存在一个最大值
    exit_max = False
    for d in y_p:
        if d == max(y_p) and exit_max == False:
            index = y_p == d
            data_y[index] = 1
            exit_max = True
    return torch.equal(data_y,y_t)

def main():
    # 配置参数
    epoch_num = 20 # 训练次数
    batch_size = 20 # 每次训练的样本个数
    train_num = 5000 # 每轮训练总共训练的样本总数
    input_size = 5 # 输入的向量维度
    learning_rate = 0.001 # 学习率
    criterion = torch.nn.CrossEntropyLoss()
    #建立模型
    model = TorchModel(criterion)

    # 选择优化器
    optim  = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # 存储日志
    log = []
    #先生成epoch_num份的训练样本分别保存在文件中
    createDatasetFile(epoch_num,train_num,DATASET_TEST_FOLDER)

    #训练过程,训练epoch_num遍
    for epoch in range(epoch_num):
        # 开启训练模式
        model.train()
        # 储存损失值
        watch_loss = []
        # 获取训练集
        train_x, train_y = getDatasetFile(epoch,DATASET_TEST_FOLDER)
        #一轮训练进行250次
        #一次训练输入20个样本个数
        for batch_index in range(train_num//batch_size):
            start_index = batch_index * batch_size
            end_index = (batch_index + 1) * batch_size
            x = train_x[start_index: end_index]
            y = train_y[start_index: end_index]
            loss = model.forward(x,y) # 计算loss,等效于 model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("\n第%d轮平均loss:%f"%(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model,epoch) # 测试本轮模型的结果
        # np.mean 计算平均值
        log.append([acc,float(np.mean(watch_loss))])
    # 保存此次训练的模型
    torch.save(model.state_dict(),"work2_model.pt")
    print("\n模型参数:%s"%(model.state_dict()))
    # print(log)
    #画图,x轴为训练次数,y轴为预测结果的准确率
    plt.plot(range(len(log)),[ly[0] for ly in log],label='acc') #画acc曲线
    # x轴为训练次数,y轴为此次训练的平均loss
    plt.plot(range(len(log)),[ly[1] for ly in log],label ='loss') #画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def execute_predict(model_path,input_vec):
    criterion = torch.nn.CrossEntropyLoss()
    model = TorchModel(criterion)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())
    model.eval() # 设置为执行模式
    with torch.no_grad(): #不计算梯度
        result = model.forward(input_vec)
    for vec,res in zip(input_vec,result):
        # round((float(res))) 概率值为0. ~ 1.0 所以此处四舍五入后,大于0.5为正样本,小于0.5为负样本
        # print("输入: %s, 预测类别: %d, 概率值: %f"%(vec,round((float(res))),res))
        index = vec == max(vec)
        print("输入: %s" % (vec))
        print("输出: %s" % (res))
        print("输入的最大项: %s"%(torch.where(res == max(res))[0]))
        print("输出的最大项: %s"%(torch.where(vec ==max(vec))[0]))
        print("概率值: %f" % (res[index]*100))
        print("=======================\n")

def predict():
    createDatasetFile(1,10,EXECUTE_FOLDER)
    X, Y = getDatasetFile(0, EXECUTE_FOLDER)
    execute_predict("work2_model.pt",X)

if __name__ == "__main__":
    t = int(input("1测试, 2预测,请选择操作:"))
    if t ==1:
        main()
    elif t==2:
        predict()
    else:
        print("无效输入")
