#coding:utf8
import time
import torch
import torch.nn as nn
import numpy as np

#num个5维数组
now = time.time()
def build_dataset(num, if_save = False):

    data = []
    for i in range(num):
        x = np.random.random(5)
        data.append(x)

    true = []
    tmp_date = np.array(data)
    for i in tmp_date:
        #y = np.index(max(i.tolist))
        y = np.argmax(i)
        true.append(y)
    
    if if_save:
        data_file_name = "./"+ str(num) + "_" + str(now) + "_dataTensor" +  ".pt"
        true_file_name = "./"+ str(num) + "_" + str(now) + "_trueTensor" +".pt"
        torch.save(true,true_file_name)
        torch.save(data,data_file_name)
    
    return torch.FloatTensor(data), torch.LongTensor(true)

#模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1=7, hidden_size2=5):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) #w：5 x 7
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 7 x 5

        # self.sigmod = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.layer1(x)   #shape: (batch_size, input_size) -> (batch_size, hidden_size1) 
        # x = self.sigmod(x)
        y_pred = self.layer2(x) #shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2) 
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p) == y_t:
                correct += 1  
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main(train_x, train_y, save_model = False):
    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    # 配置参数
    epoch_num = 1000  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    if save_model:
        model_file = "./" + str(now) + "model.bin"
        torch.save(model.state_dict(), model_file)

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        print("result:",result)
    for vec, res in zip(input_vec, result):
        loss = nn.CrossEntropyLoss()
        s = torch.argmax(res)
        print("s:",s)
        r = loss(res,s)
        
        print("输入：%s, 预测类别：%d, 损失：%f" % (vec, torch.argmax(res), r))  # 打印结果
        
if __name__ == "__main__":
    # main()
    file_name_date = "200_1732984123.9257965_dataTensor.pt"
    file_name_true = "200_1732984123.9257965_trueTensor.pt"

    data = torch.load(file_name_date)
    true = torch.load(file_name_true)

    # 训练
    #main(data,true,save_model = true)

    # 预测
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
            [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
            [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
            [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("./1733016353.236413model.bin", test_vec)

    

