# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset

'''
RNN多分类任务，判断‘你我他’字符出现的位置
需要完成：
1 样本的自动生成
2 RNN模型，采用交叉熵
3 模型训练与预测
'''

'''
1 样本自动生成
'''
def build_vocab(): #词表是a - z加上'你我他'
    chars = "abcdefghijklmnopqrstuvwxyz你我他"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #30
    return vocab

#随机生成固定长度的样本，'你我他'可能出现在不同的位置或不出现，共n+1类，'你我他'出现的概率是50%
def build_sample(vocab, sentence_length = 5):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())[:26]) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if random.random() < 0.5:
        k = random.randint(0,sentence_length-1)
        x[k] = random.choice(list(vocab.keys())[26:])
        y = k
    else:
        y = sentence_length #交叉熵只接受非负整数的真值
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

def BuildDataset(vocab, total_sample_num, sentence_length = 5):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x,y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

'''
2,定义一个RNN模型
'''
class RNNModel(nn.Module):
    def __init__(self, vocab, embedding_dim = 16, hidden_size = 10):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

'''
3,定义一个计算预测值准确度的函数，应用于每轮模型训练之后
'''
def accuracy(preds, labels):
    '''
    preds: 张量，x计算出的预测值
    labels: 张量，对应的真值
    return: 准确率
    '''
    score, pred_classes = torch.max(preds, 1) #在第二个维度上返回最大值及其序号（batch_size, num_classes)
    acc_num = (pred_classes == labels).sum().item() #逐项对比，并累加True的数量，即正确个数
    return acc_num / labels.size(0)

'''
模型训练时，需要定义：
1）训练轮数，
2）每轮使用随机的mini_batch进行训练，需要指定batch_size,
3) loss function和Optimizer
4) 学习率
'''

def main():
    total_sample_num = 5000
    epoch_num = 30
    batch_size = 20
    learning_rate = 0.001
    vocab = build_vocab()
    RNN_model = RNNModel(vocab)
    criterion = nn.CrossEntropyLoss()
    optimizier = torch.optim.Adam(RNN_model.parameters(), lr = learning_rate)

    X,Y = BuildDataset(vocab, total_sample_num)
    train_set = TensorDataset(X, Y)

    log = []
    for epoch in range(epoch_num):
        RNN_model.train()
        train_loader = DataLoader(train_set, batch_size, shuffle = True) #每轮都随机打乱数据集
        watch_loss = [] #记录当轮每个mini_batch的loss
        running_train_acc = 0 #累加每轮期间的准确度
        for inputs, labels in train_loader:
            optimizier.zero_grad()
            preds = RNN_model(inputs) #自动调用forward()
            loss = criterion(preds, labels)
            loss.backward()
            optimizier.step()
            watch_loss.append(loss.item())
            running_train_acc += accuracy(preds, labels)

        mean_loss = float(np.mean(watch_loss))
        #每轮训练完计算train的平均准确度，在新建一个测试集测试模型准确度
        mean_train_acc = running_train_acc / len(train_loader)

        X_Test, Y_Test = BuildDataset(vocab, 100)
        RNN_model.eval()
        with torch.no_grad(): #测试中不需要反向传播，不需要计算梯度
            Preds_test = RNN_model(X_Test)
            test_acc = accuracy(Preds_test, Y_Test)

        #记录当轮的loss和acc，打印信息
        log.append([mean_loss, mean_train_acc, test_acc])
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, mean_loss))

    torch.save(RNN_model.state_dict(), 'RNN_model.pt')
    plt.plot(range(len(log)), [l[0] for l in log], label="mean_loss")
    plt.plot(range(len(log)), [l[1] for l in log], label="train_acc")
    plt.plot(range(len(log)), [l[2] for l in log], label="test_acc")
    plt.legend()
    plt.show()
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


'''
加载训练好的模型，并进行预测
'''

if __name__ == "__main__":
    main()
    vocab = json.load(open("vocab.json", "r", encoding="utf8"))
    test_x, test_y = BuildDataset(vocab, 1)
    test_model = RNNModel(vocab)
    test_model.load_state_dict(torch.load("RNN_model.pt"))
    test_model.eval()
    with torch.no_grad():
        result = test_model(test_x)
        predicted_class = torch.argmax(result, dim=1).item()
        char_list = 'abcdefghijklmnopqrstuvwxyz你我他'
        input_char = ''
        for i in test_x.numpy()[0]:
            input_char += char_list[i-1]
        print("输入：%s, 预测类别：%d, 概率值：%f, 真值：%d" % (input_char, predicted_class, result[0][predicted_class], test_y.item()))





