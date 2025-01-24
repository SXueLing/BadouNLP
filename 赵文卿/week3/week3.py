'''
Author: Zhao
Date: 2024-12-22 20:48:14
LastEditTime: 2024-12-22 22:12:08
FilePath: NLPDemo.py
Description: 基于pytorch的网络编写
            实现一个网络找出特定字符在字符串中的位置
'''
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

#初始化父类
class MyTorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(MyTorchModel, self).__init__()
        #创建RNN层
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        #创建池化层
        #self.pool = nn.AvgPool1d(sentence_length)
        # RNN
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        #创建线性分类层
        self.classify = nn.Linear(vector_dim, sentence_length + 1) # +1 为可能有某个词不存在的情况
        #损失函数采用交叉熵
        self.loss = nn.functional.cross_entropy
    
    #前向传播
    def forward(self, x, y = None):
        #输入嵌入
        x = self.embedding(x)
        
        #使用pooling的情况，先使用pooling池化层会丢失模型语句的时序信息
        # x = x.transpose(1, 2)
        # x = self.pool(x) 
        # x = x.squeeze()

        #使用rnn的情况
        #rnn_out：每个字对应的向量  hidden：最后一个输出的隐含层对应的向量
        rnn_out, hidden = self.rnn(x)
        #从RNN的输出中提取最后一个时间步(最后一维)的特征向量
        #x = rnn_out[:, -1, :]
        x = hidden.squeeze()

        #分类
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#定义字符集
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    #初始化一个字典 vocab，其中键为 'pad'，值为 0
    vocab = {"pad":0}
    #遍历分配序号
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) #特殊的键 'unk'，其值为当前字典的长度
    return vocab

#随机生成一个样本
def build_sample(vocab, sentence_length):
    #sample，是不放回的采样，每个字母不会重复出现，但是要求字符串长度要小于词表长度
    x = random.sample(list(vocab.keys()), sentence_length)
    #指定正样本
    if "a" in x:
        y = x.index("a")
    #未出现为负样本
    else:
        y = sentence_length
    #转换
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型 
# vocab：词汇表，通常是一个包含所有字符或单词的列表或字典
# char_dim：字符的维度，即每个字符在嵌入层中的向量长度
# sentence_length：句子的最大长度
def build_model(vocab, char_dim, sentence_length):
    model = MyTorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
def evaluate(model, vocab, sample_length):
    model.eval() #评估模式，禁用 dropout 等训练时的行为
    x, y = build_dataset(200, vocab, sample_length) 
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad(): #禁用梯度计算
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y): #与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)




#模型训练
def main():
    #配置参数:训练轮数epoch_num
    # 批量大小batch_size, 训练样本数train_sample, 字符维度char_dim, 句子长度sentence_length, 学习率learning_rate
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器 使用 Adam 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #绘图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(),"model.pt")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))   #加载字符表
    model = build_model(vocab, char_dim, sentence_length)       #建立模型
    model.load_state_dict(torch.load(model_path,weights_only=True))               #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string]) #将输入序列化
    model.eval() #测试模式
    with torch.no_grad():#不计算梯度
        result = model.forward(torch.LongTensor(x)) #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) 


if __name__ == "__main__":
    # main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pt", "vocab.json", test_strings)
    """
    输入：fnvfee, 预测类别：tensor(6), 概率值：tensor([-3.1587, -4.1056, -3.3105, -3.9832, -4.0782, -3.8415,  4.5434])
    输入：wz你dfg, 预测类别：tensor(6), 概率值：tensor([-3.6921, -4.0846, -3.4727, -3.5593, -3.5003, -3.4130,  5.1879])
    输入：rqwdeg, 预测类别：tensor(6), 概率值：tensor([-3.6922, -4.0852, -3.4731, -3.5598, -3.5008, -3.4131,  5.1884])
    输入：n我kwww, 预测类别：tensor(6), 概率值：tensor([-3.6449, -4.0485, -3.4316, -3.5193, -3.4419, -3.3601,  5.1412])
    """
