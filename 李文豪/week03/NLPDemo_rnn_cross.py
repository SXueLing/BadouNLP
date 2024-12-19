import  torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self,vector_dim,sentence_lenth,vocab):
        """
        :param vector_dim: 字符转化向量的维度
        :param sentence_lenth: 字符样本的长度
        :param vocab: 所有的字符集
        """
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(sentence_lenth,vector_dim,bias=True, batch_first=True) # 传参怎么穿
        # self.pool = nn.AvgPool1d(sentence_lenth)
        self.layer = nn.Linear(vector_dim , sentence_lenth)
        # self.activation = nn.functional.sigmoid
        self.loss = nn.CrossEntropyLoss()
    def forward(self,x,y=None):
        # print("最开始的shape：",x.shape)
        # print("embedding的shape",self.embedding.weight.shape)
        x = self.embedding(x)
        # print("embedding后的shape：",x.shape)
        # print(x)
        x= x.transpose(1,2)
        # print("转置后的shape：",x.shape)
        # print("all_weights的shape：",self.rnn.all_weights)
        x,x1 = self.rnn(x)
        # print("-------------------------------")
        # print("x,x1:",x.shape,x1.shape)
        x1 = x1.squeeze()
        # print("x1的shape:",x1.shape)
        y_pred = self.layer(x1)
        # print("y_pred的shape：",y_pred.shape)
        # print("y的shape：",y.shape)
        # y_pred = self.activation(x)
        # print("===============================")
        # print(y_pred,y)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_vocab():
    chars="你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index ,char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(chars)
    return vocab

def build_sample(vocab,sentence_length):
    x = []
    while len(x) < sentence_length:
        str_ = random.choice(list(vocab.keys()))
        if str not in x:
            x.append(str_)
    y = -1
    for i in range(len(x)):
        if x[i] == '我':
            y = i
    x = [vocab.get(word,vocab['unk']) for word in x]
    return x,y

def build_dataset(sample_length,vocab,sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        y= -1
        x= []
        while y == -1:
            x,y = build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # print("x,y:",dataset_x,dataset_y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

def evaluate(model,vocab,sentence_lenth):
    model.eval()
    x,y = build_dataset(20,vocab,sentence_lenth)
    print("本次预测集中共有%d个样本"%(len(y)))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if np.argmax(y_p) == y_t:
                correct +=1
            else:
                wrong+=1
    print("正确测试个数：%d，正确率：%f"%(correct,correct/(correct + wrong)))
    return correct/(correct + wrong)

def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005

    vocab = build_vocab()

    model = TorchModel(char_dim,sentence_length ,vocab)

    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)

    log=[]

    for epoch in range(epoch_num):
        model.train()
        watch_loss= []
        for batch in range(int(train_sample / batch_size)):
            x,y = build_dataset(batch_size,vocab, sentence_length)
            optim.zero_grad()
            loss = model.forward(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("==========\n第%d轮平均loss：%f"%(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model,vocab,sentence_length)
        log.append([acc,np.mean(watch_loss)])

    #画图
    plt.plot(range(len(log)),[l[0] for l in log] , label="acc")
    plt.plot(range(len(log)),[l[1] for l in log] , label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict() ,"model1.pth")
    writer = open("vocab1.json","w",encoding="utf-8")
    writer.write(json.dumps(vocab , ensure_ascii= False,indent=2))
    writer.close()
    return

if __name__ == "__main__":
    main()
