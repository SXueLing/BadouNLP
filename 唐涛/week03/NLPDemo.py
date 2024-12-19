import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim,padding_idx=0)
        self.pool=nn.AvgPool1d(sentence_length)     #池化层
        # self.classify=nn.Linear(vector_dim,1)       #线性层
        # self.activation=torch.sigmoid
        # self.loss=nn.functional.mse_loss
        self.layer=nn.RNN(vector_dim,6,bias=False,batch_first=True)
        self.loss=nn.CrossEntropyLoss()


    def forward(self,x,y=None):
        x=self.embedding(x)
        x=x.transpose(1,2)
        x=self.pool(x)
        x=x.squeeze()
        # x=self.classify(x)
        # y_pred=self.activation(x)
        output, h=self.layer(x)
        if y is not None:
            return self.loss(output,y)
        else:
            return output


def build_vocab():
    chars="你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char]=index+1
    vocab['unk']=len(vocab)
    return vocab

def build_sample(vocab,sentence_length):
    #随机从列表中取sentence_length个字，可重复
    # y=list(vocab.keys())
    x = random.sample(list(vocab.keys()),sentence_length)

    # x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    # if set("你我他") & set(x):
    #     y = 1
    # else:
    #     y = 0
    y=[]
    for index, char in enumerate(x):
        if set("你我他") & set(x[index]):
            y.append(index)
        else:
            y.append(0)
    x = [vocab.get(word, vocab['unk']) for word in x]
    if sum(y)==0:
        y=[6,6,6,6,6,6]

    return x,y

def bulid_dataset(sample_length,vocab,sentence_length):
    dataset_x=[]
    dataset_y=[]
    for i in range(sample_length):
        x,y=build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.FloatTensor(dataset_y)


def build_model(vocab,char_dim,sentence_length):
    model=TorchModel(char_dim,sentence_length,vocab)
    return model

def evaluate(model,vocab,sample_length):
    model.eval()
    x,y=bulid_dataset(200,vocab,sample_length)
    # print("%d个正样本，%d个负样本"%(sum(y),200-sum(y)))
    correct,wrong=0,0
    flag=0
    with torch.no_grad():
        y_pred=model(x)     #模型预测
        for y_p,y_t in zip(y_pred,y):
            for index in range(sample_length):
                if y_p[index]<0 and int(y_t[index])==6:
                    flag = 1
                elif y_p[index]>0 and int(y_t[index])!=0:
                    flag = 1
                else:
                    flag = 0
            if flag==1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num=20        #训练轮数
    batch_size=10
    train_sample=5000    #样本总数
    char_dim=20         #每个字的维度
    sentence_length=6   #样本长度
    learing_rate=0.005  #学习率

    #建立字表
    vocab=build_vocab()
    print(vocab)

    #建立模型
    model=build_model(vocab,char_dim,sentence_length)
    #优化器
    optim=torch.optim.Adam(model.parameters(),lr=learing_rate)
    log=[]

    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch in range(int(train_sample/batch_size)):
            x,y=bulid_dataset(batch_size,vocab,sentence_length)
            optim.zero_grad()   #梯度归零
            loss=model(x,y)
            loss.backward()     #计算梯度
            optim.step()        #更新权重
            watch_loss.append(loss.item())

        print("\n第%d轮平均loss:%f"%(epoch+1,np.mean(watch_loss)))
        acc=evaluate(model,vocab,sentence_length)   #测试本轮模型结果
        log.append([acc,np.mean(watch_loss)])
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    # plt.legend()
    # plt.show()
    # #保存模型
    torch.save(model.state_dict(),"model.pth")
    #保存词表
    writer=open("vocab.json","w",encoding="utf8")
    writer.write(json.dumps(vocab,ensure_ascii=False,indent=2))
    writer.close()
    return

def predict(model_path,vocab_path,input_strings):
    char_dim=20
    sentence_length=6
    vocab=json.load(open(vocab_path,"r",encoding="utf8"))
    model=build_model(vocab,char_dim,sentence_length)
    model.load_state_dict(torch.load(model_path))       #加载训练好的权重
    x=[]
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()    #测试模式
    with torch.no_grad():   #不计算梯度
        result=model.forward(torch.LongTensor(x))
    for i,input_string in enumerate(input_strings):
        print("输入：%s,预测类别：%s"%(input_string,result[i]))

if __name__=="__main__":
    # main()
    test_strings = ["fnvfee", "wz你hfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
