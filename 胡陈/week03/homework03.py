
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class FindPositionModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocabulary):
        super(FindPositionModel, self).__init__()
        self.embedding = nn.Embedding(len(vocabulary), vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim,sentence_length+1)
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):
         x=self.embedding(x)
         rnn_out, hidden = self.rnn(x)
         x=rnn_out[:,-1,:]
         y_pred = self.classify(x)
         if y is not None:
             return self.loss(y_pred,y)
         else:
             return y_pred

def build_vocabulary():
    charsets = "abcdefghijklmnopqrstuvwxyz"
    vocabulary = {"padding":0}
    for i, ch in enumerate(charsets):
        vocabulary[ch] = i + 1
    vocabulary['UnKnown'] = len(vocabulary)
    return vocabulary

def build_one_sample(vocabulary, sentence_length):
    x=random.sample(list(vocabulary.keys()),sentence_length)
    if "a" in x:
        y=x.index("a")
    else:
        y=sentence_length
    x=[vocabulary.get(word,vocabulary['UnKnown']) for word in x]
    return x,y

def build_dataset(sample_length, vocabulary,sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x,y=build_one_sample(vocabulary,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

def build_model(vocabulary, char_dim, sentence_length):
    model = FindPositionModel(char_dim, sentence_length, vocabulary)
    return model

def evaluate(model, vocabulary, sample_length):
    model.eval()#设置为评估模型
    x,y=build_dataset(100,vocabulary,sample_length)
    print("总测试样本数：%d"%(len(y)))
    right,wrong=0,0
    with torch.no_grad():
        y_pred=model(x)
        for yp,yt in zip(y_pred,y):
            if int(torch.argmax(yp)) == int(yt):
                right+=1
            else:
                wrong+=1
    total_num=right+wrong
    success_rate=right/total_num
    print("预测正确的个数：%d, 正确率：%.5f"%(right, success_rate))
    return success_rate



def main():
    epoch_num=20
    batch_size=40
    train_sample=2000 #1000  训练样本总数
    char_dim=50
    sentence_length=10
    lr=0.001

    vocabulary = build_vocabulary()
    findPositionModel = build_model(vocabulary,char_dim,sentence_length)
    opt=torch.optim.Adam(findPositionModel.parameters(),lr=lr)
    log=[]

    for epoch in range(epoch_num):
        findPositionModel.train()
        watch_loss=[]
        for batch in range(int(train_sample/batch_size)):
            x,y=build_dataset(batch_size, vocabulary,sentence_length)
            opt.zero_grad()#置零
            loss=findPositionModel(x,y)
            loss.backward()
            opt.step()#更新权重 w？
            watch_loss.append(loss.item())
        print("\n/*********************/")
        print("第%d轮的loss值取平均：%f"%(epoch+1,np.mean(watch_loss)))
        acc=evaluate(findPositionModel,vocabulary,sentence_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    torch.save(findPositionModel.state_dict(),"findPositionModel.pth")
    writer=open("vocabulary.json","w",encoding="utf8")
    writer.write(json.dumps(vocabulary,ensure_ascii=False,indent=2))
    writer.close()
    return

#预测函数和评估函数的区别？

def predict(model_path,vocabulary_path,input_strings):
    char_dim = 50
    sentence_length=10
    vocabulary=json.load(open(vocabulary_path,"r",encoding="utf8"))
    testModel=build_model(vocabulary,char_dim,sentence_length)
    testModel.load_state_dict(torch.load(model_path))
    x=[]
    for in_str in input_strings:
        x.append([vocabulary[ch] for ch in in_str])
    testModel.eval()
    with torch.no_grad():
        result = testModel.forward(torch.LongTensor(x))
    for i,in_str in enumerate(input_strings):
        print("输入：%s，预测类别：%s，概率值：%s"%(input_strings,torch.argmax(result[i]),result[i]))

if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict("findPositionModel.pth", "vocabulary.json", test_strings)









