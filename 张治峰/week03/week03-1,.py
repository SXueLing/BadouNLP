# 第三周作业
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
# 寻找 长度小于6的 字符串中 字符为“张” 的位置 信息 若不存在返回 -1
class FindIndexModel(nn.Module):
    def __init__(self,vocab,dim_size,sentence_length):
        super(FindIndexModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),dim_size,padding_idx=0)   # 使用内嵌层 将 词汇表中的字符转成向量
        self.rnn =  nn.RNN(dim_size, dim_size, bias=False, batch_first=True)  
        self.pool = nn.AvgPool1d(dim_size) # 池化
        self.linear = nn.Linear(sentence_length,sentence_length+1) # 使用linear 将结果矩阵为 batchSize * sentence_length+1
        self.loss = nn.functional.cross_entropy   # soft max 函数不需要使用 交叉商函数计算 会自动调用
    def forward(self,x,y=None):
        # 初始 x shape = batch_size,sentence_length
        x = self.embedding(x)# shape = batch_size ,sentence_length , dim_size
        x , h = self.rnn(x) # shape = batch_size ,sentence_length , dim_size
        x = self.pool(x)  # shape = batch_size ,sentence_length (如果先 tranport（1，2）结果会变的不准确)
        x = torch.squeeze(x) # batch_size , sentence_length
        pred  = self.linear(x)  # batch_size , sentence_length+1
        if y is not None:
            return self.loss(pred,y)
        else:
            return pred

# 生成 词汇表
def generator_vocab():
    chars = "张abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

# 生成测试数据样本
def generator_data(vocab,sentence_length):
    # 生成长度为6的字符串
    x = [np.random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 获取  字符 "张" 在 列表中的位置 没有则 -1
    try:
        index =  x.index("张")
    except ValueError:
        index = -1  # 如果字符不在列表中，返回-1
    x = [vocab[x[i]] for i in range(len(x))]
    y = np.zeros(sentence_length+1)
    y[index] = 1
    return x,y
# 生成测试样本数据集
def generator_dataset(vocab,sentence_length,size):
    X,Y = [],[]
    for _ in range(size):
       x,y = generator_data(vocab,sentence_length)
       X.append(x)
       Y.append(y)
    return torch.LongTensor(X),torch.FloatTensor(Y)


# 测试每轮模型的准确率
def test(model, vocab, sentence_length):
    model.eval()
    # 200个测试样本
    x, y = generator_dataset(vocab, sentence_length,200)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if np.argmax(y_p.numpy())  == np.argmax(y_t.numpy()) :
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def train():
    # 生成词汇表
    vocab = generator_vocab()
    epoch_num = 60
    batch_size = 20
    train_data_size = 1000
    lr = 0.001
    sentence_length = 6
    dim_size = 12
    model = FindIndexModel(vocab,dim_size,sentence_length)
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    log = []
    for epoch_index in range(epoch_num):
        model.train()
        train_x, train_y = generator_dataset(vocab, sentence_length, train_data_size)
        watch_loss = []
        for batch_index in range(train_data_size//batch_size):
            x = train_x[batch_index * batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad
            watch_loss.append(loss.item())
        print("轮次:%d,loss:%f"%(epoch_index,np.mean(watch_loss)))
        acc = test(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
     #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "week03_1.model")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()


#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    dim_size = 12  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = FindIndexModel(vocab,dim_size,sentence_length) #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    X = []
    for input_string in input_strings:
        x = [vocab.get(char,vocab["unk"]) for char in input_string] #将输入序列化
        while(len(x)<sentence_length):
            x.append(vocab["pad"])
        X.append(x)    
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(X))  #模型预测
    for i, input_string in enumerate(input_strings):
        index = np.argmax(result[i].numpy())
        if(index==6):
            index = -1
        print("输入：%s, 预测位置：%d, 结果值：" % (input_string,  index), result[i]) #打印结果



if __name__ == '__main__':
   train()
   print("=====开始预测结果=====")
   test_strings = ["张wzsd", "w张zsdf",  "nc张ww","ncw张w","","ncxcc"]
   predict("week03_1.model", "vocab.json", test_strings)
