import torch
import torch.nn as nn
import torch.optim as optim
import random
"""
1.得到数据集
2.定义模型
3.训练模型
4.测试模型
"""
#词典
class Net(nn.Module):
    def __init__(self,vocab_size,word_dim,hidden_size,words_len):
        super(Net, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.words_len = words_len
        self.word_embed = nn.Embedding(vocab_size,word_dim)
        self.rnn = nn.RNN(word_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,words_len)
    def forward(self,x):
        x = self.word_embed(x)
        _, state = self.rnn(x)
        output = self.fc(state)
        return output


def build_vacob():
    word2idx = {chr(i):i - ord('a') for i in range(ord('a'),ord('z')+1)}
    word2idx['unk'] = len(word2idx)
    word2idx['余'] = len(word2idx)
    idx2word = {v:k for k,v in word2idx.items()}
    return word2idx ,idx2word

def build_dataset(batchs,batch_size,words_len):
    for i in range(batchs):
        X = []
        Y = []
        for j in range(batch_size):
            x = [random.randint(0,len(vocab) - 2 ) for _ in range(words_len)]
            y = random.randint(0,words_len-1)
            x[y] = vocab['余']
            X.append(x)
            Y.append(y)
        yield torch.tensor(X),torch.tensor(Y)

    pass

def train(epochs,batch_size,batchs,words_len,net):
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    cross_entropy = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss_sum = 0
        acc_sum = 0
        for x,y in build_dataset(batchs,batch_size,words_len):
            optimizer.zero_grad()
            out = net(x)

            loss = cross_entropy(out.squeeze(),y.long())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            acc_sum += (out.argmax(-1) == y).sum().item()
        print("epoch:{},    loss:{},    acc:{}".format(epoch+1,loss_sum/batchs,acc_sum/(batchs*batch_size)))

def test(net,vocab):
    with torch.no_grad():
        for x,y in build_dataset(1,4,6):
            out = net(x).squeeze()
            pred = out.argmax(-1)
            x = x.tolist()
            print(x[0][0])
            print([[ vocab[j] for j in i] for i in x])
            print("预测结果")
            print(pred)
            print("真实结果")
            print(y)

if __name__ == '__main__':
    vocab ,idx2word= build_vacob()
    words_len = 6
    word_dim = 100
    hidden_size = 256
    batch_size = 32
    batchs = 100
    net = Net(len(vocab),word_dim,hidden_size,words_len)
    train(10,batch_size,batchs,words_len,net)
    test(net,idx2word)
