import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import random

class multi_class(nn.Module):
    def __init__(self, vocab, seq_len, embedding_size, hidden_size, out_size):
        super(multi_class, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, out_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)    #(batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        _, x = self.rnn(x)       #(batch_size, seq_len, embedding_size) -> (batch_size, 1, hidden_size)
        x = x.squeeze()          #(batch_size, 1, hidden_size) -> (batch_size, hidden_size)
        y_pred = self.classifier(x)   #(batch_size, hidden_size) -> (batch_size, out_size)
        if y is not None:
            loss = self.loss(y_pred,y)
            return loss
        else:
            return y_pred

def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    #随机从字表选取seq_len个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #"你"出现在哪个位置就属于第几类
    try:
        index = x.index("你")
    except ValueError:
        index = -1
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    y = np.zeros(len(x))
    y[index] = 1
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, seq_len, embedding_size, hidden_size, out_size):
    model = multi_class(vocab, seq_len, embedding_size, hidden_size, out_size)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    #print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1   #负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    batch_size = 10
    epoch_num = 40
    train_sample = 500
    vocab = build_vocab()
    seq_len = 6
    embedding_size = 20
    hidden_size = 10
    out_size = seq_len

    model = build_model(vocab, seq_len, embedding_size, hidden_size, out_size)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for j in range(int(train_sample/batch_size)):
            x, y = build_dataset(batch_size, vocab, seq_len)
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            optimizer.zero_grad()
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, seq_len)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

if __name__ == "__main__":
    main()


