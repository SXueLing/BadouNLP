
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.layer = nn.RNN(input_size=20, hidden_size=7, bias=False, batch_first=True)
        self.loss = nn.CrossEntropyLoss()  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        y_pred,h = self.layer(x)
        h = h.squeeze()
        if y is not None:
            return self.loss(h, y)   #预测值和真实值计算损失
        else:
            return h                 #输出预测结果



def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab


def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = 0

    random_num=[i for i in range(2)]
    p=random.choice(random_num)
    if p!=0:
        x[random.choice([i for i in range(6)])]='你'
    for string in x:
        i = 0
        for string in x:
            if string == '你':
                break
            else:
                i += 1
        y = i
    #print(x)
    #指定哪些字出现时为正样本
    #print(y)
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x , y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        # print(x)
        y_pred = model(x)      #模型预测
        # print(y_pred)
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            y_ft = y_p.argmax()
            # print(x)
            # print(y_ft)
            # print(y)
            if int(y_ft) ==int(y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)



def main():
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率



    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model.forward(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
        # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.path")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path,weights_only=True))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    print(x)
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
        print(result)
        print(torch.max(result,1).indices+1)
    # for i, input_string in enumerate(input_strings):
    #     print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.path", "vocab.json", test_strings)
