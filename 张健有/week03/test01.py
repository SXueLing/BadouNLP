from email.quoprimime import header_length

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

##定义一个RNN的模型
class TorchRnn(nn.Module):
    ##这里input_size 代表输入rnn的特征向量维度
    ## (batch_size, sequence_length, input_size) sequence_length 代表每个序列中有几个字符，input_size 代表字符的维度
    def __init__(self, input_size, sentence_length,hidden_size,vocab):
        super(TorchRnn, self).__init__()
        #batch_first=True 指定输入和输出的张量维度排列：
        """
        bias=False 
        是否使用偏置项。
        这里设置为 False，表示 RNN 中的权重矩阵计算不包含偏置。
        
        batch_first=True 
        指定输入和输出的张量维度排列：
        如果设置为True，输入和输出的形状为(batch_size, sequence_length, input_size)；
        如果为False，输入和输出的形状为(sequence_length, batch_size, input_size)
        """
        self.embedding = nn.Embedding(len(vocab), input_size, padding_idx=0)  # embedding层
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)
        self.classify = nn.Linear(hidden_size, sentence_length+1)  # 线性层
        self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    def forward(self, x,y=None):
        # 当输入真实标签，返回loss值；无真实标签，返回预测值
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x,_ = self.layer(x)
        x = self.classify(x)
        y_pred = self.activation(x,dim=1)  # softmax 进行归一化
        if y is not None:
            batch_size, sen_len = y.size()
            y_pred = y_pred.view(batch_size * sen_len, -1)  # 展平y_pred
            y = y.view(batch_size * sen_len)  # 展平目标标签y
            return self.loss(y_pred, y)  # 计算损失
        else:
            return y_pred  # 输出预测结果

vocab = {
  "pad": 0,
  "你": 1,
  "我": 2,
  "他": 3,
  "d": 4,
  "e": 5,
  "f": 6,
  "g": 7,
  "h": 8,
  "i": 9,
  "j": 10,
  "k": 11,
  "l": 12,
  "m": 13,
  "n": 14,
  "o": 15,
  "p": 16,
  "q": 17,
  "r": 18,
  "s": 19,
  "t": 20,
  "u": 21,
  "v": 22,
  "w": 23,
  "x": 24,
  "y": 25,
  "z": 26,
  "unk": 27
}
## 生成单个样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    target_char = "x"
    if target_char in x:
        y = x.index(target_char)
    else:
        y = sentence_length+1
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    print(x,y)
    return x, y

##  创建数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, sample_length, sentence_length):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sentence_length)  # 生成测试数据集
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for i in range(len(y_pred)):  # 遍历所有样本
            pred_idx = y_pred[i].argmax().item()  # 获取最大值索引（即预测的位置）
            true_idx = y[i].item()  # 真实标签的标量值
            if pred_idx == true_idx:
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print(f"准确率：{accuracy * 100:.2f}%")
    return accuracy


def main():
    # 配置参数
    epoch_num = 1000 # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    hidden_size = 64 # 隐藏层的大小
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    model = TorchRnn(char_dim, sentence_length,hidden_size,vocab)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.4f}")
        acc = evaluate(model, vocab, 200, sentence_length)  # 测试模型结果
        log.append([acc, np.mean(watch_loss)])

    # 画图
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


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    hidden_size = 64
    sentence_length = 6 # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = TorchRnn(char_dim, sentence_length,hidden_size,vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        pred_idx = result[i].argmax().item()  # 预测位置
        if pred_idx == sentence_length:
            print(f"输入：{input_string}, 预测：未找到目标字符")
        else:
            print(f"输入：{input_string}, 预测目标字符位置：{pred_idx}")

if __name__ == "__main__":
    main()
    test_strings = ["fnxfee", "wz你dfx", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
