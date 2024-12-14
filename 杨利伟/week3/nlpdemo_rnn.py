import random
import torch
import torch.nn as nn
import torch.optim as optim
# from torchinfo import summary

class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualRNN, self).__init__()
        
        # 定义模型参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 定义 RNN 的权重
        self.Wx_to_h = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)  # U 输入到隐藏层的权重
        self.Wh_to_h = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)  # W 隐藏层到隐藏层的权重
        self.Wh_to_y = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)  # 隐藏层到输出层的权重
        
        # 偏置项
        self.bh_to_h = nn.Parameter(torch.zeros(hidden_size))  # 隐藏层偏置
        self.bh_to_y = nn.Parameter(torch.zeros(output_size))  # 输出层偏置
        
    def forward(self, x):
        # 初始化隐藏状态
        h = torch.zeros(x.size(0), self.hidden_size).cuda()  # Batch size, hidden_size
        
        # RNN 前向传播
        # x : (batch_size, sequence_length, input_size)
        for t in range(x.size(1)):  # 迭代序列中的每个时间步
            x_t = x[:, t, :] # 获取时间步t的输入  (batch_size, input_size)
            # print("x_t.shape:", x_t.shape, "h.shape:", h.shape, "Wx_to_h.shape:", self.Wx_to_h.shape, "Wh_to_h.shape:", self.Wh_to_h.shape, "bh_to_h.shape:", self.bh_to_h.shape)
            # (batch_size, input_size) * (input_size, hidden_size) + (batch_size, hidden_size) * (hidden_size, hidden_size) + (hidden_size)
            # ht = tanh(W @ ht-1 + U @ xt + b)
            h = torch.tanh(torch.mm(x_t, self.Wx_to_h) + torch.mm(h, self.Wh_to_h) + self.bh_to_h)  # 更新隐藏状态

        # 计算输出
        # (batch_size, hidden_size) * (hidden_size, output_size) + (output_size)
        y = torch.mm(h, self.Wh_to_y) + self.bh_to_y  # 最后一个隐藏状态的输出

        return y
    
    def cuda(self, device = None):
        self.Wx_to_h = self.Wx_to_h.cuda(device)
        self.Wh_to_h = self.Wh_to_h.cuda(device)
        self.Wh_to_y = self.Wh_to_y.cuda(device)
        self.bh_to_h = self.bh_to_h.cuda(device)
        self.bh_to_y = self.bh_to_y.cuda(device)
        return super().cuda(device)
    
class NlpRNN(nn.Module):
    def __init__(self, vector_dim, vocab_size, output_size, hidden_size_rnn=16, hidden_size_fc=16):
        super(NlpRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)  #embedding层
        self.rnn = ManualRNN(vector_dim, hidden_size_rnn, hidden_size_fc)
        self.fc = nn.Linear(hidden_size_fc, output_size)
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.embedding(x)  # 将输入的句子转换为词嵌入表示
        x = self.rnn(x)  # 将词嵌入表示输入到RNN中
        x = self.fc(x)  # 将RNN的输出输入到全连接层中
        x = self.activation(x)  # 对输出进行归一化
        if y is not None:
            return self.loss(x, y)
        return x

def test_rnn():
    # 参数设置
    input_size = 10  # 输入特征维度
    hidden_size = 20  # 隐藏层大小
    output_size = 1  # 输出维度

    # 创建模型
    model = ManualRNN(input_size, hidden_size, output_size)

    # 打印模型结构
    print(model)

    # 测试模型
    # 假设输入的batch size为32，序列长度为5，输入特征的维度为10
    x = torch.randn(32, 5, input_size)  # Batch size=32, sequence length=5, input size=10
    output = model(x)

    # 打印输出的形状
    print(output.shape)  # 应该是 (32, output_size)
    # summary(model, input_size=(32, 5, 10))  # 输入尺寸: (sequence_length, input_size)

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) 
    return vocab

stoi = build_vocab()
itos = {v: k for k, v in stoi.items()}

#随机生成一个样本
#在所有字符中选取sentence_length个字
#在其中随机插入一个 unk 字符, 标签为 unk 的位置
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    choice = [i for i in range(list(itos.keys()).__len__() - 1)]
    x = [random.choice(choice) for _ in range(sentence_length-1)]
    y = random.randint(0, sentence_length-1)
    x.insert(y, vocab['unk'])
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#训练模型 dataset: (x, y) evaluate_dataset: (x, y)
def train(model, dataset, epochs=10, batch_size=320, evaluate_dataset=None, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_data, y_data = dataset
    for epoch in range(epochs):
        loss_sum = 0
        for data in range(0, len(x_data[0]), batch_size):
            optimizer.zero_grad()
            x, y = x_data[data:data+batch_size], y_data[data:data+batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print("epoch:{}, loss:{}".format(epoch, loss_sum / (len(x_data) // batch_size)), end="  ")
        print("acc:", evaluate(model, evaluate_dataset))

def evaluate(model, evaluate_dataset):
    x, y = evaluate_dataset
    outs = model(x)
    right_num = 0
    for output, target in zip(outs, y):
        if torch.argmax(output) == target:
            right_num += 1
    return right_num / 20000


def main_train():
    hidden_size = 32
    sentence_length = 10
    vocab_size = len(stoi)
    vector_dim = 16
    output_size = sentence_length
    model = NlpRNN(vector_dim, vocab_size, output_size, hidden_size_rnn=hidden_size, hidden_size_fc=hidden_size)
    model = model.cuda()

    train_dataset = build_dataset(320*1000, stoi, sentence_length)
    train_dataset = (train_dataset[0].cuda(), train_dataset[1].cuda())
    evaluate_dataset = build_dataset(20000, stoi, sentence_length)
    evaluate_dataset = (evaluate_dataset[0].cuda(), evaluate_dataset[1].cuda())
    for i in range(10):
        print(f"第{i}次训练")
        train(model, dataset=train_dataset, evaluate_dataset=evaluate_dataset)
    acc = evaluate(model, evaluate_dataset)
    torch.save(model.state_dict(), f"model/model_nlp_{acc}.pth")

main_train()