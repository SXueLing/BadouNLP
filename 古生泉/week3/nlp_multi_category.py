import torch
import torch.nn as nn
import random
import jieba
import numpy as np
from pyecharts.charts import Line
import pyecharts.options as opts
import json

# 构建数据集
"""
数据集类型：X  -> [语句1,语句2,....] ->[[词11, 词12...],[词21, 词22...]...]
          Y  -> [分类1,分类N,....]
"""


def data_set(data_num):
    data_x = []
    data_y = []
    sport_list = ["足球", "篮球", "运动员", "比赛", "训练", "冠军", "场馆", "球迷", "战术"]
    recreat_list = ["电影", "明星", "歌手", "演唱会", "电视剧", "红毯", "角色", "剧情", "粉丝"]
    internet_list = ["人工智能", "电子产品", "互联网", "软件", "创新", "研发", "技术", "数据", "算法"]

    for _ in range(data_num):
        rand = random.randint(0, 2)
        if rand == 0:
            data_sport_x = ''.join(random.sample(sport_list, random.randint(4, 6)))
            data_sport_x = list(jieba.cut(data_sport_x))
            data_x.append(data_sport_x)
            data_y.append(0)
        if rand == 1:
            data_sport_x = ''.join(random.sample(recreat_list, random.randint(4, 6)))
            data_sport_x = list(jieba.cut(data_sport_x))
            data_x.append(data_sport_x)
            data_y.append(1)
        if rand == 2:
            data_sport_x = ''.join(random.sample(internet_list, random.randint(4, 6)))
            data_sport_x = list(jieba.cut(data_sport_x))
            data_x.append(data_sport_x)
            data_y.append(2)
    return data_x, data_y


# 构建词汇表
def data_vocab(data_set_x):
    data_vocab_set = {}
    data_vocab_set['unk'] = 0
    indx = 1
    for sentence in data_set_x:
        for word in sentence:
            if word not in data_vocab_set:
                data_vocab_set[word] = indx
                indx += 1
    return data_vocab_set


def sentence_vocab_idx(sentence, data_vocab):
    return [data_vocab.get(sent, 0) for sent in sentence]


def collate_fn(data_set_x, data_set_y, data_vocab):
    max_dim = max((len(data_x) for data_x in data_set_x), default=0)
    data_x = []
    for sentence in data_set_x:
        sent = sentence_vocab_idx(sentence, data_vocab)
        zero = torch.zeros(1, max_dim, dtype=int).squeeze(0).tolist()
        zero[: len(sent)] = sent
        data_x.append(zero)
    return torch.LongTensor(data_x), torch.LongTensor(data_set_y)


# 定义模型
class RnnModel(nn.Module):
    def __init__(self, data_vocab, emb_dim, hidden_size):
        super(RnnModel, self).__init__()
        self.emb = nn.Embedding(len(data_vocab), emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)
        self.linre = nn.Linear(hidden_size, 3)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.emb(x)  #(语句长度, 单词长度) -> (每批语句长度, 一批语句长度, 单个embdding维度)
        x_h0, x_h1 = self.rnn(x)  # x_h0:(每批语句长度, 一批语句长度, 单个embdding维度) # x_h1 (每批语句长度,1,单个embdding维度)
        # print("x_h0:", x_h0)
        # print("x_h1:", x_h1)
        x = x_h1.squeeze(0)  #  (每批语句长度,单个embdding维度)
        y_pre = self.linre(x)  #  (每批语句长度,linear输出维度)
        if y is None:
            return y_pre
        else:
            return self.loss(y_pre, y)


# 定义验证数据
def verify(model, test_num, data_vocab_m):
    model.eval()
    wrong, sum_tatol = 0, 0
    with torch.no_grad():
        # 再生成200条验证数据
        data_test_x, data_test_y = data_set(test_num)
        # data_vocab_test_m = data_vocab(data_test_x)
        data_ten_test_x, data_ten_test_y = collate_fn(data_test_x, data_test_y, data_vocab_m)
        y_pred = model(data_ten_test_x)
        for y_t, y_p in zip(data_ten_test_y, y_pred):
            sum_tatol += 1
            # print(y_p)
            if y_t == torch.argmax(y_p):
                wrong += 1
        print(f"本次共测试{test_num},正确率为{wrong / sum_tatol}")
    return wrong / sum_tatol


# 模型预测
def predict(modet_path, vocab_path, input_strings):
    emb_dim = 10
    hidden_size = 4
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    model = RnnModel(vocab, emb_dim, hidden_size)
    model.load_state_dict(torch.load(modet_path))
    x = []
    y = []
    for input_string in input_strings:
        x_0 = list(jieba.cut(input_string))
        x.append(x_0)
        y.append(-1)
    x_1, y_1 = collate_fn(x, y, vocab)
    model.eval()
    with torch.no_grad():
        retult = model.forward(x_1)
    for i, pred in enumerate(input_strings):
        print(f"文本：{input_strings[i]},预测的比率分布{retult[i]},预测类别为：{torch.argmax(retult[i])}")


if __name__ == '__main__':
    # 训练轮数
    epcho = 5
    # 训练数据数量
    test_count = 10000
    # 每批训练数据量
    batch_count = 20

    # 先生成1万条训练数据
    data_x, data_y = data_set(test_count)
    data_vocab_m = data_vocab(data_x)
    data_ten_x, data_ten_y = collate_fn(data_x, data_y, data_vocab_m)

    model = RnnModel(data_vocab_m, 10, 4)
    # 定义优化器
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    log = []
    for epcho in range(epcho):
        model.train()
        watch_loss = []
        for batch_idx in range(test_count // batch_count):
            x = data_ten_x[batch_idx * batch_count: (batch_idx + 1) * batch_count]
            y = data_ten_y[batch_idx * batch_count: (batch_idx + 1) * batch_count]
            opt.zero_grad()
            loss = model(x, y)
            loss.backward()
            opt.step()
            watch_loss.append(loss.item())
        print(f"第{epcho + 1}轮,loss值为{np.mean(watch_loss)}")
        ver = verify(model, 100, data_vocab_m)
        log.append([ver, np.mean(watch_loss)])
    (
        Line()
        .add_xaxis(range(len(log)))
        .add_yaxis(series_name="验证成功比率", y_axis=[i[0] for i in log], label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis(series_name="loss值", y_axis=[i[1] for i in log], label_opts=opts.LabelOpts(is_show=False))
        .render("log_axis.html")
    )
    # 保存模型
    torch.save(model.state_dict(), "nlp_multi.pth")
    writer = open("nlp_vocab.json", "w", encoding="utf-8")
    writer.write(json.dumps(data_vocab_m, ensure_ascii=False, indent=2))
    writer.close()

    data_yc_x = ["运动员训练比赛战术", "歌手明星演唱会"]
    # predict
    predict("nlp_multi.pth", "nlp_vocab.json", data_yc_x)
