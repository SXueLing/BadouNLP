import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class MyNLPModel(nn.Module):
  # vector_dim:每个字符的向量维度
  # sentence_length:字符串长度
  # vocab:字符集长度
  def __init__(self, vector_dim, sentence_length, vocab):
    super(MyNLPModel, self).__init__()
    self.embeddiing = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
    self.rnn = nn.RNN(input_size=vector_dim, hidden_size = 2 * vector_dim, bias=True, batch_first=True)
    self.linear = nn.Linear(2 * vector_dim, sentence_length)
    self.loss = nn.functional.cross_entropy

  def forward(self, x, y=None):
    # print(20, x, x.shape)
    x = self.embeddiing(x) # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
    # print(22, x, x.shape)
    # 经过RNN层后，返回一个元组，这个元组包含两个元素，第一个是(batch_size, sen_len, 2 * vector_dim)，包含中间所有的输出； 
    # 第二个是(1, batch_size, 2 * vector_dim)，是最终输出，取元组中的第二个最终输出继续往后计算
    x = self.rnn(x)[-1] # (batch_size, sen_len, vector_dim) -> (1, batch_size, 2 * vector_dim)
    # print(26, x, x[0].shape, x[1].shape)
    x = x.squeeze() # (1, batch_size, 2 * vector_dim) -> (batch_size, 2 * vector_dim)
    # print(28, x, x.shape)
    y_pred = self.linear(x) # (batch_size, 2 * vector_dim) -> (batch_size, sen_len)
    # print(32, y_pred)
    if y is not None:
      return self.loss(y_pred, y)
    else:
      return y_pred

# 生成字符集
def build_vocab():
  chars = '你我他defghijklmnopqrstuvwxyz'
  vocab = {'pad': 0}
  for index, char in enumerate(chars):
    vocab[char] = index + 1
  vocab['unk'] = len(vocab)
  return vocab

# 随机生成样本
def build_sample(vocab, sentence_length):
  x = []
  while not (set('你') & set(x)):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
  index = x.index('你')
  x = [vocab.get(word, vocab['unk']) for word in x]
  y = index
  return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
  dataset_x = []
  dataset_y = []
  for i in range(sample_length):
    x, y = build_sample(vocab, sentence_length)
    dataset_x.append(x)
    dataset_y.append(y)
  return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
  model = MyNLPModel(char_dim, sentence_length, vocab)
  return model

# 每一轮测试的准确率
def evaluate(model, vocab, sentence_length):
  model.eval()
  x, y = build_dataset(200, vocab, sentence_length)
  # print(f'本次预测集有{sum(y)}个正样本，{200 - sum(y)}个负样本')
  correct, wrong = 0, 0
  with torch.no_grad():
    y_pre = model.forward(x)
    for y_P, y_T in zip(y_pre, y):
      _, p = torch.max(y_P, dim=0)
      _, t = torch.max(y_T, dim=0)
      # print(76, y_P, p, y_T)
      if p == y_T:
          correct += 1
      else:
          wrong += 1
  print(f'正确预测个数：{correct}，正确率：{correct / (correct + wrong)}')
  return correct/(correct + wrong)


def main():
  # 训练轮数
  epoch_num = 10
  # 每次训练样本个数
  batch_size = 20
  # 每轮总共训练样本总数
  train_sample = 500
  # 每个字符的维度
  char_dim = 20
  # 样本文本长度
  sentence_length = 8
  # 学习率
  learning_rate = 0.005
  # 建立字表
  vocab = build_vocab()
  # 建立模型
  model = build_model(vocab, char_dim, sentence_length)
  # 选择优化器
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
  # 记录loss和acc数据
  log = []
  for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    for batch in range(train_sample // batch_size):
      x, y = build_dataset(batch_size, vocab, sentence_length)
      optim.zero_grad()
      loss = model(x, y)
      loss.backward()
      optim.step()
      watch_loss.append(loss.item())
    print(f'第{epoch+1}轮loss：{np.mean(watch_loss)}')
    acc = evaluate(model, vocab, sentence_length)
    log.append([acc, np.mean(watch_loss)])
  # 画图
  plt.plot(range(len(log)), [l[0] for l in log], label="acc")
  plt.plot(range(len(log)), [l[1] for l in log], label="loss")
  plt.legend()
  plt.show()
  # 保存模型
  torch.save(model.state_dict(), "model.pth")
  # 保存词表
  writer = open('vocab.json', 'w', encoding='utf-8')
  writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
  writer.close()
  return

def predict(model_path, vocab_path, input_strings):
  # 每个字的维度
  char_dim = 20
  # 样本文本长度
  sentence_length = 8
  # 加载字符集
  vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
  # 建立模型
  model = MyNLPModel(char_dim, sentence_length, vocab)
  # 加载训练好的权重
  model.load_state_dict(torch.load(model_path))

  x = []
  for input_string in input_strings:
    x.append([vocab[char] for char in input_string])
  # 测试模式
  model.eval()
  print(158, x)
  with torch.no_grad():
    # 模型预测
    result = model.forward(torch.LongTensor(x))
  for input_string, res in zip(input_strings, result):
    y_pre = torch.max(res, dim=0)[1]
    print(f'输入：{input_string}，预测类别：{y_pre}')


if __name__ == '__main__':
  main()
  test_strings = ["fnvf你eop", "wz你dfgop", "你qwdeonp", "n你kwwwop"]
  predict("model.pth", "vocab.json", test_strings)
