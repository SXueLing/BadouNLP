import time

import jieba
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn, optim

# 1. 加载并分词（使用jieba）
data = pd.read_csv('../文本分类练习.csv')  # 假设文件名为 'data.csv'
data = shuffle(data)

X = data['review']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 加载本地的Word2Vec模型
w2v_model = Word2Vec.load('./model.w2v')


# 3. 使用jieba分词
def jieba_tokenize(text):
    return list(jieba.cut(text))


# 4. 估计嵌入维度：通过词汇的向量维度来确定（这里假设嵌入维度为100）
embedding_dim = 100  # 你可以根据需要调整为100, 200等


# 5. 将每个句子转换为嵌入向量（取平均词向量）
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
                # vector = np.max([vector, model.wv[word]], axis=0)
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def sentence_to_vec(sentence):
    words = jieba_tokenize(sentence)
    vecs = torch.zeros(w2v_model.vector_size)
    for word in words:
        try:
            vecs += torch.tensor(w2v_model.wv[word])
        except KeyError:
            vecs += torch.zeros(w2v_model.vector_size)

    return vecs / len(words) if len(words) > 0 else vecs

X_train_vec = torch.stack([sentence_to_vec(text) for text in X_train])
X_test_vec = torch.stack([sentence_to_vec(text) for text in X_test])


# 6. 定义 RNN 和 LSTM 模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])  # 使用最后一个时间步的输出
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
        return output


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # 将输入映射到隐藏空间
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出层，二分类

    def forward(self, x):
        # 输入的 x 形状: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # x 变为 [batch_size, seq_len, hidden_dim]
        x = x.permute(1, 0, 2)  # 改变为 [seq_len, batch_size, hidden_dim]，符合 Transformer 的输入要求
        transformer_out = self.transformer(x)  # Transformer 编码
        output = transformer_out[-1, :, :]  # 使用最后一个时间步的输出
        output = self.fc(output)  # 输出二分类结果 [batch_size, 2]
        return output


# 7. 训练模型
def train_model(model, train_data, test_data, learning_rate=0.001, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.to("cuda").train()
        optimizer.zero_grad()
        outputs = model(train_data[0])
        loss = criterion(outputs, train_data[1])
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

    # 评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(test_data[0])
        _, preds = torch.max(outputs, 1)
        accuracy = accuracy_score(test_data[1].numpy(), preds.numpy())

    return model, accuracy, train_time


# 8. 训练 RNN 和 LSTM 模型
rnn_model = RNNModel(input_dim=embedding_dim, hidden_dim=128, output_dim=2)
lstm_model = LSTMModel(input_dim=embedding_dim, hidden_dim=128, output_dim=2)

rnn_model, rnn_accuracy, rnn_train_time = train_model(rnn_model, train_data=(
    X_train_vec.unsqueeze(1), torch.tensor(y_train.values)),
                                                      test_data=(X_test_vec.unsqueeze(1), torch.tensor(y_test.values)))
lstm_model, lstm_accuracy, lstm_train_time = train_model(lstm_model, train_data=(
    X_train_vec.unsqueeze(1), torch.tensor(y_train.values)), test_data=(
    X_test_vec.unsqueeze(1), torch.tensor(y_test.values)))
transformer_model = TransformerModel(input_dim=w2v_model.vector_size, hidden_dim=128, output_dim=2)

transformer_model, transformer_accuracy, transformer_train_time = train_model(
    transformer_model,
    train_data=(X_train_vec.unsqueeze(1), torch.tensor(y_train.values)),
    test_data=(X_test_vec.unsqueeze(1), torch.tensor(y_test.values))
)
# 9. 输出总结表格
results = [
    ["RNN", 0.001, rnn_accuracy, rnn_train_time],
    ["LSTM", 0.001, lstm_accuracy, lstm_train_time],
["transformer", 0.001, transformer_accuracy, transformer_train_time]
]

# 输出结果表格
results_df = pd.DataFrame(results, columns=["Model", "Learning Rate", "Accuracy", "Training Time (seconds)"])
print(results_df)
