import json
import random
import torch
import jieba
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# 数据加载器
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_triplet_sample()  # 生成三元组样本
        else:
            return self.data[index]

    # 生成三元组样本 (Anchor, Positive, Negative)
    def random_triplet_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 随机选择一个类别作为 Anchor 和 Positive 的来源
        anchor_label = random.choice(standard_question_index)
        # 确保该类别的样本数量足够
        if len(self.knwb[anchor_label]) < 2:
            return self.random_triplet_sample()
        # 随机选择 Anchor 和 Positive
        anchor, positive = random.sample(self.knwb[anchor_label], 2)
        # 随机选择一个不同的类别作为 Negative 的来源
        negative_label = random.choice(standard_question_index)
        while negative_label == anchor_label:
            negative_label = random.choice(standard_question_index)
        negative = random.choice(self.knwb[negative_label])
        return [anchor, positive, negative]


# 加载词汇表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0 留给 padding 位置，所以从 1 开始
    return token_dict


# 加载标签映射
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用 torch 自带的 DataLoader 类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


# 定义 Triplet Loss
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算 Anchor 和 Positive 的距离
        distance_positive = F.pairwise_distance(anchor, positive)
        # 计算 Anchor 和 Negative 的距离
        distance_negative = F.pairwise_distance(anchor, negative)
        # 计算 Triplet Loss
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


# 定义文本匹配模型
class TextMatchingModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextMatchingModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)  # 对序列进行平均池化
        output = self.fc(pooled)
        return output


# 训练函数
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        # 获取模型的输出
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        # 计算 Triplet Loss
        loss = criterion(anchor_output, positive_output, negative_output)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


# 主程序
if __name__ == "__main__":
    from config import Config

    # 加载配置
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loader = load_data("train_data.json", config, shuffle=True)
    test_loader = load_data("test_data.json", config, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = TextMatchingModel(config["vocab_size"], config["embedding_dim"], config["hidden_dim"]).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 训练模型
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {train_loss:.4f}")
