import random
import torch
import torch.nn as nn
import json
import numpy as np
from transformers import BertModel, BertTokenizer

"""
在Bert预训练模型的基础上实现SFT训练
"""

seed = 987
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def load_sample(data_path, tokenizer):
    data = []
    with open(data_path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            x = title + "[SEP]" + content
            y = title + content + "[SEP]"
            x = tokenizer.encode(x, add_special_tokens=False, truncation=True, max_length=50, padding='max_length')
            y = tokenizer.encode(y, add_special_tokens=False, truncation=True, max_length=50, padding='max_length')
            mask = torch.ones(50, 50)
            mask[:len(title), len(title):] = 0
            tril_matrix = torch.tril(torch.ones(50-len(title), 50-len(title)))
            mask[len(title):, len(title):] = tril_matrix
            for i in range(len(title)):
                y[i] = -1
            data.append([x, y, mask])
    return data

def build_dataset(sample_nums, data):
    X = []
    Y = []
    masks = []
    for _ in range(sample_nums):
        x, y, mask = random.choice(data)
        X.append(x)
        Y.append(y)
        masks.append(mask)
    return torch.LongTensor(X), torch.LongTensor(Y), torch.stack(masks, dim=0)

class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        vocab_size = len(vocab)
        self.bert = BertModel.from_pretrained(r'E:\BaiduNetdiskDownload\八斗精品课\第六周 语言模型\bert-base-chinese', return_dict=False)
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, mask=None, y=None):
        if mask is not None:
            x, _ = self.bert(x, attention_mask=mask)
        else:
            x, _ = self.bert(x)
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

def train():
    epoch_nums = 20
    train_nums = 10000
    batch_size = 128
    lr = 1e-3
    window_size = 10
    tokenizer = BertTokenizer.from_pretrained(r'E:\BaiduNetdiskDownload\八斗精品课\第六周 语言模型\bert-base-chinese')
    vocab = tokenizer.vocab
    data = load_sample(r'E:\BaiduNetdiskDownload\八斗精品课\第十周 文本生成\week10 文本生成问题\transformers-生成文章标题\sample_data.json', tokenizer)
    model = LanguageModel(vocab)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epoch_nums):
        model.train()
        train_loss = []
        for idx in range(train_nums//batch_size):
            x, y, mask = build_dataset(batch_size, data)
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()
            loss = model(x, mask, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            # if idx % ((train_nums//batch_size)//2) == 0:
            #     print("batch loss: %f" % loss.item())
        print("========\n第%d轮epoch的loss: %f" % (epoch + 1, np.mean(train_loss)))
        print(generate_text("北美洲发现肥皂人", tokenizer, model, window_size))

def generate_text(openings, tokenizer, model, window_size):
    model.eval()
    pred_char = ""
    while pred_char != "\n" and len(openings) <= 30:
        openings += pred_char
        window = openings[-window_size:]
        window = tokenizer.encode(window, add_special_tokens=False)
        window = torch.LongTensor([window])
        if torch.cuda.is_available():
            window = window.cuda()
        with torch.no_grad():
            pred = model(window)[0][-1]
        index = sample_strategy(pred)
        pred_char = tokenizer.decode(index)
    return openings

def sample_strategy(prob):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "random"

    if strategy == "greedy":
        return int(torch.argmax(prob, dim=-1))
    else:
        prob = prob.cpu().detach().numpy()
        return np.random.choice(list(range(len(prob))), p=prob)     # 切记这里的prob要为numpy格式

if __name__ == "__main__":
    train()
