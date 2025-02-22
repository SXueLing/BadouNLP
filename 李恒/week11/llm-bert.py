import json
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader


class QAModel(torch.nn.Module):
    def __init__(self, _bert_path_chinese):
        super(QAModel, self).__init__()
        # 加载BERT模型
        bert_model = BertModel.from_pretrained(_bert_path_chinese, return_dict=False, attn_implementation='eager')
        self.bert = bert_model
        # 分类器：输出一个类别（start和end位置）
        hidden_size = bert_model.config.hidden_size
        vocab_size = self.bert.config.vocab_size
        # print("hidden size:", hidden_size)
        # print("vocab size:", vocab_size)
        self.classify = torch.nn.Linear(hidden_size, vocab_size)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None, mask=None):
        if y is not None:
            # 训练时，前向传播BERT
            # print(mask.shape)
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


class QADataset(Dataset):
    def __init__(self, data_path, _tokenizer, max_length):
        self.tokenizer = _tokenizer
        self.max_length = max_length
        self.questions = []
        self.answers = []

        with open(data_path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                title = line["title"]
                _context = line["content"]
                self.questions.append(title)
                self.answers.append(_context)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        _question = self.questions[idx]
        _answer = self.answers[idx]
        prompt_encode = self.tokenizer.encode(_question, add_special_tokens=False)
        answer_encode = self.tokenizer.encode(_answer, add_special_tokens=False)

        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))

        # padding
        x = x[:self.max_length] + [0] * (self.max_length - len(x))
        y = y[:self.max_length] + [0] * (self.max_length - len(y))
        x, y = torch.LongTensor(x), torch.LongTensor(y)
        mask = pad_mask(mask, (self.max_length, self.max_length))

        return x, y, mask


# 构造掩码，输入两个字符串的长度
def create_mask(s1, s2):
    len_s1 = s1 + 2  # cls + sep
    len_s2 = s2 + 1  # sep
    # 创建掩码张量
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # 遍历s1的每个token
    for i in range(len_s1):
        # s1的当前token不能看到s2的任何token
        mask[i, len_s1:] = 0
        # 遍历s2的每个token
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2 token
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask


def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result


# 文本生成测试代码
def generate_sentence(openings, model, _tokenizer, max_length):
    model.eval()
    openings = _tokenizer.encode(openings)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    with torch.no_grad():
        # 生成文本超过30字则终止迭代
        while len(openings) <= max_length:
            x = torch.LongTensor([openings])
            x = x.to(device)
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return _tokenizer.decode(openings)


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train_model(_bert_path_chinese, _tokenizer, save_weight=False):
    # 定义超参数
    epoch_num = 10  # 训练轮数
    batch_size = 32
    learning_rate = 1e-5
    max_length = 50

    # 创建数据集和DataLoader
    train_data_path = r"data/train_tag_news.json"
    dataset = QADataset(train_data_path, _tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = QAModel(_bert_path_chinese)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)

    # 初始化优化器
    optim = AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y, mask in dataloader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            # 前向传播
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, _tokenizer, max_length))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, _tokenizer, max_length))
    if not save_weight:
        return
    else:
        base_name = f"bert-llm-{epoch_num}.pth"
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # 加载BERT tokenizer和模型
    bert_path_chinese = r"/Users/liheng/Downloads/my/百度网盘/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(bert_path_chinese)
    # 开始训练
    train_model(bert_path_chinese, tokenizer, True)



