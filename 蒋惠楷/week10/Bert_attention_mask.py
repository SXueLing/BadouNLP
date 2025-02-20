#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import logging
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
基于pytorch的Bert_attention_mask语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        config = BertConfig.from_pretrained("E:\\AIGC\\NLP算法\\【9】序列标注\\课件\\ner_bert\\ner_bert\\bert-base-chinese")
        # config.is_decoder = True  # 将模型配置为解码器，自动启用 causal mask
        config.position_embedding_type = "relative_key_query"  # 使用相对位置编码
        self.bert = BertModel.from_pretrained("E:\\AIGC\\NLP算法\\【9】序列标注\课件\\ner_bert\\ner_bert\\bert-base-chinese", config=config)
        self.hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(self.hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy


    def forward(self, x, y=None):
        device = x.device
        batch_size, seq_len = x.shape

        # 构造标准的 padding mask，Bert默认 pad token 的 id 为 0
        attention_mask = (x != 0).long()  # [batch_size, seq_len]，自动启动causal mask则不需要以下代码

        attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        # 构造 causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
        # 组合 mask
        final_mask = attention_mask & (~causal_mask)  # [batch_size, seq_len, seq_len]
        
        outputs = self.bert(x, attention_mask=final_mask)
        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)
        y_pred = self.classify(hidden_states[:, -1, :])  # 预测最后一个位置

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1), ignore_index=0)
        else:
            # return torch.softmax(y_pred, dim=-1)
            return y_pred

#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def encode_sentence_Bert(text, max_length=10):
    """ 使用 BertTokenizer 进行编码，固定 max_length，填充/截断 """
    return tokenizer.encode(text, 
                            add_special_tokens=False,  # 不加 [CLS] 和 [SEP]
                            max_length=max_length,     # 固定长度
                            padding="max_length",      # 填充到 max_length
                            truncation=True)           # 超出 max_length 就截断

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    x = encode_sentence_Bert(window, max_length=10)
    y = encode_sentence_Bert(target, max_length=10)
    return x, y[-1]

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = encode_sentence_Bert(openings[-window_size:], max_length=window_size)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0]
            # y = int(torch.argmax(y)) # 不使用采样策略则直接返回权重最大位置（贪婪搜索）
            y = sampling_strategy_1(y)
            pred_char = tokenizer.decode([y])
    return openings

# 采样策略1：温度采样-topk采样
def sampling_strategy_1(y_pred, temperature=0.8, top_k=50):
    y_pred = y_pred / temperature
    top_k = min(top_k, y_pred.size(-1))
    if top_k > 0:
        # 筛选出概率前top_k个token
        kth_value = torch.topk(y_pred, top_k)[0][..., -1, None]
        indices_to_remove = y_pred < kth_value
        y_pred = y_pred.masked_fill(indices_to_remove, -float('Inf'))
    probs = torch.softmax(y_pred, dim=-1)    

    # 随机采样一个 token
    next_token = torch.multinomial(probs, 1).item()

    # 如果采样的是 [UNK]，选择概率第二大的 token
    unk_token_id = tokenizer.unk_token_id
    if next_token == unk_token_id:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        next_token = sorted_indices[1].item()  # 选择概率第二大的 token
    # # 获取排序信息（从高到低）
    # sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # print("\n=== Sorted predictions ===")
    # for i in range(min(10, len(sorted_indices))):  # 只打印前10个最高概率的词
    #     token_id = sorted_indices[i].item()
    #     token_prob = sorted_probs[i].item()
    #     print(f"Rank {i+1}: Token {token_id} (p={token_prob:.5f})")

    # # 采样
    # next_token = torch.multinomial(probs, 1).item()
    # print(f"\n=== Selected token ===\nToken {next_token} (Decoded: {tokenizer.decode([next_token])})")
    return next_token

# 采样策略2：贪婪搜索-随机采样
def sampling_strategy_2(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = torch.softmax(prob_distribution, dim=-1).cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

#计算文本ppl
def calc_perplexity(sentence, model, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = encode_sentence_Bert(window, max_length=window_size)
            x = torch.LongTensor([x])
            target = sentence[i]
            # target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0, -1]
            target_prob_id = encode_sentence_Bert(sentence[i], max_length=1)[0]
            target_prob = torch.softmax(y, dim=-1)[target_prob_id]
            prob += torch.log(target_prob)
    perplexity = torch.exp(-prob / len(sentence))
    return perplexity

def train(corpus_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 80000   #每轮训练总共训练的样本总数
    # char_dim = 256        #每个字的维度
    window_size = 10       #样本文本长度
    # vocab = build_vocab("vocab.txt")       # 建立字表
    corpus = load_corpus(corpus_path)     # 加载语料
    vocab_size = tokenizer.vocab_size     # 加载 BertTokenizer 自带的词表大小
    model = LanguageModel(vocab_size)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01) #建立优化器
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=500,
        num_training_steps=epoch_num * (train_sample//batch_size)
    )
    # optim = torch.optim.Adam(model.parameters(), lr=2e-5)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
            scheduler.step()
        print("=========================")
        logging.info("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("E:\\AIGC\\NLP算法\\【9】序列标注\课件\\ner_bert\\ner_bert\\bert-base-chinese") # 加载bert分词器
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", True)
