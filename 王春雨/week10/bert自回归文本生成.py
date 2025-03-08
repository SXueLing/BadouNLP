#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from transformers import BertTokenizer
"""
基于pytorch的LSTM语言模型
"""
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),  # 输出到日志文件
                        logging.StreamHandler()  # 输出到控制台
                    ])

#bert_path =  r"D:\八斗\课件\第六周 语言模型\bert-base-chinese"

bert_path =  r"D:\八斗\课件\第六周 语言模型\bert-base-chinese"
#voc_path  =  r"vocab.txt"
text_path =  r"corpus.txt"

tokenizer = BertTokenizer.from_pretrained(bert_path)

class LanguageModel(nn.Module):
    def __init__(self,input_dim):
        super(LanguageModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert  = BertModel.from_pretrained(bert_path,return_dict=False)
        self.bert.config.s_decoder = True
        self.bert.config.add_cross_attention = False
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.bert.config.max_position_embeddings, 
                      self.bert.config.max_position_embeddings)))
        self.classify = nn.Linear(input_dim, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)  #loss采用交叉熵损失


    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None,attention_mask=None):
        seq_len = x.size(1)
        # 生成注意力掩码
        causal_attention_mask = self.causal_mask[:seq_len, :seq_len]
        sequence_output, pooler_output = self.bert(x,
                                                   attention_mask=attention_mask,
                                                   encoder_attention_mask=causal_attention_mask)
        if y is not None:
            y_pred = self.classify(sequence_output)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            y_pred = self.classify(sequence_output[:, -1, :])  # 最后一个位置预测
            return y_pred

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    
    return corpus.replace(" ", "")

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample( window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
  
    encodingx =  tokenizer.encode_plus(window,  max_length=window_size+2, padding='max_length',add_special_tokens=True) #将字转换成序号
    encodingy =  tokenizer.encode_plus(target,  max_length=window_size+2, padding='max_length',add_special_tokens=True) #将字转换成序号
    
    # 获取input_ids和attention_mask
    x = encodingx['input_ids']
    # 使用列表推导式生成 attention_mask
    #attention_mask = [1 if id != 0 else 0 for id in x]
    attention_mask = encodingx['attention_mask']
    y = encodingy['input_ids']
    # 掩码处理（忽略特殊标y记）x
    y = torch.where(torch.tensor(x) == tokenizer.pad_token_id, 
                       -100, torch.tensor(y)).tolist()  # -100被CrossEntropy忽略

    return x, y,attention_mask

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length,  window_size, corpus):
    dataset_x = []

    dataset_y = []
    attention_mask_x = []
    for i in range(sample_length):
        x, y,attention_mask = build_sample( window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_mask_x.append(attention_mask)

        # 调试信息
        #print(f"x length: {len(x)}, y length: {len(y)}, attention_mask length: {len(attention_mask)}")
       

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y),torch.LongTensor(attention_mask_x)

#建立模型
def build_model(input_dim):
    model = LanguageModel(input_dim)
    return model




def generate_sentence( prompt,model, max_length=50, temperature=1.0, top_k=50):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    for _ in range(50):
        with torch.no_grad():
            logits = model(input_ids)
            
        # 采样策略
        logits = logits / temperature
        top_k = min(top_k, logits.size(-1))
        top_logits, top_indices = logits.topk(top_k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        
        next_token = top_indices[0, torch.multinomial(probs[0], num_samples=1)]
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token == tokenizer.sep_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


#文本生成测试代码
def generate_sentence1(openings, model, window_size):
    logging.info("generate_sentence=========")
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            inputs =  tokenizer.encode_plus(openings[-window_size:], return_tensors='pt', max_length=window_size+2, padding='max_length',add_special_tokens=True) #将字转换成序号
            x = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x,attention_mask=attention_mask)#[0][-1]
            predicted_token = tokenizer.convert_ids_to_tokens(y.item())
            # 将tokens组合成字符串
            pred_char = tokenizer.convert_tokens_to_string(predicted_token)
    return openings



def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 500   #每轮训练总共训练的样本总数
    char_dim = 768        #每个字的维度
    window_size = 10       #样本文本长度
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y,attention_mask = build_dataset(batch_size, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y,attention_mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        logging.info("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        logging.info(generate_sentence("让他在半年之前，就不能做出 [MASK] ", model,  window_size))
        logging.info(generate_sentence("李慕站在山路上，深深的呼吸 [MASK]", model,  window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train(text_path, False)
