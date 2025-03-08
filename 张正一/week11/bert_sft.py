import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import json
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

class Loader:
    def __init__(self, title_max_length, content_max_length):
        self.bert_pretrained_path = r'C:\Users\81080\Documents\Python_Study\AI\week06\bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrained_path)
        # self.max_length = max_length
        self.load(title_max_length, content_max_length)
    
    def encode_sentence(self, sentence, max_length):
        ids = self.tokenizer.encode(sentence, max_length=max_length, add_special_tokens=True, padding='max_length')
        return ids

    def load(self, title_max_length, content_max_length):
        with open('sample_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            sep_token = self.tokenizer.sep_token
            self.data = []
            for line in data:
                title, content = line['title'], line['content']
                # input_ids = self.encode_sentence(title + ' ' + sep_token + ' ' + content)
                input_title_ids = self.encode_sentence(title, title_max_length)
                input_content_ids = self.encode_sentence(content, content_max_length)
                input_ids = input_title_ids + input_content_ids
                self.data.append(input_ids)

    def build_sentence(self, sentence, window_size, title_max_length):
        title_ids = sentence[:title_max_length]
        input_title_start_index = np.random.randint(0, len(title_ids) - window_size)
        input_title_end_index = input_title_start_index + window_size
        input_title_ids = title_ids[input_title_start_index:input_title_end_index]
        
        content_ids = sentence[title_max_length:]
        input_content_start_index = np.random.randint(0, len(content_ids) - 1 - window_size)
        input_content_end_index = input_content_start_index + window_size
        input_content_ids = content_ids[input_content_start_index:input_content_end_index] # 问题部分不移位
        output_content_ids = content_ids[input_content_start_index + 1: input_content_end_index + 1] # 答案部分移位
        
        input_id = input_title_ids + input_content_ids
        output_id = [-100] * window_size + output_content_ids # 问题部分不计算loss
        return input_id, output_id
    
    def build_dataset(self, batch_size, window_size, title_max_length):
        start_index = np.random.randint(0, len(self.data) - batch_size)
        end_index = start_index + batch_size
        input_ids = []
        output_ids = []
        for i, sentence in enumerate(self.data[start_index:end_index]):
            input_id, output_id = self.build_sentence(sentence, window_size, title_max_length)
            input_ids.append(input_id)
            output_ids.append(output_id)
        return torch.LongTensor(input_ids), torch.LongTensor(output_ids)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def generate_sentence(model, start_sentence):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(start_sentence) <= 30:
            start_sentence += pred_char
            input_ids = model.tokenizer.encode(start_sentence, add_special_tokens=False)
            input_ids = torch.LongTensor([input_ids])
            # print(54, input_ids)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            output = model(input_ids)
            # print(66, output, output.shape)
            # print(67, output[0])
            # print(68, output[0][-1])
            # print(58, output)
            output_id = sampling_strategy(output[0][-1])
            # print(60, output_ids)
            output_char = model.tokenizer.decode(output_id)
            # print(76, ''.join(output_char))
            pred_char = ''.join(output_char)
            # print(62, output_sentence)
    return start_sentence

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


class BERT_SFT(nn.Module):
    def __init__(self, batch_size, hidden_size, dropout=0.1):
      super().__init__()
      self.bert_preetrained_path = r'C:\Users\81080\Documents\Python_Study\AI\week06\bert-base-chinese'
      self.bert = BertModel.from_pretrained(self.bert_preetrained_path, return_dict=False)
      self.tokenizer = BertTokenizer.from_pretrained(self.bert_preetrained_path)
      self.classify = nn.Linear(hidden_size, len(self.tokenizer.vocab))
      self.dropout = nn.Dropout(dropout)
      self.loss = nn.CrossEntropyLoss(ignore_index=-100)

      
    def forward(self, x, y=None):
        if y is not None:
            ones = torch.ones((x.shape[0], x.shape[1] // 2)) # 全1矩阵 title部分
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1] // 2)))  # 下三角矩阵 content部分
            mask = torch.cat((ones, mask), dim=1) # 拼接成mask梯形矩阵，问题可以看到自己，答案只能看到前面的内容
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)  # [batch_size, seq_len, hidden_size]
            y_pred = self.classify(x)  # [batch_size, seq_len, vocab_size]
            # y_pred.view(-1, len(self.tokenizer.vocab)) [batch_size * seq_len, vocab_size]  y.view(-1).shape [batch_size * seq_len]
            return self.loss(y_pred.view(-1, len(self.tokenizer.vocab)), y.view(-1))
        else:
            # 预测不用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)
    
    
    
def train():
    epochs = 50
    char_dim = 768
    batch_size = 80
    window_size = 10
    title_max_length = 50
    content_max_length = 150
    model = BERT_SFT(batch_size, char_dim)
    loader = Loader(title_max_length, content_max_length)
    train_data = loader.data
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch in range(len(train_data) // batch_size):
            input_ids, output_ids = loader.build_dataset(batch_size, window_size, title_max_length)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                output_ids = output_ids.cuda()
            optimizer.zero_grad()
            loss = model(input_ids, output_ids)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f'Epoch {epoch+1} loss: {np.mean(watch_loss)}')
        print(generate_sentence(model, '阿根廷歹徒抢服装尺码不对拿回店里换'))
        print(generate_sentence(model, '邓亚萍：互联网要有社会担当'))
    
if __name__ == '__main__':
    train()
