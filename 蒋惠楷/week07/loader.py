import pandas as pd
import jieba
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split
from config import *
from transformers import BertTokenizer, XLNetTokenizer

class TextTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.pad_token = '<PAD>'
        self.pad_index = vocab[self.pad_token]

    def text_to_sequence(self, text):
        # 使用 jieba 对文本进行分词
        tokens = list(jieba.cut(text))
        sequence = [self.vocab.get(word, self.vocab.get('<UNK>', 0)) for word in tokens]  # 未在词汇表中的词将映射到索引 0（UNK）
        return sequence

class DataProcessor:
    def __init__(self, file_path, vocab=None, model_type=None, max_length=MAX_LENGTH):
        self.file_path = file_path
        self.model_type = model_type
        self.max_length = max_length
        self.tokenizer = TextTokenizer(vocab) if vocab else None  # 如果没有传入vocab则为None
        self.texts, self.labels = self.load_data(file_path)

        if model_type == 'Bert':
            self.bert_xlnet_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        elif model_type == 'XLNet':
            self.bert_xlnet_tokenizer = XLNetTokenizer.from_pretrained("hfl/chinese-xlnet-base")
    
    def load_data(self, file_path):
        """加载数据"""
        df = pd.read_csv(file_path)
        return df['review'], df['label']

    def build_vocab(self, texts, min_freq=1):
        """构建词汇表"""
        all_words = []
        for text in texts:
            words = jieba.cut(text)  # 使用 jieba 分词
            all_words.extend(words)

        # 统计词频
        word_freq = Counter(all_words)

        # 构建词汇表，只保留出现频率大于 min_freq 的单词
        vocab = {word: idx+1 for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
        vocab['<PAD>'] = 0  # 添加填充符（PAD）
        return vocab
    
    def encode_data_bert_xlnet(self, texts, labels):
        """BERT and XLNet 数据预处理"""
        inputs = self.bert_xlnet_tokenizer(list(texts), truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        inputs['labels'] = torch.tensor(list(labels)).long()
        return inputs
    
    def encode_data_lstm(self, texts, labels):
        """LSTM 数据预处理"""
        input_ids = []
        for text in texts:
            tokens = self.tokenizer.text_to_sequence(text)  # 获取文本的词索引序列
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]  # 截断
            else:
                tokens = tokens + [self.tokenizer.pad_index] * (self.max_length - len(tokens))  # 填充
            input_ids.append(tokens)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(list(labels)).long()
        return input_ids, labels

    def prepare_data(self):
        """根据模型类型选择不同的预处理方法"""
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=RANDOM_SEED)

        if self.model_type == 'Bert' or self.model_type == 'XLNet':
            train_data = self.encode_data_bert_xlnet(X_train, y_train)
            test_data = self.encode_data_bert_xlnet(X_test, y_test)

            train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_data['labels'])
            test_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'], test_data['labels'])

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16)

        elif self.model_type == 'LSTM' or self.model_type == "CNN":
            train_data = self.encode_data_lstm(X_train, y_train)
            test_data = self.encode_data_lstm(X_test, y_test)

            train_dataset = TensorDataset(train_data[0], train_data[1])
            test_dataset = TensorDataset(test_data[0], test_data[1])

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return train_loader, test_loader
    
