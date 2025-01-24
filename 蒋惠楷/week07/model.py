import torch.nn as nn
from transformers import BertForSequenceClassification, XLNetForSequenceClassification
from config import *

# 选择模型函数
def choosemodel(model_type, vocab):
    if model_type == "Bert":
        model = BertModel(model_name='bert-base-chinese').bert_model()  # 加载BERT模型
    elif model_type == 'LSTM':
        model = LSTMModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_labels=2)
    elif model_type == "XLNet":
        model = XLNetModel(model_name="hfl/chinese-xlnet-base", num_labels=2)
    elif model_type == "CNN":
        model = CNNModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, num_filters=NUM_FILTERS, filter_sizes=FILTER_SIZE, num_labels=2, dropout=DROUPOUT)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


# 预训练BERT模型
class BertModel:
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
    
    def bert_model(self):
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        return model

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_labels=2, max_length=MAX_LENGTH):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=DROUPOUT)
        self.fc = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, input_ids):
        # 获取词嵌入
        x = self.embedding(input_ids)
        x, (hidden, cell) = self.lstm(x)
        output = self.fc(hidden[-1])

        return output

# XLNet模型
class XLNetModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(XLNetModel, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 传入 labels 参数时，模型将返回 loss 和 logits
        outputs = self.xlnet(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits
    
# CNN模型
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, num_filters=NUM_FILTERS, filter_sizes=FILTER_SIZE, num_labels=2, dropout=DROUPOUT):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        conv_results = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # (batch_size, num_filters, seq_len - filter_size + 1)
        pooled_results = [torch.max(conv, dim=2)[0] for conv in conv_results]  # (batch_size, num_filters)

        # 连接所有池化后的特征
        x = torch.cat(pooled_results, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        return x
