import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from loader import load_data
from config import Config
from torchcrf import CRF


class TorchModel(nn.Module):
    def __init__(self, Config):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config['bert_model_path'], return_dict=False)
        self.fc = nn.Linear(Config['hidden_size'], Config['num_classes'])
        self.loss = nn.CrossEntropyLoss(ignore_index=Config['pad_label'])
        self.crf_layer = CRF(Config['num_classes'], batch_first=True)
    def forward(self, x, y=None):
        x = self.bert(x)[0]  # [batch_size, seq_len, hidden_size]
        x = self.fc(x)  # [batch_size, seq_len, num_classes]
        if y is not None:
            if Config['use_crf']:
                mask = y.gt(Config['pad_label']) # [batch_size, seq_len]
                # mask = y != Config['pad_label']  # [batch_size, seq_len]
                # print(mask, mask.shape)
                return -self.crf_layer(x, y, mask, reduction='mean') 
            else:
                x = x.view(-1, Config['num_classes'])  # [batch_size * seq_len, num_classes]
                y = y.view(-1)  # [batch_size * seq_len]
                return self.loss(x, y)
        else:
            if Config['use_crf']:
                return self.crf_layer.decode(x)
            else:
                return x
          
    
    
model = TorchModel(Config)