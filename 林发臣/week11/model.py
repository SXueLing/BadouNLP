import torch.nn as nn
import torch
import json
from transformers import BertTokenizer
from transformers import BertModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageBertToSftModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.layer = nn.Linear(self.bert.config.hidden_size, self.tokenizer.vocab_size, bias=True)
        self.cuda_enable = torch.cuda.is_available()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        logger.info('init模型')

    def forward(self, x, x_mask=None, y=None):  # x = (bs,s1+s2) x_mask = (bs,l(s1+s2),l(s1+s2)) y =(bs,s1+s2)
        if all(i is not None for i in [x, x_mask, y]):
            # print('===============================================================')
            # print(x[0, :])
            # print(y[0, :])
            # print()
            x, _ = self.bert(x, attention_mask=x_mask)
            pred = self.layer(x)
            pred = pred.view(-1, pred.shape[-1])
            y = y.view(-1)
            return self.loss(pred, y)
        else:
            x, _ = self.bert(x)
            pred = self.layer(x)
            return torch.softmax(pred, dim=-1)


if __name__ == '__main__':
    pass
