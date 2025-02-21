import torch
from torch import nn
from transformers import BertModel

class BERT_SFT_Model(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BERT_SFT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
