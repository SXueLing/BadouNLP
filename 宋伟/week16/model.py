import torch
from torch import nn
from transformers import BertModel

class DialogueModel(nn.Module):
    def __init__(self, config):
        super(DialogueModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.get_model_checkpoint())
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Example: Binary classification (e.g., "repeat" or "not repeat")

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
