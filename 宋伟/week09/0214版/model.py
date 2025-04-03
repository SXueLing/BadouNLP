import torch
from transformers import BertForTokenClassification

class NERModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(NERModel, self).__init__()
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
