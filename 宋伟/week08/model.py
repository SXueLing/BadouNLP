import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name, embedding_dim):
        super(TextEmbeddingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.encoder.config.hidden_size, embedding_dim)

    def forward(self, texts):
        # inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', clean_up_tokenization_spaces=True)

        outputs = self.encoder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return self.fc(embeddings)