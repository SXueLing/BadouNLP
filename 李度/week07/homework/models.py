import torch.nn as nn
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(
        self,
        vocab,
        core_model="bert",
        pooling="max",
        classify_type="simple",
        num_layers=5,
        use_bert_pooler=False,
    ):
        super().__init__()
        self.use_bert = core_model == "bert"
        self.use_bert_pooler = use_bert_pooler
        self.pooling_style = pooling
        self.embedding = nn.Embedding(len(vocab), 768)
        if core_model == "bert":
            self.core = BertModel.from_pretrained(
                "/Users/duli/Desktop/UT Austin/自学内容/八斗课件/NLP/six-lm/bert-base-chinese",
                return_dict=False,
            )
        elif core_model == "RNN":
            self.core = nn.RNN(
                768,
                768,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1,
            )
        elif core_model == "LSTM":
            self.core = nn.LSTM(
                768,
                768,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1,
            )
        else:
            self.core = CNN(hidden_size=768, num_layers=num_layers)

        if classify_type == "simple":
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(768, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )

    def forward(self, x):
        if self.use_bert_pooler and self.use_bert:
            _, x = self.core(x) # 直接用pooler的结果
            return self.classifier(x)
        elif self.use_bert:
            x, _ = self.core(x)
        else:
            x = self.embedding(x)
            x = self.core(x)
            x = x[0] if isinstance(x, tuple) else x
        if self.pooling_style == "max":
            self.pooler = nn.MaxPool1d(x.shape[1])
        else:
            self.pooler = nn.AvgPool1d(x.shape[1])
        x = self.pooler(x.transpose(1,2)).squeeze()
        return self.classifier(x)


class CNN(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.encoders = []
        self.kernel_size = 3
        self.pad_len = self.kernel_size // 2
        for _ in range(num_layers):
            self.encoders.append(
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=self.kernel_size,
                    padding=self.pad_len,
                )
            )
        self.encoders = nn.Sequential(*self.encoders)


    def forward(self, x):
        # B N D -> B D N
        x = x.transpose(1,2)
        return self.encoders(x).transpose(1,2)

