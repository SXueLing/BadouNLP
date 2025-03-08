import torch
import torch.nn as nn
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        self.use_bert = config["model_type"].startswith("bert")
        if not self.use_bert:
            self.embedding = nn.Embedding(config["vocab_size"] + 1, config["hidden_size"], padding_idx=0)
        self.encoder = self._build_encoder()
        self.hidden_size = self.encoder.config.hidden_size if self.use_bert else config["hidden_size"]
        self.classify = nn.Linear(self.hidden_size, config["class_num"])
        self.loss = nn.functional.cross_entropy
        self.pooling_style = config["pooling_style"]

    def _build_encoder(self):
        model_type = self.config["model_type"]
        if model_type == "fast_text":
            return lambda x: x
        elif model_type in ["lstm", "gru", "rnn"]:
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[model_type]
            return rnn_cls(self.config["hidden_size"], self.config["hidden_size"],
                           num_layers=self.config["num_layers"], batch_first=True)
        elif model_type.startswith("bert"):
            return BertModel.from_pretrained(self.config["pretrain_model_path"], return_dict=False)
        else:
            # For CNN-based models and others
            encoder_class = {"cnn": CNN, "gated_cnn": GatedCNN, "stack_gated_cnn": StackGatedCNN,
                             "rcnn": RCNN, "bert_lstm": BertLSTM, "bert_cnn": BertCNN,
                             "bert_mid_layer": BertMidLayer}.get(model_type)
            if encoder_class is None:
                raise ValueError(f"Unknown model type: {model_type}")
            return encoder_class(self.config)

    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)[0]  # only take sequence output for BERT
        else:
            x = self.encoder(self.embedding(x))
        
        if isinstance(x, tuple):  # RNN 类模型处理
            x = x[0]
        
        x = self._apply_pooling(x)
        predict = self.classify(x)
        
        return self.loss(predict, target.squeeze()) if target is not None else predict
    
    def _apply_pooling(self, x):
        pooling_layer = nn.MaxPool1d(x.shape[1]) if self.pooling_style == "max" else nn.AvgPool1d(x.shape[1])
        return pooling_layer(x.transpose(1, 2)).squeeze()

# 定义其他编码器类（CNN, GatedCNN, StackGatedCNN, RCNN, BertLSTM, BertCNN, BertMidLayer）
# 这些类可以根据之前的定义保持不变或做相应调整...

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(config["hidden_size"], config["hidden_size"], config["kernel_size"], padding='same')

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = torch.sigmoid(self.gate(x))
        return torch.mul(a, b)

class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                GatedCNN(config),
                nn.LayerNorm(config["hidden_size"]),
                nn.ReLU(),
                nn.Linear(config["hidden_size"], config["hidden_size"]),
                nn.ReLU(),
                nn.Linear(config["hidden_size"], config["hidden_size"]),
                nn.LayerNorm(config["hidden_size"])
            ) for _ in range(config["num_layers"])
        ])

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        return x

class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        self.rnn = nn.RNN(config["hidden_size"], config["hidden_size"], batch_first=True)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.cnn(x)

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        return self.cnn(x)

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        hidden_states = self.bert(x)[2]  # (13, batch, len, hidden)
        return hidden_states[-2].add(hidden_states[-1]).div(2)  # Average the last two layers

def choose_optimizer(config, model):
    optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[config["optimizer"]]
    return optimizer_class(model.parameters(), lr=config["learning_rate"])

if __name__ == "__main__":
    from config import Config
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))
