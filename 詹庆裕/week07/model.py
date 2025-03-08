import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.optim import AdamW, SGD  # 改用AdamW优化器

class OptimizedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = None
        self.config = config
        self._init_components()


    def _init_components(self):
        if self.config["model_type"].startswith("bert"):
            self.init_bert_based()
        else:
            self.init_vanilla_model()
        self.pooling = {
            "max": nn.AdaptiveMaxPool1d(1),
            "avg": nn.AdaptiveAvgPool1d(1)
        }[self.config["pooling_type"]]
        self.classify = nn.Sequential(
            nn.Dropout(self.config.get("dropout", 0.1)),
            nn.Linear(self.hidden_size, self.config["class_num"])
        )


    def init_bert_based(self):
        bert_config = BertConfig.from_pretrained(self.config["pretrain_model_path"],
                    output_hidden_states=True if "mid_layer" in self.config["model_type"] else False)
        self.bert = BertModel.from_pretrained(self.config["pretrain_model_path"],
                                              config=bert_config)
        self.hidden_size = bert_config.hidden_size
        if "lstm" in self.config["model_type"]:
            self.rnn = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                num_layers=1,
                                batch_first=True)
        elif "cnn" in self.config["model_type"]:
            self.cnn = nn.Conv1d(bert_config.hidden_size,
                                bert_config.hidden_size,
                                kernel_size= self.config["kernel_size"],
                                padding=1)


    def init_vanilla_model(self):
        self.embedding = nn.Embedding(self.config["vocab_size"],
                                      self.config["hidden_size"],
                                      padding_idx=0)
        self.hidden_size = self.config["hidden_size"]
        if self.config["model_type"] in ["lstm", "rnn", "gru"]:
            rnn_class = {
                "lstm": nn.LSTM,
                "rnn": nn.RNN,
                "gru": nn.GRU
            }[self.config["model_type"]]

            self.encoder = rnn_class(
                input_size=self.config["hidden_size"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                batch_first=True,
                bidirectional=self.config.get("bidirectional", False)  # 添加双向支持
            )
            if self.config.get("bidirectional", False):
                self.hidden_size *= 2
        elif self.config["model_type"] == "gated_cnn":
            self.encoder = GatedCNN(self.config)
        elif self.config["model_type"] == "stack_gated_cnn":
            self.encoder = OptimizedGatedCNN(self.config)
        elif self.config["model_type"] == "rcnn":
            self.encoder = RCNN(self.config)

    def forward(self, x, y=None):
        if hasattr(self, "bert"):
            output = self.bert(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"]
            )
            x = output.last_hidden_state
            if hasattr(self, "rnn"):
                x, _ = self.rnn(x)
            elif hasattr(self, "cnn"):
                x = x.transpose(1, 2)
                x = self.cnn(x).transpose(1,2)
        else:
            x = self.embedding(x)
            x, _ = self.encoder(x) if isinstance(self.encoder, nn.RNNBase) else (self.encoder(x), None)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        logits = self.classify(x)
        if y is not None:
            return F.cross_entropy(logits, y.view(-1))
        return F.softmax(logits, dim=-1)

    def _register_buffer(self):
        """预注册空缓冲区应对动态模块"""
        self.register_buffer('dummy', torch.tensor(0))


class RCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            batch_first=True
        )
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        return x


class GatedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_cnn = OptimizedCNN(config)
        self.forget_cnn = OptimizedCNN(config)

    def forward(self, x):
        output_gate = self.output_cnn(x)
        forget_gate = torch.sigmoid(self.forget_cnn(x))
        return torch.mul(output_gate, forget_gate)



class OptimizedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config["hidden_size"],
            out_channels=config["hidden_size"],
            kernel_size=config["kernel_size"],
            padding=(config["kernel_size"]-1)//2,
            groups=4  # 添加分组卷积
        )
        self.activation = nn.GELU()  # 更优的激活函数

    def forward(self, x):
        # 优化维度转换
        return self.activation(self.conv(x.transpose(1,2))).transpose(1,2)


class OptimizedGatedCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = OptimizedCNN(config)
        self.gate_conv = nn.Conv1d(
            config["hidden_size"],
            config["hidden_size"],
            kernel_size=1  # 使用1x1卷积生成门控信号
        )

    def forward(self, x):
        gate = torch.sigmoid(self.gate_conv(x.transpose(1,2)))
        return self.conv(x) * gate.transpose(1,2)


def choose_optim(model, config):
    optim = {
        "adamw": AdamW(model.parameters(), lr=config["learning_rate"]),
        "sgd": SGD(model.parameters(), lr=config["learning_rate"])
    }[config["optim_type"]]
    return optim
