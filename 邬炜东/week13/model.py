import torch
from transformers import BertModel
import torch.nn as nn
from config import Config
from loader import loader
from torchcrf import CRF
from evaluate import Evaluation


class All_Models(nn.Module):
    def __init__(self, config):
        super(All_Models, self).__init__()
        self.config = config
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.model_type = self.config["model_type"]
        if self.config["use_crf"]:
            self.crf = CRF(self.config["class_num"], batch_first=True)
        self.is_bert = False
        self.pooling_type = self.config["pooling_style"]
        if self.pooling_type == "max":
            self.pool = nn.MaxPool1d(self.config["max_length"])
        elif self.pooling_type == "avg":
            self.pool = nn.AvgPool1d(self.config["max_length"])
        else:
            self.pool = lambda x: x[:, :, -1]
        self.load_model()

    # 三个算损失，一个算嵌入
    def load_model(self):
        # 模型为lstm、bilstm、rnn、birnn和bert-base-chinese
        if self.model_type == "lstm":
            self.embedding = nn.Embedding(self.config["vocab_size"], self.hidden_size, padding_idx=0)
            self.network = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False, num_layers=self.num_layers,
                                   batch_first=True)
        elif self.model_type == "bilstm":
            self.embedding = nn.Embedding(self.config["vocab_size"], self.hidden_size, padding_idx=0)
            self.network = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True, num_layers=self.num_layers,
                                   batch_first=True)
            self.hidden_size = 2 * self.hidden_size
        elif self.model_type == "rnn":
            self.embedding = nn.Embedding(self.config["vocab_size"], self.hidden_size, padding_idx=0)
            self.network = nn.RNN(self.hidden_size, self.hidden_size, bidirectional=False, num_layers=self.num_layers,
                                  batch_first=True)
        elif self.model_type == "birnn":
            self.embedding = nn.Embedding(self.config["vocab_size"], self.hidden_size, padding_idx=0)
            self.network = nn.RNN(self.hidden_size, self.hidden_size, bidirectional=True, num_layers=self.num_layers,
                                  batch_first=True)
            self.hidden_size = 2 * self.hidden_size
        elif self.model_type == "bert":
            self.network = BertModel.from_pretrained(self.config["pretrain_model_path"], return_dict=False)
            self.is_bert = True
            self.hidden_size = self.network.config.hidden_size
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.classify = nn.Linear(self.hidden_size, self.config["class_num"])
        self.activation = torch.softmax

    def forward(self, x, y=None):
        if not self.is_bert:
            x = self.embedding(x)
        x = self.network(x)[0]
        y_pred = self.classify(x)
        if self.is_bert:
            y_pred = y_pred[:, 1:-1, :]
        if y is not None:
            if self.is_bert:
                y = y[:, 1:-1]
            if self.config["use_crf"]:
                mask = y.gt(-1)
                return -self.crf(y_pred, y, mask, reduction="mean")
            else:
                return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1))
        else:
            if self.config["use_crf"]:
                return self.crf.decode(y_pred)
            else:
                return torch.argmax(self.activation(y_pred, dim=-1), dim=-1)


# 优化器选择
def choose_optim(model, config):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])


# 测试代码是否可用
def test(config):
    # config["use_crf"] = False
    config["model_type"] = "bert"
    DL = loader(config["train_data_path"], config)
    model = All_Models(config)
    if torch.cuda.is_available():
        model.cuda()
    eval = Evaluation(config, model)
    for index, batch_data in enumerate(DL):
        if torch.cuda.is_available():
            batch_data = [d.cuda() for d in batch_data]
        sentence, label = batch_data
        loss = model(sentence, label)
        pred = model(sentence)
        eval.evaluator()
        print("=====================================")


if __name__ == "__main__":
    test(Config)
