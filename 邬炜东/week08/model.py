import torch
from transformers import BertModel
import torch.nn as nn
from config import Config
from loader import loader
from evaluate import Evaluation


class All_Models(nn.Module):
    def __init__(self, config):
        super(All_Models, self).__init__()
        self.config = config
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.vocab_size = self.config["vocab_size"]
        self.model_type = self.config["model_type"]
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
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        if self.model_type == "lstm":
            self.network = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=False, num_layers=self.num_layers,
                                   batch_first=True)
        elif self.model_type == "bilstm":
            self.network = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=True, num_layers=self.num_layers,
                                   batch_first=True)
            self.hidden_size = 2 * self.hidden_size
        elif self.model_type == "rnn":
            self.network = nn.RNN(self.hidden_size, self.hidden_size, bidirectional=False, num_layers=self.num_layers,
                                  batch_first=True)
        elif self.model_type == "birnn":
            self.network = nn.RNN(self.hidden_size, self.hidden_size, bidirectional=True, num_layers=self.num_layers,
                                  batch_first=True)
            self.hidden_size = 2 * self.hidden_size
        elif self.model_type == "bert":
            self.network = BertModel.from_pretrained(self.config["pretrain_model_path"], return_dict=False)
            self.is_bert = True
            self.hidden_size = self.network.config.hidden_size
        self.loss = nn.functional.triplet_margin_loss
        self.activation = torch.softmax

    def forward(self, anchor, pos=None, neg=None):
        if pos is None:
            if self.is_bert:
                anchor = self.network(anchor)
            else:
                anchor = self.embedding(anchor)
                anchor = self.network(anchor)
            return self.pool(anchor[0].transpose(1, 2)).squeeze(-1)
        else:
            if self.is_bert:
                anchor = self.network(anchor)
                anchor = self.pool(anchor[0].transpose(1, 2)).squeeze(-1)
                pos = self.network(pos)
                pos = self.pool(pos[0].transpose(1, 2)).squeeze(-1)
                neg = self.network(neg)
                neg = self.pool(neg[0].transpose(1, 2)).squeeze(-1)
            else:
                anchor = self.embedding(anchor)
                anchor = self.network(anchor)
                anchor = self.pool(anchor[0].transpose(1, 2)).squeeze(-1)
                pos = self.embedding(pos)
                pos = self.network(pos)
                pos = self.pool(pos[0].transpose(1, 2)).squeeze(-1)
                neg = self.embedding(neg)
                neg = self.network(neg)
                neg = self.pool(neg[0].transpose(1, 2)).squeeze(-1)
            return self.loss(anchor, pos, neg)


# 优化器选择
def choose_optim(model, config):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])


# 测试代码是否可用
def test(config):
    # DL, _, _ = loader(config["train_data_path"], config)
    DL, data_dict = loader(config["train_data_path"], config)
    model = All_Models(config)
    # evaluate = Evaluation(config, model, data_dict)
    # evaluate.evaluator()

    for index, batch_data in enumerate(DL):
        anchor, pos, neg = batch_data
        neg = neg.transpose(1, 2).squeeze()
        print(anchor.shape)
        print(pos.shape)
        print(neg.shape)
        print(model(anchor, pos, neg))
        print(model(anchor).shape)
        print("=====================================")


if __name__ == "__main__":
    # # Config["class_num"] = 3
    # # Config["vocab_size"] = 20
    # # Config["max_length"] = 5
    # Config["model_type"] = "bert"
    # model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    # x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    # sequence_output, pooler_output = model(x)
    # print(x, type(x), len(x))
    # print("=============")
    test(Config)
