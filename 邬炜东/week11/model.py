import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from config import Config
from loader import loader
import numpy as np
from evaluate import Evaluation

class All_Models(nn.Module):
    def __init__(self, config):
        super(All_Models, self).__init__()
        self.config = config
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.model_type = self.config["model_type"]
        self.is_bert = False
        self.load_model()

    # 三个算损失，一个算嵌入
    def load_model(self):
        # 模型为lstm、bilstm、rnn、birnn和bert-base-chinese
        self.network = BertModel.from_pretrained(self.config["pretrain_model_path"], return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.is_bert = True
        self.hidden_size = self.network.config.hidden_size
        self.output = self.tokenizer.vocab_size
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.classify = nn.Linear(self.hidden_size, self.output)
        self.activation = nn.functional.softmax

    def forward(self, x, y=None):
        try:
            mask = generate_mask(x.shape[0])
            x, _ = self.network(x, attention_mask=mask)
        except:
            x, _ = self.network(x)

        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1))
        else:
            return self.activation(y_pred, dim=-1)


def generate_mask(batch_size):
    total = Config["max_length_content"] + len(Config["bridge_token"]) + Config["max_length_title"] - 2
    mask = np.zeros((total, total))
    for i in range(Config["max_length_content"] + len(Config["bridge_token"]) - 1):
        for j in range(Config["max_length_content"] + len(Config["bridge_token"]) - 1):
            mask[i][j] = 1
    for i in range(Config["max_length_content"] + len(Config["bridge_token"]) - 1, total):
        for j in range(i + 1):
            mask[i][j] = 1
    mask = torch.LongTensor(mask)
    mask = mask.repeat(batch_size, 1, 1)
    return mask.cuda()


# 优化器选择
def choose_optim(model, config):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])


# 测试代码是否可用
def test(config):
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
        print(loss)
    print(eval.evaluator("浅色的中式边几搭配金属镜面花瓶，体现出一种低调的生活态度，落落大方的百合让窗景成为一幅生动的画作。"))
    print("=====================================")


if __name__ == "__main__":
    test(Config)
