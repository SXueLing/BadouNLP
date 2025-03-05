import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam


class LanguageModel(nn.Module):
    """docstring for LanguageModel"""

    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_path'],
                                              return_dict=False,
                                              attn_implementation='eager')

        self.classify = nn.Linear(self.bert.config.hidden_size,
                                  self.bert.config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, mask=None, labels=None):
        # x : batch,seq
        # y : batch,seq
        # mask : batch,seq,seq
        # 训练时
        if labels is not None:
            inputs, _ = self.bert(inputs, attention_mask=mask)
            y_pred = self.classify(inputs)  # [batch,seq,vocab_size]
            return self.loss(y_pred.view(-1, y_pred.size(-1)), labels.view(-1))
        # 预测时
        else:
            inputs, _ = self.bert(inputs)
            y_pred = self.classify(inputs)  # batch,seq,vocab
            return torch.softmax(y_pred, dim=-1)


def choose_optimizer(config, model: LanguageModel):
    if config['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(),
                         lr=config['learning_rate'])
        return optimizer

    pass


if __name__ == '__main__':
    from config import Config
    from loader import data_load
    # data
    train_data = data_load(Config)

    # mdoel
    model = LanguageModel(Config)

    for i, batch_data in enumerate(train_data):
        # loss = model(**batch_data)
        # print(loss)
        y_pred = model(batch_data['inputs'])
        print(y_pred.shape)
        break
