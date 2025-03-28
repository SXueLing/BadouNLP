import torch
import torch.nn as nn
from transformers import BertModel
# from torchcrf import CRF
from torchcrf import CRF
from torch.optim import Adam


# 解析config,一般而言config,本身既是字典，复杂一点的则需要进行解析
class ConfigWrapper(object):
    """docstring for ConfigWrapper"""

    def __init__(self, config):
        super(ConfigWrapper, self).__init__()
        self.config = config

    def to_dict(self):
        return self.config


class TorchModel(nn.Module):
    """docstring for TorchModel"""

    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = ConfigWrapper(config).config

        class_num = config['class_num']
        self.bert = BertModel.from_pretrained(
            self.config['bert_path'], return_dict=False)

        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)

        self.crf_layer = CRF(class_num, batch_first=True)  # 生成一个标签到标签的转移概率矩阵

        self.use_crf = self.config['use_crf']

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.config['label_padding_token'])

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        # 其中x :[batch_size,seq_len]
        input_ids, _ = self.bert(input_ids)  # [bsz,sl,h]
        predict = self.classify(input_ids)  # [bsz,sl,class_num]

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1).bool()  # 获取掩码，对标签大于-1,认定为有效的，即crf,也需要掩码标签
                return - self.crf_layer(predict, labels, mask)
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), labels.view(-1))
        else:  # 进行预测
            # 使用crf，那么
            if self.use_crf:
                predictions = self.crf_layer.decode(predict)
                return predictions # 返回最大的序列，对与输入而言不需对齐     
            else:
                return predict


def choose_optimizer(config, model: TorchModel):
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    return optimizer


if __name__ == '__main__':
    from config import Config
    model = TorchModel(Config)
    x = torch.randint(0, 100, [10, 100])
    predict = model(x)
    print(predict)
