'''
Author: Zhao
Date: 2025-01-20 20:34:45
LastEditTime: 2025-01-22 13:08:34
FilePath: model.py
Description: 建立模型结构

'''
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam, SGD
from torchcrf import CRF

class TorchModel(nn.Module):
    def __init__(self, config, logger):
        super(TorchModel, self).__init__()

        self.logger = logger

        # BERT
        self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict = False)
        
        self.hidden_size = self.encoder.config.hidden_size # 获取 BERT 的 hidden_size
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
            
        # 分类层：将将 BERT 的输出映射到标签空间
        self.classify = nn.Linear(in_features=self.hidden_size, out_features=class_num)
        
        # CRF层：条件随机场层，用于序列标注
        self.crf = CRF(num_tags=class_num, batch_first=True)
        
        # 损失函数：交叉熵损失函数，用于计算损失值
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, x, labels=None):
        # BERT 模型的输出
        x = self.encoder(x)
        predict = self.classify(x[0])
        
        if labels is not None:
            if self.use_crf:
                # 使用 attention_mask 生成布尔掩码
                mask = labels.gt(-1) 
                log_likelihood = self.crf(predict, labels, mask=mask, reduction="mean")
                return -log_likelihood
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                mask = labels.gt(-1) 
                # 如果使用CRF，解码预测结果
                predict = self.crf.decode(predict, mask=mask)
                return predict
            else:
                return predict

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    # 选择优化器：根据配置选择Adam或SGD优化器
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

# if __name__ == "__main__":
#     #from config import Config
#     config = { 
#         "hidden_size": 128, 
#         "vocab_size": 5000, 
#         "class_num": 10, 
#         "num_layers": 2, 
#         "max_length": 50, 
#         "use_crf": True, 
#         "optimizer": "adam", 
#         "learning_rate": 0.001 
#         }
#     model = TorchModel(config)

#     batch_size = 32 
#     x = torch.randint(0, config["vocab_size"], (batch_size, config["max_length"]), dtype=torch.long) 
#     target = torch.randint(0, config["class_num"], (batch_size, config["max_length"]), dtype=torch.long)
#     # 测试前向传播 
#     output = model(x, target) 
#     # 初始化优化器 
#     optimizer = choose_optimizer(config, model) 
#     print("模型输出：", output)
#     #模型输出： tensor(115.1977, grad_fn=<NegBackward0>)
            

