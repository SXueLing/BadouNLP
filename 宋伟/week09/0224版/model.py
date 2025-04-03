
import sys
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel


class TorchModel(nn.Module):
    """docstring for TorchModel"""

    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        vocab_size = config.get("vocab_size",1024)   # 未加载数据前，用于测试使用
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=hidden_size,
                                      padding_idx=0)
        # 使用多层双向LSTM进行获取序列信息,进行编码
        if config['use_bert']:
            self.layer = BertModel.from_pretrained(config['bert_path'],return_dict=False)
            self.classify = nn.Linear(self.layer.config.hidden_size,class_num)
        else:
            self.layer = nn.LSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 batch_first=True,
                                 bidirectional=True,
                                 num_layers=num_layers)
            self.classify = nn.Linear(hidden_size*2,class_num)

        self.cry_layer = CRF(class_num,batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)   # 这个-1代表标签的填充值

    def forward(self,x,target=None):
        # x:[batch,sequence],如果使用了bert,这里是一个字典还将记录attention_mask
        # taget:[batch,sequence]
        if self.config['use_bert']:
            x,_  = self.layer(**x,return_dict=False)
        else:
            x = self.embedding(x) # [batch_len,seqence_len,embddeing_size]
            x, _ = self.layer(x)  # [batch,sequence,2*embedding]

        predict = self.classify(x) # [bacth,sequence,num_tag] 每个字分布`num_tag`维度分布

        # 进行训练
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.cry_layer(predict,target,mask,reduction="mean")
            else: 
                # 预测数据：[batch,seq,num_class]->[batch*seq,num_class]
                # 目标标签： [batch,seq]->(batch*sequence)
                return self.loss(predict.view(-1,predict.shape[-1]),target.view(-1))  # 返回一个带梯度的标量
        # 进行预测
        else:
            if self.use_crf:
                return self.cry_layer.decode(predict)
            else:
                return predict

def choose_optimizer(config,model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer=="adam":
        return Adam(model.parameters(),lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    
if __name__ == '__main__':
    from config import Config
    from loader import MyDataSet,data_load
    dataset = MyDataSet(Config['train_data_path'],Config)
    model_input,label = dataset[0]
    model = TorchModel(Config)
    model_input['input_ids'] = model_input['input_ids'].unsqueeze(0)
    # quit()
    # quit()
    loss = model(model_input['input_ids'])
    print("进行损失值计算",loss)
    