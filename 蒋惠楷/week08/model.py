import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam, SGD

'''建立网络模型结构'''

# class BERTSentenceEncoder(nn.Module):
#     def __init__(self, config):
#         super(BERTSentenceEncoder, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-chinese')
#         self.hidden_size = self.bert.config.hidden_size  # 768
#         self.dropout = nn.Dropout(0.5)
    
#     def forward(self, x):
#         # BERT expects input shape (batch_size, sequence_length)
#         output = self.bert(input_ids=x)
#         # 提取了 [CLS] token 的嵌入表示，作为每个输入文本的句子表示
#         # output.last_hidden_state 是三维的 (batch_size, sequence_length, hidden_size)
#         cls_embedding = output.last_hidden_state[:, 0, :] # [batch_size, hidden_size]
#         return self.dropout(cls_embedding)

class BERTSentenceEncoder(nn.Module):
    def __init__(self, config):
        super(BERTSentenceEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.hidden_size = self.bert.config.hidden_size  # 768
        
        # 小数据集-冻结BERT的所有参数，避免在训练时修改其预训练权重
        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # BERT expects input shape (batch_size, sequence_length)
        output = self.bert(input_ids=x)
        
        # 提取[CLS] token的嵌入表示，作为每个输入文本的句子表示
        # output.last_hidden_state是三维的 (batch_size, sequence_length, hidden_size)
        cls_embedding = output.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        x = self.fc(cls_embedding)
        x = self.dropout(x)
        
        return x

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

class TripletNetwork(nn.Module):
    def __init__(self, config):
        super(TripletNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss(margin=0.1, p=2) # p:计算距离的范数, 1为曼哈顿距离（L1范数），2为欧氏距离
    
    '''计算余弦距离 1-cos(a,b)'''
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine
    
    '''计算Triplet Loss'''
    '''只计算那些违反三元组约束的损失'''
    def cosine_triplet_loss_1(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) # diff[diff.gt(0)] 过滤掉了小于等于 0 的部分
    
    '''每个样本都贡献一定的损失'''
    def cosine_triplet_loss_2(self, anchor, positive, negative, margin=0.1):
        ap = self.cosine_distance(anchor, positive)
        an = self.cosine_distance(anchor, negative)
        loss = torch.mean(torch.clamp(ap - an + margin, min=0))  # 小于 0 的部分，损失为 0
        return loss
    
    '''如果传入了正样本和负样本，调用 cosine_triplet_loss 来计算损失。否则，仅返回锚点的句子嵌入表示'''
    def forward(self, anchor, positive=None, negative=None):
        if positive is not None and negative is not None:
            anchor_embedding = self.sentence_encoder(anchor)
            positive_embedding = self.sentence_encoder(positive)
            negative_embedding = self.sentence_encoder(negative)
            return self.cosine_triplet_loss_2(anchor_embedding, positive_embedding, negative_embedding)
        else:
            return self.sentence_encoder(anchor)

'''选择优化器'''
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        return SGD(model.parameters(), lr=learning_rate)
