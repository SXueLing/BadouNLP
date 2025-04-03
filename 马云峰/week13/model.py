import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, PreTrainedModel
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
#TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"])
class TorchModel(PreTrainedModel):
    config_class = Config
    def __init__(self, config):
        super(TorchModel, self).__init__(config)
        max_length = config.max_length
        class_num = config.class_num
        self.bert = BertModel.from_pretrained(config.pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config.use_crf
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x, _ = self.bert(x)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict

def choose_optimizer(config, model):
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
if __name__ == "__main__":
    from config import Config
    config=Config()
    # 访问配置值
    print("Model Path:", config.model_path)
    print("Train Data Path:", config.train_data_path)
    print("Valid Data Path:", config.valid_data_path)
    print("Schema Path:", config.schema_path)
    print("Vocab Path:", config.vocab_path)
    print("Model Type:", config.model_type)
    print("Max Length:", config.max_length)
    print("Class Num:", config.class_num)
    print("Use CRF:", config.use_crf)
    print("Hidden Size:", config.hidden_size)
    print("Kernel Size:", config.kernel_size)
    print("Num Layers:", config.num_layers)
    print("Epoch:", config.epoch)
    print("Batch Size:", config.batch_size)
    print("Tuning Tactics:", config.tuning_tactics)
    print("Pooling Style:", config.pooling_style)
    print("Optimizer:", config.optimizer)
    print("Learning Rate:", config.learning_rate)
    print("Pretrain Model Path:", config.pretrain_model_path)
    print("Seed:", config.seed)
    model = TorchModel(config)
    for name, param in model.named_parameters(): #遍历模型参数
        print(name, param.shape)
