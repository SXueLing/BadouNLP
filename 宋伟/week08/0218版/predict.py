'''
根据训练好的模型，进行预测

[description]
'''

import torch
from loader import load_data
from config import Config
from model import SiameseNetwork,choose_optimizer
import jieba

class Predictor(object):
    """docstring for Predictor"""
    def __init__(self, config,model,knwb_data):
        super(Predictor, self).__init__()
        self.config = config
        self.model = model
        self.train_data = knwb_data
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.knwb_to_vector()

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y,x) for x,y in self.schema.items())
        for standard_question_index,question_ids  in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrix = torch.stack(self.question_ids,dim=0)
            if torch.cuda.is_available():
                question_matrix = question_matrix.cuda()
            self.knwb_vectors = self.model(question_matrix)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors,dim=-1)
        return
    
    def encode_sentence(self,text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word,self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char,self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self,input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0]*(self.config["max_length"]-len(input_id))
        return input_id
        
    def predict(self,sentence):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor(input_id).unsqueeze(0)
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector = self.model(input_id)  # [batch,embdeding] [1,128]
            # self.knwb_vectors [batch,embdeding] ,记录所有的问题，属于many to one 的一个模型
            # print(self.knwb_vectors.shape)
            result = torch.mm(test_question_vector,self.knwb_vectors.t())
            hit_index = int(torch.argmax(result.squeeze()))
            hit_index = self.question_index_to_standard_question_index[hit_index]
        return self.index_to_standard_question[hit_index]

if __name__ == '__main__':
    knwb_data = load_data(Config["train_data_path"],Config)
    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load("./model_output/epoch_cos_loss_10.pth"))
    predictor = Predictor(Config,model,knwb_data)
    sentence = "请问，宽带密码如何进行修改"  # 固定宽带服务密码修改
    result = predictor.predict(sentence)
    print(result)

    

