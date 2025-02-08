# -*- coding: utf-8 -*-
import os
import torch
from config import CONFIG
from dataloader import load_data
from model import SiameseNetwork


class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        self.train_data = knwb_data
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.knwb_to_vector()
        return
    
    def predict(self, sentence):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector = self.model(input_id)
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))
            hit_index = self.question_index_to_standard_question_index[hit_index]
        return  self.index_to_standard_question[hit_index]
    
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.index_to_standard_question = dict((y, x) for x, y in self.schema.items())
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return
    
    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_id
    

if __name__ == "__main__":
    config = CONFIG
    knwb_data = load_data(config, 'train')
    model = SiameseNetwork(config)
    model_path = os.path.join(config['model_path'], 'epoch_%d.pth' % config['n_epoches'])
    model.load_state_dict(torch.load(model_path))
    predict = Predictor(config, model, knwb_data)

    while True:
        sentence = input("请输入问题：")
        if sentence == "退出":
            break
        res = predict.predict(sentence)
        print(res)
