import torch
from loader import loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluation:
    def __init__(self, config, model, data_dict):
        self.model = model.to(device)
        self.config = config
        self.data = loader(self.config["eval_data_path"], self.config, is_predict=True, shuffle=True)
        self.data_dict = data_dict
        self.correct = 0
        self.wrong = 0

    def get_result_dict(self):
        self.all_question_index = []
        self.result_dict = {}
        for question_type, questions_index in self.data_dict.items():
            for question_index in questions_index:
                self.result_dict[len(self.all_question_index)] = question_type
                self.all_question_index.append(question_index)

    def evaluator(self):
        self.correct = 0
        self.wrong = 0
        self.model.eval()
        self.get_result_dict()
        with torch.no_grad():
            self.all_question_embedding = self.model(torch.LongTensor(self.all_question_index).to(device))
            self.all_question_embedding = torch.nn.functional.normalize(self.all_question_embedding, dim=-1)
            self.get_result()
        acc = self.result_show()
        return acc

    def get_result(self):
        for index, input_question in enumerate(self.data):
            questions_index, questions_type = input_question
            for question_index, question_type in zip(questions_index, questions_type):
                question_embedding = self.model(question_index.unsqueeze(0).to(device))
                result_matrix = torch.mm(question_embedding, self.all_question_embedding.T)
                pred = self.result_dict[int(torch.argmax(result_matrix.squeeze(0)))]
                if int(question_type) == int(pred):
                    self.correct += 1
                else:
                    self.wrong += 1

    def result_show(self):
        print(
            f"一共有{self.correct + self.wrong}个样本, 预测正确{self.correct}个，预测错误{self.wrong}个，正确率为{self.correct / (self.correct + self.wrong)}")
        return self.correct / (self.correct + self.wrong)
