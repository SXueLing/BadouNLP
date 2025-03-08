from home_loader import load_data
import torch
import torch.nn as nn


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config)
        self.train_data = load_data(config["train_data_path"], config)

    def knwb_to_vectors(self):
        self.quesiton_to_stardard_question_idx = {}
        self.questions = []
        for standard_question_id, questions in self.train_data.dataset.knwb.items():
            for question in questions:
                self.quesiton_to_stardard_question_idx[len(self.questions)] = standard_question_id
                self.questions.append(question)
        questions_matrix = torch.stack(self.questions, dim=0)
        if torch.cuda.is_available():
            questions_matrix = questions_matrix.cuda()
        with torch.no_grad():
            self.knwb_vectors = self.model(questions_matrix)
        self.knwb_vectors = nn.functional.normalize(self.knwb_vectors, dim=-1)

    def eval(self, epoch):
        self.model.eval()
        self.knwb_to_vectors()
        self.logger.info("开始第%d轮模型评估" % (epoch))
        self.state_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            inputs, labels = batch_data
            with torch.no_grad():
                vectors = self.model(inputs)
            self.write_stats(vectors, labels)
        self.show_stats()

    def write_stats(self, vectors, labels):
        for vector, label in zip(vectors, labels):
            vector = nn.functional.normalize(vector, dim=-1)    #归一化
            res = torch.mm(vector.unsqueeze(0), self.knwb_vectors.T)
            hint_idx = int(res.argmax())
            hint_idx = self.quesiton_to_stardard_question_idx[hint_idx]
            if hint_idx == int(label):
                self.state_dict["correct"] += 1
            else:
                self.state_dict["wrong"] += 1

    def show_stats(self):
        correct, wrong = self.state_dict["correct"], self.state_dict["wrong"]
        all_nums = correct + wrong
        self.logger.info("预测总数为：%d" % (all_nums))
        self.logger.info("预测正确的数量为：%d，错误的为：%d" % (correct, wrong))
        self.logger.info("预测正确率为：%f" % (correct / all_nums))
        self.logger.info("--------------------")
        return correct / all_nums
