import torch
from loader import loader
import re


class Evaluation:
    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.data = loader(self.config["eval_data_path"], self.config, shuffle=False)

    def evaluator(self):
        self.results = {}
        self.results["all_pred"] = 0
        self.results["all_true"] = 0
        self.results["correct"] = 0
        self.results["all_pred_location"] = 0
        self.results["all_pred_organization"] = 0
        self.results["all_pred_person"] = 0
        self.results["all_pred_time"] = 0
        self.results["all_true_location"] = 0
        self.results["all_true_organization"] = 0
        self.results["all_true_person"] = 0
        self.results["all_true_time"] = 0
        self.results["correct_location"] = 0
        self.results["correct_organization"] = 0
        self.results["correct_person"] = 0
        self.results["correct_time"] = 0
        for index, batch_data in enumerate(self.data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            inputs, labels = batch_data
            self.model.eval()
            with torch.no_grad():
                all_pred = self.model(inputs)
                if self.config["model_type"] == "bert":
                    labels = labels[:, 1:-1]
                if self.config["use_crf"]:
                    all_pred = torch.LongTensor(all_pred)
                for sentence_t, sentence_p in zip(labels, all_pred):
                    result_p, result_p_location, result_p_organization, result_p_person, result_p_time = self.decode(
                        sentence_p)
                    result_t, result_t_location, result_t_organization, result_t_person, result_t_time = self.decode(
                        sentence_t)
                    self.get_result(result_p, result_p_location, result_p_organization, result_p_person, result_p_time,
                                    result_t, result_t_location, result_t_organization, result_t_person, result_t_time)
        micro_precision, micro_recall = self.result_show()
        return micro_precision, micro_recall

    def get_result(self, result_p, result_p_location, result_p_organization, result_p_person, result_p_time, result_t,
                   result_t_location, result_t_organization, result_t_person, result_t_time):
        self.results["all_pred"] += len(result_p)
        self.results["all_true"] += len(result_t)
        self.results["correct"] += len(result_p.intersection(result_t))
        self.results["all_pred_location"] += len(result_p_location)
        self.results["all_pred_organization"] += len(result_p_organization)
        self.results["all_pred_person"] += len(result_p_person)
        self.results["all_pred_time"] += len(result_p_time)
        self.results["all_true_location"] += len(result_t_location)
        self.results["all_true_organization"] += len(result_t_organization)
        self.results["all_true_person"] += len(result_t_person)
        self.results["all_true_time"] += len(result_t_time)
        self.results["correct_location"] += len(result_p_location.intersection(result_t_location))
        self.results["correct_organization"] += len(result_p_organization.intersection(result_t_organization))
        self.results["correct_person"] += len(result_p_person.intersection(result_t_person))
        self.results["correct_time"] += len(result_p_time.intersection(result_t_time))

    def decode(self, sentence):
        strs = ''.join([str(int(number)) for number in sentence])
        result = []
        result_location = []
        result_organization = []
        result_person = []
        result_time = []
        for location in re.finditer("(04+)", strs):
            start, end = location.span()
            result.append(''.join(str(i) for i in range(start, end)) + str(end))
            result_location.append(''.join(str(i) for i in range(start, end)) + str(end))

        for location in re.finditer("(15+)", strs):
            start, end = location.span()
            result.append(''.join(str(i) for i in range(start, end)) + str(end))
            result_organization.append(''.join(str(i) for i in range(start, end)) + str(end))

        for location in re.finditer("(26+)", strs):
            start, end = location.span()
            result.append(''.join(str(i) for i in range(start, end)) + str(end))
            result_person.append(''.join(str(i) for i in range(start, end)) + str(end))

        for location in re.finditer("(37+)", strs):
            start, end = location.span()
            result.append(''.join(str(i) for i in range(start, end)) + str(end))
            result_time.append(''.join(str(i) for i in range(start, end)) + str(end))
        return set(result), set(result_location), set(result_organization), set(result_person), set(result_time)

    def result_show(self):
        micro_precision = self.results["correct"] / (self.results["all_pred"] + 1e-5)
        micro_recall = self.results["correct"] / (self.results["all_true"] + 1e-5)
        macro_location_precision = self.results["correct_location"] / (self.results["all_pred_location"] + 1e-5)
        macro_location_recall = self.results["correct_location"] / (self.results["all_true_location"] + 1e-5)
        macro_organization_precision = self.results["correct_organization"] / (
                    self.results["all_pred_organization"] + 1e-5)
        macro_organization_recall = self.results["correct_organization"] / (
                    self.results["all_true_organization"] + 1e-5)
        macro_person_precision = self.results["correct_person"] / (self.results["all_pred_person"] + 1e-5)
        macro_person_recall = self.results["correct_person"] / (self.results["all_true_person"] + 1e-5)
        macro_time_precision = self.results["correct_time"] / (self.results["all_pred_time"] + 1e-5)
        macro_time_recall = self.results["correct_time"] / (self.results["all_true_time"] + 1e-5)
        F1_micro = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        F1_macro_location = (2 * macro_location_precision * macro_location_recall) / (
                    macro_location_precision + macro_location_recall + 1e-5)
        F1_macro_organization = (2 * macro_organization_precision * macro_organization_recall) / (
                    macro_organization_precision + macro_organization_recall + 1e-5)
        F1_macro_person = (2 * macro_person_precision * macro_person_recall) / (
                    macro_person_precision + macro_person_recall + 1e-5)
        F1_macro_time = (2 * macro_time_precision * macro_time_recall) / (
                    macro_time_precision + macro_time_recall + 1e-5)
        F1_macro = (F1_macro_location + F1_macro_organization + F1_macro_person + F1_macro_time) / 4

        print(
            f"""
总共有{self.results["all_true"]}个实体；
LOCATION实体有{self.results["all_true_location"]}个
ORGANIZATION实体有{self.results["all_true_organization"]}个
PERSON实体有{self.results["all_true_person"]}个
TIME实体有{self.results["all_true_time"]}个
预测了{self.results["all_pred"]}个实体；
预测了{self.results["all_pred_location"]}个LOCATION实体
预测了{self.results["all_pred_organization"]}个ORGANIZATION实体
预测了{self.results["all_pred_person"]}个PERSON实体
预测了{self.results["all_pred_time"]}个TIME实体
正确了{self.results["correct"]}个实体；
正确了{self.results["correct_location"]}个LOCATION实体
正确了{self.results["correct_organization"]}个ORGANIZATION实体
正确了{self.results["correct_person"]}个PERSON实体
正确了{self.results["correct_time"]}个TIME实体
location准确率为：{macro_location_precision}；
location召回率为：{macro_location_recall}；
organization准确率为：{macro_organization_precision}；
organization召回率为：{macro_organization_recall}；
person准确率为：{macro_person_precision}；
person召回率为：{macro_person_recall}；
time准确率为：{macro_time_precision}；
time召回率为：{macro_time_recall}；
macro_F1为：{F1_macro}
micro准确率为：{micro_precision}；
micro召回率为：{micro_recall}；
micro_F1为：{F1_micro}
            """
        )
        return micro_precision, micro_recall
