import torch
import re
import numpy as np
from collections import defaultdict
from home_loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle = False)

    def eval(self, epoch):
        self.logger.info("开始第%d轮模型的评估：" %(epoch))
        self.model.eval()
        self.state_dict = {
        "LOCATION": defaultdict(int),
        "ORGANIZATION": defaultdict(int),
        "PERSON": defaultdict(int),
        "TIME": defaultdict(int)
        }
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input, labels = batch_data
            with torch.no_grad():
                pre_labels = self.model(input)
            self.write_stats(labels, pre_labels, sentences)
        self.show_stats()

    def write_stats(self, labels, pre_labels, sentences):
        if not self.config["use_crf"]:
            pre_labels = torch.argmax(pre_labels, dim = -1)
        for true_label, pre_label, sentence in zip(labels, pre_labels, sentences):
            if not self.config["use_crf"]:
                pre_label = pre_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            # print(len(true_label))
            # # print(pre_label)
            true_entities = self.decode(sentence, true_label)
            pre_entities = self.decode(sentence, pre_label)

            for key in self.state_dict.keys():
                self.state_dict[key]["正确识别"] += len([ent for ent in pre_entities[key] if ent in true_entities[key]])
                self.state_dict[key]["样本实体数"] += len(true_entities[key])
                self.state_dict[key]["识别出实体数"] += len(pre_entities[key])


    def show_stats(self):
        F1_scores = []
        for key in self.state_dict.keys():
            precision = self.state_dict[key]["正确识别"] / (1e-5 + self.state_dict[key]["识别出实体数"])
            recall = self.state_dict[key]["正确识别"] / (1e-5 + self.state_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (1e-5 + precision + recall)
            F1_scores.append(F1)
            self.logger.info("%s类实体的准确率为：%f，召回率为：%f，F1：%f" %(key, precision, recall, F1))
        self.logger.info("Macro-F1：%f" %(np.mean(F1_scores)))
        correct_pred = sum([self.state_dict[key]["正确识别"] for key in self.state_dict.keys()])
        total_pred = sum([self.state_dict[key]["识别出实体数"] for key in self.state_dict.keys()])
        true_enti = sum([self.state_dict[key]["样本实体数"] for key in self.state_dict.keys()])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro_F1：%f" %(micro_f1))
        self.logger.info("----------------")

    def decode(self, sentence, labels):
        # labels = "".join([str(x) for x in labels[:len(sentence)]])
        labels = "".join([str(x) for x in labels[1:len(sentence) + 1]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):   #re.finditer遍历所有满足条件的
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
