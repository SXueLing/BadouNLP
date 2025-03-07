import torch
import re
import numpy as np

from home_loader import load_data
from collections import defaultdict

class Evaluator:
    def __init__(self, config, model, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.model.eval()
        self.logger.info("开始%d轮模型测试" % epoch)
        self.state_dict = {
            "LOCATION": defaultdict(int),
            "PERSON": defaultdict(int),
            "TIME": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            inputs, labels = batch_data
            with torch.no_grad():
                pred_labels = self.model(inputs)
            self.write_stats(sentences, pred_labels, labels)
        self.show_stats()

    def write_stats(self, sentences, pred_labels, labels):
        if not self.config["use_crf"]:
            pred_labels = torch.argmax(pred_labels, dim=-1)
        for sentence, pred_label, true_label in zip(sentences, pred_labels, labels):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)

            for key in self.state_dict.keys():
                self.state_dict[key]["正确识别数"] += len([ent for ent in true_entities[key] if ent in pred_entities[key]])
                self.state_dict[key]["预测实体数"] += len(true_entities[key])
                self.state_dict[key]["实际实体数"] += len(pred_entities[key])

    def decode(self, sentence, label):
        sentence = "$" + sentence
        labels = ''.join([str(x) for x in label[:len(sentence)]])
        results = defaultdict(list)
        for loc in re.finditer("(04+)", labels):
            s, e = loc.span()
            results["LOCATION"].append(sentence[s:e])
        for org in re.finditer("(15+)", labels):
            s, e = org.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for per in re.finditer("(26+)", labels):
            s, e = per.span()
            results["PERSON"].append(sentence[s:e])
        for tim in re.finditer("(37+)", labels):
            s, e = tim.span()
            results["TIME"].append(sentence[s:e])
        return results

    def show_stats(self):
        F1_scores = []
        for key in self.state_dict.keys():
            precision = self.state_dict[key]["正确识别数"]/(1e-5 + self.state_dict[key]["预测实体数"])
            recall = self.state_dict[key]["正确识别数"] / (1e-5 + self.state_dict[key]["实际实体数"])
            f1 = 2 * precision * recall / (1e-5 + precision + recall)
            F1_scores.append(f1)
            self.logger.info("%s类实体的准确率为：%f，召回率为：%f，f1分数为：%f" % (key, precision, recall, f1))
        self.logger.info("Macro_F1: %f" % np.mean(F1_scores))
        all_true = sum([self.state_dict[key]["实际实体数"] for key in self.state_dict.keys()])
        all_pred = sum([self.state_dict[key]["预测实体数"] for key in self.state_dict.keys()])
        all_cort = sum([self.state_dict[key]["正确识别数"] for key in self.state_dict.keys()])
        all_precision = all_cort / (1e-5 + all_pred)
        all_recall = all_cort / (1e-5 + all_true)
        Micro_F1 = 2 * all_precision * all_recall / (1e-5 + all_precision + all_recall)
        self.logger.info("Micro_F1: %f" % Micro_F1)
        self.logger.info("----------------")
