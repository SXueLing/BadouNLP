# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
Model effect test
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  # Used to store test results

    def eval(self, epoch):
        self.logger.info("Start testing the %dth round of model effects：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # Clear the previous round results
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            # Unpack batch_data based on its length
            if len(batch_data) == 3:  # With attention_mask
                input_ids, attention_mask, labels = batch_data
            elif len(batch_data) == 2:  # Without attention_mask
                input_ids, labels = batch_data
                attention_mask = None
            else:
                self.logger.error(f"Unexpected batch_data format: {batch_data}")
                continue

            with torch.no_grad():
                # Include attention_mask if present
                if attention_mask is not None:
                    pred_results = self.model(input_ids, attention_mask)
                else:
                    pred_results = self.model(input_ids)

            self.write_stats(labels, pred_results)

        acc = self.show_stats()
        return acc

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("Total number of predicted collection items：%d" % (correct +wrong))
        self.logger.info("The predicted correct entries are: %d, the predicted incorrect entries are: %d" % (correct, wrong))
        self.logger.info("Prediction accuracy：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_logits in zip(labels, pred_results):
            pred_label = torch.argmax(pred_logits)  # Get the class with the highest score
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
