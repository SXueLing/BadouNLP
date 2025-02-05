import os
import re
from collections import defaultdict
from typing import Any

import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler, EVAL_DATALOADERS
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics import Metric
from transformers import BertModel
from config import Config
from loader import load_data




class TorchModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.class_num = config["class_num"]
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.fc = nn.Linear(self.bert.config.hidden_size, self.class_num)
        self.config = config

    def forward(self, x):
        out = self.bert(x)
        out = self.fc(out.last_hidden_state) # B N D -> B N CLS
        return out


class LightBert(L.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.encoder = TorchModel(config)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_id, labels = batch
        input_id, labels = input_id.to(self.device), labels.to(self.device)
        outputs = self.encoder(input_id)
        loss = F.cross_entropy(
            outputs.view(-1, self.encoder.class_num),
            labels.view(-1),
            ignore_index=-1
        )
        loss_val = {"loss": loss}
        self.log(
            "Loss",
            loss_val["loss"],
            prog_bar=True,
        )
        return loss_val

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.encoder.config["learning_rate"],
            weight_decay=0.02,
        )
        return optimizer

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_id, labels = batch
        input_id, labels = input_id.to(self.device), labels.to(self.device)
        outputs = self.encoder(input_id)
        sentences = self.trainer.val_dataloaders.dataset.sentences[
            batch_idx * Config["batch_size"]: (batch_idx + 1) * Config["batch_size"]
        ]
        evaluator = Evaluator(Config)
        result = evaluator.cal(labels, outputs, sentences)
        self.log_dict(result, prog_bar=True)
        return result


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int),
        }

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[1 : len(sentence) + 1]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
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

    def cal(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

        F1_scores = []
        result = {}
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            result[f"{key}_precision"] = precision
            result[f"{key}_recall"] = recall
            result[f"{key}_f1"] = F1
        result["macro_f1"] = torch.mean(torch.as_tensor(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        result["micro_f1"] = micro_f1
        return result