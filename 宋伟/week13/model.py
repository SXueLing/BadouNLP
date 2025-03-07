import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model
from config import Config

class LoRANERModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        base_model = AutoModelForTokenClassification.from_pretrained(Config.model_name, num_labels=num_labels)
        lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            lora_dropout=Config.lora_dropout,
            bias="none",
            target_modules=["query", "value"]  # 选择 LoRA 适配的层
        )
        self.model = get_peft_model(base_model, lora_config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
