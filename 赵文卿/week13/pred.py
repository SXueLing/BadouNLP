'''
Author: Zhao
Date: 2025-03-05 18:12:51
LastEditors: Please set LastEditors
LastEditTime: 2025-03-05 18:52:05
FilePath: pred.py
Description: 

'''
import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from evaluate import Evaluator
from config import Config

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
tuning_tactics = Config["tuning_tactics"]
logger.info(f"Start training with {tuning_tactics} tuning tactics")

if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
elif tuning_tactics == "p_tuning":
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

# 读取模型
model = TorchModel(Config)
# print(model.state_dict().keys())

# 获取模型
model = get_peft_model(model, peft_config)
# print(model.state_dict().keys())

# 加载预训练模型
state_dict = model.state_dict()


# 加载微调模型
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('week13/output/lora_tuning.pth')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('week13/output/p_tuning.pth')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('week13/output/prompt_tuning.pth')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('week13/output/prefix_tuning.pth')

# 更新模型参数
state_dict.update(loaded_weight)

# 加载模型参数
model.load_state_dict(state_dict)

# 评估
model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)