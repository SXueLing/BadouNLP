import torch
import logging
from model import NerModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from evaluate import Evaluator
from config import Config

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

peft_config = LoraConfig(
            task_type="TOKEN_CLS",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )

#重建模型
model = NerModel

model = get_peft_model(model, peft_config)
model.to(device)
state_dict = model.state_dict()

#将微调部分权重加载
loaded_weight = torch.load('model_path/lora_tuning.pth')

print(loaded_weight.keys())
state_dict.update(loaded_weight)

#权重更新后重新加载到模型
model.load_state_dict(state_dict)

#进行一次测试
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)
