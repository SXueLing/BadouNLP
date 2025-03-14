import torch
import torch.nn as nn
from loader import load_data
from config import Config
from model import TorchModel
import numpy as np
from evaluate import Evaluator
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
def main():
    train_data = load_data(Config, Config['data_path'])
    model = TorchModel()
    #大模型微调策略
    tuning_tactics = Config["tuning_tactics"]
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
    
    
    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())

    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule("fc").parameters():
            param.requires_grad = True
     # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        print("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    model.train()
    # 加载优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'])
    # 加载模型测试
    evaluate = Evaluator(Config, model, logger)
    for epoch in range(Config['epochs']):
        watch_loss = []
        for index, data in enumerate(train_data):
            if cuda_flag:
                data = [d.cuda() for d in data]
            inputs, labels = data
            optimizer.zero_grad()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print(f"Epoch {epoch+1} loss: {np.mean(watch_loss)}")
        evaluate.eval(epoch+1)
    
if __name__ == '__main__':
    main()