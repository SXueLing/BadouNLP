import os

import torch.nn
from peft import get_peft_model,LoraConfig
from config import Config
from model import NerModel,choose_optimizer
import logging
from loader import load_data
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import numpy as np
from evaluate import Evaluator


"""
模型训练主程序
"""
def main(config):
    # 模型保存路径 不存在则创建
    if not os.path.isdir(config["model_save_path"]):
        os.mkdir(config["model_save_path"])

    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

    train_data = load_data(config["train_data_path"],config)

    tuning_tactics = config["tuning_tactics"]
    num_labels = config["num_labels"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            task_type="TOKEN_CLS",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )

    model = get_peft_model(NerModel,peft_config=peft_config).to(device)

    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True
    # 选择优化器
    optimizer =  choose_optimizer(config,model)

    for epoch in range(config["epoch"]):
        model.train()
        train_loss = []
        for index,batch_data in enumerate(train_data):
            input_ids,labels = batch_data
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output =  model(input_ids)[0]
            loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(output.view(-1,num_labels),labels.view(-1))
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_save_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  # 保存模型权重
    return acc

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)
if __name__ == "__main__":
    main(Config)
