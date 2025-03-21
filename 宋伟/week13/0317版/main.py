import torch
from config import Config
from loader import load_data
from model import TorchModel,choose_optimizer
from evaluate import Evaluator
import numpy as np
import logging
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
import os


logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 随机种子
seed=Config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def peft_wrapper(model):
    peft_config = LoraConfig(r=8,lora_alpha=32,lora_dropout=0.1,target_modules=["query","value"])
    return get_peft_model(model=model,peft_config=peft_config)


def main(config):
    # 创建模型保存目录：
    if not os.path.isdir(config['model_path']):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config['train_data'],config)
    # 加载模型
    model = TorchModel(config)
    model = peft_wrapper(model)

    # 是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，将模型迁移到gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config,model)
    # 加载测试类
    evaluator = Evaluator(config,model,logger)

    # 进行训练
    for epoch in range(config['epoch']):
        epoch += 1
        logger.info(f"epoch {epoch} start:")
        model.train()
        train_loss = []
        for idx,batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = {key:data.cuda() for key,data in batch_data.items()}
            loss = model(**batch_data)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if idx % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)

    return 

if __name__ == '__main__':
    main(Config)