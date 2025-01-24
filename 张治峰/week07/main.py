# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
from config import Config
from dataset import load_data
from model import TorchModel
import logging
from evaluate import Evaluator
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""
seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    # 标识是否使用gpu
   cuda_flag = torch.cuda.is_available()
   model_path =  config["model_path"]
   if not os.path.isdir(model_path):
      os.mkdir(model_path)
   train_sample,valid_sample = build_sample(config["data_path"],config["train_data_percentage"]) 
   train_data = load_data(train_sample,config)

   model = TorchModel(config)
   if cuda_flag:
       logger.info("gpu可以使用，迁移模型至gpu")
       model = model.cuda()
   epochs = config["epoch"]
   optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"]) 
    #加载效果测试类
   evaluator = Evaluator(config,valid_sample, model, logger)
   for epoch in range(epochs):
      model.train()
      train_loss = []
      for index, batch_data in enumerate(train_data):
         if cuda_flag:
              batch_data = [d.cuda() for d in batch_data]
         optimizer.zero_grad()
         input_ids, labels = batch_data
         loss = model(input_ids,labels)
         loss.backward()
         optimizer.step()
         train_loss.append(loss.item())
         if index % int(len(train_data) / 2) == 0:
            logger.info("batch loss %f" % loss)
      logger.info("epoch %d average loss: %f" %(epoch+1 ,np.mean(train_loss)))  
      acc = evaluator.eval(epoch)
   return acc

def build_sample(data_path,percentage): # 根据百分比 划分训练集和测试集
    with open(data_path,encoding="utf-8") as f:
           lines = f.readlines()
           positives_sample = []
           negative_sample = []
           for line in lines[1:]:
               data = line.strip().split(",")
               label = int(data[0])
               content = data[1]
               if label==1:
                    positives_sample.append([label,content])
               else:
                    negative_sample.append([label,content])
    positives_index = int(len(positives_sample)*percentage)
    negative_index = int(len(negative_sample)*percentage)
    train_sample = positives_sample[0:positives_index]+negative_sample[0:negative_index]
    valid_sample = positives_sample[positives_index:]+negative_sample[negative_index:]
    return train_sample,valid_sample
if __name__ == "__main__":
   # main(Config)
   output_header = ["model_type", "learning_rate", "hidden_size" ,"batch_size", "pooling_style", "accuracy"]
   with open(Config["result_path"], 'w') as file:  # 结果写入文件
       file.write(" ".join(map(str, output_header)) + "\n")
   # 对比所有模型
   for model in ['lstm', 'gru', 'rnn', 'bert']:
       Config["model_type"] = model
       for lr in [1e-3,1e-4]:
           Config["learning_rate"] = lr
           for hidden_size in [128]:
               Config["hidden_size"] = hidden_size
               for batch_size in [128]:
                   Config["batch_size"] = batch_size
                   for pooling_style in ["avg","max"]:
                       Config["pooling_style"] = pooling_style
                       accuracy = main(Config)
                       print("最后一轮准确率：", accuracy, "当前配置：", Config)
                       with open(Config["result_path"], 'a') as file:  # 结果写入文件
                           file.write(" ".join(map(str, [model, lr, hidden_size, batch_size, pooling_style, accuracy])) + "\n")








