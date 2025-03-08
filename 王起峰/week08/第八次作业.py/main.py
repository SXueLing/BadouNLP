import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
模型训练主程序
"""
def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %.4f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return
if __name__ == "__main__":
    main(Config)
"""
训练轮数设置为 20 ，过程日志如下
2025-01-16 17:07:16,669 - __main__ - INFO - epoch 1 begin
2025-01-16 17:07:16,796 - __main__ - INFO - epoch average loss: 0.0607
2025-01-16 17:07:19,549 - __main__ - INFO - 开始测试第1轮模型效果：
2025-01-16 17:07:19,744 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:19,744 - __main__ - INFO - 预测正确条目：398，预测错误条目：66
2025-01-16 17:07:19,744 - __main__ - INFO - 预测准确率：0.857759
2025-01-16 17:07:19,744 - __main__ - INFO - --------------------
2025-01-16 17:07:19,744 - __main__ - INFO - epoch 2 begin
2025-01-16 17:07:22,850 - __main__ - INFO - epoch average loss: 0.0402
2025-01-16 17:07:29,417 - __main__ - INFO - 开始测试第2轮模型效果：
2025-01-16 17:07:29,579 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:29,580 - __main__ - INFO - 预测正确条目：399，预测错误条目：65
2025-01-16 17:07:29,580 - __main__ - INFO - 预测准确率：0.859914
2025-01-16 17:07:29,580 - __main__ - INFO - --------------------
2025-01-16 17:07:29,580 - __main__ - INFO - epoch 3 begin
2025-01-16 17:07:29,709 - __main__ - INFO - epoch average loss: 0.0308
2025-01-16 17:07:29,710 - __main__ - INFO - 开始测试第3轮模型效果：
2025-01-16 17:07:29,869 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:29,869 - __main__ - INFO - 预测正确条目：406，预测错误条目：58
2025-01-16 17:07:29,869 - __main__ - INFO - 预测准确率：0.875000
2025-01-16 17:07:29,869 - __main__ - INFO - --------------------
2025-01-16 17:07:29,869 - __main__ - INFO - epoch 4 begin
2025-01-16 17:07:33,024 - __main__ - INFO - epoch average loss: 0.0145
2025-01-16 17:07:33,024 - __main__ - INFO - 开始测试第4轮模型效果：
2025-01-16 17:07:33,179 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:33,180 - __main__ - INFO - 预测正确条目：404，预测错误条目：60
2025-01-16 17:07:33,180 - __main__ - INFO - 预测准确率：0.870690
2025-01-16 17:07:33,180 - __main__ - INFO - --------------------
2025-01-16 17:07:33,180 - __main__ - INFO - epoch 5 begin
2025-01-16 17:07:33,273 - __main__ - INFO - epoch average loss: 0.0151
2025-01-16 17:07:33,273 - __main__ - INFO - 开始测试第5轮模型效果：
2025-01-16 17:07:33,435 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:33,435 - __main__ - INFO - 预测正确条目：400，预测错误条目：64
2025-01-16 17:07:33,435 - __main__ - INFO - 预测准确率：0.862069
2025-01-16 17:07:33,435 - __main__ - INFO - --------------------
2025-01-16 17:07:33,435 - __main__ - INFO - epoch 6 begin
2025-01-16 17:07:33,516 - __main__ - INFO - epoch average loss: 0.0220
2025-01-16 17:07:33,516 - __main__ - INFO - 开始测试第6轮模型效果：
2025-01-16 17:07:33,671 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:33,671 - __main__ - INFO - 预测正确条目：400，预测错误条目：64
2025-01-16 17:07:33,671 - __main__ - INFO - 预测准确率：0.862069
2025-01-16 17:07:33,671 - __main__ - INFO - --------------------
2025-01-16 17:07:33,672 - __main__ - INFO - epoch 7 begin
2025-01-16 17:07:33,761 - __main__ - INFO - epoch average loss: 0.0037
2025-01-16 17:07:33,761 - __main__ - INFO - 开始测试第7轮模型效果：
2025-01-16 17:07:33,914 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:33,914 - __main__ - INFO - 预测正确条目：401，预测错误条目：63
2025-01-16 17:07:33,914 - __main__ - INFO - 预测准确率：0.864224
2025-01-16 17:07:33,914 - __main__ - INFO - --------------------
2025-01-16 17:07:33,915 - __main__ - INFO - epoch 8 begin
2025-01-16 17:07:33,970 - __main__ - INFO - epoch average loss: 0.0007
2025-01-16 17:07:33,970 - __main__ - INFO - 开始测试第8轮模型效果：
2025-01-16 17:07:34,122 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:34,123 - __main__ - INFO - 预测正确条目：403，预测错误条目：61
2025-01-16 17:07:34,123 - __main__ - INFO - 预测准确率：0.868534
2025-01-16 17:07:34,123 - __main__ - INFO - --------------------
2025-01-16 17:07:34,123 - __main__ - INFO - epoch 9 begin
2025-01-16 17:07:34,181 - __main__ - INFO - epoch average loss: 0.0021
2025-01-16 17:07:34,182 - __main__ - INFO - 开始测试第9轮模型效果：
2025-01-16 17:07:34,333 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:34,334 - __main__ - INFO - 预测正确条目：403，预测错误条目：61
2025-01-16 17:07:34,334 - __main__ - INFO - 预测准确率：0.868534
2025-01-16 17:07:34,334 - __main__ - INFO - --------------------
2025-01-16 17:07:34,334 - __main__ - INFO - epoch 10 begin
2025-01-16 17:07:34,401 - __main__ - INFO - epoch average loss: 0.0045
2025-01-16 17:07:34,401 - __main__ - INFO - 开始测试第10轮模型效果：
2025-01-16 17:07:34,553 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:34,553 - __main__ - INFO - 预测正确条目：404，预测错误条目：60
2025-01-16 17:07:34,553 - __main__ - INFO - 预测准确率：0.870690
2025-01-16 17:07:34,553 - __main__ - INFO - --------------------
2025-01-16 17:07:34,554 - __main__ - INFO - epoch 11 begin
2025-01-16 17:07:34,621 - __main__ - INFO - epoch average loss: 0.0046
2025-01-16 17:07:34,621 - __main__ - INFO - 开始测试第11轮模型效果：
2025-01-16 17:07:34,779 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:34,779 - __main__ - INFO - 预测正确条目：407，预测错误条目：57
2025-01-16 17:07:34,779 - __main__ - INFO - 预测准确率：0.877155
2025-01-16 17:07:34,779 - __main__ - INFO - --------------------
2025-01-16 17:07:34,779 - __main__ - INFO - epoch 12 begin
2025-01-16 17:07:34,835 - __main__ - INFO - epoch average loss: 0.0023
2025-01-16 17:07:34,835 - __main__ - INFO - 开始测试第12轮模型效果：
2025-01-16 17:07:34,997 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:34,997 - __main__ - INFO - 预测正确条目：408，预测错误条目：56
2025-01-16 17:07:34,997 - __main__ - INFO - 预测准确率：0.879310
2025-01-16 17:07:34,997 - __main__ - INFO - --------------------
2025-01-16 17:07:34,998 - __main__ - INFO - epoch 13 begin
2025-01-16 17:07:35,078 - __main__ - INFO - epoch average loss: 0.0134
2025-01-16 17:07:35,078 - __main__ - INFO - 开始测试第13轮模型效果：
2025-01-16 17:07:35,229 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:35,229 - __main__ - INFO - 预测正确条目：408，预测错误条目：56
2025-01-16 17:07:35,229 - __main__ - INFO - 预测准确率：0.879310
2025-01-16 17:07:35,229 - __main__ - INFO - --------------------
2025-01-16 17:07:35,229 - __main__ - INFO - epoch 14 begin
2025-01-16 17:07:35,294 - __main__ - INFO - epoch average loss: 0.0036
2025-01-16 17:07:35,294 - __main__ - INFO - 开始测试第14轮模型效果：
2025-01-16 17:07:35,458 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:35,459 - __main__ - INFO - 预测正确条目：409，预测错误条目：55
2025-01-16 17:07:35,459 - __main__ - INFO - 预测准确率：0.881466
2025-01-16 17:07:35,459 - __main__ - INFO - --------------------
2025-01-16 17:07:35,459 - __main__ - INFO - epoch 15 begin
2025-01-16 17:07:35,513 - __main__ - INFO - epoch average loss: 0.0051
2025-01-16 17:07:35,514 - __main__ - INFO - 开始测试第15轮模型效果：
2025-01-16 17:07:35,676 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:35,676 - __main__ - INFO - 预测正确条目：409，预测错误条目：55
2025-01-16 17:07:35,676 - __main__ - INFO - 预测准确率：0.881466
2025-01-16 17:07:35,676 - __main__ - INFO - --------------------
2025-01-16 17:07:35,676 - __main__ - INFO - epoch 16 begin
2025-01-16 17:07:35,731 - __main__ - INFO - epoch average loss: 0.0014
2025-01-16 17:07:35,731 - __main__ - INFO - 开始测试第16轮模型效果：
2025-01-16 17:07:35,892 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:35,892 - __main__ - INFO - 预测正确条目：411，预测错误条目：53
2025-01-16 17:07:35,893 - __main__ - INFO - 预测准确率：0.885776
2025-01-16 17:07:35,893 - __main__ - INFO - --------------------
2025-01-16 17:07:35,893 - __main__ - INFO - epoch 17 begin
2025-01-16 17:07:35,937 - __main__ - INFO - epoch average loss: 0.0000
2025-01-16 17:07:35,937 - __main__ - INFO - 开始测试第17轮模型效果：
2025-01-16 17:07:36,096 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:36,096 - __main__ - INFO - 预测正确条目：411，预测错误条目：53
2025-01-16 17:07:36,096 - __main__ - INFO - 预测准确率：0.885776
2025-01-16 17:07:36,097 - __main__ - INFO - --------------------
2025-01-16 17:07:36,097 - __main__ - INFO - epoch 18 begin
2025-01-16 17:07:36,142 - __main__ - INFO - epoch average loss: 0.0000
2025-01-16 17:07:36,142 - __main__ - INFO - 开始测试第18轮模型效果：
2025-01-16 17:07:36,299 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:36,299 - __main__ - INFO - 预测正确条目：411，预测错误条目：53
2025-01-16 17:07:36,299 - __main__ - INFO - 预测准确率：0.885776
2025-01-16 17:07:36,300 - __main__ - INFO - --------------------
2025-01-16 17:07:36,300 - __main__ - INFO - epoch 19 begin
2025-01-16 17:07:36,341 - __main__ - INFO - epoch average loss: 0.0000
2025-01-16 17:07:36,341 - __main__ - INFO - 开始测试第19轮模型效果：
2025-01-16 17:07:36,494 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:36,494 - __main__ - INFO - 预测正确条目：411，预测错误条目：53
2025-01-16 17:07:36,494 - __main__ - INFO - 预测准确率：0.885776
2025-01-16 17:07:36,494 - __main__ - INFO - --------------------
2025-01-16 17:07:36,495 - __main__ - INFO - epoch 20 begin
2025-01-16 17:07:36,533 - __main__ - INFO - epoch average loss: 0.0000
2025-01-16 17:07:36,533 - __main__ - INFO - 开始测试第20轮模型效果：
2025-01-16 17:07:36,675 - __main__ - INFO - 预测集合条目总量：464
2025-01-16 17:07:36,675 - __main__ - INFO - 预测正确条目：411，预测错误条目：53
2025-01-16 17:07:36,675 - __main__ - INFO - 预测准确率：0.885776
2025-01-16 17:07:36,676 - __main__ - INFO - --------------------
"""
