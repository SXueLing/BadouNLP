import os

import numpy as np
import torch
from loader import load_vocab, load_corpus, load_data, DataLoader
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from transformers import BertTokenizer


def main(config, save_weight=True):
    epoch_num = config["epoch_num"]        #训练轮数
    batch_size = config["batch_size"]       #每次训练样本个数
    train_sample = config["train_sample"]   #每轮训练总共训练的样本总数
    vocab = load_vocab("vocab.txt")       #建立字表
    corpus = load_corpus(config["corpus_path"])     #加载语料
    load = DataLoader(config, vocab)
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    model = TorchModel(config)    #建立模型
    evaluator = Evaluator(config, model)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = choose_optimizer(config, model)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            print("batch，%d" % batch)
            x, y = load_data(config, corpus, tokenizer) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(evaluator.eval("让他在半年之前，就不能做出", tokenizer))
        print(evaluator.eval("李慕站在山路上，深深的呼吸", tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(config["corpus_path"]).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == '__main__':
    from config import Config
    main(Config, False)
