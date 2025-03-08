from config import Config
from model import getModel
from loader import load_vocab, build_dataset
import torch
import os
import numpy as np
from evaluate import generate_sentence


def main(save_weight=True):
    epochs = Config['epochs']
    batch_size = Config['batch_size']
    train_sample = 10000
    vocab = load_vocab(Config)
    # corpus = load_corpus(Config['data_path'])
    data = build_dataset(Config)
    model = getModel(Config, vocab)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epochs):
        model.train()
        watch_loss = []

        for index, batch_data in enumerate(data):
            x, y, mask = batch_data 
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            # causal_mask = torch.tril(torch.ones((batch_size, x.shape[1], x.shape[1])))
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
            
        # print(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        print(generate_sentence("第一套人民币估值四百多万 全国不超过10套", model, vocab))
        print(generate_sentence("盈亏指数分析：拜仁客场有险 尤文指数不利难取胜", model, vocab))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(Config['data_path']).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    main(False)
