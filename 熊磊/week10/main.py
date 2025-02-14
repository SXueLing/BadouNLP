from config import Config
from model import getModel
from loader import load_vocab, load_corpus,build_dataset
import torch
import os
import numpy as np
from evaluate import generate_sentence


def main(save_weight=True):
    epochs = Config['epochs']
    batch_size = Config['batch_size']
    train_sample = 10000
    window_size = Config['window_size']
    vocab = load_vocab(Config['vocab_path'])
    corpus = load_corpus(Config['data_path'])
    model = getModel(Config, vocab)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, corpus, window_size) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            causal_mask = torch.tril(torch.ones((batch_size, x.shape[1], x.shape[1])))
            loss = model(x, y, causal_mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        # print(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # break
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(Config['data_path']).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    main(False)
