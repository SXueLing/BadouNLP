from Config import config
from loader import Dataloader, vanilla_encode, load_vocab, produce_mask
from model import Model, optim
import torch


def predict(sentence, model):
    vocab = load_vocab(config["vocab_path"])
    reverse_vocab = list(vocab.keys())
    pre_char = ''
    with torch.no_grad():
        while len(sentence) < 30:
            sentence += pre_char
            x = vanilla_encode([sentence])
            index = model(x)[0][-1]
            pre_char = reverse_vocab[index]
        print(sentence)


if __name__ == "__main__":
    train_data = Dataloader()
    model = Model(config)
    optimizer = optim(config, model)
    model.train()
    for epoch in range(config["epochs"]):
        loss_item = []
        for x, y in train_data.dataloader:
            loss = model(x=x, y=y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item.append(loss.item())
        print(f"第%d轮的loss为：%f" % (epoch+1, sum(loss_item)/len(loss_item)))
        s1 = "李清面色微凝，不确信道"
        predict(s1, model)






