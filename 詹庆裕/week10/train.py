import torch
import random
from init_model import  model, choose_optimizer
from config_file import Config
from loader import all_data
from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np

def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def encode_str(query, tokenizer):
    encode = tokenizer.encode_plus(
        query,
        max_length=30,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encode["input_ids"], encode["attention_mask"]



def train():
    sftModel = model()
    optimizer = choose_optimizer(sftModel)
    train_data = all_data()
    sftModel.train()
    for i in range(Config["epoch"]):
        loss_item = []
        for batch in tqdm(train_data):
            input_ids = batch["input_ids"]
            masks = batch["masks"]
            labels = batch["target"]
            loss = sftModel(input_ids, masks, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item.append(loss.item())
        print("第%d轮的loss为%f：" % (i+1, sum(loss_item)/len(loss_item)))
        tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
        result = generate_sentence("怀孕后为啥总拉肚子 孕妇拉肚子吃什么",sftModel, tokenizer)
        print(result)


if __name__ == "__main__":
    train()
