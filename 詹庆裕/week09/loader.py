from torch.utils.data import DataLoader, Dataset
import torch
from Config import config
import random
from transformers import BertTokenizer

"""
单个样本获取类
"""
class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        self._init_components()

    def _init_components(self):
        sentences = []
        tags = []
        data = load_corpus(config["corpus_path"])
        for i in range(config["train_num"]):
            start_index = random.randint(0, len(data) -1 -config["window_size"])
            end_index = start_index + config["window_size"]
            input_id = data[start_index:end_index]
            label = data[start_index + 1: end_index + 1]
            sentences.append(input_id)
            tags.append(label)
        self.input_ids, self.labels = vanilla_encode(sentences, tags)


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.LongTensor(self.input_ids[index]), torch.LongTensor(self.labels[index])

def vanilla_encode(sentences, tags=None):
    input_ids = []
    labels = []
    vocab = load_vocab(config["vocab_path"])
    if tags is None:
        for sentence in sentences:
            sentence_id = [vocab.get(word, vocab["<UNK>"]) for word in sentence]
            input_ids.append(sentence_id)
        return torch.LongTensor(input_ids)
    for sentence, tag in zip(sentences, tags):
        sentence_id = [vocab.get(word, vocab["<UNK>"]) for word in sentence]
        tag_id = [vocab.get(word, vocab["<UNK>"]) for word in tag]
        input_ids.append(sentence_id)
        labels.append(tag_id)
    return input_ids, labels

"""
批量获取数据
"""
class Dataloader:
    def __init__(self):
        self.dataset = MyDataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"])


"""
获取mask掩码
"""
def produce_mask():
    attention_mask = torch.ones((256, 10))  # 形状 [256, 10]
    return attention_mask


"""
获取词表
"""
def load_vocab(path):
    vocab_dict = {"pad":0}
    with open(path, 'r', encoding='utf-8') as f:
        vocab = f.read().split('\n')
        for index, word in enumerate(vocab):
            vocab_dict[word] = index + 1
    return vocab_dict

"""
获取语料
"""
def load_corpus(path):
    data = ''
    with open(path, 'r') as f:
        for line in f:
            data += line.strip()
    return data
