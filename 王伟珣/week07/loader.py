import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer


class TextClassifyDataset(Dataset):
    def __init__(self, config):
        self.model_type = config['model_type']
        self.max_sentence_len = config['max_sentence_len']
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_bert_model_path'])

        self.vocab = load_vocab(config['vocab_path'])

        self.data = []
        data = pd.read_csv(config['data_path'])
        for i in range(len(data)):
            x = data['review'][i]
            y = data['label'][i]
            if self.model_type == 'bert':
                x = self.tokenizer.encode(x, max_length=self.max_sentence_len, pad_to_max_length=True)
            else:
                x = self.encode_sentence(x)

            self.data.append(
                [torch.LongTensor(x), torch.LongTensor([y])]
            )
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return x, y
    
    def encode_sentence(self, text):
        text = text if len(text) <= self.max_sentence_len else text[:self.max_sentence_len]
        encode_chars = []
        for char in text:
            encode_chars.append(self.vocab.get(char, self.vocab['[UNK]']))
        if len(encode_chars) < self.max_sentence_len:
            encode_chars += [0] * (self.max_sentence_len - len(encode_chars))
        return encode_chars
    

def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding='utf8') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            vocab[word] = idx + 1
    return vocab


def load_data(config):
    total_dataset = TextClassifyDataset(config)
    N = total_dataset.__len__()
    n_train = int(N * config['train_percent'])
    n_test = N - n_train
    train_dataset, test_dataset = random_split(total_dataset, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    print("Preprocess for load data ...")
    length_dict = {}
    length_array = []
    data = pd.read_csv("TextClassifyData.csv")
    for i in range(len(data)):
        x = data['label'][i]
        y = data['review'][i]
        sentence_len = len(y)
        length_array.append(sentence_len)
        if sentence_len not in length_dict.keys():
            length_dict[sentence_len] = 1
        else:
            length_dict[sentence_len] += 1
    
    print(max(length_array))

    sorted_keys = sorted(length_dict.keys())

    import matplotlib.pyplot as plt

    num, total = 0, len(data)
    labels, values, percent = [], [], []
    for key in sorted_keys:
        labels.append(key)
        values.append(length_dict[key])
        cur_num = length_dict[key]
        num += cur_num
        print(key, num / total, cur_num, total-num, total)

    plt.bar(labels, values)
    plt.show()
