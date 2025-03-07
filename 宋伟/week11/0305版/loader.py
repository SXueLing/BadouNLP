import json
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class MyDataSet(Dataset):
    """docstring for MyDataSet"""

    def __init__(self, config):
        super(MyDataSet, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.corpus = load_corpus(config['corpus_path'])
        self.max_length = config['max_length']

        self.load()

    def load(self):
        self.dataset = []
        for i, single_data in enumerate(self.corpus):
            data: dict = {}
            prompt = single_data['title']
            answer = single_data['content']

            prompt_encode = self.tokenizer.encode(
                prompt, add_special_tokens=False)
            answer_encode = self.tokenizer.encode(
                answer, add_special_tokens=False)

            x = [self.tokenizer.cls_token_id] +\
                prompt_encode +\
                [self.tokenizer.sep_token_id] +\
                answer_encode +\
                [self.tokenizer.sep_token_id]

            y = len(prompt_encode)*[-1] +\
                [-1] +\
                answer_encode +\
                [self.tokenizer.sep_token_id] +\
                [-1]

            # mask 矩阵，prompt 信息可以获取，answer仅可以获取单向信息
            mask = create_mask(len(prompt_encode), len(answer_encode))
            # padding 填充序列
            x = x[:self.max_length] + [0]*(self.max_length-len(x))
            y = y[:self.max_length] + [0]*(self.max_length-len(y))

            x = torch.LongTensor(x)
            y = torch.LongTensor(y)  # [seequence,]
            # 将掩码也进行填充处理
            mask = pad_mask(mask, (self.max_length,self.max_length))
            data['inputs'] = x
            data['labels'] = y
            data['mask'] = mask
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return self.dataset[idx]



def load_corpus(path):
    # 获取数据，单条数据为{'title':str,'content':str}
    corpus = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            corpus.append({'title': line['title'], 'content': line['content']})
    return corpus


def create_mask(prompt_len, answer_len):
    len_prompt = prompt_len + 2
    len_answer = answer_len + 1
    mask = torch.ones(len_prompt+len_answer, len_answer+len_prompt)
    for i in range(len_prompt):
        mask[i, len_prompt:] = 0

    for i in range(len_answer):
        mask[len_prompt+i, len_prompt+i+1:] = 0
    return mask


def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_height, target_width = target_shape

    result = torch.zeros(target_shape,
                         dtype=tensor.dtype,
                         device=tensor.device)
    h_start, w_start = 0, 0
    h_end, w_end = min(height, target_height), min(width, target_width)

    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start,
                                                  :w_end - w_start]
    return result                                           


def data_load(config,shuffle=True):
    dataset = MyDataSet(config)
    dataloader = DataLoader(dataset,batch_size= config['batch_size'],shuffle=shuffle)

    return dataloader



if __name__ == '__main__':
    from config import Config
    # dataset = MyDataSet(Config)
    dataloader = data_load(Config)
    for idx,batch_data in enumerate(dataloader):
        print(batch_data['inputs'].shape)   # [32,50]
        break

    
