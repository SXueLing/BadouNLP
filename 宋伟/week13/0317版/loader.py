from transformers import BertTokenizer
import json
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataSet(Dataset):
    """docstring for MyDataSet"""

    def __init__(self, data_path, config):
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.schema = self.load_schema(config['schema_path'])

        self.load()

    def load(self):
        self.data = []
        self.sentences = []
        # 内部的每一条数据，是一个字典形式，分别储存，inputs和labels,内部是大量的数据
        # 每个句子  #
        with open(self.data_path, encoding='utf-8') as f:
            segments = f.read().split('\n\n')
            for segment in segments:
                data = {}  # {inputs:tokens_id ,labels:schema_ids},存储句子，及其对应的标签
                sentence = []
                label_ids = [self.schema['O']]  # 前置一个cls对应的无效标签
                for line in segment.split('\n'):
                    if line.strip() == '':
                        continue
                    char, label = line.split(' ')
                    sentence.append(char)
                    label_ids.append(self.schema[label])
                sentence = ''.join(sentence)
                self.sentences.append(sentence)
                input_ids = self.encode_sentence(sentence)
                label_ids = self.padding(label_ids, self.config['label_padding_token'])  # 使用-1进行空余位置的填充
                data['input_ids'] = input_ids
                data['labels'] = label_ids
                self.data.append(data)
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_schema(self, schema_path):
        with open(schema_path, encoding='utf-8') as f:
            schema_dict = json.load(f)
        return schema_dict

    def encode_sentence(self, sentence: list, padding=True)->torch.Tensor:
        # 返回整型一维张量，[seq_len,]，同时实现填充，切割，其中填充0
        sentence_ids = self.tokenizer.encode(sentence, return_tensors='pt',
                                               max_length=self.config['max_length'],
                                               padding = 'max_length',
                                               truncation = True).squeeze(0)  # 进行填充和截断
        return sentence_ids


    def padding(self, label_ids: list,pad_token)->torch.Tensor:
        # 进行标签的填充处理，这里的标签填充不能是-1，
        # 返回一个一维张量；[seq,]
        label_ids = label_ids[:self.config['max_length']]
        label_ids += [pad_token]*(self.config['max_length']- len(label_ids))

        label_ids = torch.LongTensor(label_ids)
        return label_ids

def load_data(data_path,config,shuffle=True):
    # 每次批次的数据 [batch,seq]
    dataset = MyDataSet(data_path,config)
    dataloader = DataLoader(dataset=dataset,batch_size=config['batch_size'],shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    from config import Config
    # dataset = MyDataSet(Config['train_data'],Config)
    train_data = load_data(Config['train_data'],Config)
    for i,batch_data in enumerate(train_data):
        print(batch_data['input_ids'].shape)
        print(batch_data['labels'].shape) 
        break
