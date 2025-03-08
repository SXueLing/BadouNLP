"""
@Project ：cgNLPproject 
@File    ：loader_my.py
@Date    ：2025/1/6 15:51 
"""
import torch
from transformers import BertTokenizer
import json
import jieba
from torch.utils.data import DataLoader,Dataset

class DatasetGenerator:
    def __init__(self, config, data_path):
        # super(Dataset, self).__init__()
        self.config = config
        self.vocab = load_vocab(config['vocab_path'])
        self.config["vocab_length"] = len(self.vocab)
        self.model_type = config['model_type']
        self.label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                       5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                       10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                       14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_index = dict((v, k) for k,v in self.label.items())
        self.config["class_num"] = len(self.label_index)
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        self.build_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def build_data(self, data_path):
        self.data = []
        # 读文件
        with open(data_path, encoding='utf8') as f:
            for line in f:
                # 拆解数据
                sentence = json.loads(line)
                tag = sentence['tag']
                title = sentence['title']
                if self.model_type == 'bert':
                    # 若是bert模型，则入参交由bert处理
                    input_ids = self.tokenizer.encode(title, max_length=self.config['max_length'], pad_to_max_length=True)
                else:
                    # 若是其他，则自己处理
                    input_ids = self.encode_input(title)
                # 转换为tensor
                input_ids = torch.LongTensor(input_ids)
                # 之所以在label加一个[]，是因为input_ids是一个输入（一个list），
                # 所以label是一个输出，也是list。
                # 也可以转换思维，label实际是[0,0,0,...,1,0]的list，只不过简化了，以对应位置的索引作为输出是[16]
                label = torch.LongTensor([self.label_index[tag]])
                self.data.append([input_ids, label])
        return


    def encode_input(self, sentence):
        # 将句子分词，然后转换成对应index
        input_ids = []
        words = jieba.lcut(sentence)
        for word in words:
            input_ids.append(self.vocab.get(word, self.vocab.get('[UNK]')))
        # 对输入数据进行长度统一
        input_ids = self.sentence_padding_index(input_ids)
        return input_ids

    def sentence_padding_index(self, input_ids):
        max_length = self.config['max_length']
        # 长的截断
        input_ids = input_ids[:max_length]
        # 短的补齐
        input_ids += [0] * (max_length - len(input_ids))
        return input_ids

def load_vocab(vocab_path):
    vocab_dic = {}
    with open(vocab_path, encoding='utf8') as f:
        for index, line in enumerate(f):
            word = line.strip()
            vocab_dic[word] = index + 1 #0留给padding位置，所以从1开始
    return vocab_dic


def load_data(config,data_path):
    dataset = DatasetGenerator(config,data_path)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataloader


if __name__ == "__main__":
    from config_my import Config
    dg = DatasetGenerator(Config,Config['valid_data_path'])
    print(dg[1])
