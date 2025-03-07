
import torch
from torch.utils.data import Dataset, DataLoader
import json
import jieba
from transformers import BertTokenizer


class MyDataSet(Dataset):
    """docstring for MyDataSet"""

    def __init__(self, data_path, config):
        super(MyDataSet, self).__init__()
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])

        # 使用bert 就需要使用其对应的分词器
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        # print(list(self.vocab.items())[782])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []  # 原数据格式不太合适，这里获取句子的文本
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    # 加载为{标签：类别}
    def load_schema(self, schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 加载数据，主要是为了获取创建self.data =[inputs,labels]

    def load(self):
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = [8]  # 现在首位进行一个占位lable
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split(" ")
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence, use_bert=self.config['use_bert'])
                labels = self.padding(labels, -1)
                self.data.append([input_ids, torch.LongTensor(labels)])

    # 将一句话进行id化，注意padding中的截断或补全
    def encode_sentence(self, sentence: list, padding=True, use_bert=False):
        if use_bert:
            model_inputs = self.tokenizer.encode_plus(sentence,
                                             return_tensors='pt',
                                             max_length=self.config["max_length"],
                                             padding='max_length',
                                             truncation=True)
            model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
            model_inputs['attention_mask']  =model_inputs['attention_mask'].squeeze(0)
            model_inputs.pop('token_type_ids')
            input_id = model_inputs
        else:
            input_id = []
            if self.config["vocab_path"] == "words.txt":
                for word in jieba.cut("".join(sentence)):
                    word_id = self.vocab.get(word, self.vocab["[UNK]"])
                    input_id.append(word_id)
            elif self.config["vocab_path"] == "chars.txt":
                for char in sentence:
                    char_id = self.vocab.get(char, self.vocab["[UNK]"])
                    input_id.append(char_id)
            if padding:
                input_id = self.padding(input_id)
            input_id = torch.LongTensor(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        # 补全句子时，使用0进行补全
        # 补全lables时，使用-1 进行补全
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token]*(self.config["max_length"]-len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 将0号位置留给padding
    return token_dict


def data_load(data_path, config, shuffle=True):
    dataset = MyDataSet(data_path, config)
    dataloader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    from config import Config
    # Config['use_bert'] = False
    dataloader = data_load(
        Config['train_data_path'], Config)   # 其中的batch-size,是16

    # batch_data ,·[batch,sequence],貌似是进行了unsqueeze的操作
    for i, batch_data in enumerate(dataloader):
        # print("输入一个批次的数据尺寸", batch_data[0]["attention_mask"].cuda())
        print("对应一个批次的标签尺寸", batch_data[1].cuda())
        pass
        break

    quit()
    x = "今天天气不错"
    Config['use_bert'] = True
    Config['max_length'] = 30
    dataset = MyDataSet(Config["train_data_path"], Config)


    """
    entences[0]:  cls     他    说    :     中   国     政    府    对     目   前     南    亚    出   现    的    核    军     备   sep       竞赛的局势深感忧虑和不安。
    sentence_id :[ 101,  800, 6432,  131,  704, 1744, 3124, 2424, 2190, 4680, 1184, 1298, 762, 1139, 4385, 4638, 3417, 1092, 1906,  102]
    label_id:    [ 8,     8,   8,     8,     1,   5,     5,   5,   8,     3 ,  7,     0,    4,    8,   8,     8,    8,     8,    8,   8]
    """
    print(dataset[0])
    # not bert
    sentence_id = dataset.encode_sentence(x, use_bert=False)
    # [257, 978, 978,  2183, 165, 4231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sentence_id = dataset.encode_sentence(x, use_bert=True)
    # [101, 791, 1921, 1921, 3698, 679, 7231, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # [2,   5,   0,    0,    0,     0,  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    # print(sentence_id)
