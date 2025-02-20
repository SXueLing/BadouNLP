import torch.nn as nn
import torch
import json
from transformers import BertTokenizer
from transformers import BertModel
import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
sft测试的读取数据层

'''


class LanguageBertToSftLoader:
    def __init__(self, config, file_path):
        self.config = config
        self.path = file_path
        # 初始化参数
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.data = []
        self.json_data = []
        # 先前置读取数据
        self.load_data_pre()
        # 做数据处理
        self.load()
        logger.info('读取数据完毕')

    def load_data_pre(self):
        with open(self.path, 'r', encoding='utf8') as f:
            read_lines = f.readlines()
            max_length = 0
            for line in read_lines:
                json_str = json.loads(line)
                len_content = len(json_str['content'])
                len_content += len(json_str['title'])
                if len_content > max_length:
                    max_length = len_content
                self.json_data.append(json.loads(line))
            print(max_length)
            f.close()

    def load(self):
        for line in self.json_data:
            question = line['title']
            len_q = len(question)
            ans = line['content']
            len_a = self.config['max_length'] - len_q - 3
            input_ids = self.tokenizer.encode_plus(question, ans, pad_to_max_length=True)
            input_ids = input_ids['input_ids']
            input_ids = input_ids[:self.config['max_length']]
            labels_id = self.tokenizer.encode_plus(question, ans, pad_to_max_length=True)
            labels_id = labels_id['input_ids']
            labels_id.pop(0)
            labels_id = labels_id[:self.config['max_length']]
            for i in range(len_q + 1):
                labels_id[i] = -1
            # 组装mask
            '''
            q为问题字符  s为[sep]   c为[cls] a为回答字符  1代表能看到  0代表看不到  分为左上 左下  右上  右下
              c q q q s | a a a s
            c 1 1 1 1 1 | 0 0 0 0
            q 1 1 1 1 1 | 0 0 0 0
            q 1 1 1 1 1 | 0 0 0 0
            q 1 1 1 1 1 | 0 0 0 0
            s 1 1 1 1 1 | 0 0 0 0
            _ _ _ _ _ _ _ _ _ _ _ 
            a 1 1 1 1 1 | 1 0 0 0
            a 1 1 1 1 1 | 1 1 0 0
            a 1 1 1 1 1 | 1 1 1 0
            s 1 1 1 1 1 | 1 1 1 1
            '''
            left_up = torch.ones([len_q + 2, len_q + 2])  # 左上
            right_up = torch.zeros([len_q + 2, len_a + 1])  # 右上
            left_down = torch.ones([len_a + 1, len_q + 2])  # 左下
            right_down = torch.tril(torch.ones([len_a + 1, len_a + 1]))  # 右下
            up = torch.concat([left_up, right_up], dim=-1)  # 上面
            down = torch.concat([left_down, right_down], dim=-1)  # 下面
            mask = torch.concat([up, down], dim=0)  # all
            # 为了和x形状一样 h后面还需要做padding
            msg_length = (len_q + len_a + 3)
            retain_dim = self.config['max_length'] - msg_length
            if retain_dim > 0:
                retain_dim_right = torch.zeros([msg_length, retain_dim])
                retain_dim_down = torch.zeros([retain_dim, self.config['max_length']])
                mask = torch.concat([mask, retain_dim_right], dim=-1)
                mask = torch.concat([mask, retain_dim_down], dim=0)  # mask  = (max_length ,max_length)
            self.data.append([torch.LongTensor(input_ids), mask, torch.LongTensor(labels_id)])

    def sen_to_v(self, input):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(config: dict, path):
    ll = LanguageBertToSftLoader(config, path)
    dl = DataLoader(ll, config['batch_size'], shuffle=True)
    return dl


if __name__ == '__main__':
    from config import Config

    dl = load_data(Config, Config['train_data_path'])
    for index, value in enumerate(dl):
        print(index, value)
    pass
