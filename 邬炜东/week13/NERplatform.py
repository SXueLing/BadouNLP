import torch
from transformers import BertTokenizer
from config import Config
from model import All_Models
import jieba
import json
import re
from peft import LoraConfig, get_peft_model


class NER_platform:
    def __init__(self, config):
        self.config = config
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.model = self.load_model()
        self.vocab = self.load_vocab(self.config["vocab_path"])
        self.schema = self.schema_load(self.config["schema_data_path"])

    def load_model(self):
        model = All_Models(self.config)
        if self.config["is_lora"]:
            lora = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]
            )
            model = get_peft_model(model, lora)
            for param in model.crf.parameters():
                param.requires_grad = True
            for param in model.classify.parameters():
                param.requires_grad = True
        trainable_parameters = {
            name: param for name, param in model.named_parameters() if param.requires_grad
        }
        lora_parameters = torch.load(self.config["model_path"])
        print(trainable_parameters)
        origin_parameters = model.state_dict()
        origin_parameters.update(lora_parameters)
        model.load_state_dict(origin_parameters)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def encode_sentence(self, sentence):
        if self.config["model_type"] == "bert":
            input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], padding='max_length',
                                             truncation=True, return_attention_mask=True)
        else:
            input_id = []
            if self.config["vocab_path"] == "words.txt":
                sentence = jieba.lcut(sentence)
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id, self.vocab["padding"])
        input_id = torch.LongTensor(input_id)
        input_id = input_id.reshape(1, -1).cuda()
        return input_id

    def padding(self, input_id, padding_idx):
        input_id = input_id[:self.config["max_length"]]
        input_id += [padding_idx] * (self.config["max_length"] - len(input_id))
        return input_id

    def load_vocab(self, vocab_path):
        vocab = {}
        vocab["padding"] = 0
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                vocab[line.strip()] = index + 1
            self.config["vocab_size"] = len(vocab)
        return vocab

    # schema是schema文件的字典格式
    def schema_load(self, schema_path):
        with open(schema_path, encoding='utf-8') as f:
            schema = json.loads(f.read())
            return schema

    def get_predict(self, input_sentence):
        input_id = self.encode_sentence(input_sentence)
        predict = self.model(input_id)
        self.get_ner(predict[0], input_sentence)

    def get_ner(self, predict, input_sentence):
        all_str = ''.join([str(int(number)) for number in predict])
        for location in re.finditer("(04+)", all_str):
            start, end = location.span()
            print(f"location:{input_sentence[start:end]}")

        for location in re.finditer("(15+)", all_str):
            start, end = location.span()
            print(f"organization:{input_sentence[start:end]}")

        for location in re.finditer("(26+)", all_str):
            start, end = location.span()
            print(f"person:{input_sentence[start:end]}")

        for location in re.finditer("(37+)", all_str):
            start, end = location.span()
            print(f"time:{input_sentence[start:end]}")


if __name__ == "__main__":
    ner = NER_platform(Config)
    sentence = "在1月23日晚间的直播中，王子异透露自己在法国巴黎直播求婚被马保国骂了五分钟，他说黄圣依在电话里骂：“你太过分了，为什么不跟我商量。"
    ner.get_predict(sentence)
