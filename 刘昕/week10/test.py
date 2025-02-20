from transformers import BertTokenizer
import re
def split_numbers(text):
    """将连续的数字拆分成单个数字并用空格分隔"""
    # 在连续数字的每个字符间插入空格（保留原文本结构）
    processed = re.sub(r'([a-zA-Z\d])(?=[a-zA-Z\d])', r'\1 ', text)
    return processed

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# text = "我lsd25岁了"
# text = split_numbers(text)
# print(text)
# print(tokenizer.encode(text))
# corpus = ""
# with open('corpus.txt', encoding="gbk") as f:
#     for line in f:
#         corpus += line.strip()
# print(split_numbers(corpus[2006299:2006309]))
import torch
a = torch.tril(torch.ones(3, 3))
print(a)
