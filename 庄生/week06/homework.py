import numpy as np
from transformers import BertModel

'''

手动实现Bert结构 （内含12层）
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"F:\巴豆NLP\week06 语言模型和预训练\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
total_params = 0
for value in state_dict.values():
    val_shape = value.shape
    if len(val_shape) == 1:
        total_params += val_shape[0]
    else:
        a, b = val_shape
        total_params += a * b
print(total_params)  # 102267648
