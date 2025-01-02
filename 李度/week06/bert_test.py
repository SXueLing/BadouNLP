from bert import BERT
import torch
import math
import numpy as np
from transformers import BertModel

def main():
    bert = BertModel.from_pretrained(r"./bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    bert.eval()
    x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
    torch_x = torch.LongTensor([x])  # pytorch形式输入
    seqence_output, pooler_output = bert(torch_x)
    print(seqence_output.shape, pooler_output.shape)
    print(bert.state_dict().keys())
    self_bert = BERT(state_dict)
    output, pooler_output = self_bert(x)
    print(output, pooler_output)

if __name__ == "__main__":
    main()
