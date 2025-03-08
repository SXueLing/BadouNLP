# -*- coding : UTF-8 -*- #
from transformers import BertModel
bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
print(state_dict.keys())


def sum_num1( tuple1):
    a = 1
    for i in tuple1:
        a = a * i
    return a

def count_num(name):
    num=sum_num1(name.shape)
    return num


sum_num=0
for key in state_dict.keys():
    num=count_num(state_dict[key])
    sum_num+=num

print(sum_num)
print("一共",sum_num/1000000000,"B")


# embedding层  +  1  *  transformers
