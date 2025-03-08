"""
3个模型在不同学习率 和pooling_style 准确率表现
model	leanring_rate	hide_size	pooling_style 	acc	    预测100条耗时
rnn 	0.001	        128	       avg	          0.866	  0.03
rnn	  0.0001	      128	       avg	          0.853	  0.03
rnn	  0.001	        128	       max	          0.8796	0.029
rnn	  0.0001	      128	       max	          0.8489	0.032
bert	0.001	        128	       avg	          0.772	  0.12
bert	0.0001	      128	       avg	          0.89	  0.101
bert	0.001	        128	       max	          0.658	  0.095
bert	0.0001	      128	       max	          0.896	  0.093
lstm	0.001	        128	       avg	          0.876	  0.014
lstm	0.0001	      128	       avg	          0.854	  0.014
lstm	0.001	        128      	 max	          0.878	  0.015
lstm	0.0001	      128	       max	          0.853	  0.014

"""

import csv
import json
import random

## 读取csv  随机挑选20%数据做验证集
def split_data_for_validation(data, validation_percentage=0.2):
    # 计算验证集的大小
    validation_size = int(len(data) * validation_percentage)

    # 随机挑选验证集
    validation_set = random.sample(data, validation_size)

    # 计算训练集
    training_set = [item for item in data if item not in validation_set]

    return training_set, validation_set
def load_csv(path):
    data = []
    with open(path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # 逐行读取CSV文件内容
        for index, row in enumerate(reader):
            if index == 0:
                continue
            row[0] = int(row[0])
            data.append(row)

    training_set, validation_set = split_data_for_validation(data)
    with open("../data/training_set.json","w", encoding="utf-8") as f:
        f.write(json.dumps(training_set, ensure_ascii=False, indent=4))
    with open("../data/validation_set.json","w", encoding="utf-8") as f:
        f.write(json.dumps(validation_set, ensure_ascii=False, indent=4))

load_csv("datas.csv")
