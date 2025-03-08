# 拆分数据集，保存为json格式，用于训练和测试
# 数据集文件为cvs格式，包含label和review两列

# 读取文件，统计数据
import pandas as pd
import os
from config import Config
print("当前工作目录：", os.getcwd())
data = pd.read_csv('week7/work/data/文本分类练习.csv') # 读取csv文件
print(data['label'].value_counts()) # 统计标签分布

# 拆分数据集
train_data = data.sample(frac=0.8, random_state=Config['seed']) # frac=0.8表示提取80%的数据作为训练集，random_state=Config['seed']表示随机种子
test_data = data.drop(train_data.index)
# 统计拆分后的数据,正负样本数，文本长度分布,并保存为json格式
print(train_data['label'].value_counts())
print(test_data['label'].value_counts())
print(train_data['review'].apply(lambda x: len(x)).describe())
print(test_data['review'].apply(lambda x: len(x)).describe())
# 保存为json格式
train_data.to_json('week7/work/data/train.json', orient='records', lines=True, force_ascii=False) # orient='records'表示每行为一个json对象，lines=True表示每行为一个json对象，force_ascii=False表示不转义非ascii字符
test_data.to_json('week7/work/data/test.json', orient='records', lines=True, force_ascii=False)
print('数据集拆分完成')
