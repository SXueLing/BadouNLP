import json
import csv
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
with open('..\文本分类练习数据集\文本分类练习.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    data = [{'label': int(row[0]), 'review': row[1]} for row in reader]

# 根据 label 分割数据
label_1_data = [item for item in data if item['label'] == 1]
label_0_data = [item for item in data if item['label'] == 0]

# 划分数据集
train_label_1, valid_label_1 = train_test_split(label_1_data, test_size=0.1, random_state=42)
train_label_0, valid_label_0 = train_test_split(label_0_data, test_size=0.1, random_state=42)

# 合并数据集
train_data = train_label_1 + train_label_0
valid_data = valid_label_1 + valid_label_0

# 修改保存路径
train_json_path = r".\data\train_tag_news.json"
valid_json_path = r".\data\valid_tag_news.json"

# 保存数据到 JSON 文件
with open(train_json_path, 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

with open(valid_json_path, 'w', encoding='utf-8') as f:
    for item in valid_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')