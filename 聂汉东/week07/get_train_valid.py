import csv
import json

import pandas as pd

path = './data/文本分类练习.csv'
train_json_path = './data/train_tag_news.json'
valid_json_path = './data/valid_tag_news.json'


def writ_to_json(path, json_data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in json_data:
            # 将每个JSON对象转换为字符串并写入文件
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def writ_to_csv(path,data):
    with open(path,'a+',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def get_train_valid(path, train_ratio=0.8):
    data = pd.read_csv(path)
    # 数据集总数
    all_label_count = len(data)
    # 标签为1的数量
    true_label_count = data[data['label'] == 1]
    # 标签为0的数量
    false_label_count = data[data['label'] == 0]

    max_count = data['label'].idxmax()
    print(all_label_count, len(true_label_count), len(false_label_count))

    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()

    label_counts = data['label'].value_counts()
    for label, count in label_counts.items():
        label_data = data[data['label'] == label]
        # 训练集数量
        train_size = int(count * train_ratio)
        # 验证集数量
        valid_size = count - train_size
        # 开始划分数据集
        train_label_data = label_data.sample(n=train_size, random_state=42)
        valid_label_data = label_data.drop(train_label_data.index)
        # 合并数据集
        train_data = pd.concat([train_data, train_label_data])
        valid_data = pd.concat([valid_data, valid_label_data])

    # 将数据转换为字典列表
    train_dict = train_data.to_dict(orient='records')
    valid_dict = valid_data.to_dict(orient='records')

    return train_dict, valid_dict

if __name__ == '__main__':
    train_data, valid_data = get_train_valid(path)
    writ_to_json(train_json_path, train_data)
    writ_to_json(valid_json_path, valid_data)
    print("数据集划分完毕")
