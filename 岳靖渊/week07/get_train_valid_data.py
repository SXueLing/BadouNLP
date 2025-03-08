import pandas as pd
import random
from sklearn.model_selection import train_test_split
import json
import os

def get_train_valid_data(path):
    # 判断对应的路径下是否以及存在json文件
    if os.path.exists(r"D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/data"):
        print("数据已经存在")
        return 
    
    df = pd.read_csv(path)
    # 确定好评差评数量
    all_len = len(df)
    label_1_len = len(df[df["label"] == 1])
    label_0_len = len(df[df["label"] == 0])

    label_counts = df['label'].value_counts()
    max_count = label_counts.max()  # 较多类别的数量

    balanced_data = []
    for label, count in label_counts.items():
        # 获取某一类别数据
        label_data = df[df['label'] == label]
        if count < max_count:
            label_data_oversampled = label_data.sample(max_count, replace=True, random_state=42)
        else:
            label_data_oversampled = label_data

        balanced_data.append(label_data_oversampled)

    # 合并所有类别的数据，并打乱顺序
    df = pd.concat(balanced_data).sample(frac=1, random_state=42).reset_index(drop=True)


    # 划分训练集和验证集，9:1 比例
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

    # 转换为 JSON 格式
    train_json = train_df.to_dict(orient='records')
    valid_json = valid_df.to_dict(orient='records')

    # 保存为 JSON 文件
    train_json_path = r"D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/data/train_tag_news.json"  # 修改保存路径
    valid_json_path = r"D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/week07/data/valid_tag_news.json"

    # 判断是否有对应的data文件夹
    if not os.path.exists(os.path.dirname(train_json_path)):
        os.makedirs(os.path.dirname(train_json_path))

    # 将训练集每条记录逐行写入
    with open(train_json_path, 'w', encoding='utf-8') as f:
        for record in train_df.to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')  # 每行一个 JSON 对象

    # 将验证集每条记录逐行写入
    with open(valid_json_path, 'w', encoding='utf-8') as f:
        for record in valid_df.to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')  # 每行一个 JSON 对象


    print("训练集和验证集划分完成")

if __name__ == '__main__':
    get_train_valid_data("D:/python_project_git/ai_study/week7 文本分类问题/week7 文本分类问题/文本分类练习数据集/文本分类练习.csv")
