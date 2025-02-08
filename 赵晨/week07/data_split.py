import pandas as pd
from sklearn.model_selection import train_test_split
import json

# 定义一个函数来读取数据并创建DataFrame
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 分割每行的第一个字符（标签）和剩余部分（评论）
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                label, review = parts
                data.append({'label': int(label), 'review': review})
    return pd.DataFrame(data)

# 将DataFrame保存为JSON Lines格式
def save_as_json_lines(df, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in df.to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False))
            f.write('\n')

# 加载数据
data_df = load_data(r'H:\pyProj1\nlp_learn\week07\文本分类练习数据集\文本分类练习.csv')

# 将数据分为训练集和验证集
train_df, valid_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['label'])

# 保存为JSON Lines格式的文件
save_as_json_lines(train_df, r'H:\pyProj1\nlp_learn\week07\data\train_tag_homework.json')
save_as_json_lines(valid_df, r'H:\pyProj1\nlp_learn\week07\data\valid_tag_homework.json')
