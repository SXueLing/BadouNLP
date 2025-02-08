import pandas as pd
import json
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
df = pd.read_csv(r"../文本分类练习.csv")

# 将数据转换为字典列表
data_list = [{"tag": row["label"], "title": row["review"]} for _, row in df.iterrows()]

# 划分数据集：80% 训练集, 20% 测试集
train_data, valid_data = train_test_split(data_list, test_size=0.2, random_state=42)

# 保存训练集到 JSON 文件，每行一个 JSON 对象
with open("../data/train_text_classify.json", "w", encoding="utf-8") as train_file:
    for item in train_data:
        json.dump(item, train_file, ensure_ascii=False)
        train_file.write("\n")  # 每个 JSON 对象占一行

# 保存测试集到 JSON 文件，每行一个 JSON 对象
with open("../data/valid_text_classify.json", "w", encoding="utf-8") as valid_file:
    for item in valid_data:
        json.dump(item, valid_file, ensure_ascii=False)
        valid_file.write("\n")  # 每个 JSON 对象占一行

