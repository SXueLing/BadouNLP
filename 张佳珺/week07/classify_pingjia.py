import pandas as pd
import json
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
df = pd.read_csv(r"D:/八斗课程/AI人工智能培训/第七周 文本分类/week7 文本分类问题/文本分类练习.csv")

# 将数据转换为字典列表
data_list = [{"tag": row["label"], "title": row["review"]} for _, row in df.iterrows()]

# 划分数据集：80% 训练集, 20% 测试集
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# 保存训练集到 JSON 文件，每行一个 JSON 对象
with open("train_data.json", "w", encoding="utf-8") as train_file:
    for item in train_data:
        json.dump(item, train_file, ensure_ascii=False)
        train_file.write("\n")  # 每个 JSON 对象占一行

# 保存测试集到 JSON 文件，每行一个 JSON 对象
with open("test_data.json", "w", encoding="utf-8") as test_file:
    for item in test_data:
        json.dump(item, test_file, ensure_ascii=False)
        test_file.write("\n")  # 每个 JSON 对象占一行

print("训练集和测试集已保存为独立的 JSON 对象格式")



