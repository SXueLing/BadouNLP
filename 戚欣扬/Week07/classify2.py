import pandas as pd
import json
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"E:/八斗课程/第七周/文本分类练习.csv")

data_list = [{"tag": row["label"], "title": row["review"]} for _, row in df.iterrows()]
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)


with open("train_data.json", "w", encoding="utf-8") as train_file:
    for item in train_data:
        json.dump(item, train_file, ensure_ascii=False)
        train_file.write("\n")  

with open("test_data.json", "w", encoding="utf-8") as test_file:
    for item in test_data:
        json.dump(item, test_file, ensure_ascii=False)
        test_file.write("\n")  
