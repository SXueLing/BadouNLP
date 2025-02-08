import json
import pandas as pd

file_path = "文本分类练习.csv"
df = pd.read_csv(file_path)
# 计算分割点
labels = df['label']
features = df['review']
data_size = len(features)
split_index = int(0.8 * data_size)
print(features)
# 分割数据集为训练集和测试集
X_train, X_test = features[:split_index], features[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

# 将训练集数据写入文件，使用UTF-8编码
with open('./train_data.json', 'w', encoding='utf-8') as file:
    for feature, label in zip(X_train, y_train):
        data_dict = {"label": label, "feature": feature}
        json_str = json.dumps(data_dict, ensure_ascii=False)  # 将字典转换为JSON字符串，并确保非ASCII字符正确显示
        file.write(f"{json_str}\n")  # 写入文件并换行

# 将测试集数据写入文件，使用UTF-8编码
with open('./test_data.json', 'w', encoding='utf-8') as file:
    for feature, label in zip(X_test, y_test):
        data_dict = {"label": label, "feature": feature}
        json_str = json.dumps(data_dict, ensure_ascii=False)  # 将字典转换为JSON字符串，并确保非ASCII字符正确显示
        file.write(f"{json_str}\n")  # 写入文件并换行
