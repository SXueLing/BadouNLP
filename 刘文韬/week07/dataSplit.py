import pandas as pd
from sklearn.model_selection import train_test_split

file_path = r'E:\日常\学习\八斗\第七周 文本分类\week7 文本分类问题\文本分类练习.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 第一列是 label，第二列是 review
X = df.iloc[:, 1]
y = df.iloc[:, 0]

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.DataFrame({'label': y_train, 'review': X_train})
test_data = pd.DataFrame({'label': y_test, 'review': X_test})

print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)