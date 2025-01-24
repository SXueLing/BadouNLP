import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def get_text_length(text):
    return len(text)


# 读取数据
file_path = 'data/文本分类练习.csv'
data = pd.read_csv(file_path)
X = data['review']
y = data['label']

# 分层划分训练集和测试集
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in stratified_split.split(X, y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

# 打印正负样本数
positive_count_train = (y_train == 1).sum()
negative_count_train = (y_train == 0).sum()
positive_count_test = (y_test == 1).sum()
negative_count_test = (y_test == 0).sum()

print(f"训练集正样本数: {positive_count_train}")
print(f"训练集负样本数: {negative_count_train}")
print(f"测试集正样本数: {positive_count_test}")
print(f"测试集负样本数: {negative_count_test}")

# 计算文本平均长度
train_text_lengths = X_train.apply(get_text_length)
average_length_train = train_text_lengths.mean()
test_text_lengths = X_test.apply(get_text_length)
average_length_test = test_text_lengths.mean()

print(f"训练集文本平均长度: {average_length_train}")
print(f"测试集文本平均长度: {average_length_test}")

# 将训练集和测试集保存为CSV文件
train_df = pd.DataFrame({'label': y_train,'review': X_train})
test_df = pd.DataFrame({'label': y_test,'review': X_test})

train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)
