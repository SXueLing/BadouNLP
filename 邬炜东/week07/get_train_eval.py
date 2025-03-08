import pandas as pd
from config import Config

train_index = Config["train_index"]
file_path = './dataset/texts.csv'
df = pd.read_csv(file_path)

# 区分正样本和负样本
count_0 = (df['label'] == 0).sum()
count_1 = (df['label'] == 1).sum()

# 计算最长文本以及根据比例选择max_len长度
max_length_review = df['review'].str.len().max()
to_long = (df['review'].str.len() > 100).sum()
to_long_ratio = to_long / len(df)
print(f"too_long:{to_long}")
print(f"to_long_ratio:{to_long_ratio}")

# 打印统计信息
print(f"Label 0 count: {count_0}")
print(f"Label 1 count: {count_1}")
print(f"Longest review length: {max_length_review}")

train_df_0 = df[df['label'] == 0].head(int(count_0 * train_index))
train_df_1 = df[df['label'] == 1].head(int(count_1 * train_index))

eval_df_0 = df[df['label'] == 0].iloc[len(train_df_0):]
eval_df_1 = df[df['label'] == 1].iloc[len(train_df_1):]

train_df = pd.concat([train_df_0, train_df_1])
eval_df = pd.concat([eval_df_0, eval_df_1])

# 按照train_index按比例划分训练集和验证集合
train_df.to_csv('./dataset/train.csv', index=False)
eval_df.to_csv('./dataset/eval.csv', index=False)

