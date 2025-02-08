import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('文本分类练习.csv')

row_lengths = df.apply(lambda row: sum(len(str(item)) for item in row), axis=1)
print(row_lengths)
ave_row_length = row_lengths.mean()
print(ave_row_length)
print(row_lengths.max())
print(sum(row_lengths < 40))
df_0 = df[df['label']==0]
df_1 = df[df['label']==1]
print(len(df_0))
print(len(df_1))
print(df)
## 数据集总样本 11987  label=1正向样本7987   label=0负向样本4000

## 按照8:2 划分
train_set, test_set = train_test_split(df, test_size=0.2,random_state=42)
train_set.to_csv('train_set.csv',index=False)
test_set.to_csv('test_set.csv',index=False)


