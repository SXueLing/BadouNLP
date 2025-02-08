
"""
根据data.csv里面的数据随机划分为训练集和验证集 训练集占比80%
生成两个文件 train_tag.csv valid_data.csv
"""
import pandas as pd;

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv')
    df = df.sample(frac=1, random_state=42)
    train = df.iloc[:int(len(df)*0.8)]
    valid = df.iloc[int(len(df)*0.8):]
    train.to_csv('data/train_tag.csv', index=False)
    valid.to_csv('data/valid_data.csv', index=False)