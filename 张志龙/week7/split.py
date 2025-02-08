import pandas as pd
import json
import jieba
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'./张志龙/week7/文本分类练习.csv', encoding='utf-8')

X = data['review'].values
Y = data.label.values

# train-8：test-2
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=18)

# 将分好的train和test保存为csv文件
train_data = pd.DataFrame({'label':y_train,'review':x_train})
test_data = pd.DataFrame({'label':y_test,'review':x_test})

train_data.to_csv('./张志龙/week7/train.csv',index=False,encoding='utf-8')
test_data.to_csv('./张志龙/week7/test.csv',index=False,encoding='utf-8')
