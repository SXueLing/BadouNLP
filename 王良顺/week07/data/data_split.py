# -*- coding = utf-8 -*-
# @Time : 2025-01-06 15:00
# @Author : WLS
# @File : data_split.py
# @Software : PyCharm

import pandas as pd
import json
import jieba
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'.\文本分类练习.csv', encoding='utf-8')

X = data['review'].values
Y = data.label.values

#test_size:测试集的大小，如果是小数的话，值在（0,1）之间，表示测试集所占有的比例；
#如果是整数，表示的是测试集的具体样本数；
#如果train_size也是None的话，会有一个默认值0.25
#train_size也一样
#random_state:随机种子
#shuffle:是否重洗数据（洗牌）,是否把数据打散重新排序
#stratify:假设原始的结果集中有2种分类，A：B=1:2
#我们在随机分配的时候，是无法保证训练集和测试集中的A与B的比例
#设置stratify=Y，就可以让测试集和训练集中的结果集也保证这种分布

# train-8:test-2
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=18)
# valid-1:test-1
x_valid, x_test, y_valid, y_test = train_test_split(x_test,y_test,test_size=0.5,shuffle=True,stratify=y_test,random_state=18)
#train:test:valid = 8:1:1

print("训练集样本数 = ", len(y_train))
print("训练集中正样本数 = ", len([w for w in y_train if w == 1]))
print("训练集中负样本数 = ", len([w for w in y_train if w == 0]))
print("验证集样本数 = ", len(y_valid))
print("验证集中正样本数 = ", len([w for w in y_valid if w == 1]))
print("验证集中负样本数 = ", len([w for w in y_valid if w == 0]))
print("测试集样本数 = ", len(y_test))
print("测试集中正样本数 = ", len([w for w in y_test if w == 1]))
print("测试集中负样本数 = ", len([w for w in y_test if w == 0]))

# 用jieba对各数据集分词
def tokenizer(data):
    # 得到文本数据
    text = []
    for i in range(data.shape[0]):
        text.append(str(data[i]))

    comment = '\n'.join(text)

    # 清洗文本数据-用正则表达式删去数字、字母、标点符号、特殊符号等
    import re
    # symbols = "[0-9\!\%\,\。\.\，\、\～\~\?\(\)\（\）\？\！\“\”\:\：\;\"\；\……&\-\_\|\．\Ａ．Ｂ．Ｃ\*\^\/'\，\x08]"
    r = "[z0-9_.!！+-=——,$%^，。？、~～@#￥%……&*《》<>「」{}【】()/\\\[\]'\"\\x08]"
    comment = re.sub(r, ' ', comment)
    sen_text = re.compile(u'[\u4E00-\u9FA5|\s\w]').findall(comment)
    sentece = "".join(sen_text)

    comments = re.sub(sentece, '', comment)

    comments_list = jieba.cut(comments)  # 精确模式
    # comments_list = jieba.cut_for_search(comments)#搜索引擎模式
    data_tokenizer = ' '.join([x for x in comments_list])  # 用空格连接分好的词
    return data_tokenizer

# 划分成json文件
test_dir = "./test.json"
train_dir = "./train.json"
valid_dir = "./valid.json"


# 对各数据集分词
x_test = tokenizer(x_test)
x_train = tokenizer(x_train)
x_valid = tokenizer(x_valid)

x_train = x_train.split('\n')
x_test = x_test.split('\n')
x_valid = x_valid.split('\n')

print(x_test)
print(type(x_test))
print(len(x_test))

# {
# "tag": "0",
# "comment": "是不是 因为 外卖 的 关系 怕 泡 的 时间 长 面条 都 没 怎么 熟好 硬 "
# }

# {"tag": "0","comment": "是不是 因为 外卖 的 关系 怕 泡 的 时间 长 面条 都 没 怎么 熟好 硬 "}

def writeFile(x,y,dir):
    with open(dir, 'a+', encoding='utf-8') as f:
        for i, j in zip(x, y):
            line_data = {'tag': str(j),'comment': str(i)}
            f.write(str(line_data).replace('"',"'"))
            f.write('\n')
    f.close()

writeFile(x_train,y_train,train_dir)
writeFile(x_test,y_test,test_dir)
writeFile(x_valid,y_valid,valid_dir)



