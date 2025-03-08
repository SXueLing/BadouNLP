# -*- coding: utf-8 -*-
# @Time    : 2025/1/7 10:01
# @Author  : WLS
# @File    : data_length.py
# @Software: PyCharm

# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 设置显示风格
plt.style.use('fivethirtyeight')

# 分别读取训练tsv和验证tsv
data = pd.read_csv(r'.\文本分类练习.csv', encoding='utf-8')
# 添加新的句子长度列, 每个元素的值都是对应的句子列的长度
data["sentence_length"] = list(map(lambda x: len(x), data["review"]))

# 绘制句子长度列的数量分布图
sns.countplot(x=data['sentence_length'], data=data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
# title显示中文时需要带字体才能显示
plt.title("文本分类练习的评论长度分布",fontproperties='SimHei')
plt.show()

