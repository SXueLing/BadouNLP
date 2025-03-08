import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
结论：
评论共计11987条，正样本4000条，负样本7987条
评论最大长度：463，平均长度：25.048552598648534
绝大多数评论（98.09%）长度都在100字以内，基本都在200字以内，超过200字的只是个例，max_length选择90
"""

# 计算每条评论的长度
df = pd.read_csv(r"E:\日常\学习\八斗\第七周 文本分类\week7 文本分类问题\文本分类练习.csv")
label = df["label"].tolist()
review = df["review"].tolist()

print("评论共计{}条，正样本{}条，负样本{}条".format(len(label), label.count(1), label.count(0)))
print("评论最大长度：{}，平均长度：{}".format(max([len(x) for x in review]), sum([len(x) for x in review])/len(review)))

comment_lengths = [len(comment) for comment in review]

# 统计长度不超过100字的评论数量
max_length = 100
short_comments_count = sum(1 for length in comment_lengths if length <= max_length)

total_comments = len(review)
short_comments_ratio = short_comments_count / total_comments

print(f"评论长度不超过 {max_length} 字的比例: {short_comments_ratio:.2%}")



# 绘制评论长度的分布图
df = pd.DataFrame(comment_lengths, columns=['Length'])

plt.figure(figsize=(10, 6))

ax = sns.histplot(df['Length'], bins=20, kde=True, color='blue')
for p in ax.patches:
    height = p.get_height()  # 获取柱状图的高度（即频数）
    if height > 0:  # 只显示高度大于0的柱状图标签
        ax.annotate(
            f'{int(height)}',  # 显示的文本内容
            (p.get_x() + p.get_width() / 2, height),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )
plt.title('评论长度分布')
plt.xlabel('评论长度')
plt.ylabel('频率')
plt.show()

