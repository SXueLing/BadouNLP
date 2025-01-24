import csv
import random

with open(r'文本分类练习.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # 存储文本和标签
    labels = []
    texts = []
    for row in reader:
        labels.append(int(row['label']))  # 标签为整数类型
        texts.append(row['review'])
# 去除空标签
labels = [label for label in labels if label is not None]
# 去除空文本
texts = [text for text in texts if text.strip()]

# 计算正负样本数、文本评论的长度
positive_label = 0
negative_label = 0
positive_reviews = []
negative_reviews = []
positive_reviews_len = 0
negative_reviews_len = 0
for i in range(len(labels)):
    if labels[i] == 1:
        positive_label += 1
        positive_reviews.append(texts[i])
        positive_reviews_len += len(texts[i])
    elif labels[i] == 0:
        negative_label += 1
        negative_reviews.append(texts[i])
        negative_reviews_len += len(texts[i])
# 计算评论的平均长度
average_length = (positive_reviews_len +
                  negative_reviews_len) / (positive_label + negative_label)
# 计算正负评论的平均长度
average_length_positive = positive_reviews_len / positive_label
average_length_negative = negative_reviews_len / negative_label

# 划分训练集和验证集，假设按 80% 划分训练集，20% 划分验证集
data_size = len(texts)
train_size = int(0.8 * data_size)
indices = list(range(data_size))  # 创建一个包含从 0 到 data_size - 1 的整数列表
random.shuffle(indices)  # 随机打乱数据索引

# 划分训练集和验证集
train_indices = indices[:train_size]
valid_indices = indices[train_size:]
X_train = [texts[i] for i in train_indices]
y_train = [labels[i] for i in train_indices]
X_valid = [texts[i] for i in valid_indices]
y_valid = [labels[i] for i in valid_indices]

# 保存训练集
with open('train_tag_reviews.csv', mode='w', encoding='utf-8',
          newline='') as train_file:
    writer = csv.writer(train_file)
    writer.writerow(['label', 'text'])  # 写入表头
    for i in range(len(X_train)):
        writer.writerow([y_train[i], X_train[i]])

# 保存验证集
with open('valid_tag_reviews.csv', mode='w', encoding='utf-8',
          newline='') as valid_file:
    writer = csv.writer(valid_file)
    writer.writerow(['label', 'text'])  # 写入表头
    for i in range(len(X_valid)):
        writer.writerow([y_valid[i], X_valid[i]])

if __name__ == "__main__":
    # 打印分析结果
    print(f"正面评论数: {len(positive_reviews)}")
    print(f"负面评论数: {len(negative_reviews)}")
    print(f"评论的平均长度: {average_length:.2f}")
    print(f"正面评论的平均长度: {average_length_positive:.2f}")
    print(f"负面评论的平均长度: {average_length_negative:.2f}")
    # 打印训练集和验证集大小
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_valid)}")
