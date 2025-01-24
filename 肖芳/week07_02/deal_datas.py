import csv 
import math
# 处理训练数据

X = []
Y = []
# 分X和Y

# 读取csv，分成X和Y
def read_csv_datas(path):
  with open(path, 'r') as f:
      csv_reader = csv.reader(f)
      next(csv_reader)
      for row in csv_reader:
          X.append(row[1])
          Y.append(row[0])
  return X, Y

# 统计正负样本数量
def count_pos_neg():
    print("正样本数量：", Y.count('1'))
    print("负样本数量：", Y.count('0'))

# 计算文本平均长度
def avg_text_length():
    print("文本平均长度：", math.floor(sum(len(text) for text in X) / len(X)))

# 计算最大文本长度
def max_text_length():
    print("最大文本长度：", max(len(text) for text in X))



NEG_X = []
NEG_Y = []
POS_X = []
POS_Y = []

# 划分正负样本
def split_neg_pos():
    for i in range(len(X)):
        if Y[i] == '0':
            NEG_X.append(X[i])
            NEG_Y.append(Y[i])
        else:
            POS_X.append(X[i])
            POS_Y.append(Y[i])
    return NEG_X, NEG_Y, POS_X, POS_Y

# 训练集验证集划分
def split_train_val():
    # 取3000个正样本和3000个负样本拼到一起作为训练集
    train_X = POS_X[:3000] + NEG_X[:3000]
    train_Y = POS_Y[:3000] + NEG_Y[:3000]
    # 剩下的作为验证集
    val_X = POS_X[3000:] + NEG_X[3000:]
    val_Y = POS_Y[3000:] + NEG_Y[3000:]

    # 将训练集和测试集分别保存到csv中
    with open('data/train_datas.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['label', 'text'])
        for i in range(len(train_X)):
            csv_writer.writerow([train_Y[i], train_X[i]] )
    with open('data/val_datas.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['label', 'text'])
        for i in range(len(val_X)):
            csv_writer.writerow([val_Y[i], val_X[i]])

def gen_test_csv(X, Y):
    with open('data/test_datas.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['label', 'text'])
        for i in range(10):
            csv_writer.writerow([Y[i], X[i]])

if __name__ == '__main__':
    X, Y = read_csv_datas('data/all_datas.csv')
    count_pos_neg()
    avg_text_length()
    max_text_length()
    split_neg_pos()
    split_train_val()
# 运行结果
# 正样本数量： 4000
# 负样本数量： 7987
# 文本平均长度： 25
# 最大文本长度： 463


    
