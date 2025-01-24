#读取csv文件转换为json文件
import csv
import json

#打开csv文件
with open("E:/nlp_learn/practice/week07/nn_pipline/文本分类练习.csv", "r", encoding = "utf8") as csvf:
    #创建csv读取器
    reader = csv.reader(csvf)
    csv_data = list(reader)
    print(csv_data)
    #初始化一个列表来存储所有转换后的数据
    data = []
    
    #遍历csv文件中的每一行
    for row in csv_data[1:]:  #跳过表头
        #print(row[1])
        #将每行数据转换为json格式文件
        label= row[0] 
        review = row[1]
    #将转换后的数据添加到列表中
        data.append({"label":label,"review":review})
      
#将数据写入json文件中
with open("E:/nlp_learn/practice/week07/nn_pipline/data/text_output.json", "w", encoding = "utf8") as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
    
print(f"CSV 文件已成功转换为 JSON 文件：{f}")
    


#训练集和验证集的划分
import json
import random

#读取json文件
with open("E:/nlp_learn/practice/week07/nn_pipline/data/text_output.json", 'r', encoding = "utf8") as jsonf:
    #读取所有数据
    data = jsonf.read().split("\n")
    #print(data)
    
#去除空字符串
data = [item for item in data if item]

#打乱数据顺序
random.shuffle(data)

#计算划分索引
split_index = int(len(data) * 0.8)

#划分训练集和验证集
train_data = data[: split_index]
val_data = data[split_index:]

#将训练集、验证集读取到json文件
with open("E:/nlp_learn/practice/week07/nn_pipline/data/train_textdata.json", "w", encoding = "utf8") as trainf:
    trainf.write("\n".join(train_data))
    

with open("E:/nlp_learn/practice/week07/nn_pipline/data/val_textdata.json", "w", encoding = "utf8") as valf:
    valf.write("\n".join(val_data))    
    
    
#数据分析
import json
def count_samples(file_path):
    #统计计数
    positive_count = 0
    negative_count = 0
    
    #统计正负样本个数
    with open(file_path, "r", encoding= "utf8") as f:
        for line in f:
            #解析每行json数据
            sample = json.loads(line)
            label = sample['label']
            if label == "1":
                positive_count += 1
            elif label == "0":
                negative_count += 1
    return positive_count, negative_count
#统计文件正负样本的个数
train_postive, train_negative = count_samples("./data/train_textdata.json")
print(f"训练集：正样本的个数{train_postive}个·，负样本的个数{train_negative}个")

#统计验证集正负样本的个数
val_postive, val_negative = count_samples("./data/val_textdata.json")
print(f"验证集：正样本的个数{val_postive}个，负样本的个数{val_negative}个")

def avg_sample(file_path):
    #统计平均文本长度
    length_sample = 0
    sum_length_review = 0
    max_length = 0
    i = 0
    with open(file_path, 'r', encoding = "utf8") as f:
        for line in f:
            sample = json.loads(line)
            review = sample["review"]
            length_review = len(review)
            if length_review > max_length:
                max_length = length_review
            #print(length_review)
            
            sum_length_review = sum_length_review + length_review
            i += 1
            #print(i)
            length_sample = sum_length_review / i
    return length_sample, max_length

length_sample, max_length = avg_sample("./data/text_output.json")
print(f"平均文本长度为：{length_sample},最长文本长度为：{max_length}")
