import csv
import json
import random


# 打开CSV文件进行读取,获取总样本
def deal_with_csv(csv_file_path, all_data):
    all_data_list = []  # 所有样本数据
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        id = -1
        with open(all_data, 'w', encoding='utf-8') as f:
            for row in csv_reader:
                # print(row) #['label', 'review']
                if row[0] == 'label':
                    continue
                else:
                    id += 1
                    all_data_list.append({'id': id, 'label': row[0], 'review': row[1]})
                    json_str = json.dumps({'id': id, 'label': row[0], 'review': row[1]}, ensure_ascii=False) + '\n'
                    f.write(json_str)
    print("总样本集合大小=", len(all_data_list))
    # 求平均句子长度/正样本个数/负样本个数
    positives_list = []
    negative_list = []
    ele_len_list = []
    for ele in all_data_list:
        ele_len_list.append(len(ele["review"]))
        if ele["label"]=="1":
            positives_list.append(ele)
        else:
            negative_list.append(ele)
    sorted_ele_len_list = sorted(ele_len_list)
    print("正样本个数=", len(positives_list))
    print("负样本个数=", len(negative_list))
    print("最小句子长度=", sorted_ele_len_list[0])
    print("最大句子长度=", sorted_ele_len_list[len(sorted_ele_len_list) - 1])
    print("平均句子长度=", sum(sorted_ele_len_list) / len(sorted_ele_len_list))


# 获取100个测试
def split_file_randomly(all_data, valid_data, train_data, num_lines):
    # 读取所有行到列表中
    with open(all_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查文件行数是否少于请求的随机行数
    if len(lines) <= num_lines:
        raise ValueError("文件行数少于请求的随机行数")

    # 随机选择不重复的行索引
    selected_indices = random.sample(range(len(lines)), num_lines)

    # 根据索引获取选定的行和剩余的行
    selected_lines = [lines[i] for i in selected_indices]  # 测试集
    remaining_lines = [lines[i] for i in range(len(lines)) if i not in selected_indices]  # 训练集

    # 将选定的行写入第一个输出文件
    with open(valid_data, 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)

    # 将剩余的行写入第二个输出文件
    with open(train_data, 'w', encoding='utf-8') as f:
        f.writelines(remaining_lines)


if __name__ == "__main__":
    csv_file_path = r'D:\Chen\八斗学院资料\本期培训\直播课程\7-第7周\week7 文本分类问题\week7 文本分类问题\文本分类练习数据集\文本分类练习.csv'  # CSV文件的路径
    all_data = r'tag_news.json'  # 所有样本数据
    train_data = r'train_tag_news.json'  # 训练集
    valid_data = r'valid_tag_news.json'  # 测试集
    deal_with_csv(csv_file_path, all_data)
    split_file_randomly(all_data, valid_data, train_data, 100)
    print("=============样本数据处理成功====================")
