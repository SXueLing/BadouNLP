import json
import random


def load_data(file_path):
    """加载json文件中的数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return [json.loads(line) for line in data]


def split_data(data, train_ratio=0.8):
    """打乱数据并按照比例切分训练集和测试集"""
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_set = data[:split_index]
    test_set = data[split_index:]
    return train_set, test_set


def save_data(data, file_path):
    """将数据保存到json文件中"""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # 文件路径
    input_file = r'/week10 文本生成问题/data.json'
    train_file = r'F:\NLP\资料\week10 文本生成问题\week10 文本生成问题\语言模型\homework\train.json'
    test_file = r'F:\\NLP\资料\week10 文本生成问题\week10 文本生成问题\语言模型\homework\test.json'

    # 加载数据
    data = load_data(input_file)

    # 切分数据
    train_set, test_set = split_data(data, train_ratio=0.8)

    # 保存训练集和测试集
    save_data(train_set, train_file)
    save_data(test_set, test_file)


if __name__ == '__main__':
    main()
