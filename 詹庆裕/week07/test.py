import random
import json
import os

def load():
    data = []
    with open("文本分类练习.csv", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 确保行不为空
                label = line.split(",")[0].strip().replace("1","好评").replace("0","差评")
                content = ''.join(line.split(",")[1:]).strip().replace('\n', "")
                data.append({"label": label, "content": content})

    # 打乱数据
    random.shuffle(data)

    # 划分数据集
    train_ratio = 0.8  # 80% 训练集
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 确保目录存在
    os.makedirs("nn_pipeline/profile", exist_ok=True)

    # 将训练集写入文件
    with open(r"nn_pipeline/profile/train.jsonl", "w", encoding="utf-8") as f1:
        for item in train_data:
            f1.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 将测试集写入文件
    with open(r"nn_pipeline/profile/test.jsonl", "w", encoding="utf-8") as f2:
        for item in test_data:
            f2.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    load()
