#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def full_segment(text, dictionary, path=None):
    if path is None:
        path = []

    # 如果文本为空，则返回路径
    if not text:
        return [path]

    results = []
    # 尝试从头开始切割不同长度的子串
    for i in range(1, len(text) + 1):
        word = text[:i]
        # 如果子串在字典中，则递归处理剩余部分
        if word in dictionary:
            # 继续分割剩余的文本
            results.extend(full_segment(text[i:], dictionary, path + [word]))

    return results

# 执行全切分
results = full_segment(sentence, Dict)

# 打印结果
for result in results:
    print('/'.join(result))

