#分词方法：全切分

def all_cut(sentence, Dict):
    if not sentence:
        return [[]]

    words = []
    for i in range(1, len(sentence) + 1):
        prefix = sentence[:i]
        if prefix in Dict:
            suffixes = all_cut(sentence[i:], Dict)
            for suffix in suffixes:
                words.append([prefix] + suffix)

    return words


#加载词典
Dict = {
    "经常": 0.1,
    "经": 0.1,
    "有": 0.1,
    "常": 0.1,
    "有意见": 0.1,
    "歧": 0.1,
    "意见": 0.1,
    "分歧": 0.1,
    "见": 0.1,
    "意": 0.1,
    "见分歧": 0.1,
    "分": 0.1
}

#待切分文本
sentence = "经常有意见分歧"


# 获取所有可能的分词组合
words =  all_cut(sentence, Dict)[::-1]

# 打印结果
for word in words:
    print((word))
