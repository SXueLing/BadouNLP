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

# 待切分文本
sentence = "经常有意见分歧"

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    def dfs(start, path, result):
        if start == len(sentence):
            result.append(path[:])
            return
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                dfs(end, path, result)
                path.pop()

    result = []
    dfs(0, [], result)
    return result

# 调用函数并打印结果
target = all_cut(sentence, Dict)
print(target)
