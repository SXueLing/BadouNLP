#week4作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "常":0.001,
        '有':0.01,
        "有意见":0.1,
        "歧":0.001,
        "分歧":0.2,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, result=None, path=None):
    if result is None:
        result = []
    if path is None:
        path = []
    if not sentence:
        result.append(list(path))
        return result
    ans = []
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in Dict:
            path.append(word)
            ans.extend(all_cut(sentence[i:], Dict, result, path))
            path.pop()
    return result

#调用函数并打印结果
print(all_cut(sentence, Dict))



