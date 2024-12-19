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
def all_cut(sentence, Dict):
    def defs(start, path):
        # 如果当前路径的长度等于句子的长度，说明找到了一个完整的切分方式
        if start == len(sentence):
            result.append(path[:])
            return
        for end in range(start + 1, len(sentence) + 1):
            # 检查当前切分的词是否在词典中
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                defs(end, path)
                # 回溯，移除当前词，尝试其他切分方式
                path.pop()

    result = []
    defs(0, [])
    return result

# 调用函数
print(all_cut(sentence, Dict))
