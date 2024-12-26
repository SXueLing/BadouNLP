# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}

# 待切分文本
sentence = "经常有意见分歧"

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def helper(start):
        if start == len(sentence):
            return [[]]  # 切分完成，返回一个空列表
        result = []
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:  # 如果字典中存在当前片段
                for rest in helper(end):  # 递归切分剩余部分
                    result.append([word] + rest)
        return result
    return helper(0)
  
result = all_cut(sentence, Dict)
for r in result:
    print(r)
