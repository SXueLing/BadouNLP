"""
根据已有词典，利用回溯分割子串的方法实现全切分，并根据每种切分方式的总词频分数，从高到底排列结果
"""

#词典
Dict = {"经常":0.5,
        "经":0.01,
        "有":0.1,
        "常":0.01,
        "有意见":0.4,
        "歧":0.01,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.01}

sentence = "经常有意见分歧"

#实现全切分函数
def all_cut(sentence, Dict):
    path = []
    res = []
    # 回溯算法
    def backtrack(startIndex, tmp_sum):
        if startIndex == len(sentence):
            path.append(tmp_sum)
            res.append(path[:])
            path.pop()
            return
        for i in range(startIndex, len(sentence)):
            tmp_s = sentence[startIndex: i + 1]
            if tmp_s in Dict.keys():
                tmp_sum += Dict[tmp_s]
                path.append(tmp_s)
                backtrack(i + 1, tmp_sum)
                path.pop()
                tmp_sum -= Dict[tmp_s]
    backtrack(0, 0)
    return res

res = all_cut(sentence, Dict)
res.sort(key = lambda x: -x[-1])
for i in res:
    print(i[:-1]) # 打印结果


