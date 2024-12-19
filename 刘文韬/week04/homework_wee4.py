#week3作业
# 实现了正向、反向最大匹配以及全切分（递归）

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

# 正向最大匹配
def forward_maximum_matching(sentence, Dict):
    res = []
    max_len = len(max(Dict, key=len))
    i = 0
    while i <= len(sentence) - 1:
        index = -1
        for j in range(i + 1, i + max_len + 1):
            if j <= len(sentence):
                if sentence[i:j] in Dict:
                    index = j
                else:
                    continue
        if index != -1:
            res.append(sentence[i:index])
            i = index
        else:
            res.append(sentence[i])
            i += 1
    return res


# 反向最大匹配
def backward_maximum_matching(sentence, Dict):
    res = []
    max_len = len(max(Dict, key=len))
    i = len(sentence) - 1
    while i >= 0:
        index = -1
        for j in range(i - 1, i - max_len - 1, -1):
            if j >= 0:
                if sentence[j:i+1] in Dict:
                    index = j
                else:
                    continue
        if index != -1:
            res.append(sentence[index:i+1])
            i = index - 1
        else:
            res.append(sentence[i])
            i -= 1
    res.reverse()
    return res


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 存储所有可能的切分结果
    target = []

    # 递归实现全切分
    def recursive_cut(sentence, current_segment):
        # 如果句子为空，将当前已有的切分结果加入列表
        if not sentence:
            target.append(current_segment)
            return

        for i in range(1, len(sentence) + 1):
            # 取出前i个字符组成的词
            word = sentence[:i]
            # 如果这个词在词表中，则进行递归
            if word in Dict:
                recursive_cut(sentence[i:], current_segment + [word])

    recursive_cut(sentence, [])
    return target

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

print(forward_maximum_matching(sentence, Dict))   #['经常', '有意见', '分歧']
print(backward_maximum_matching(sentence, Dict))   #['经常', '有', '意', '见分歧']
all_cut_res = all_cut(sentence, Dict)
print(all(x in target for x in all_cut_res) and all(x in all_cut_res for x in target) and len(target) == len(all_cut_res))   #True
