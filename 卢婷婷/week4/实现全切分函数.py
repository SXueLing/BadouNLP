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
sentence = "常经有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式

def all_cut(sentence, Dict, start=0, memo=None):
    if memo is None:
        memo = {}

    if start == len(sentence):
        return [[]]

    if start in memo:  # 如已计算过，直接返回
        return memo[start]

    cuts = set()  # 使用集合避免重复

    for end in range(start + 1, len(sentence) + 1):
        word = sentence[start:end]
        if word in Dict:  # 如果当前子串在字典中
            for cut in all_cut(sentence, Dict, end, memo):  # 递归切分剩余部分
                cuts.add((word, *cut))  # 以元组形式添加，便于集合去重

    memo[start] = [list(cut) for cut in cuts]  # 将当前位置的切分结果存储到记忆化字典中
    return memo[start]

# 调用函数并存储结果
cuts = all_cut(sentence, Dict)

print("总共有 {} 种切分方式：".format(len(cuts)))
for cut in cuts:
    print(cut)
final_cuts = set(tuple(cut) for cut in cuts)

#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]

