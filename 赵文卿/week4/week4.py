'''
Author: Zhao
Date: 2024-12-25 22:34:52
LastEditTime: 2024-12-26 19:52:28
FilePath: homework_readme.py
Description: week4作业,实现全切分函数，输出根据字典能够切分出的所有的切分方式

'''
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
        
def all_cut(sentence, Dict):
    target = []
    segment_split(sentence,Dict,[],target)
    return target

def segment_split(str,Dict,current, result):
    if not str:
        result.append(current[:])
        return
    for i in range (1, len(str) +1):
        index = str[:i]
        if index in Dict:
            current.append(index)
            segment_split(str[i:], Dict, current, result)
            current.pop()

# 验证
# words = all_cut(sentence, Dict)
# for word in words: 
#     print(word)

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
