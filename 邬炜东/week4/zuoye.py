# #week3作业
#
# #词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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
#
# #待切分文本
# sentence = "常经有意见分歧"
#
# #实现全切分函数，输出根据字典能够切分出的所有的切分方式
# def all_cut(sentence, Dict):
#     #TODO
#     return target
#
# #目标输出;顺序不重要
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


sentences = "经常有意见分歧"
max_length = 0
for char in Dict.keys():
    max_length = max(max_length, len(char))
print(max_length)

def all_cut(sentences, Dict, max_length):
    results = []

    def backtrack(sentence, result, max_length):
        if sentence == "":
            results.append(result[:])
            return
        for i in range(1, max_length + 1):
            if len(sentence) >= i and sentence[:i] in Dict.keys():
                result.append(sentence[:i])
                backtrack(sentence[i:], result, max_length)
                result.pop()
        # if sentence == "":
        #     results.append(result[:])
        #     return
        # if len(sentence) >= 1 and sentence[:1] in Dict:
        #     result.append(sentence[:1])
        #     backtrack(sentence[1:], result)
        #     result.pop()
        # if len(sentence) >= 2 and sentence[:2] in Dict:
        #     result.append(sentence[:2])
        #     backtrack(sentence[2:], result)
        #     result.pop()
        # if len(sentence) >= 3 and sentence[:3] in Dict:
        #     result.append(sentence[:3])
        #     backtrack(sentence[3:], result)
        #     result.pop()

    backtrack(sentences, [], max_length)
    return results


for lst in all_cut(sentences, Dict, max_length):
    print(lst)
