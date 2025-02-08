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
# sentence = "经"


def backtracking(sentence, Dict, temp, target):


    if not sentence:
        target.append(temp[:])#避免引用传递
        return



#把下面这三个if语句换成for循环
    # if sentence[0] in Dict:
    #     temp.append(sentence[0])
    #     backtracking(sentence[1:], Dict, temp, target)
    #     temp.pop()
    #
    # if len(sentence) > 1 and sentence[0] + sentence[1]  in Dict:
    #     temp.append(sentence[0] + sentence[1])
    #     backtracking(sentence[2:], Dict, temp, target)
    #     temp.pop()
    #
    # if len(sentence) > 2 and sentence[0] + sentence[1] + sentence[2] in Dict:
    #     temp.append(sentence[0] + sentence[1] + sentence[2])
    #     backtracking(sentence[3:], Dict, temp, target)
    #     temp.pop()


    for i in range(len(max(Dict))):
        if len(sentence) > i and sentence[:i+1] in Dict:
        # if  sentence[:i+1] in Dict:
            temp.append(sentence[:i+1])
            backtracking(sentence[i+1:], Dict, temp, target)
            temp.pop()




    return target



#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):




    target = backtracking(sentence, Dict, [], [])


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
# print(list(sentence))
# print(len(max(Dict)))
# print(str(list(sentence)[0] + list(sentence)[1]) in Dict)
#
#
# print(  list(sentence)[0] + list(sentence)[1]   )


print(all_cut(sentence, Dict))


