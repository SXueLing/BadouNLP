# week04
# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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


# 全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []


    def cut_function(position_now, temp):
        end_notice = len(sentence)
        if position_now == end_notice:
            # 当递归到达句子末尾时，将当前切分结果添加到最终结果中
            target.append(temp[:])
            return target
        # 未到末尾继续切分
        for i in range(position_now + 1, end_notice + 1):
            word = sentence[position_now:i]
            if word in Dict:
                temp.append(word)
                cut_function(i, temp)
                temp.pop()  # 回溯，撤销上一次的添加
    # 从句子开头开始递归
    cut_function(0, [])
    return target


# 调用函数并打印结果
target = all_cut(sentence, Dict)
for cut in target:
    print(cut)


# 目标输出;顺序不重要
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
