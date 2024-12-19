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
    # 用于存储所有的切分结果
    result = []

    # 递归函数，current_index是当前检查的索引，current_cut是当前的切分路径
    def cut_from_index(current_index, current_cut):
        # 如果当前索引已经是句子的末尾，说明切分完毕，保存当前切分结果
        if current_index == len(sentence):
            result.append(current_cut)
            return

        # 遍历从当前索引开始的所有可能子串
        for end_index in range(current_index + 1, len(sentence) + 1):
            word = sentence[current_index:end_index]
            if word in Dict:
                # 如果子串在字典中，继续递归处理剩余部分
                cut_from_index(end_index, current_cut + [word])

    # 从句子的开始位置开始切分
    cut_from_index(0, [])

    return result

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
result = all_cut(sentence, Dict)

# 输出结果
for r in result:
    print(r)
