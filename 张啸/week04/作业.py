# week4作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
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
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 递归函数，start 是当前处理的其实索引， path 是当前的切分路径， result 是存储所有切分结果的列表。
    def backtrack(start, path, result):
        # 结果收集：当 start 达到句子长度时，表示找到一种完整的切分方式，将其加入result
        if start == len(sentence):
            result.append(path[:])
            return
        # 循环遍历，从 start + 1 到 len(sentence) + 1 遍历所有可能的结束索引，提取字串word
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 检查子串：如果 word 在词典 Dict 中，则将其加入 path，并递归调用 backtrack 处理剩余部分
            if word in Dict:
                path.append(word)
                backtrack(end, path, result)
                # 回溯
                path.pop()

    result = []
    backtrack(0, [], result)
    return result


if __name__ == '__main__':
    print(all_cut(sentence, Dict))

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
