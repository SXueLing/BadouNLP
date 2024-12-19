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

# 实现全切分函数，输出根据字典能够切分出的所有切分方式
def all_cut(sentence, Dict):
    n = len(sentence)
    dp = [[] for _ in range(n + 1)]
    dp[0].append([])  # 初始化空字符串的切分结果

    for i in range(1, n + 1):
        for j in range(i):
            word = sentence[j:i]
            if word in Dict:
                for prev in dp[j]:
                    dp[i].append(prev + [word])

    return dp[n]

# 实现全切分函数，输出根据字典能够切分出的所有切分方式
def all_cut2(sentence, Dict):
    # 按最大长度优先切分
    max_len = max(len(word) for word in Dict)
    def helper(sub_sentence):
        if not sub_sentence:
            return [[]]  # 返回包含空列表的结果
        cuts = []
        #使用一个循环来遍历从 1 到 min(max_len, len(sub_sentence)) 的所有可能切分长度
        for i in range(1, min(max_len, len(sub_sentence)) + 1):
            #对于每一次循环，提取子字符串 sub_sentence 从开头到第 i 个字符的部分，并检查这个部分是否在 Dict 中
            word = sub_sentence[:i]
            # 如果在字典中找到这个词，则进行下一步的切分
            if word in Dict:
                # 如果找到了一个有效的词（word），则调用 helper 函数对剩下的子字符串（sub_sentence[i:]）进行递归切分
                for right in helper(sub_sentence[i:]):
                    #返回的结果 right 被添加到 cuts 列表中。这里组合了当前找到的词和后续的切分结果
                    cuts.append([word] + right)
        return cuts

    return helper(sentence)

# 实现全切分函数，输出根据字典能够切分出的所有切分方式
def all_cut3(sentence, Dict):
    # 按最大长度优先切分
    max_len = max(len(word) for word in Dict)
    target = set()  # 使用集合来避免重复结果

    if not sentence:  # 基本情况处理
        return [[]]  # 返回包含空列表的结果

    for i in range(max_len, 0, -1):
        for j in range(len(sentence) - i + 1):
            word = sentence[j:j+i]
            if word in Dict:
                # 找到切分点，开始递归切分
                left_cuts = all_cut3(sentence[:j], Dict)
                right_cuts = all_cut3(sentence[j+i:], Dict)
                for left in left_cuts:
                    for right in right_cuts:
                        target.add(tuple(left) + (word,) + tuple(right))

    return [list(item) for item in target]  # 将集合转换为列表形式


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

def main():
    result = all_cut3(sentence, Dict)
    print(sorted(result))
    print(sorted(target))
    assert sorted(result) == sorted(target), "结果与目标输出不符！"

if __name__ == '__main__':
    main()