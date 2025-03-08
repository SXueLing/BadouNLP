# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
    "经常": 0.1,
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
    "分": 0.1
}

# 待切分文本
sentence = "经常有意见分歧"

# 记忆化缓存
memo = {}

# 实现全切分函数，输出根据字典能够切分出的所有切分方式
def all_cut(sentence, Dict):
    # 使用 memo 缓存，避免重复计算
    if sentence in memo:
        return memo[sentence]

    # 如果句子为空，返回空列表
    if not sentence:
        return [[]]

    result = set()  # 使用集合避免重复结果

    # 从最大长度开始遍历
    max_len = len(sentence)
    for i in range(1, max_len + 1):
        word = sentence[:i]
        if word in Dict:
            # 找到一个有效的词，递归地切分剩余的部分
            rest_cuts = all_cut(sentence[i:], Dict)
            for rest in rest_cuts:
                result.add(tuple([word] + rest))

    # 记忆化缓存当前结果
    memo[sentence] = [list(item) for item in result]
    return memo[sentence]

# 目标输出;顺序不重要
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
    result = all_cut(sentence, Dict)
    print(sorted(result))
    print(sorted(target))
    assert sorted(result) == sorted(target), "结果与目标输出不符！"

if __name__ == '__main__':
    main()
