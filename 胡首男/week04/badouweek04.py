def all_cut(sentence, Dict):
    # 定义一个缓存字典，用于记忆化搜索
    memo = {}

    # 辅助函数，递归的核心
    def helper(s):
        # 如果已经计算过这个子字符串的所有切分方式，直接返回缓存结果
        if s in memo:
            return memo[s]

        # 初始化当前子串的所有切分结果
        result = []

        # 如果整个子串是一个字典中的有效词，直接返回它自己作为一种切分方式
        if s in Dict:
            result.append([s])

        # 递归分割子字符串
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            # 如果前缀在字典中，递归处理剩余部分
            if prefix in Dict:
                suffix = s[i:]
                suffix_cut = helper(suffix)
                # 对于每种后续切分方式，加入当前前缀
                for cut in suffix_cut:
                    result.append([prefix] + cut)

        # 将计算结果缓存
        memo[s] = result
        return result

    # 调用辅助函数，获取整个句子的切分方式
    return helper(sentence)


# 词典
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

# 调用全切分函数
result = all_cut(sentence, Dict)

# 输出结果
for res in result:
    print(res)
