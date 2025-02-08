def all_cut(sentence, Dict):
    def helper(sub_sentence, current_result):
        if not sub_sentence:  # 如果子句为空，表示切分完成
            results.append(current_result)
            return
        for i in range(1, len(sub_sentence) + 1):  # 遍历子句的所有前缀
            word = sub_sentence[:i]
            if word in Dict:  # 如果前缀在词典中
                helper(sub_sentence[i:], current_result + [word])  # 递归处理剩余部分

    results = []
    helper(sentence, [])
    return results

# 词典和文本
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
    "分": 0.1,
}

sentence = "经常有意见分歧"

# 调用全切分函数
target = all_cut(sentence, Dict)

# 打印结果
for t in target:
    print(t)
