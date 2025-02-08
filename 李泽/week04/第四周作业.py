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


def all_cut(sentence, Dict):
    results = []

    def backtrack(start, path):
        if start == len(sentence):
            # 将路径中的词用逗号连接成字符串后添加到结果列表
            result_str = ",".join(path)
            results.append(result_str)
            return
        # 先尝试长词切分（包括从当前位置开始的长词）
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                backtrack(end, path)
                path.pop()
        # 这里不再单独处理单字切分，因为长词切分已经包含了单字情况

    backtrack(0, [])
    return results


# 获取切分结果并输出
target = all_cut(sentence, Dict)
for t in target:
    print(f"['{t}']")
