# 使用回溯法实现全切割
def all_cut(text, dictionary):
    def backtrack(path, start):
        if start == len(text):
            results.append(path[:])
            return

        for end in range(start + 1, len(text) + 1):
            sub_string = text[start:end]
            if sub_string in dictionary:
                path.append(sub_string)
                backtrack(path, end)
                path.pop()

    results = []
    backtrack([], 0)
    return results


# 示例字典
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
result = all_cut(sentence, Dict)

print(sentence[0:len(sentence)])
# 输出所有切分路径
for i, res in enumerate(result, 1):
    print(f"切分路径 {i}: {res}")
