"""
需求：实现基于词表的全切分
"""

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


def full_segment(sentence, dict_words):
    words = list(dict_words.keys())
    n = len(sentence)
    results = []

    # 递归函数，用于全切分
    def recursive_segment(start, path):
        if start == n:
            results.append(path)
            return
        for end in range(start + 1, n + 1):
            word = sentence[start:end]
            if word in words:
                recursive_segment(end, path + [word])

    recursive_segment(0, [])
    return results

# 待分词的文本
sentence = "经常有意见分歧"

# 获取所有可能的分词方式
all_possible_segments = full_segment(sentence, Dict)

print(all_possible_segments)