

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


sentence = "经常有意见分歧"



def all_cut(sentence, Dict):
    max_dict_len = max(len(word) for word in Dict)
    # 辅助函数：用来递归查找所有切分方式

    def backtrack(start):
        # 如果已经处理完了句子的末尾，返回一个包含空列表的结果
        if start == len(sentence):
            return [[]]  # 返回一个包含空列表的列表，表示当前路径的结束

        result = []
        # 从当前字符开始，逐渐扩展直到结尾
        for end in range(start + 1, min(len(sentence), start + max_dict_len) + 1):
            word = sentence[start:end]  # 获取当前子串
            if word in Dict:  # 如果子串在字典中
                # 递归查找剩余部分的所有切分方式
                for suffix in backtrack(end):
                    result.append([word] + suffix)  # 将当前子串与剩余的切分结果组合起来
        return result

    return backtrack(0)  # 从句子的开始位置开始切分


if __name__ == "__main__":
    result = all_cut(sentence, Dict)
    print(len(result))
    print(result)