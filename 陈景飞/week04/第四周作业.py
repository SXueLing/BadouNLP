def all_cut(sentence, Dict):
    word_list = list(Dict.keys())
    # print("word_list=", word_list)  # word_list= ['经常', '经', '有', '常', '有意见', '歧', '意见', '分歧', '见', '意', '见分歧', '分']

    def is_word(word):
        return word in word_list

    def segment(sentence):
        n = len(sentence)
        dp = [[] for _ in range(n + 1)]
        dp[0] = [[]]

        for i in range(1, n + 1):
            for j in range(i):
                word = sentence[j:i]
                if is_word(word):
                    for segmentation in dp[j]:
                        dp[i].append(segmentation + [word])

        return dp[n]

    results = segment(sentence)
    return results


# 词典，每个词后方存储的是其词频，仅为示例，也可自行添加
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

sentence = "经常有意见分歧"  # 示例句子
segmentations = all_cut(sentence, Dict)  # 获取所有切分可能情况

# 输出结果
for segmentation in segmentations:
    print(segmentation)
