def all_cut(sentence, Dict):
    length = len(sentence)
    # dp[i] 用于存储sentence中前i个字符的所有切分结果（索引从0开始）
    dp = [[] for _ in range(length + 1)]
    dp[0] = [[]]
    for i in range(1, length + 1):
        for j in range(i):
            word = sentence[j:i]
            if word in Dict:
                for prev_cut in dp[j]:
                    dp[i].append(prev_cut + [word])
    return dp[length]

for i in all_cut(sentence, Dict):
    print(i)
