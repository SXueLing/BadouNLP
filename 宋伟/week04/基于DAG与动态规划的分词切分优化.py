# -*- coding: utf-8 -*-
# @Date    :2024-12-25 21:06:34
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text




import math
from collections import defaultdict, deque

def build_dag(sentence, dictionary):
    n = len(sentence)
    dag = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n + 1):
            word = sentence[i:j]
            if word in dictionary:
                log_prob = math.log(dictionary[word])
                dag[i].append((j, log_prob))
    return dag

def find_all_cuts_full(sentence, dictionary):
    n = len(sentence)
    dag = build_dag(sentence, dictionary)

    # 路径记录：存储从 0 到每个节点的所有可能路径
    path_record = defaultdict(list)
    path_record[0] = [[]]  # 从起点初始化为空路径

    # 动态规划：记录每个节点的最大概率
    dp = [-float('inf')] * (n + 1)
    dp[0] = 0  # 起点概率初始化

    for i in range(n):
        for j, log_prob in dag[i]:
            # 更新路径记录，添加从 i 到 j 的路径
            for path in path_record[i]:
                path_record[j].append(path + [sentence[i:j]])

            # 更新动态规划值（最大概率路径）
            dp[j] = max(dp[j], dp[i] + log_prob)

    # 返回所有切分方式和最优概率
    return path_record[n], math.exp(dp[n])

# 示例
sentence = "经常有意见分歧"
dictionary = {"经常":0.1,
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
all_cuts, best_prob = find_all_cuts_full(sentence, dictionary)

print("所有切分方式:")
for cut in all_cuts:
    print(cut)
print("最优概率:", best_prob)
