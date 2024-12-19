'''

实现全切分函数，对于一个文本，输出根据字典能够切分出的所有的切分方式
示例: 
    Dict = set(["我", "喜", "喜欢", "自然", "语言", "处理", "自然语言处理"])
    sentence = "我喜欢自然语言处理"
    输出: [['我', '喜欢', '自然语言处理'], ['我', '喜欢', '自然', '语言', '处理']]

'''

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
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

#待切分文本
sentence = "经常有意见分歧"

# 全切分函数实现: 动态规划 时间复杂度-O(n^3)
def all_cut(sentence, Dict):
    n = len(sentence)
    # dp[i] 表示从句子开始到第 i 个字符的所有可能切分方式
    dp = [[] for _ in range(n+1)]
    dp[0] = [[]]  # 空字符串有一种切分方式，即空切分
    # 遍历句子中的每个字符
    for i in range(1, n+1):
        # 遍历所有可能的切分点
        for j in range(i):
            # 如果从 j 到 i 的子串在字典中
            if sentence[j:i] in Dict:
                # 将从 0 到 j 的切分方式与当前找到的词组合
                for cut in dp[j]:
                    dp[i].append(cut + [sentence[j:i]])
    return dp[n]

#目标输出;顺序不重要
target_true = [
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

# 测试
target_cut = all_cut(sentence, Dict) #调用全切分函数
print('全切分结果为：')
print(target_cut)
print('目标输出：')
print(target_true)
# 转为集合判断是否结果一致
target_cut = [tuple(inner_list) for inner_list in target_cut]
target_true = [tuple(inner_list) for inner_list in target_true]
print('=================================================')
if set(target_cut) == set(target_true):
    print("全切分结果正确！")
else:
    print("全切分结果错误！")

# 注：也可用递归实现，时间复杂度为O(n!)
# def all_cut(sentence, Dict):
#     target = []
#     if not sentence:
#         return [[]]
#     for i in range(len(sentence)):
#         if sentence[:i+1] in Dict:
#             for word in all_cut(sentence[i+1:], Dict):
#                 target.append([sentence[:i+1]] + word)
#     return target
