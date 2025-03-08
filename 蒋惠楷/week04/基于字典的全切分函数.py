#week3作业

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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
'''
方法一: 递归
'''
def Rec_cut(sentence, Dict):
    target = []

    def recursion(current_cut, start):
        # 递归终止条件
        if start == len(sentence):
            target.append(current_cut)
            return
        
        # 从start位置开始遍历字符串，尝试切分
        for end in range(start+1, len(sentence)+1):
            word = sentence[start:end]
            if word in Dict:
                recursion(current_cut + [word], end)

    recursion([], 0)
    return target

'''
方法二: 动态规划
'''
def DP_cut(sentence, Dict):
    n = len(sentence)
    dp = [[] for _ in range(n+1)]
    dp[0] = [[]]

    for i in range(1, n+1):
        # print(i)
        for j in range(0, i):
            word = sentence[j:i]
            if word in Dict:
                for prev in dp[j]:
                    dp[i].append(prev + [word])
                # print(dp[i])
    
    return dp[n]

#目标输出;顺序不重要
target = [
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

# cuts1 = Rec_cut(sentence, Dict)
# for cut in cuts1:
#     print(cut)

cuts2 = DP_cut(sentence, Dict)
for cut in cuts2:
    print(cut)

