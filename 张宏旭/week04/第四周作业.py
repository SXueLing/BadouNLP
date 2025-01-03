def all_cut(sentence, Dict):
    def backtrack(start, path):
        # 如果已经到达句子的末尾，将当前路径添加到结果集中
        if start == len(sentence):
            results.append(path[:])
            return

        # 从当前起点开始，尝试每一种可能的切分
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict: 
                path.append(word)
                backtrack(end, path)
                path.pop()  

    results = []
    backtrack(0, [])
    return results

# 测试
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

sentence = "经常有意见分歧"
result = all_cut(sentence, Dict)
for r in result:
    print(r)
