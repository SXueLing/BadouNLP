#week4作业

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
sentence = "常经有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    
    target = []

    def backtrack(start, path):  # 回溯函数

        if start == len(sentence):  # 如果起点等于字符串末尾，表示切分完成，记录该路径
            target.append(path[:])  # 深拷贝保存
            return
        
        for end in range(start + 1, len(sentence) + 1):  # 在当前起点 start 到字符串末尾之间，枚举每个可能的结束点 end：
            word = sentence[start:end]
            if word in Dict:  # 如果 word 在字典中
                path.append(word)  # 添加到当前路径 path.append(word)
                backtrack(end, path)  # 递归搜索下一部分 backtrack(end, path)
                path.pop()  # 回溯，移除该词 path.pop()

    backtrack(0, [])

    return target

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


# 输出切分结果
result = all_cut(sentence, Dict)
for r in result:
    print(r)


