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
sentence2 = "意见经常有分歧"
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    target = [] # 存储结果
    backtracking(sentence, Dict, target, [], 0)
    return target

def backtracking(sentence, Dict, target, path, start):  # 回溯函数
    if start == len(sentence): # 切分到句子末尾时
        target.append(path[:]) #添加每一次切分结果
        return
    # 从 start 开始尝试不同长度的子串
    for end in range(start + 1, len(sentence) + 1): #end 切分结束标志
        word = sentence[start:end]  # 获取子串
        if word in Dict:  # 如果子串在字典中
            path.append(word)  # 选择当前子串
            backtracking(sentence, Dict, target, path, end)  # 递归切分剩余部分 上一次结束作为下一次开始 end->start
            path.pop()  # 撤销选择，回溯

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

if __name__ == '__main__':
    print(all_cut(sentence, Dict))
    print(all_cut(sentence2, Dict))
