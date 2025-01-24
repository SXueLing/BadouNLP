#week3作业
import numpy as np
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

print('经常' in Dict, '-----------------')
#待切分文本
sentence = "经常有意见分歧"
def getMaxLenKey() :
    return max(len(key) for key in Dict.keys())
def isWord(word):
    return word in Dict

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    res = []
    maxLen = getMaxLenKey()
    def traceback(path, start, totalLen):
        for i in range(start, min(totalLen, start + maxLen)):
            temp = sentence[start:i + 1]
            tempIsWord = isWord(temp)
            # print(temp, tempIsWord)
            if(tempIsWord):
                path.append(sentence[start:i + 1])
                if(i+1 == totalLen):
                    res.append(path[:])
                else:
                    traceback(path, i + 1, totalLen)
                path.pop()

    traceback([], 0, len(sentence))
    return res


result = all_cut(sentence, Dict)

print(result)

print('[')
for row in result:
    print(' [', ", ".join(map(str, row)), ']')
print(']')
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

