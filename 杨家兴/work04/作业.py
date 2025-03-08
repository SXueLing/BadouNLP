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
sentence1 = '有意见'
max_length = 0 # 字典里最长的字符
for i in Dict.keys():
    max_length = max(max_length, len(i))

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []
    # 在函数里定义一个递归函数
    def getStrCut(str, result):
        # return
        if len(str) > 0:
            index = 1
            # while str[0:index] in Dict:
            while index <= max_length and index <= len(str): 
                if str[0:index] in Dict:
                    # print(str[0:index], str, index)
                    mergeList = result + [str[0:index]]
                    #递归剩下的字符，将已经找到的保存到mergeList里, 比如mergeList是['经常']， 那么第一个参数就传'有意见分歧'
                    getStrCut(str[index:len(sentence)], mergeList)
                index += 1
                
        else:
            # 遍历完最后一个字符时，result是其中一个拆分结果，添加到target里
            target.append(result)
            return
    getStrCut(sentence, [])
    return target
    
result = all_cut(sentence, Dict)
for i in result:
    print(i)

print(len(result))


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

