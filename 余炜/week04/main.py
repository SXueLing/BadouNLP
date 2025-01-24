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
sentence = "常经有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    """
    全切分函数
    使用2进制表示虽有的结果，0代表不是边界，1代表是边界
    :param sentence:
    :param Dict:
    :return:
    """
    length = len(sentence)
    return_list  = []
    for i in range(2**length):
        if i % 2 == 0:
            continue
        binary = bin(i)[2:].zfill(length)
        start = 0
        end = 1
        words = []
        flag  = True

        for j in range(length):
            if binary[j] == '1':
                end = j+1
                if sentence[start:end] in Dict:
                    words.append(sentence[start:end])
                    start = end

                else:
                    flag = False
                    break
        if flag:
            return_list.append(words)
    return return_list

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
result = all_cut(sentence, Dict)
for i in result:
    print(i)
