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

#待切分文本
sentence = "经常有意见分歧"

word_list = []
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence,Dict,index, word_cut):

    if len(sentence) == index:
        word_cut_copy = word_cut.copy()
        word_list.append(word_cut_copy)
        return

    for i in range(index,len(sentence)):
        while index <= len(sentence)+1:
            word = sentence[i : index]
            if word in Dict:
                word_cut.append(word)
                all_cut(sentence,Dict,index,word_cut)
                word_cut.pop()
            index += 1
    return word_list

if __name__ == "__main__":
    word_cut = []
    my_target = all_cut(sentence, Dict, 0, word_cut)
    print(my_target[::-1])
