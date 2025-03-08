import copy
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

def all_cut(sentence, Dict):
    list1=[]
    target=[]
    for i in range(1,len(sentence)+1):
        sen2=sentence[:i]
        sentence_len=len(sen2)
        if sen2 in Dict:
            if i == len(sentence):
                list1.append(sen2)
                target.append(list1)
                list1 = []
            else:
                for j in range(sentence_len+1,len(sentence)+1):
                    sen3 = sentence[i:j]
                    sentence_len = len(sen2)+len(sen3)
                    if sen3 in Dict:
                        if j == len(sentence):
                            list1.append(sen2)
                            list1.append(sen3)
                            target.append(list1)
                            list1 = []
                        else:
                            for k in range(sentence_len+1, len(sentence) + 1):
                                sen4 = sentence[j:k]
                                sentence_len = len(sen2) + len(sen3) + len(sen4)
                                if sen4 in Dict:
                                    if k == len(sentence):
                                        list1.append(sen2)
                                        list1.append(sen3)
                                        list1.append(sen4)
                                        target.append(list1)
                                        list1 = []
                                    else:
                                        for q in range(sentence_len + 1, len(sentence) + 1):
                                            sen5 = sentence[k:q]
                                            sentence_len = len(sen2) + len(sen3) + len(sen4) + len(sen5)
                                            if sen5 in Dict:
                                                if q==len(sentence):
                                                    list1.append(sen2)
                                                    list1.append(sen3)
                                                    list1.append(sen4)
                                                    list1.append(sen5)
                                                    target.append(list1)
                                                    list1 = []
                                                else:
                                                    for w in range(sentence_len + 1, len(sentence) + 1):
                                                        sen6 = sentence[q:w]
                                                        sentence_len = len(sen2) + len(sen3) + len(sen4) + len(sen5) + len(sen6)
                                                        if sen6 in Dict:
                                                            if w==len(sentence):
                                                                list1.append(sen2)
                                                                list1.append(sen3)
                                                                list1.append(sen4)
                                                                list1.append(sen5)
                                                                list1.append(sen6)
                                                                target.append(list1)
                                                                list1 = []
                                                            else:
                                                                for e in range(sentence_len + 1, len(sentence) + 1):
                                                                    sen7 = sentence[w:e]
                                                                    sentence_len = len(sen2) + len(sen3) + len(
                                                                        sen4) + len(sen5) + len(sen6) + len(sen7)
                                                                    if sen7 in Dict:
                                                                        if e == len(sentence):
                                                                            list1.append(sen2)
                                                                            list1.append(sen3)
                                                                            list1.append(sen4)
                                                                            list1.append(sen5)
                                                                            list1.append(sen6)
                                                                            list1.append(sen7)
                                                                            target.append(list1)
                                                                            list1 = []
                                                                        else:
                                                                            for r in range(sentence_len + 1, len(sentence) + 1):
                                                                                sen8 = sentence[e:r]
                                                                                if sen8 in Dict:
                                                                                    if r == len(sentence):
                                                                                        list1.append(sen2)
                                                                                        list1.append(sen3)
                                                                                        list1.append(sen4)
                                                                                        list1.append(sen5)
                                                                                        list1.append(sen6)
                                                                                        list1.append(sen7)
                                                                                        list1.append(sen8)
                                                                                        target.append(list1)
                                                                                        list1 = []
    return target

target = all_cut(sentence, Dict)

print(target)
# #目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]
