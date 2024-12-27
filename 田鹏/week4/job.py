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
'''
找到词表中最大词的长度和最小词的长度
截取字符串中第一个最大的数据和词表中的进行对比，如果匹配，则前几个字符的匹配数据算一类方式，
                                               余下的字符串与词表中的继续匹配。
                                  如果不匹配。则减少匹配数据，逐渐到最小字符串
                                  如果都不匹配，则进行截取，跳转到下一个字符
递归函数
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
'''

target = []
# index_num = 0
list_exc = []
for _, i in enumerate(Dict):
    list_exc.append(i)
max_num = max(len(i) for i in list_exc)
min_num = min(len(i) for i in list_exc)
target_list = [[]]
# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    while True:
        index_num = 0
        for i in range(max_num, min_num-1, -1):
            str_exc = sentence[:i]
            # print(i, str_exc)
            if str_exc in list_exc or i == (min_num-1):
                target_list[index_num].append(str_exc)
                sentence = sentence[i:]
            else:
                sentence = sentence
        if len(sentence) != 0:
            return all_cut(sentence, Dict)
        else:
            return target_list

a = all_cut(sentence, Dict)

print(a)
#目标输出;顺序不重要
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

