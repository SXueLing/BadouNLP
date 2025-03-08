#week3作业
import  torch
import numpy
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


# list_ = [1,2,3,4,5]
# for index,i in enumerate(list_):
#     # print(list_)
#     print(index,i)
#     if index%2 ==1:
#         del list_[index]
# print(list_)

# print(type(Dict.keys()))
# print(type(list(Dict.keys())))
# print(type(Dict))

# print(max(list(map(len,list(Dict.keys())))))
# print(Dict['分'])
# print(Dict['12345'])

# print(Dict.get('分'))
# print(Dict.get('12345'))

# print('123/456/45'.rfind('/'))
# print('123/456/45'['123/456/45'.rfind('/')+1  : '123/456/45'.rfind('/')+1 +1])
a = '123/456/45'
# a = a[:a.rfind('/')+1+1]+'/'+a[a.rfind('/')+1+1:]
# print(a)
# print(a.rfind('/'))
# print(a[a.rfind('/'):])


#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    key_list = list(Dict.keys())
    max_length = max(list(map(len,key_list)))
    list_cir_start = []
    list_cir_end = []
    list_result = []
    list_cir_start.append(sentence)
    i = 0
    while len(list_cir_start)>0:
        i+=1
        # print(i)
        # if i%10 == 0:
            # print(list_result)
            # print(list_cir_start)
        for i_str in list_cir_start:
            # print(i_str)
            for lenth in range(max_length):
                lenth = lenth+1
                index = i_str.rfind('/') # if i_str.rfind('/') != -1 else 0 #这句是有问题的
                if index == -1:
                    if len(i_str) == lenth:
                        list_result.append(i_str)
                    if len(i_str) > lenth:
                        if (Dict.get(i_str[0: lenth])):
                            list_cir_end.append(i_str[:lenth] + '/' + i_str[lenth:])
                else:
                    # 这里要区分 index 是否是-1
                    if len(i_str[index+1:]) == lenth:
                        list_result.append(i_str)
                    if len(i_str[index+1:]) > lenth:
                        if(Dict.get(i_str[index+1: index+1+lenth])):
                            list_cir_end.append(i_str[:index+1+lenth]+'/'+i_str[index+1+lenth:])
        list_cir_start = list_cir_end
        list_cir_end = []
    return list_result
str_list = all_cut(sentence, Dict)
list_list = [s.split('/') for s in str_list]
# print(numpy.array(list_list))
for s in list_list:
    print(s)
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

