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
#待切分文本
sentence = "经常有意见分歧"
#实现全切分函数，输出根据字典能够切分出的所有的切分方式

# 正向
def cut_method1(string, word_dict, max_len):
    words = []
    while string != '':
        lens = min(max_len, len(string))
        word = string[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1]
        words.append(word)
        string = string[len(word):]
    return words
# 反向  
# def cut_method2(string, word_dict, max_len):
#     words = []
#     while string != '':
#         lens = min(max_len, len(string))
#         word = string[-lens:]
#         while word not in word_dict:
#             if len(word) == 1:
#                 break
#             word = word[-lens-1:]
#         words.append(word)
#         string = string[len(word):]
#     return words

def all_cut(sentence, Dict):
    word_dict = [key for key in Dict]
    word_dict = sorted(word_dict, key=lambda x:len(x), reverse=True)
    max_word_length = len(word_dict[0])
    target = []

    result = cut_method1(sentence, word_dict, max_word_length)
    # result2 = cut_method2(sentence, word_dict, max_word_length)
    target.append(result)
    cut_number = len(result)
    
    while max_word_length > 1:
        cut_number = cut_number - 1
        result_sub = []
        result_sub2 = []

        if(cut_number == 0):
            result = target[len(target)-1]
            cut_number = len(result)-1

        for i in range(len(result)):
            max_len = len(result[i])
            if(i == cut_number):
                max_len = max_len-1
            result_sub += cut_method1(result[i], word_dict, max_len)
            # result_sub2 += cut_method2(result[i], word_dict, max_len)
        target.append(result_sub)
        # target.append(result_sub2)
    print(target)   
    return target

all_cut(sentence, Dict)
