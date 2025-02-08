#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

#待切分文本
sentence = "经常有意见分歧"


#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    target = set_arr(sentence, Dict,0,[],[])
    return target


def set_arr(sentence, Dict,start_index,all_text_list,cur_text_list):
    counter = 1
    end_index = start_index + 1
    origin_text_list = list(cur_text_list)
    origin_text_length = start_index

    if start_index == len(sentence):
        all_text_list.append(origin_text_list)
        return all_text_list
    while origin_text_length <= len(sentence):
        text_list = list(origin_text_list)
        origin_text_length = start_index + counter
        if Dict.get(sentence[start_index:end_index]) is not None:
            text_list.append(sentence[start_index:end_index])
            all_text_list = set_arr(sentence, Dict, end_index, all_text_list, text_list)
        else:
            origin_text_length += 1
        counter += 1
        end_index += 1
    return all_text_list






#目标输出;顺序不重要

my_list = all_cut(sentence, Dict)
print(my_list)
# my_list即为最终输出

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
