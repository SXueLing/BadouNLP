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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    dict_list = []
    result = []
    flag = False
    for key in Dict.keys():
        dict_list.append(key)
    backtracking(dict_list, [], result, sentence, flag)
    return result

def backtracking(dict_list, target_list, result, sentence, flag):
    if flag:
        check_str = "经常有意见分歧"
        for j in range(len(target_list)):
            check_str = check_str.replace(target_list[j], "")
        # 去重以及检查是否符合要求，check_str最终长度为0，则说明里面的元素已经全部取出，可作为结果
        if len(check_str) == 0 and target_list not in result:
            result.append(target_list[:])
        return
    for i in range(len(dict_list)):
        if dict_list[i] in sentence:
            sentence_item = sentence.replace(dict_list[i], "")
        else:
            continue
        if len(sentence_item) == 0:
            flag = True
        target_list.append(dict_list[i])
        backtracking(dict_list, sorted(target_list), result, sentence_item, flag)
        target_list.pop()


if __name__ == '__main__':
    print(all_cut(sentence, Dict))
