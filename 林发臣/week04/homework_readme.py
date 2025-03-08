# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # TODO
    # 使用窗口,窗口往由挪动
    result_map = {}
    for index, value in enumerate(sentence):
        start_index = index
        #start_index++的时候 窗口往右移动
        # 窗口从start_index作为开始起点，然后慢慢变大 ，从1到3
        for win in range(1, 4):
            #窗口的尾索引 end_index
            end_index = start_index + win
            #看看当前字符数量index 是否有list
            current_list = result_map.get(index)
            #窗口拿到的字符
            data = sentence[index:end_index]
            #check_result 判断窗口拿到的字符是否属于分词 不是分词跳过本次循环 然后窗口变大 继续拿字符串
            check_result = check_belong_to_dic(data)
            if not check_result:
                continue
            if not current_list:
                #当前字符数量对应的map没有任何元素 则添加一个
                new_list = [data]
                length = get_str_len(new_list)
                result_map[length] = []
                result_map[length].append(new_list)
            else:
                # 当前字符数量对应的map有多个元素  然后循环 每一个元素都要添加本次拿到的分词 ，
                # 添加成功以后 更新map  新的元素的字符数量对应新的结果集
                for item in current_list:
                    c_list = list.copy(item)
                    c_list.append(data)
                    length = get_str_len(c_list)
                    new_list = result_map.get(length)
                    if not new_list:
                        result_map[length] = []
                    result_map[length].append(c_list)
    # 移动窗口生成了个别重复数据，去重
    result_list = result_map.get(len(sentence))
    tuples = map(tuple, result_list)
    unique_tuples = set(tuples)
    or_list = [list(t) for t in unique_tuples]
    list(map(print, or_list))
    return or_list


def get_str_len(data: list):
    return sum((len(i) for i in data))


def check_belong_to_dic(data: str):
    return data in Dict


if __name__ == '__main__':
    all_cut(sentence, Dict)

# 目标输出;顺序不重要
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
