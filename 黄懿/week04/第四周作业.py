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
sentence = "常经有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def calc(sentence, Dict):
    my_dict = {}
    length = len(sentence)
    for i in range(length):
        my_list = []
        j = i
        slice = sentence[i]
        while j < length:
            if slice in Dict:
                my_list.append(j)
            j += 1
            slice = sentence[i: j + 1]
        if not my_list:
            my_list.append(i)
        my_dict[i] = my_list
    return my_dict


class Transfer:
    def __init__(self, sentence, Dict):
        self.sentence = sentence
        self.length = len(sentence)
        self.my_dict = calc(sentence, Dict)
        self.path_to_do = [[]]
        self.path_ok = []

    def transfer_path(self, path):
        path_len = len("".join(path))
        if path_len == self.length:
            self.path_ok.append(path)
            return
        values = self.my_dict[path_len]
        path_list = []
        for value in values:
            path_list.append(path + [self.sentence[path_len: value + 1]])
        self.path_to_do += path_list
        return

    def transfer(self):
        while self.path_to_do != []:
            path = self.path_to_do.pop()
            self.transfer_path(path)
        return self.path_ok


def all_cut(sentence, Dict):
    transfer = Transfer(sentence, Dict)
    target = transfer.transfer()
    return target


target = all_cut(sentence, Dict)
print(target)

# 目标输出;顺序不重要
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
