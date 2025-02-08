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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    return target

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
# 先找到
#递归思路 ：当前层的递归：取大分词，然后剩余的用递归，本层是个循环，然后是次最大分词，剩下取递归，本层到最小分词的时候返回
#递归终止判断，当传入的是一个最小分词的时候返回
def get_longest_divide(sentence, maxLen):
    if len(sentence) < maxLen:
        return sentence
    str_fit = ''
    for length in range(maxLen, 0, -1):  # 从最大长度开始尝试匹配
        word = sentence[:length]
        if word in Dict:
            return word
    # for i in range(maxLen):
    #     str_cut = sentence[:i]
    #     if str_cut in Dict:
    #         str_fit = str_cut
    return ''

def all_cut(sentence, Dict):
    #TODO
    target1 = []
    max_len_word = 0
    for key in Dict.keys():
        if len(key) > max_len_word:
            max_len_word = len(key)

    for i in range(1,max_len_word + 1):
        # 获取符合长度限制的最长匹配词
        str_fit = get_longest_divide(sentence[:i], i)
        if len(str_fit) != i:
            continue
        if str_fit == sentence:
            return target1 + [[sentence]]
        # 剩余部分递归
        arr_result = all_cut(sentence[len(str_fit):], Dict)
        for elem in arr_result:
            elem.insert(0, str_fit)
            target1.append(elem)
        # if len(arr_result) > 0:
        #     target1.append(arr_result)

    return target1


def main():
    arr_result = all_cut(sentence, Dict)
    for elem in arr_result:
        print(elem)

    return

if __name__ == '__main__':
    main()
