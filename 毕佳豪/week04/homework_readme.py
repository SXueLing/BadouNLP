#week4作业
import jieba
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
    # target = jieba.cut(sentence)
    # print("/".join(target))

    # 存放所有集合
    all_cut_list = []
    # 获取词表所有的key
    dict_keys = Dict.keys()
    # 递归调用
    package_date(0, sentence, dict_keys, all_cut_list, [])
    return all_cut_list

def package_date(index, sentence, dict_keys, all_cut_list, cut_list):
    # 索引等于文本长度时加入集合中
    if index == len(sentence):
        all_cut_list.append(cut_list)
        return

    for key in range(index, len(sentence)):
        sentence_str = sentence[index : key+1]
        if sentence_str in dict_keys:
            newcut_list = cut_list.copy()
            newcut_list.append(sentence_str)
            package_date(key+1, sentence, dict_keys, all_cut_list, newcut_list)

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

if __name__ == '__main__':
    value = all_cut(sentence, Dict)
    print(value)