# -*- coding: utf-8 -*-
# @Date    :2024-12-19 22:36:58
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text

from functools import lru_cache

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

def get_split_words(sentence:list)->list:
    ans = []
    n = len(sentence)

    @lru_cache
    def backtrack(index,path:tuple):
        if index==n:
            ans.append(list(path))
            return
        for i in range(index,n):
            temp = sentence[index:i+1]
            backtrack(i+1,path+(temp,))
    backtrack(0,())
    return ans

def filter_words_by_Dict(words:list,Dict:dict)->list:
    ans = []
    for pair in words:
        if set(pair) <= set(Dict.keys()):
            ans.append(pair)
    return ans

def all_cut(sentence, Dict):
    words = get_split_words(sentence)
    filter_words = filter_words_by_Dict(words,Dict)
    return filter_words

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
    target = all_cut(sentence,Dict)
    print(target)