#week3作业
import  copy

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
def count_words(list):
    total = 0
    for element in list:
        total += len(element)
    return total

def longest_word(list):
    maxi = 0
    for i in list:
        maxi = max(maxi, len(i))
    return maxi

def main(result_final,result):
    for a in range(len(result_final)):
        for i in range(1,min(4,len(result_final[a][-1])+1)):
            word = result_final[a][-1][:i]
            if word in Dict:
                pool = result_final[a][:-1]
                pool.append(word)
                if count_words(pool) != len(sentence):
                    pool.append(sentence[count_words(pool):])
                result.append(pool)

    return result_final, result

def all_cut(sentence, Dict):
    result_final = []
    result = []
    for i in range(1,4):
        word = sentence[:i]
        if word in Dict:
            pool = [word, sentence[i:]]
            result_final.append(pool)

    while longest_word(result_final) != len(sentence):
        main(result_final,result)
        result_final = copy.deepcopy(result)
        result.clear()

    target = result_final

    unique_target = [list(x) for x in set(tuple(x) for x in target)]

    return unique_target

print(all_cut(sentence, Dict))
print("当前list长度：",len(all_cut(sentence, Dict)))
# print(longest_word(all_cut(sentence, Dict)))

#目标输出;顺序不重要
test_target = [
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

print("目标长度：", len(test_target))
