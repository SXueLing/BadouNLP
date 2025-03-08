#week3作业
import copy

from AI.week4.homework_readme import all_cut

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
    max_length = 0
    sentence_group = [[sentence]]
    for each_world in Dict:
        if len(each_world) > max_length:
            max_length = len(each_world)
    target= []
    while True:
        if sentence_group == []:
            break
        for each in sentence_group:
            sentence_to_be_delete = copy.deepcopy(each)
            for i in range(max_length):
                word_already_cut = each[:-1]
                cut_word = each[-1][:i+1]
                remain_word = each[-1][i+1:]
                if remain_word == '':
                    finish_word_list = word_already_cut
                    finish_word_list.append(cut_word)
                    target.append(finish_word_list)
                    break
                if cut_word in  Dict :
                    new_sentence_list = word_already_cut
                    new_sentence_list.append(cut_word)
                    new_sentence_list.append(remain_word)
                    sentence_group.append(new_sentence_list)
                # elif cut_word in Dict and remain_word == []:
                #     sentence_group.append( remain_word, each[-1][i:])
            # for each in sentence_before:
            sentence_group.remove(sentence_to_be_delete)
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

if __name__ == '__main__':
    target2 = all_cut(sentence,Dict)
    for each in target2:
        print(each)
