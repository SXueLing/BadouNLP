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

sentence = "经常有意见分歧"

def word_segment(sentence, index=0, current_result=[]):
    """
    对句子进行全切分的函数
    :param sentence: 待切分的句子
    :param index: 当前处理的字符索引
    :param current_result: 当前已经切分出来的结果列表
    :return: 所有可能的切分结果列表
    """
    if index == len(sentence):
        return [current_result]
    results = []
    for length in range(1, len(sentence) - index + 1):
        word = sentence[index:index + length]
        if word in Dict:
            sub_results = word_segment(sentence, index + length, current_result + [word])
            results.extend(sub_results)
    return results


target = word_segment(sentence)
for result in target:
    print(result)
