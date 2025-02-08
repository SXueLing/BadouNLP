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

def all_cut(sentence, Dict):
    if not sentence:
        return [[]]
    result = []
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in Dict or not Dict:
            sub_cuts = all_cut(sentence[i:], Dict)
            for sub_cut in sub_cuts:
                result.append([word] + sub_cut)
    return result


cuts = all_cut(sentence, Dict)
for cut in cuts:
    print("/".join(cut))
