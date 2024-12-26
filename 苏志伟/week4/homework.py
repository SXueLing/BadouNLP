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
    target = []
    max_word_len = 3
    def get_word(s, t=[]):
        if len(s) == 0:
            target.append(t)
            return
        for i in range(1, max_word_len+1):
            if len(s) >= i:
                w = s[:i]
                if w in Dict.keys():
                    get_word(s[i:], t + [s[:i]])
    get_word(sentence)
    return target
targets = all_cut(sentence, Dict)
targets = sorted(targets, reverse=True)
for t in targets:
    print(t)
