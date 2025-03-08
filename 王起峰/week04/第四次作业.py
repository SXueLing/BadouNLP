Dict = {
    "经常": 0.1,
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
    "分": 0.1
}

sentence = "经常有意见分歧"

def all_cut(sentence, Dict):
    result = []

    def cut_from_index(current_index, current_cut):
        if current_index == len(sentence):
            result.append(current_cut)
            return
        for end_index in range(current_index + 1, len(sentence) + 1):
            word = sentence[current_index:end_index]
            if word in Dict:
                cut_from_index(end_index, current_cut + [word])

    cut_from_index(0, [])
    return result

result = all_cut(sentence, Dict)

# 输出结果
for r in result:
    print(r)
