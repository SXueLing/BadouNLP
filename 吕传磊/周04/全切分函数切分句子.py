# 定义字典，这里使用集合来存储，方便快速判断元素是否存在
word_dict = {
    "一切",
    "都会",
    "好",
    "会好",
    "会好的",
    "好的",
    "我",
    "我一直",
    "一直相信",
    "一直",
    "相信",
    "我一直相信"
}

word_dict2 = {
    "待",
    "待我",
    "苦涩",
    "苦涩的",
    "苦涩的眼睛",
    "眼睛",
    "长满",
    "春天",
    "我",
    "我再",
    "向你",
    "你",
    "诉说",
    "所见",
    "我所见的",
    "所见的",
    "的",
    "盎然",
    "我所见的的盎然"
}

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

sentence1 = "一切都会好的我一直相信"

sentence2 = "待我苦涩的眼睛长满春天我再向你诉说我所见的盎然"

sentence3 = "经常有意见分歧"

def segmentation(sentence, word_dict):
    result = []
    if sentence == "":
        return [[]]
    for end in range(1, len(sentence) + 1):
        word = sentence[:end]
        if word in word_dict:
            # 递归获取剩余字符串的切分结果
            sub_results = segmentation(sentence[end:], word_dict)
            for sub_result in sub_results:
                result.append([word] + sub_result)
    return result

all_cuts1 = segmentation(sentence1, word_dict)
all_cuts2 = segmentation(sentence2, word_dict2)
all_cuts3 = segmentation(sentence3, Dict)

for word in all_cuts1:
    print(" ".join(word))

print("————————————————————————————————————————")

for cut2 in all_cuts2:
    print(" ".join(cut2))

print("————————————————————————————————————————")

for cut3 in all_cuts3:
    print(" ".join(cut3))
