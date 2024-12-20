# 定义分词字典
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

word_dict = Dict.keys()

# 待分词的文本
text = "经常有意见分歧"

# 用于存放分词结果的列表
result = []
i = 0
while i < len(text):
    found_word = False
    for length in range(len(text) - i, 0, -1):
        current_word = text[i:i + length]
        if current_word in word_dict:
            result.append(current_word)
            i += length
            found_word = True
            break
    if not found_word:
        i += 1

print(result)
