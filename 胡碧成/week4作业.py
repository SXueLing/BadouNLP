
def fully_segment(text, dictionary):
    results = []
    n = len(text)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = text[i:j]
            if substring in dictionary:
                results.append(substring)
    def backtrack(start, path, res):
        if start == n:
            res.append(' '.join(path))
            return
        for word in results:
            if text.startswith(word, start):
                end = start + len(word)
                backtrack(end, path + [word], res)

    all_segmentations = []
    backtrack(0, [], all_segmentations)
    return all_segmentations

dictionary = {"经常":0.1,
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
text = "经常有意见分歧"

# 调用全切分函数
segmentations = fully_segment(text, dictionary)

for segmentation in segmentations:
    print(segmentation)
