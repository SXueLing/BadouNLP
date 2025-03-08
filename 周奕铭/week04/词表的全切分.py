def full_segmentation(sentence, word_dict):
    result = []
    if len(sentence) == 0:
        return [[]]
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in word_dict:
            # 递归处理剩余的句子
            remaining_segments = full_segmentation(sentence[i:], word_dict)
            for segments in remaining_segments:
                result.append([word] + segments)
    return result

# 示例词表
word_dict = {"我们", "是", "中国人", "中国", "人"}
# 示例句子
sentence = "我们是中国人"

# 进行全切分
segmentations = full_segmentation(sentence, word_dict)

# 输出所有可能的分词结果
for seg in segmentations:
    print(seg)
