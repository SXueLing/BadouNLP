# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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
def all_cut(sentence, dict_value):
    def segment_helper(remaining_text, current_segmentation, all_segmentations):
        # 基本情况：如果字符串已经被处理完，则把当前分词结果添加到结果列表中
        if not remaining_text:
            all_segmentations.append(current_segmentation[:])
            return

        # 尝试所有可能的分词方式
        for i in range(1, len(remaining_text) + 1):
            word = remaining_text[:i]
            # 如果当前词在字典中，则继续递归处理剩余部分
            if word in dict_value:
                current_segmentation.append(word)
                segment_helper(remaining_text[i:], current_segmentation, all_segmentations)
                # 回溯，将当前词从当前分词结果中移除
                current_segmentation.pop()

    # 初始化存储所有分词结果的列表
    all_segmentations = []
    current_segmentation = []
    # 调用递归辅助函数开始分词
    segment_helper(sentence, current_segmentation, all_segmentations)

    # 返回所有可能的分词结果
    return all_segmentations


dict_value = list(Dict.keys())
all_segmentations = all_cut(sentence, dict_value)
for segment in all_segmentations:
    print(segment)

# 目标输出;顺序不重要
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
