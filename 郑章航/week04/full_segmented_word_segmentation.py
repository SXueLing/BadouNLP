# 词典示例，实际使用中可替换为真实完整的词典数据
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


def all_cut(sentence, Dict):
    """
    使用滑动窗口方式，依据词典里词的长度对句子进行全切分
    :param sentence: 待切分的句子字符串
    :param Dict: 词典，以字典形式存储词语及其对应的词频（词频此处暂不实际使用）
    :return: 包含所有切分方式的列表，每个切分方式为一个词语列表
    """
    results = []
    word_lengths = sorted(set(len(word) for word in Dict), reverse=True)  # 获取词典中词的长度集合并排序
    print("词典中词的长度集合:", word_lengths)
    # 用于存储待处理的节点，每个节点包含当前切分结果和剩余待切分的句子
    nodes = [([], sentence)]
    print("初始节点:", nodes)
    while nodes:
        current_result, remaining_sentence = nodes.pop()
        print("当前处理节点 - 已切分结果:", current_result, " 剩余句子:", remaining_sentence)
        if not remaining_sentence:
            results.append(current_result)
            continue
        for length in word_lengths:
            if length <= len(remaining_sentence):
                word = remaining_sentence[:length]
                if word in Dict:
                    print("找到匹配词:", word, " 新切分结果:", current_result + [word], " 新剩余句子:",
                          remaining_sentence[length:])
                    new_result = current_result + [word]
                    new_remaining = remaining_sentence[length:]
                    nodes.append((new_result, new_remaining))
                    print("添加新节点:", (new_result, new_remaining), " 当前节点列表:", nodes)
    return results


def main():
    # 待切分文本示例
    sentence = "经常有意见分歧"
    result = all_cut(sentence, Dict)
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
    for r in result:
        print(r)
    # 验证结果是否和目标结果一致（仅验证元素相同，不考虑顺序）
    result_set = [tuple(sorted(x)) for x in result]
    target_set = [tuple(sorted(x)) for x in target]
    print("验证结果是否一致:",set(result_set) == set(target_set))


if __name__ == "__main__":
    main()
