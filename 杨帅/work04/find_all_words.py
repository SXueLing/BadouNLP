#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AIwork 
@File    ：find_all_words.py
@IDE     ：PyCharm 
@Author  ：杨帅
@Date    ：2024/12/18 星期三 21:47 
'''

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

# 待切分文本
sentence = "经常有意见分歧"


def all_segmentations(sentence, vocab, memo=None):
    if memo is None:
        memo = {}

    if sentence in memo:
        return memo[sentence]

    # 如果句子为空，返回一个空列表
    if not sentence:
        return [[]]

    results = []
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in vocab or len(word) == 1:  # 单个字符总是被视为合法词汇
            # 递归处理剩余部分
            for rest in all_segmentations(sentence[i:], vocab, memo):
                results.append([word] + rest)

    memo[sentence] = results
    return results


if __name__ == '__main__':
    find_words = all_segmentations(sentence, Dict)
    print(find_words)
