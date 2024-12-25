#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：AIwork 
@File    ：find_all_words.py
@IDE     ：PyCharm 
@Author  ：杨帅
@Date    ：2024/12/18 星期三 21:47 
'''


# 加载词前缀词典
# 用0和1来区分是前缀还是真词
# 需要注意有的词的前缀也是真词，在记录时不要互相覆盖
def load_prefix_word_dict(path):
    prefix_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict:  # 不能用前缀覆盖词
                    prefix_dict[word[:i]] = 0  # 前缀
            prefix_dict[word] = 1  # 词
    return prefix_dict


def cut_method2(string, prefix_dict):
    if string == "":
        return []
    words = []  # 准备用于放入切好的词

    for i in range(len(string)):
        start_index = i
        end_index = i + 1
        window = string[start_index:end_index]  # 从第一个字开始

        while start_index < len(string):
            find_word = window
            if find_word not in prefix_dict or end_index > len(string):
                break
            elif prefix_dict[window] == 1:
                words.append(find_word)
                end_index += 1
                window = string[start_index:end_index]
            elif prefix_dict[window] == 0:
                end_index += 1
                window = string[start_index:end_index]

    return words


if __name__ == '__main__':
    test_data = "王羲之草书《平安帖》共有九行"
    word_dict = load_prefix_word_dict("dict.txt")
    find_words = cut_method2(test_data, word_dict)
    print(find_words)
