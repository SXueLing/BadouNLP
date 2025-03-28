import os
import re
import jieba
from collections import Counter, defaultdict


def read_txt_files_to_string(directory):
    """
    读取指定目录下的所有 .txt 文件，并将内容合并为一个长字符串。
    :param directory: 包含 .txt 文件的目录路径
    :return: 合并后的长字符串
    """
    combined_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                text = file.read()
                combined_text += text  # 直接拼接文本内容
    return combined_text


def remove_punctuation(text):
    """
    去除中文和英文标点符号。
    :param text: 输入的文本
    :return: 去除标点符号后的文本
    """
    # 定义中文和英文标点符号的正则表达式
    punctuation_pattern = r"[^\w\s\u4e00-\u9fff]"  # 保留汉字、字母、数字和空格
    cleaned_text = re.sub(punctuation_pattern, "", text)
    return cleaned_text


def tokenize_chinese(text):
    """
    使用 jieba 对中文文本进行分词。
    :param text: 输入的中文文本
    :return: 分词后的单词列表
    """
    words = jieba.lcut(text)  # 使用 jieba 进行分词
    return words


def get_stats(vocab):
    """
    统计当前词汇表中所有相邻字符对的频率。
    :param vocab: 当前词汇表，格式为 {word: frequency}
    :return: 字符对频率统计，格式为 {(char1, char2): frequency}
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """
    合并高频字符对，更新词汇表。
    :param pair: 要合并的字符对 (char1, char2)
    :param vocab: 当前词汇表，格式为 {word: frequency}
    :return: 更新后的词汇表
    """
    new_vocab = {}
    bigram = " ".join(pair)  # 将字符对转换为字符串形式
    replacement = "".join(pair)  # 合并后的子词
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab


def learn_bpe(data, num_merges):
    """
    使用 BPE 算法学习子词词汇表。
    :param data: 训练数据，格式为 [word1, word2, ...]
    :param num_merges: 合并次数（即词汇表扩展的次数）
    :return: 子词词汇表
    """
    # 初始化词汇表：将每个单词拆分为字符序列，并在末尾添加 <EOS>
    vocab = {" ".join(list(word)) + " </w>": freq for word, freq in Counter(data).items()}

    bpe_merges = []  # 记录每次合并的字符对
    for i in range(num_merges):
        pairs = get_stats(vocab)  # 统计字符对频率
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)  # 找出频率最高的字符对
        vocab = merge_vocab(best_pair, vocab)  # 合并高频字符对
        bpe_merges.append(best_pair)  # 记录本次合并的字符对

    return bpe_merges, vocab


# 主函数
if __name__ == "__main__":
    # 指定包含 .txt 文件的目录
    data_directory = "E:\BaiduNetdiskDownload\人工智能课程\第十四周 大模型相关内容第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"

    # 读取所有 .txt 文件的内容，合并为一个长字符串
    combined_text = read_txt_files_to_string(data_directory)

    # 去除标点符号
    cleaned_text = remove_punctuation(combined_text)

    # 使用 jieba 对清洗后的中文文本进行分词
    words = tokenize_chinese(cleaned_text)

    # 设置合并次数
    num_merges = 5000  # 根据需求调整

    # 学习 BPE 词汇表
    bpe_merges, final_vocab = learn_bpe(words, num_merges)

    # 输出结果
    print("BPE Merges:")
    for i, merge in enumerate(bpe_merges):
        print(f"Merge {i + 1}: {merge}")

    print("\nFinal Vocabulary:")
    for word, freq in final_vocab.items():
        print(f"{word}: {freq}")

    # 保存 BPE 合并规则和词汇表到文件
    with open("bpe_merges.txt", "w", encoding="utf-8") as f:
        for merge in bpe_merges:
            f.write(f"{merge[0]} {merge[1]}\n")

    with open("final_vocab.txt", "w", encoding="utf-8") as f:
        for word, freq in final_vocab.items():
            f.write(f"{word} {freq}\n")