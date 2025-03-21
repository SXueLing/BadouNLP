# import re
# from collections import defaultdict
import os
# # 统计字节对的频率
# def get_stats(vocab):
#     pairs = defaultdict(int)
#     for word, freq in vocab.items():
#         symbols = list(word)
#         for i in range(len(symbols) - 1):
#             pair = symbols[i] + symbols[i + 1]
#             pairs[pair] += freq
#     return pairs
#
# def load_hero_data(folder_path):
#     hero_data = ''
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".txt"):
#             with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
#                 intro = file.read()
#                 hero_data += intro
#     return hero_data
#
# # def merge(ids, pair):
# #   # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
# #   newids = []
# #   i = 0
# #   while i < len(ids):
# #     if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
# #       newids.append(idx)
# #       i += 2
# #     else:
# #       newids.append(ids[i])
# #       i += 1
# #   return newids
# def merge_vocab(v_in,pair):
#     v_out = {}
#     for sentence, freq in v_in.items():
#         new_sentence = sentence.replace(pair[:1] + pair[1:], ''.join(pair))
#         v_out[new_sentence] = freq
#     return v_out
#
# def bpe(vocab, num_merges):
#     """
#     执行 BPE 算法
#     :param vocab: 初始词表
#     :param num_merges: 合并次数
#     :return: 最终词表
#     """
#     for i in range(num_merges):
#         pairs = get_stats(vocab)
#         if not pairs:
#             break
#         best = max(pairs, key=pairs.get)
#         # print('best==========',best , vocab)
#         print(f"merging {best} into a new token ")
#         vocab = merge_vocab(vocab ,best)
#         print('vocab==========', vocab)
#     return vocab
#
# # 示例文本
# heroData = load_hero_data('T')




# vocab = [char for char in heroData]
# print('vocab==========',vocab)
# puncs_zh = ['。', '，', '？', '！', '；', '：', '、', '（', '）', '「',
#             '」', '“', '”', '‘', '’', '《', '》', '【', '】', '…', '—', '～', '　']
# puncs_en = ['.', ',', '?', '!', ';', ':',
#             '(', ')', '"', '"', '\'', '\'', '<', '>', '[', ']', '...', '~']
# puncs = [*puncs_zh, *puncs_en]
#
# pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
# # 使用 re.split() 方法根据标点符号分割文本
# corpus = re.split(pattern, heroData.strip())
# print('corpus==========',corpus)
# # 初始化词表
# vocab = defaultdict(int)
# for sen in corpus:
#     # 统计句子
#     vocab[sen] += 1
# print('vocab==========',vocab.items())
# # # 执行 BPE 算法，合并 3 次
# num_merges = 3
# final_vocab = bpe(vocab, num_merges)
#
# print("final_vocab:", final_vocab)
# # 提取最终词表
# final_tokens = set()
# for word in final_vocab:
#     final_tokens.update(str(word))
#
# print("最终词表:", final_tokens)


import re
from collections import defaultdict

def load_hero_data(folder_path):
    hero_data = ''
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                intro = file.read()
                hero_data += intro
    return hero_data
# 统计字节对的频率
import re
from collections import defaultdict

# 统计字节对的频率
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

# 合并最频繁的字节对
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 执行 BPE 算法
def bpe(vocab, num_merges):
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

heroData = load_hero_data('T')

# vocab = [char for char in heroData]

pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
# 使用 re.split() 方法根据标点符号分割文本
corpus = re.split(pattern, heroData.strip())
print('corpus==========',corpus)
# 初始化词表
vocab = defaultdict(int)
for sentence in corpus:
    for word in sentence:
        # 在每个单字后添加空格和结束标记
        vocab[word + ' </w>'] += 1

# 执行 BPE 算法，合并 5 次
num_merges = 15
final_vocab = bpe(vocab, num_merges)

# 提取最终词表
final_tokens = set()
for word in final_vocab:
    final_tokens.update(word.split())

print("最终词表:", final_tokens)

