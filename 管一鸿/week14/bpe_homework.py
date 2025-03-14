import re
from collections import defaultdict, Counter

"""
以下为未编码情况下进行bpe实现，用于学习思路


def load_text(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        return f.readlines()
    
def train_npe(data,num_merges=17000):
    
    #1. 对原始数据进行单词频率统计,建立vocab
    # 2. 进行BPE合并
    vocab = defaultdict(int)
    merges = defaultdict(int)
    for line in data:
        words = line.strip().split()  # ['合借']
        for word in words: 
            vocab[' '.join(word)] += 1 # {'合 借': 1})
        
    for _ in range(num_merges):
        pairs = get_stats(vocab) # ('利', '用'): 1
        if not pairs:
                break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges[best_pair] = ''.join(best_pair)  # 记录合并规则 {('银', '行'): '银行'})
    return vocab,merges

def merge_vocab(pair,vocab):
    new_vocab =  defaultdict(int)
    pair_str_temp = ' '.join(pair) # ('利', '用') -> '利 用'
    merged_pair = ''.join(pair_str_temp) # '利 用' -> 利用
    for word in vocab: # {'合 借': 1})
        new_word = word.replace(pair_str_temp,merged_pair) # 所有带有'利 用'的词表都变成‘利用’ 如‘利 用 率’ 变成‘利用 率’
        new_vocab[new_word] = vocab[word]  # 更新词表，保持词频
    return new_vocab

def get_stats(vocab):
    pairs = defaultdict(int)
    for words,freq in vocab.items(): # {'合 借': 1})
        splited = words.split() 
        for i in range(len(splited) - 1):
            pairs[splited[i],splited[i+1]] += freq # 两两字符对的频率
    return pairs

def encode(word,merges):
    word = ' '.join(word)
    for pair, merge in merges.items():
            word = word.replace(' '.join(pair), merge) # 先分再换
    return word.split()
if __name__ == "__main__":
    texts = load_text("words.txt")
    vocab,merge = train_npe(texts)

    #测试
    test_word = '没什么'
    print(encode(test_word,merge))

"""


# 读取文本并进行 UTF-8 编码
def read_and_encode(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return list(text.encode('utf-8'))  # 转换为字节列表

# 统计频率
def get_byte_pair_frequencies(byte_list):
    pairs = defaultdict(int)
    for word in byte_list:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
    return pairs

# 进行 BPE 合并
def merge_byte_pairs(byte_list, num_merges=5000):
    byte_list = list(map(tuple, byte_list))  # 确保是元组，便于哈希
    vocab = Counter(byte_list)  # 统计频率
    merges= []  # 记录合并规则

    for _ in range(num_merges):
        # 统计字节对的频率
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq

        if not pairs:
            break

        # 选择最频繁的字节对
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)  # 记录合并规则

        # 构造新的词表
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(best_pair)  # 合并字节对
                    i += 2  # 跳过合并的两个字节
                else:
                    new_word.append((word[i],))
                    i += 1
            new_vocab[tuple(sum(new_word, ()))] = freq
        vocab = new_vocab

    return vocab, merges

def apply_bpe(text, merge_rules):
    # 将文本转为 UTF-8 字节
    byte_sequence = list(text.encode("utf-8"))
    token_list = list(map(tuple, byte_sequence))  # 确保是元组，便于哈希

    for pair in merge_rules:
        new_token_list = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == pair:
                new_token_list.append(pair)  # 进行合并
                i += 2
            else:
                new_token_list.append(token_list[i])
                i += 1
        token_list = new_token_list  # 更新 token 序列

    return token_list

# 读取 & 编码
encoded_text = read_and_encode("words.txt")

# 运行 BPE
final_vocab,merges = merge_byte_pairs([encoded_text])

# 输出最终 vocab
# print(final_vocab)
test_text = "没什么"
bpe_encoded = apply_bpe(test_text, merges)
print(bpe_encoded)
