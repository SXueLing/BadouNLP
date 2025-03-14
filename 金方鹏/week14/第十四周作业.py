import re
from collections import defaultdict
import os

def merge_txt_files(input_dir, output_file):
    """
    将指定目录下的所有.txt文件合并为一个语料文件

    参数:
        input_dir (str): 包含.txt文件的目录路径
        output_file (str): 合并后的输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历输入目录
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(input_dir, filename)
                try:
                    # 读取并写入每个文件内容
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n')  # 文件间添加换行符
                    print(f"已合并文件: {filename}")
                except Exception as e:
                    print(f"无法处理文件 {filename}: {str(e)}")

    print(f"所有文件已合并到: {output_file}")




class BPE:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.vocab = defaultdict(int)
        self.merges = {}  # 合并规则
        self.vocab_ids = {}  # 子词到ID的映射
        self.id_to_vocab = {}  # ID到子词的映射
        self.pattern = re.compile(r"(\w+|\S)", re.UNICODE)
        self.unk_id = 0  # 未知符号ID

    def preprocess(self, text):
        """添加单词结束符"""
        return [' '.join(list(word) + ['</w>']) for word in text.split()]

    def get_vocab(self, corpus):
        """构建初始词汇表"""
        vocab = defaultdict(int)
        for text in corpus:
            preprocessed = self.preprocess(text)
            for word in preprocessed:
                vocab[' '.join(word.split())] += 1
        return vocab

    def get_pairs(self, word):
        """获取相邻字符对"""
        symbols = word.split()
        return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

    def merge_vocab(self, best_pair, vocab):
        """合并指定字符对"""
        new_vocab = {}
        bigram = re.escape(' '.join(best_pair))
        replacement = ''.join(best_pair)
        for word in vocab:
            new_word = re.sub(r'(?<!\S)' + bigram + r'(?!\S)', replacement, word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def build_vocab_mapping(self):
        """构建子词到数字ID的映射"""
        token_freq = defaultdict(int)

        # 统计所有子词频率
        for word, freq in self.vocab.items():
            tokens = word.split()
            for token in tokens:
                token_freq[token] += freq

        #print(token_freq)
        # 添加特殊标记
        special_tokens = {'<UNK>': 0, '<PAD>': 1}
        current_id = len(special_tokens)

        # 按频率排序
        sorted_tokens = sorted(
            token_freq.items(),
            key=lambda x: (-x[1], x[0])
        )
        print(sorted_tokens)
        # 构建映射字典
        self.vocab_ids = {**special_tokens}
        self.id_to_vocab = {v: k for k, v in special_tokens.items()}

        for token, _ in sorted_tokens:
            if token not in self.vocab_ids:
                self.vocab_ids[token] = current_id
                self.id_to_vocab[current_id] = token
                current_id += 1

    def bpe_train(self, corpus_path):
        # 读取语料
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.readlines()

        # 初始化词汇表
        self.vocab = self.get_vocab(corpus)

        # 执行合并操作
        for i in range(self.num_merges):
            pairs = defaultdict(int)

            # 统计字符对频率
            for word, freq in self.vocab.items():
                for pair in self.get_pairs(word):
                    pairs[pair] += freq

            if not pairs:
                break

            # 选择最高频字符对
            best_pair = max(pairs, key=lambda x: (pairs[x], -len(x)))
            self.merges[best_pair] = i  # 记录合并顺序
            self.vocab = self.merge_vocab(best_pair, self.vocab)

        # 构建数字映射表
        self.build_vocab_mapping()

    def encode_word(self, word):
        """编码单个单词为子词"""
        word = ' '.join(list(word) + ['</w>'])
        while True:
            pairs = self.get_pairs(word)
            if not pairs:
                break

            # 根据合并顺序选择最佳对
            best_pair = None
            min_rank = float('inf')
            for pair in pairs:
                if pair in self.merges:
                    if self.merges[pair] < min_rank:
                        best_pair = pair
                        min_rank = self.merges[pair]

            if not best_pair:
                break

            word = word.replace(' '.join(best_pair), ''.join(best_pair))

        return word.split()

    def encode(self, text):
        """将文本编码为数字序列"""
        words = text.split()
        encoded = []
        for word in words:
            for subword in self.encode_word(word):
                encoded.append(self.vocab_ids.get(subword, self.unk_id))
        return encoded

    def decode(self, ids):
        """将数字序列解码为文本"""
        tokens = [self.id_to_vocab.get(id, '<UNK>') for id in ids]
        text = ' '.join(tokens)
        text = text.replace('</w>', ' ')

        # 逆合并操作
        for pair in sorted(self.merges.keys(), key=lambda x: -self.merges[x]):
            merged = ''.join(pair)
            text = text.replace(merged, ' '.join(pair))

        return text.strip()


def save_model(bpe, model_path):
    """保存模型（包含映射表）"""
    model = {
        'num_merges': bpe.num_merges,
        'merges': {f"{a}|{b}": rank for (a, b), rank in bpe.merges.items()},
        'vocab_ids': bpe.vocab_ids,
        'id_to_vocab': {str(k): v for k, v in bpe.id_to_vocab.items()}
    }
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)


def load_model(model_path):
    """加载模型"""
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)

    bpe = BPE(num_merges=model['num_merges'])

    # 恢复合并规则
    bpe.merges = {
        tuple(k.split('|')): v for k, v in model['merges'].items()
    }

    # 恢复映射表
    bpe.vocab_ids = model['vocab_ids']
    bpe.id_to_vocab = {int(k): v for k, v in model['id_to_vocab'].items()}

    return bpe


if __name__ == '__main__':
    # 合并语料
    input_directory = "../RAG/dota2英雄介绍-byRAG/Heroes"  # 替换为你的.txt文件目录
    output_corpus = os.path.abspath("bpe_Dota_corpus.txt")  # 替换为输出文件路径

    # 调用合并函数
    #merge_txt_files(input_directory, output_corpus)

    # 初始化BPE训练器
    bpe = BPE(num_merges=500)

    # 训练BPE模型（假设语料文件为corpus.txt）
    corpus_path = "bpe_Dota_corpus.txt"
    bpe.bpe_train(corpus_path)
    save_model(bpe, "bpe_model.json")
    #print(bpe.vocab)


    # 保存加载模型测试

    loaded_bpe = load_model("bpe_model.json")

    # 编码示例
    text = "技能是无敌斩"
    encoded = loaded_bpe.encode(text)
    print("编码结果:", encoded)

    # 解码示例
    decoded = loaded_bpe.decode(encoded)
    print("解码结果:", decoded)

    #print(loaded_bpe)
