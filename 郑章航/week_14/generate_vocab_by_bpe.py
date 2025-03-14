import os
from collections import defaultdict
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

class BPE:
    def __init__(self, special_tokens=None):
        self.vocab = set()
        self.merge_rules = []  # 存储合并操作的优先级
        self.special_tokens = special_tokens if special_tokens else []

    def preprocess(self, text):
        # 增强预处理：移除标点符号、数字，小写、按空格分词、添加结束符
        text = text.lower()
        # 移除标点符号和数字
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        words = re.findall(r'\w+', text)
        return [word + '</w>' for word in words]

    def train(self, corpus, target_vocab_size, min_freq=5):
        start_time = time.time()  # 记录训练开始时间

        # 预处理语料库
        processed_corpus = []
        for text in corpus:
            words = self.preprocess(text)
            processed_text = ' '.join(words)
            processed_corpus.append(processed_text)

        # 使用 TfidfVectorizer 计算 TF-IDF
        vectorizer = TfidfVectorizer(min_df=min_freq)
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)
        feature_names = vectorizer.get_feature_names_out()
        word_freq = {word: tfidf_matrix[:, idx].sum() for idx, word in enumerate(feature_names)}

        # 初始化词表为所有唯一字符和特殊字符
        self.vocab = set()
        for word in word_freq:
            self.vocab.update(list(word))
        for token in self.special_tokens:
            self.vocab.add(token)

        # 构建初始拆分
        splits = {word: list(word) for word in word_freq}

        iteration = 0  # 记录迭代次数
        while len(self.vocab) < target_vocab_size:
            iteration += 1
            # 统计字符对频率
            pair_counts = defaultdict(int)
            for word, freq in word_freq.items():
                chars = splits[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break  # 无法继续合并

            # 选择最高频的字符对，但要求频率大于某个阈值
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            if pair_counts[best_pair] < 10:  # 阈值可以根据实际情况调整
                break
            self.merge_rules.append(best_pair)

            # 合并字符对并更新splits
            merged_char = ''.join(best_pair)
            self.vocab.add(merged_char)

            for word in word_freq:
                chars = splits[word]
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                        new_chars.append(merged_char)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                splits[word] = new_chars

            # 打印处理进度
            progress = len(self.vocab) / target_vocab_size * 100
            elapsed_time = time.time() - start_time
            print(f"迭代次数: {iteration}, 词表大小: {len(self.vocab)}, 进度: {progress:.2f}%, 耗时: {elapsed_time:.2f}秒")

        end_time = time.time()  # 记录训练结束时间
        total_time = end_time - start_time  # 计算总耗时
        print(f"训练完成，总耗时: {total_time:.2f}秒")

        return self.vocab, self.merge_rules

def read_file(file_path):
    """读取单个文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []

# 使用示例
if __name__ == "__main__":
    # 清华大学自然语言处理实验室的官方网站下载https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip 语料,循环读取THUCNews文件夹下面的文件写入语料
    corpus = []
    folder_path = '财经'
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

    # 使用多线程读取文件
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_file, file_paths))
        for lines in results:
            corpus.extend(lines)

    # 初始化BPE类
    bpe = BPE(special_tokens=['<unk>', '<pad>'])

    # 训练 BPE 并生成词表
    target_vocab_size = 30000  # 目标词表大小
    vocab, rules = bpe.train(corpus, target_vocab_size)

    # 按 DeepSeek-V3 模型词表样例输出词表文件
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for i, token in enumerate(sorted(vocab)):
            f.write(f'{token} {i}\n')

    print("词表文件已生成：vocab.txt")
