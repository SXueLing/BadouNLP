import time


# Trie树节点类
class TrieNode:
    def __init__(self):
        self.is_word = False
        self.children = {}


# Trie树类
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word


# 加载词典到Trie树
def load_prefix_word_dict(path):
    trie = Trie()
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.strip()
            trie.insert(word)
            # 也将前缀插入字典
            for i in range(1, len(word)):
                trie.insert(word[:i])
    return trie


# 打印所有可能的切分结果
def cut_method2(string, trie):
    all_results = []  # 用于保存所有切分结果

    def dfs(start_index, path):
        # 如果已经到达字符串末尾，保存当前路径
        if start_index == len(string):
            all_results.append(" / ".join(path))  # 将当前路径加入所有结果
            return

        # 尝试找到一个尽可能长的词
        for end_index in range(len(string), start_index, -1):
            word = string[start_index:end_index]
            # 调试：打印当前查找的单词
            if trie.search(word):
                # 递归调用，继续切割后续部分
                dfs(end_index, path + [word])

    # 从字符串的起始位置开始切分
    dfs(0, [])

    # 打印所有结果
    for result in all_results:
        print(result)


# 主函数
def main(cut_method, input_path, output_path):
    trie = load_prefix_word_dict("dict.txt")
    writer = open(output_path, "w", encoding="utf8")
    start_time = time.time()

    with open(input_path, encoding="utf8") as f:
        for line in f:
            print(f"处理文本: {line.strip()}")
            cut_method(line.strip(), trie)  # 打印所有切分结果
            writer.write("\n")  # 输出换行符
    writer.close()
    print("耗时：", time.time() - start_time)


string = "王羲之草书《平安帖》共有九行"
trie = load_prefix_word_dict("dict.txt")
print("所有可能的切分结果：")
cut_method2(string, trie)  # 使用递归打印所有切分结果
