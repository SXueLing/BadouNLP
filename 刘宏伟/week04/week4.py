# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
import time

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
word_list = [[] for _ in sentence]
result = []


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


root = TreeNode('root')


# 根据词组构建树
def build_tree(remain_list, string, temp_list, node):
    i = 0
    for words in remain_list:
        i += 1
        for word in words:
            if string.startswith(word):
                temp_list.append(word)
                child = TreeNode(word)
                node.add_child(child)
                build_tree(remain_list[i:], string[len(word):len(string)], temp_list, child)


# 遍历所有从根到叶子的路径，添加到list中
def get_leaf(node, temp_list):
    # 没有子节点了，添加到结果中
    if len(node.children) == 0:
        result.append(temp_list)
    else:
        for i in range(len(node.children)):
            child = node.children[i]
            # i>0时需要删除上一个node
            if i != 0:
                temp_list.pop()
            temp_list.append(child.value)
            get_leaf(child, temp_list.copy())


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, dict):
    # 根据词的第一个字进行分组
    for i, char in enumerate(sentence):
        for word in dict:
            if word.startswith(char):
                word_list[i].append(word)

    temp_str = sentence
    build_tree(word_list, temp_str, [], root)
    get_leaf(root, [])
    for element in result:
        print(element)


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
if __name__ == '__main__':
    start_time = time.time()
    all_cut(sentence, Dict)
    end_time = time.time()
    print(f"{end_time - start_time:.6f}")
