import os

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)

script_dir = os.path.dirname(script_path)

os.chdir(script_dir)

# print(os.getcwd())

class bpe():
    def __init__(self, times=1, path="Heroes"):
        self.text = self.load_hero_data(path)
        self.token = self.text.encode("utf-8")
        self.token = list(map(int, self.token))
        # self.generate_state()
        # self.bp = sorted(self.state, key = lambda x: self.state[x], reverse=True)
        begin = 256

        for _ in range(times):
            self.generate_state()
            self.bp = sorted(self.state, key = lambda x: self.state[x], reverse=True)
            top_pair = self.bp[0]
            self.token = self.merge(self.token, top_pair, begin)
            print(f"merging {top_pair} into a new token {begin}, length: {len(self.token)}")
            begin += 1

    def load_hero_data(self, folder_path):
        self.hero_data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    self.hero_data.append(intro)
        self.hero_data = "\n".join(self.hero_data)
        return self.hero_data
    
    def generate_state(self):
        self.state = {}
        for pair in zip(self.token[:-1], self.token[1:]):
            self.state[pair] = self.state.get(pair, 0) + 1

    def merge(self, ids, top_pair, pad):
        newids = []
        index = 0
        while index < len(ids):
            if index < len(ids) - 1 and ids[index] == top_pair[0] and ids[index + 1] == top_pair[1]:
                newids.append(pad)
                index += 2
            else:
                newids.append(ids[index])
                index += 1
        return newids


bpe_ = bpe(20)
# print(bpe_.text)
# print(bpe_.token)
# print(bpe_.state)
# print(sorted(bpe_.state, key = lambda x: bpe_.state[x], reverse=True))
# print(bpe_.state[bpe_.bp[1]])
# print(bpe_.newtokens)
# print("length:", len(bpe_.newtokens))
