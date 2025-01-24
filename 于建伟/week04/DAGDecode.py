#用文本展示出所有切分方式
class DAGDecode:
    def __init__(self, sentence):
        self.sentence = sentence
        self.length = len(sentence)
        self.finish_path = []

    def decode_next(self, path, start):
        if start == self.length:
            self.finish_path.append(",".join(path))
            return
        for i in range(start, self.length):
            path.append(self.sentence[start:i + 1])
            self.decode_next(path, i + 1)
            path.pop()
        return

    # 递归调用序列解码过程
    def decode(self):
        self.decode_next([], 0)  # 使用该序列进行解码
sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)
