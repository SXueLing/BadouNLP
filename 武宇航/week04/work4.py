def generate_partitions(sentence, start=0, path=[]):

    if start == len(sentence):
        yield path
        return

    # 尝试从当前位置开始的所有可能的词长度
    for end in range(start + 1, len(sentence) + 1):
        # 将当前词添加到路径中
        yield from generate_partitions(sentence, end, path + [sentence[start:end]])

# 示例句子

if __name__ == "__main__":
    sentence = "我爱好电影1abc"
    # 生成所有可能的切分
    for partition in generate_partitions(sentence):
        print(" ".join(partition))
