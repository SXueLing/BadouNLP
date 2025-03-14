"""
实现bpe的词表构建
"""

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def build_merges(file_content, num_merges,):
    
    ids = list(file_content.encode("utf-8"))
    merges = {} # (int, int) -> int

    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges
        
        

def encode(text, merges): ##传入需要encode的语句和bpe构建的词表
##  # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens




def decode(ids,merges):  ##传入需要decode的语句和bpe构建的词表
    
    vocab = {xx: bytes([xx]) for xx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


        
def read_txt(file_path):

    # 使用 with 语句自动处理文件关闭
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取全部内容到字符串变量
            file_content = file.read()

        # 打印验证（可选）
        print("文件内容：\n", file_content)

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
    return file_content


if __name__ == "__main__":

    vocab_size = 300   # the desired final vocabulary size 超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256   ##  计算需要扩充的数量

    # 定义文件路径
    file_path = './dazhou.txt' #
    file_content = read_txt(file_path)##读取语料
    print("file_content:", file_content)

    merges = build_merges(file_content, num_merges)##建立merges后的词表
    print("merges", merges)

    tokenz = encode("你是谁，我是谁，ta_shi_shei", merges) ## 根据merges对文本进行编码
    print(tokenz)
    
    yuan_text = decode(tokenz, merges) ##解码回到原来文本
    print(yuan_text)




